# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
train mpc fm
"""

import sys
import numpy as np
import time
import os
import logging
import errno

import paddle
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils
import args
import mpc_network
import process_data
import evaluate_accuracy

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

def train(args):
    """
    train
    """
    # Init MPC
    role = int(args.role)
    pfl_mpc.init(mpc_protocol_name, role, "localhost", args.server, int(args.port))

    # Input and Network
    BATCH_SIZE = args.batch_size
    FIELD_NUM = args.num_field
    FEATURE_NUM = args.sparse_feature_number + 1

    feat_idx = pfl_mpc.data(
        name='feat_idx',
        shape=[BATCH_SIZE, FIELD_NUM, FEATURE_NUM],
        lod_level=1,
        dtype="int64")
    feat_value = pfl_mpc.data(
        name='feat_value',
        shape=[BATCH_SIZE, FIELD_NUM],
        lod_level=0,
        dtype="int64")
    label = pfl_mpc.data(
        name='label', shape=[BATCH_SIZE, 1], lod_level=1, dtype="int64")
    inputs = [feat_idx] + [feat_value] + [label]

    avg_cost, predict = mpc_network.FM(args, inputs, seed=2)
    infer_program = fluid.default_main_program().clone(for_test=True)
    optimizer = pfl_mpc.optimizer.SGD(args.base_lr)
    optimizer.minimize(avg_cost)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # Prepare train data
    mpc_data_dir = "./mpc_data/"
    mpc_train_data_dir = mpc_data_dir + 'train/'
    if not os.path.exists(mpc_train_data_dir):
        raise ValueError("{} is not found. Please prepare encrypted data.".
                         format(mpc_train_data_dir))
    feature_idx_reader = mpc_du.load_shares(
        mpc_train_data_dir + "criteo_feature_idx",
        id=role,
        shape=(FIELD_NUM, FEATURE_NUM))
    feature_value_reader = mpc_du.load_shares(
        mpc_train_data_dir + "criteo_feature_value",
        id=role,
        shape=(FIELD_NUM, ))
    label_reader = mpc_du.load_shares(
        mpc_train_data_dir + "criteo_label", id=role, shape=(1, ))

    batch_feature_idx = mpc_du.batch(
        feature_idx_reader, BATCH_SIZE, drop_last=True)
    batch_feature_value = mpc_du.batch(
        feature_value_reader, BATCH_SIZE, drop_last=True)
    batch_label = mpc_du.batch(label_reader, BATCH_SIZE, drop_last=True)

    loader = fluid.io.DataLoader.from_generator(
        feed_list=[feat_idx, feat_value, label], capacity=BATCH_SIZE)
    batch_sample = paddle.reader.compose(batch_feature_idx,
                                         batch_feature_value, batch_label)
    loader.set_batch_generator(batch_sample, places=place)

    # Training
    logger.info('******************************************')
    logger.info('Start Training...')
    logger.info('batch_size = {}, learning_rate = {}'.format(args.batch_size,
                                                             args.base_lr))

    mpc_model_basedir = "./mpc_model/"
    start_time = time.time()
    step = 0

    for epoch_id in range(args.epoch_num):
        for sample in loader():
            step += 1
            exe.run(feed=sample, fetch_list=[predict.name])
            batch_end = time.time()
            if step % 100 == 0:
                print('Epoch={}, Step={}, current cost time: {}'.format(
                    epoch_id, step, batch_end - start_time))

        print('Epoch={}, current cost time: {}'.format(epoch_id, batch_end -
                                                       start_time))

        # For each epoch: save infer program
        mpc_model_dir = mpc_model_basedir + "epoch{}/party{}".format(epoch_id,
                                                                     role)
        fluid.io.save_inference_model(
            dirname=mpc_model_dir,
            feeded_var_names=["feat_idx", "feat_value", "label"],
            target_vars=[predict],
            executor=exe,
            main_program=infer_program,
            model_filename="__model__")

        logger.info('Model is saved in {}'.format(mpc_model_dir))
    end_time = time.time()
    print('Mpc Training of Epoch={} Batch_size={}, epoch_cost={:.4f} s'
          .format(args.epoch_num, BATCH_SIZE, (end_time - start_time)))
    logger.info('******************************************')
    logger.info('End Training...')


if __name__ == '__main__':
    args = args.parse_args()
    train(args)
