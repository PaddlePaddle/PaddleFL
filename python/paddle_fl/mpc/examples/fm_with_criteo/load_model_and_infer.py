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
load mpc model and infer
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
import evaluate_metrics as evaluate


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

def load_model_and_infer(args):
    """
    load mpc data and model, then infer
    """
    # Init MPC
    role = int(args.role)
    pfl_mpc.init(mpc_protocol_name, role, "localhost", args.server, int(args.port))

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Input
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

    # Prepare test data
    mpc_data_dir = "./mpc_data/"
    mpc_test_data_dir = mpc_data_dir + 'test/'
    if not os.path.exists(mpc_test_data_dir):
        raise ValueError("{}is not found. Please prepare encrypted data.".
                         format(mpc_test_data_dir))
    test_feature_idx_reader = mpc_du.load_shares(
        mpc_test_data_dir + "criteo_feature_idx",
        id=role,
        shape=(FIELD_NUM, FEATURE_NUM))
    test_feature_value_reader = mpc_du.load_shares(
        mpc_test_data_dir + "criteo_feature_value",
        id=role,
        shape=(FIELD_NUM, ))
    test_label_reader = mpc_du.load_shares(
        mpc_test_data_dir + "criteo_label", id=role, shape=(1, ))

    test_batch_feature_idx = mpc_du.batch(
        test_feature_idx_reader, BATCH_SIZE, drop_last=True)
    test_batch_feature_value = mpc_du.batch(
        test_feature_value_reader, BATCH_SIZE, drop_last=True)
    test_batch_label = mpc_du.batch(
        test_label_reader, BATCH_SIZE, drop_last=True)

    test_loader = fluid.io.DataLoader.from_generator(
        feed_list=[feat_idx, feat_value, label],
        capacity=BATCH_SIZE,
        drop_last=True)
    test_batch_sample = paddle.reader.compose(
        test_batch_feature_idx, test_batch_feature_value, test_batch_label)
    test_loader.set_batch_generator(test_batch_sample, places=place)

    for i in range(args.epoch_num):
        mpc_model_dir = './mpc_model/epoch{}/party{}'.format(i, role)
        mpc_model_filename = '__model__'
        infer(test_loader, role, exe, BATCH_SIZE, mpc_model_dir,
              mpc_model_filename)


def infer(test_loader, role, exe, BATCH_SIZE, mpc_model_dir,
          mpc_model_filename):
    """
    load mpc model and infer
    """
    # Load mpc model
    logger.info('Load model from {}'.format(mpc_model_dir))
    infer_program, feed_targets, fetch_targets = mpc_du.load_mpc_model(
        exe=exe,
        mpc_model_dir=mpc_model_dir,
        mpc_model_filename=mpc_model_filename,
        inference=True)

    # Infer
    logger.info('******************************************')
    logger.info('Start Inferring...')
    mpc_infer_data_dir = "./mpc_infer_data/"
    if not os.path.exists(mpc_infer_data_dir):
        try:
            os.mkdir(mpc_infer_data_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    prediction_file = mpc_infer_data_dir + "prediction.part{}".format(role)
    if os.path.exists(prediction_file):
        os.remove(prediction_file)

    start_time = time.time()
    for sample in test_loader():
        prediction = exe.run(program=infer_program,
                             feed=sample,
                             fetch_list=fetch_targets)
        with open(prediction_file, 'ab') as f:
            f.write(np.array(prediction).tostring())
    end_time = time.time()
    logger.info('End Inferring...cost time: {}'.format(end_time - start_time))

    logger.info('Start Evaluate Accuracy...')
    cypher_file = mpc_infer_data_dir + "prediction"
    decrypt_file = mpc_infer_data_dir + 'label_mpc'
    time.sleep(0.1)
    if role == 0:
        if os.path.exists(decrypt_file):
            os.remove(decrypt_file)
        process_data.decrypt_data_to_file(cypher_file, (BATCH_SIZE, ),
                                          decrypt_file)
        evaluate.evaluate_accuracy('./mpc_infer_data/label_criteo',
                                   decrypt_file)
        evaluate.evaluate_auc('./mpc_infer_data/label_criteo', decrypt_file)

    end_time = time.time()
    logger.info('End Evaluate Accuracy...cost time: {}'.format(end_time -
                                                               start_time))
    logger.info('******************************************')


if __name__ == '__main__':
    args = args.parse_args()
    load_model_and_infer(args)
