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
MNIST CNN Demo (LeNet5)
"""

import sys
import os
import errno

import numpy as np
import time
import logging
import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

role, server, port = sys.argv[1], sys.argv[2], sys.argv[3]
# modify host(localhost).
pfl_mpc.init(mpc_protocol_name, int(role), "localhost", server, int(port))
role = int(role)

# data preprocessing
BATCH_SIZE = 128
epoch_num = 1

x = pfl_mpc.data(name='x', shape=[BATCH_SIZE, 1, 28, 28], dtype='int64')
y = pfl_mpc.data(name='y', shape=[BATCH_SIZE, 10], dtype='int64')

fc_out = pfl_mpc.layers.fc(input=x, size=10)
cost, softmax = pfl_mpc.layers.softmax_with_cross_entropy(logits=fc_out,
                                                          label=y,
                                                          soft_label=True,
                                                          return_softmax=True)

infer_program = fluid.default_main_program().clone(for_test=False)

avg_loss = pfl_mpc.layers.mean(cost)
optimizer = pfl_mpc.optimizer.SGD(learning_rate=0.1)
optimizer.minimize(avg_loss)

# prepare train and test reader
mpc_data_dir = "./mpc_data/"
if not os.path.exists(mpc_data_dir):
    raise ValueError("mpc_data_dir is not found. Please prepare encrypted data.")

# train_reader
feature_reader = mpc_du.load_shares(mpc_data_dir + "mnist10_feature", id=role, shape=(1, 28, 28))
label_reader = mpc_du.load_shares(mpc_data_dir + "mnist10_label", id=role, shape=(10,))
batch_feature = mpc_du.batch(feature_reader, BATCH_SIZE, drop_last=True)
batch_label = mpc_du.batch(label_reader, BATCH_SIZE, drop_last=True)

# test_reader
test_feature_reader = mpc_du.load_shares(mpc_data_dir + "mnist10_test_feature", id=role, shape=(1, 28, 28))
test_label_reader = mpc_du.load_shares(mpc_data_dir + "mnist10_test_label", id=role, shape=(10,))
test_batch_feature = mpc_du.batch(test_feature_reader, BATCH_SIZE, drop_last=True)
test_batch_label = mpc_du.batch(test_label_reader, BATCH_SIZE, drop_last=True)

place = fluid.CPUPlace()

# async data loader
loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=BATCH_SIZE)
batch_sample = paddle.reader.compose(batch_feature, batch_label)
loader.set_batch_generator(batch_sample, places=place)

test_loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=BATCH_SIZE)
test_batch_sample = paddle.reader.compose(test_batch_feature, test_batch_label)
test_loader.set_batch_generator(test_batch_sample, places=place)

# infer
def infer():
    """
    MPC infer
    """
    mpc_infer_data_dir = "./mpc_infer_data/"
    if not os.path.exists(mpc_infer_data_dir):
        try:
            os.mkdir(mpc_infer_data_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    prediction_file = mpc_infer_data_dir + "mnist_debug_prediction"
    prediction_file_part = prediction_file + ".part{}".format(role)

    if os.path.exists(prediction_file_part):
        os.remove(prediction_file_part)

    step = 0
    start_time = time.time()
    for sample in test_loader():
        step += 1
        prediction = exe.run(program=infer_program, feed=sample, fetch_list=[softmax])
        with open(prediction_file_part, 'ab') as f:
            f.write(np.array(prediction).tostring())
        if step % 10 == 0:
            end_time = time.time()
            logger.info('MPC infer of step={}, cost time in seconds:{}'.format(step, (end_time - start_time)))

    end_time = time.time()
    logger.info('MPC infer time in seconds:{}'.format((end_time - start_time)))

# train
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

mpc_model_basedir = "./mpc_model/"

logger.info('MPC training start...')
for epoch_id in range(epoch_num):
    step = 0
    epoch_start_time = time.time()
    for sample in loader():
        step += 1
        step_start_time = time.time()
        results = exe.run(feed=sample, fetch_list=[softmax])
        step_end_time = time.time()
        if step % 100 == 0:
            logger.info('MPC training of epoch_id={} step={},  cost time in seconds:{}'
                        .format(epoch_id, step, (step_end_time - step_start_time)))
    
    # For each epoch: infer or save infer program
    #infer()
    mpc_model_dir = mpc_model_basedir + "epoch{}/party{}".format(epoch_id, role)
    fluid.io.save_inference_model(dirname=mpc_model_dir,
        feeded_var_names=["x", "y"],
        target_vars=[softmax],
        executor=exe,
        main_program=infer_program,
        model_filename="__model__")

    epoch_end_time = time.time()
    logger.info('MPC training of epoch_id={} batch_size={}, cost time in seconds:{}'
      .format(epoch_num, BATCH_SIZE, (epoch_end_time - epoch_start_time)))

# infer
infer()

