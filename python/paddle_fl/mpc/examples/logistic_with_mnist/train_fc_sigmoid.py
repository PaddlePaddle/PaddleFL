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
MNIST Demo
"""

import sys
import os
import errno

import numpy as np
import time

import paddle
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils

mpc_protocol_name = 'aby3'
mpc_du= get_datautils(mpc_protocol_name)
role, server, port = sys.argv[1], sys.argv[2], sys.argv[3]
# modify host(localhost).
pfl_mpc.init(mpc_protocol_name, int(role), "localhost", server, int(port))
role = int(role)

# data preprocessing
BATCH_SIZE = 128
epoch_num = 2

# network
x = pfl_mpc.data(name='x', shape=[BATCH_SIZE, 784], dtype='int64')
y = pfl_mpc.data(name='y', shape=[BATCH_SIZE, 1], dtype='int64')

y_pre = pfl_mpc.layers.fc(input=x, size=1)
cost = pfl_mpc.layers.sigmoid_cross_entropy_with_logits(y_pre, y)

infer_program = fluid.default_main_program().clone(for_test=False)

avg_loss = pfl_mpc.layers.mean(cost)
optimizer = pfl_mpc.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss)

mpc_data_dir = "./mpc_data/"
if not os.path.exists(mpc_data_dir):
    raise ValueError("mpc_data_dir is not found. Please prepare encrypted data.")

# train_reader
feature_reader = mpc_du.load_shares(mpc_data_dir + "mnist2_feature", id=role, shape=(784,))
label_reader = mpc_du.load_shares(mpc_data_dir + "mnist2_label", id=role, shape=(1,))
batch_feature = mpc_du.batch(feature_reader, BATCH_SIZE, drop_last=True)
batch_label = mpc_du.batch(label_reader, BATCH_SIZE, drop_last=True)

# test_reader
test_feature_reader = mpc_du.load_shares(mpc_data_dir + "mnist2_test_feature", id=role, shape=(784,))
test_label_reader = mpc_du.load_shares(mpc_data_dir + "mnist2_test_label", id=role, shape=(1,))
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

# loss file
# loss_file = "/tmp/mnist_output_loss.part{}".format(role)

# train
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

start_time = time.time()
step = 0
for epoch_id in range(epoch_num):
    # feed data via loader
    for sample in loader():
        step += 1
        exe.run(feed=sample, fetch_list=[cost.name])
        batch_end = time.time()
        if step % 50 == 0:
            print('Epoch={}, Step={}'.format(epoch_id, step))

end_time = time.time()
print('Mpc Training of Epoch={} Batch_size={}, epoch_cost={:.4f} s'
      .format(epoch_num, BATCH_SIZE, (end_time - start_time)))

# prediction

mpc_infer_data_dir = "./mpc_infer_data/"
if not os.path.exists(mpc_infer_data_dir):
    try:
        os.mkdir(mpc_infer_data_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

prediction_file = mpc_infer_data_dir + "mnist_debug_prediction.part{}".format(role)
if os.path.exists(prediction_file):
    os.remove(prediction_file)
for sample in test_loader():
    prediction = exe.run(program=infer_program, feed=sample, fetch_list=[cost])
    with open(prediction_file, 'ab') as f:
        f.write(np.array(prediction).tostring())
