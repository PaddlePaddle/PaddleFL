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
mpc uci demo
"""

import os
import sys
import numpy as np
import time

import paddle
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils
import process_data

mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

role, server, port = sys.argv[1], sys.argv[2], sys.argv[3]

selfaddr="localhost"
if len(sys.argv) >= 5:
    selfaddr=sys.argv[4]

pfl_mpc.init(mpc_protocol_name, int(role), selfaddr, server, int(port))
#pfl_mpc.init(mpc_protocol_name, int(role), "localhost", server, int(port), "localhost:90001;localhost:90002;localhost:90003", "gloo")
#pfl_mpc.init(mpc_protocol_name, int(role), "localhost", server, int(port), "localhost:90001;localhost:90002;localhost:90003", "grpc")

role = int(role)

# data preprocessing
BATCH_SIZE = 10
mpc_data_dir = "./mpc_data/"

# generate share online
feature_reader, label_reader = process_data.generate_encrypted_data_online(role, server, port, selfaddr)

"""
# load shares from file
feature_reader = mpc_du.load_shares(
    mpc_data_dir + "house_feature", id=role, shape=(13, ))
label_reader = mpc_du.load_shares(mpc_data_dir + "house_label", id=role, shape=(1, ))
"""
batch_feature = mpc_du.batch(feature_reader, BATCH_SIZE, drop_last=True)
batch_label = mpc_du.batch(label_reader, BATCH_SIZE, drop_last=True)

x = pfl_mpc.data(name='x', shape=[BATCH_SIZE, 13], dtype='int64')
y = pfl_mpc.data(name='y', shape=[BATCH_SIZE, 1], dtype='int64')

# async data loader
loader = fluid.io.DataLoader.from_generator(
    feed_list=[x, y], capacity=BATCH_SIZE)
batch_sample = paddle.reader.compose(batch_feature, batch_label)
place = fluid.CPUPlace()
loader.set_batch_generator(batch_sample, places=place)

# network
y_pre = pfl_mpc.layers.fc(input=x, size=1)
#param_attr=fluid.ParamAttr(
#                            initializer=fluid.initializer
#                                .ConstantInitializer(0.0)))

infer_program = fluid.default_main_program().clone(for_test=False)

cost = pfl_mpc.layers.square_error_cost(input=y_pre, label=y)
avg_loss = pfl_mpc.layers.mean(cost)
optimizer = pfl_mpc.optimizer.SGD(learning_rate=0.01)
optimizer.minimize(avg_loss)

# loss file
mpc_infer_data_dir = "./mpc_infer_data/"
if not os.path.exists(mpc_infer_data_dir):
    try:
        os.mkdir(mpc_infer_data_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
loss_file = mpc_infer_data_dir + "uci_loss.part{}".format(role)

# train
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
epoch_num = 2

#mpc_loss = []

for epoch_id in range(epoch_num):
    start_time = time.time()
    step = 0

    # Method 1: feed data directly
    # for feature, label in zip(batch_feature(), batch_label()):
    #     mpc_loss = exe.run(feed={"x": feature, "y": label}, fetch_list=[avg_loss])

    # Method 2: feed data via loader
    for sample in loader():
        #print(sample[0]['x'])
        #print(sample[0]['y'])
        step_start = time.time()
        mpc_loss = exe.run(feed=sample, fetch_list=[avg_loss])
        step_end = time.time()

        #print('Epoch={}, Step={}, batch_cost={:.4f} s, Loss={},'.format(
        #    epoch_id, step, (step_end - step_start), mpc_loss))
        with open(loss_file, 'ab') as f:
            f.write(np.array(mpc_loss).tostring())
        step += 1

    end_time = time.time()
    print('Mpc Training of Epoch={} Batch_size={}, epoch_cost={:.4f} s'
          .format(epoch_id, BATCH_SIZE, (end_time - start_time)))

prediction_file = mpc_infer_data_dir + "uci_prediction.part{}".format(role)
for sample in loader():
    prediction = exe.run(program=infer_program,
                         feed=sample,
                         fetch_list=[y_pre])
    with open(prediction_file, 'ab') as f:
        f.write(np.array(prediction).tostring())
    print("revealed result: {}".format(process_data.decrypt_online(prediction, (2, BATCH_SIZE))))
    break
