# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
CUDA ImageNet Demo
"""

import sys
import numpy as np
import time
import logging

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils

import resnet

#np.set_printoptions(threshold=np.inf)

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

# choose party 0 as nccl "server", which receive nccl id of other parties
# party 0 listens on ports listed in endpoints
# other parties connect to those ports and send their nccl id
role, server = sys.argv[1], sys.argv[2]
pfl_mpc.init(mpc_protocol_name, int(role),
        net_server_addr=server, endpoints="33784,45888", network_mode="nccl")
role = int(role)

# data preprocessing
BATCH_SIZE = 32
epoch_num = 1

x = pfl_mpc.data(name='x', shape=[BATCH_SIZE, 3, 224, 224], dtype='int64')
y = pfl_mpc.data(name='y', shape=[BATCH_SIZE, 1000], dtype='int64')

class Model(object):
    """
   lenet model: alexnet, vgg-16
    """

    def __int__(self):
        """
        init
        """
        pass

    def alexnet(self):
        """
        alexnet
        """
        conv_1 = pfl_mpc.layers.conv2d(input=x, num_filters=96, filter_size=11, stride=4, act='relu')
        pool_1 = pfl_mpc.layers.pool2d(input=conv_1, pool_size=3, pool_stride=2, pool_type='max')
        conv_2 = pfl_mpc.layers.conv2d(input=pool_1, num_filters=256, filter_size=5, padding=2, act='relu')
        pool_2 = pfl_mpc.layers.pool2d(input=conv_2, pool_size=3, pool_stride=2, pool_type='avg')

        conv_3 = pfl_mpc.layers.conv2d(input=pool_2, num_filters=384, filter_size=3, padding=1, act='relu')
        conv_4 = pfl_mpc.layers.conv2d(input=conv_3, num_filters=384, filter_size=3, padding=1, act='relu')
        conv_5 = pfl_mpc.layers.conv2d(input=conv_4, num_filters=256, filter_size=3, padding=1, act='relu')
        pool_3 = pfl_mpc.layers.pool2d(input=conv_5, pool_size=3, pool_stride=2, pool_type='max')

        reshape1 = pfl_mpc.layers.reshape(pool_3, [2, BATCH_SIZE, -1])
        fc_1 = pfl_mpc.layers.fc(input=reshape1, size=4096, act='relu')
        fc_2 = pfl_mpc.layers.fc(input=fc_1, size=4096, act='relu')
        fc_out = pfl_mpc.layers.fc(input=fc_2, size=1000)
        return fc_out


    def vgg_16(self):
        conv_1 = pfl_mpc.layers.conv2d(input=x, num_filters=64, filter_size=3, stride=1, padding=1, act='relu')
        conv_2 = pfl_mpc.layers.conv2d(input=conv_1, num_filters=64, filter_size=3, stride=1, padding=1, act='relu')
        pool_1 = pfl_mpc.layers.pool2d(input=conv_2, pool_size=2, pool_stride=2, pool_type='max')
        conv_3 = pfl_mpc.layers.conv2d(input=pool_1, num_filters=128, filter_size=3, stride=1, padding=1, act='relu')
        conv_4 = pfl_mpc.layers.conv2d(input=conv_3, num_filters=128, filter_size=3, stride=1, padding=1, act='relu')
        pool_2 = pfl_mpc.layers.pool2d(input=conv_4, pool_size=2, pool_stride=2, pool_type='max')
        conv_5 = pfl_mpc.layers.conv2d(input=pool_2, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        conv_6 = pfl_mpc.layers.conv2d(input=conv_5, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        conv_7 = pfl_mpc.layers.conv2d(input=conv_6, num_filters=256, filter_size=3, stride=1, padding=1, act='relu')
        pool_3 = pfl_mpc.layers.pool2d(input=conv_7, pool_size=2, pool_stride=2, pool_type='max')
        conv_8 = pfl_mpc.layers.conv2d(input=pool_3, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        conv_9 = pfl_mpc.layers.conv2d(input=conv_8, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        conv_10 = pfl_mpc.layers.conv2d(input=conv_9, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        pool_4 = pfl_mpc.layers.pool2d(input=conv_10, pool_size=2, pool_stride=2, pool_type='max')
        conv_11 = pfl_mpc.layers.conv2d(input=pool_4, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        conv_12 = pfl_mpc.layers.conv2d(input=conv_11, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        conv_13 = pfl_mpc.layers.conv2d(input=conv_12, num_filters=512, filter_size=3, stride=1, padding=1, act='relu')
        pool_5 = pfl_mpc.layers.pool2d(input=conv_13, pool_size=2, pool_stride=2, pool_type='max')

        reshape1 = pfl_mpc.layers.reshape(pool_5, [2, BATCH_SIZE, -1])
        fc_1 = pfl_mpc.layers.fc(input=reshape1, size=4096, act='relu')
        fc_2 = pfl_mpc.layers.fc(input=fc_1, size=4096, act='relu')
        fc_out = pfl_mpc.layers.fc(input=fc_2, size=1000)
        return fc_out

model = Model()
out = model.alexnet()
#out = model.vgg_16()

#model = resnet.ResNet(layers=18)
#out = model.net(input=x, class_dim=1000)
cost, softmax = pfl_mpc.layers.softmax_with_cross_entropy(out, y, return_softmax=True, soft_label=True)

infer_program = fluid.default_main_program().clone(for_test=False)

avg_loss = pfl_mpc.layers.mean(cost)

lr = 0.01

optimizer = pfl_mpc.optimizer.SGD(learning_rate=lr)
optimizer.minimize(avg_loss)

## write main program into file for check
#with open('mpc_prog.txt', 'w') as f:
#    f.write(fluid.default_main_program().to_string(throw_on_error=False))

# dummy reader for profiling
# otherwise prepare encrypted data. please refer to MNIST demo
def dummy_reader(shape, rep):
    shape = list(shape)
    shape.insert(0, 2)
    shape = tuple(shape)
    def reader():
        for i in range(rep):
            yield np.zeros(shape, dtype=np.int64)
    return reader

feature_reader = dummy_reader((3, 224, 224), 160)
label_reader = dummy_reader((1000, ), 160)

batch_feature = mpc_du.batch(feature_reader, BATCH_SIZE, drop_last=True)
batch_label = mpc_du.batch(label_reader, BATCH_SIZE, drop_last=True)

test_batch_feature = mpc_du.batch(
    feature_reader, BATCH_SIZE, drop_last=True)
test_batch_label = mpc_du.batch(label_reader, BATCH_SIZE, drop_last=True)

place = fluid.CUDAPlace(0)

# async data loader
loader = fluid.io.DataLoader.from_generator(
    feed_list=[x, y], capacity=BATCH_SIZE)
batch_sample = paddle.reader.compose(batch_feature, batch_label)
loader.set_batch_generator(batch_sample, places=place)

test_loader = fluid.io.DataLoader.from_generator(
    feed_list=[x, y], capacity=BATCH_SIZE)
test_batch_sample = paddle.reader.compose(test_batch_feature, test_batch_label)
test_loader.set_batch_generator(test_batch_sample, places=place)

def decrypt_online(shares, shape):
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        input = pfl_mpc.data(name='input', shape=shape[1:], dtype='int64')
        out = pfl_mpc.layers.reveal(input)

        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        plain = exe.run(feed={'input': np.array(shares).reshape(shape)}, fetch_list=[out])
        return plain

# infer
def infer():
    """
    MPC infer
    """

    step = 0
    acc = 0
    start_time = time.time()
    for sample in test_loader():
        step += 1
        res = exe.run(program=infer_program,
                             feed=sample,
                             fetch_list=[softmax, y])
        pred = decrypt_online(res[0], (2, BATCH_SIZE, 1000))
        label = decrypt_online(res[1], (2, BATCH_SIZE, 1000))
        p = np.argmax(pred, axis=-1)
        l = np.argmax(label, axis=-1)
        match = BATCH_SIZE - np.count_nonzero(p - l)
        acc += match
        if step % 10 == 0:
            end_time = time.time()
            logger.info('MPC infer of step={}, cost time in seconds:{}'.format(
                step, (end_time - start_time)))
    end_time = time.time()
    logger.info('MPC infer acc:{} / {}'.format(acc, BATCH_SIZE * step))
    logger.info('MPC infer time in seconds:{}'.format((end_time - start_time)))


# train
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

logger.info('MPC training start...')
for epoch_id in range(epoch_num):
    step = 0
    epoch_start_time = time.time()
    for sample in loader():
        step += 1
        step_start_time = time.time()
        #profiler.start_profiler("All")
        results = exe.run(feed=sample)
        #profiler.stop_profiler("total", "./profile")

        if step % 10 == 0:
            step_end_time = time.time()
            logger.info(
                'MPC training of epoch_id={} step={},  cost time in seconds:{}'
                .format(epoch_id, step, (step_end_time - step_start_time)))

    ### For each epoch, save infer program
    #mpc_model_basedir = "./mpc_model/"
    #mpc_model_dir = mpc_model_basedir + "epoch{}/party{}".format(epoch_id,
    #                                                             role)
    #fluid.io.save_inference_model(
    #    dirname=mpc_model_dir,
    #    feeded_var_names=["x", "y"],
    #    target_vars=[softmax],
    #    executor=exe,
    #    main_program=infer_program,
    #    model_filename="__model__")

    epoch_end_time = time.time()
    logger.info(
        'MPC training of epoch_num={} batch_size={}, cost time in seconds:{}'
        .format(epoch_id, BATCH_SIZE, (epoch_end_time - epoch_start_time)))

    # infer on test set
    infer()
