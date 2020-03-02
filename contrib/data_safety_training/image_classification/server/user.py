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

from __future__ import print_function
import os
import paddle
import paddle.fluid as fluid
import numpy
import sys
import redis
import time
from paddle.fluid import layers
from paddle.fluid.param_attr import ParamAttr
import msgpack


def conv_bn_layer(input,
                  num_filters,
                  filter_size,
                  stride=1,
                  groups=1,
                  act=None,
                  name=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) // 2,
        groups=groups,
        act=None,
        param_attr=ParamAttr(name=name + "_weights"),
        bias_attr=False,
        name=name + '.conv2d.output.1')

    if name == "conv1":
        bn_name = "bn_" + name
    else:
        bn_name = "bn" + name[3:]
    return fluid.layers.batch_norm(
        input=conv,
        act=act,
        name=bn_name + '.output.1',
        param_attr=ParamAttr(name=bn_name + '_scale'),
        bias_attr=ParamAttr(bn_name + '_offset'),
        moving_mean_name=bn_name + '_mean',
        moving_variance_name=bn_name + '_variance', )


def load_conf(conf_file, local_dict):
    with open(conf_file) as fin:
        for line in fin:
            group = line.strip().split("=")
            if len(group) != 2:
                continue
            local_dict[group[0]] = group[1]
    return local_dict


# redis DB configuration
redis_host = "127.0.0.1"
redis_port = 6379
redis_password = ""

start_time = time.time()
# start a redis client and empty the DB
r = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password)
r.flushall()

# encoding program
images = fluid.layers.data(name='images', shape=[3, 32, 32], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
place = fluid.CPUPlace()
conv1 = conv_bn_layer(
    input=images,
    num_filters=64,
    filter_size=7,
    stride=2,
    act='relu',
    name="conv1")
pool = fluid.layers.pool2d(
    input=conv1, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')
feeder = fluid.DataFeeder(place=place, feed_list=[images, label])
pretrained_model = 'ResNet50_pretrained'
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())


# load pretrained mode and prepare datal
def if_exist(var):
    return os.path.exists(os.path.join(pretrained_model, var.name))


fluid.io.load_vars(
    exe,
    pretrained_model,
    main_program=fluid.default_main_program(),
    predicate=if_exist)

train_data = paddle.dataset.cifar.train10()
step = 0

# start encoding and uploading
for data in train_data():
    pre_data = []
    pre_data.append(data)
    res = exe.run(program=fluid.default_main_program(),
                  feed=feeder.feed(pre_data),
                  fetch_list=[pool.name])
    sample = [res[0][0].tolist(), data[1]]
    step += 1
    file = msgpack.dumps(sample)
    r.set(step, file)
    if step % 100 == 0:
        print(numpy.array(sample[0]).shape)
        print("%dstart" % step)

files = r.keys()
print("upload file numbers: %d" % len(files))
end_time = time.time()
total_time = end_time - start_time
print("total time: %d" % total_time)
