from __future__ import print_function
import os
import paddle
import paddle.fluid as fluid
import numpy
import sys
from paddle.fluid import layers
import redis
import time
from paddle.fluid.param_attr import ParamAttr
import math
import msgpack


def data_generater(samples, r):
    # data generater
    def train_data():
        for item in samples:
            sample = msgpack.loads(r.get(str(item)))
            conv = sample[0]
            label = sample[1]
            yield conv, label
        return train_data


class ResNet():
    def __init__(self, layers=50):
        self.layers = layers

    def net(self, input, class_dim=10):
        layers = self.layers
        supported_layers = [18, 34, 50, 101, 152]
        assert layers in supported_layers, \
            "supported layers are {} but input layer is {}".format(supported_layers, layers)

        if layers == 18:
            depth = [2, 2, 2, 2]
        elif layers == 34 or layers == 50:
            depth = [3, 4, 6, 3]
        elif layers == 101:
            depth = [3, 4, 23, 3]
        elif layers == 152:
            depth = [3, 8, 36, 3]
        num_filters = [64, 128, 256, 512]
        conv = input
        if layers >= 50:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    if layers in [101, 152] and block == 2:
                        if i == 0:
                            conv_name = "res" + str(block + 2) + "a"
                        else:
                            conv_name = "res" + str(block + 2) + "b" + str(i)
                    else:
                        conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.bottleneck_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        name=conv_name)

            pool = fluid.layers.pool2d(
                input=conv, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)),
                act="softmax")
        else:
            for block in range(len(depth)):
                for i in range(depth[block]):
                    conv_name = "res" + str(block + 2) + chr(97 + i)
                    conv = self.basic_block(
                        input=conv,
                        num_filters=num_filters[block],
                        stride=2 if i == 0 and block != 0 else 1,
                        is_first=block == i == 0,
                        name=conv_name)

            pool = fluid.layers.pool2d(
                input=conv, pool_type='avg', global_pooling=True)
            stdv = 1.0 / math.sqrt(pool.shape[1] * 1.0)
            out = fluid.layers.fc(
                input=pool,
                size=class_dim,
                param_attr=fluid.param_attr.ParamAttr(
                    initializer=fluid.initializer.Uniform(-stdv, stdv)),
                act="softmax")
        return out

    def conv_bn_layer(self,
                      input,
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

    def shortcut(self, input, ch_out, stride, is_first, name):
        ch_in = input.shape[1]
        if ch_in != ch_out or stride != 1 or is_first == True:
            return self.conv_bn_layer(input, ch_out, 1, stride, name=name)
        else:
            return input

    def bottleneck_block(self, input, num_filters, stride, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=1,
            act='relu',
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            stride=stride,
            act='relu',
            name=name + "_branch2b")
        conv2 = self.conv_bn_layer(
            input=conv1,
            num_filters=num_filters * 4,
            filter_size=1,
            act=None,
            name=name + "_branch2c")

        short = self.shortcut(
            input,
            num_filters * 4,
            stride,
            is_first=False,
            name=name + "_branch1")

        return fluid.layers.elementwise_add(
            x=short, y=conv2, act='relu', name=name + ".add.output.5")

    def basic_block(self, input, num_filters, stride, is_first, name):
        conv0 = self.conv_bn_layer(
            input=input,
            num_filters=num_filters,
            filter_size=3,
            act='relu',
            stride=stride,
            name=name + "_branch2a")
        conv1 = self.conv_bn_layer(
            input=conv0,
            num_filters=num_filters,
            filter_size=3,
            act=None,
            name=name + "_branch2b")
        short = self.shortcut(
            input, num_filters, stride, is_first, name=name + "_branch1")
        return fluid.layers.elementwise_add(x=short, y=conv1, act='relu')


# local redis config
redis_host = "127.0.0.1"
redis_port = 6379
redis_password = ""
r = redis.StrictRedis(
    host=redis_host, port=redis_port, password=redis_password)

# reader generation
reader = fluid.layers.py_reader(
    capacity=64, shapes=[(-1, 64, 8, 8), (-1, 1)],
    dtypes=['float32', 'int64'])

samples = r.keys()
train_data = data_generater(samples, r)

reader.decorate_paddle_reader(
    paddle.batch(
        paddle.reader.shuffle(
            train_data, buf_size=5000), batch_size=64))

conv1, label = fluid.layers.read_file(reader)

# train program
place = fluid.CUDAPlace(0)
model = ResNet(layers=50)
predicts = model.net(conv1, 10)
cost = fluid.layers.cross_entropy(input=predicts, label=label)
accuracy = fluid.layers.accuracy(input=predicts, label=label)
loss = fluid.layers.mean(cost)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(loss)

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

total_time = 0
EPOCH_NUM = 1
step = 0
train_start = time.time()
# start training 
for pass_id in range(EPOCH_NUM):
    reader.start()
    try:
        while True:
            start_time = time.time()
            loss_value, acc_value = exe.run(
                fetch_list=[loss.name, accuracy.name])
            step += 1
            if step % 10 == 0:
                print("epoch: " + str(pass_id) + "step: " + str(step) +
                      "loss: " + str(loss_value) + "acc: " + str(acc_value))
            end_time = time.time()
            total_time += (end_time - start_time)
    except fluid.core.EOFException:
        reader.reset()
train_end = time.time()
print("total time: %d" % (train_end - train_start))
print("computation time: %d" % total_time)
