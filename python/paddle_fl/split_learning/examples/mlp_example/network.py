import sys
import numpy as np
import time
import copy

import paddle
import paddle.fluid as fluid

from google.protobuf import text_format

paddle.enable_static()

def net():
    embed_size=11
    host_input = fluid.layers.data(
                    name='Host|input',
                    shape=[1],
                    dtype='int64',
                    lod_level=1)

    label = fluid.layers.data(
                name="Customer|label",
                shape=[1],
                dtype="int64")

    embed = fluid.layers.embedding(
                input=host_input,
                size=[31312, embed_size],
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    pool = fluid.layers.sequence_pool(
            input=embed,
            pool_type='max')
    fc1 = fluid.layers.fc(
            name="Host|fc1",
            input=pool,
            size=10,
            param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    prediction = fluid.layers.fc(
            name="Customer|fc2",
            input=fc1,
            size=2,
            act='softmax',
            param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(value=0.1)))
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    cost = fluid.layers.reduce_mean(cost)

    optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    optimizer.minimize(cost)
    return host_input, label, prediction, cost
