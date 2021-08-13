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
                    name='Host|x1',
                    shape=[1],
                    dtype='int64',
                    lod_level=1)

    customer_input = fluid.layers.data(
                    name='Customer|x2',
                    shape=[1],
                    dtype='int64',
                    lod_level=1)

    label = fluid.layers.data(
                name="Customer|label",
                shape=[1],
                dtype="int64")

    host_embed = fluid.layers.embedding(
                input=host_input,
                size=[31312, embed_size],
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    customer_embed = fluid.layers.embedding(
                input=customer_input,
                size=[31312, embed_size],
                param_attr=fluid.ParamAttr(
                        initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    host_pool = fluid.layers.sequence_pool(
            input=host_embed,
            pool_type='max')

    customer_pool = fluid.layers.sequence_pool(
            input=customer_embed,
            pool_type='max')

    host_fc1 = fluid.layers.fc(
            name="Host|fc1",
            input=host_pool,
            size=10,
            param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    customer_fc1 = fluid.layers.fc(
            name="Customer|fc1",
            input=customer_pool,
            size=10,
            param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(value=0.1)))

    concat_fc = fluid.layers.concat(
            name="Customer|concat",
            input=[host_fc1, customer_fc1], 
            axis=-1)

    prediction = fluid.layers.fc(
            name="Customer|fc2",
            input=concat_fc,
            size=2,
            act='softmax',
            param_attr=fluid.ParamAttr(
                    initializer=fluid.initializer.ConstantInitializer(value=0.1)))
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    cost = fluid.layers.reduce_mean(cost)

    optimizer = fluid.optimizer.SGD(learning_rate=0.01)
    optimizer.minimize(cost)
    return host_input, None, label, prediction, cost
