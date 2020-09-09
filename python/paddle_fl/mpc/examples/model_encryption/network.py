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
This module provides a linear regression network.
"""
import paddle
import paddle.fluid as fluid

UCI_BATCH_SIZE = 10
BATCH_SIZE = 10
TRAIN_EPOCH = 20
PADDLE_UPDATE_EPOCH = 10
MPC_UPDATE_EPOCH = TRAIN_EPOCH - PADDLE_UPDATE_EPOCH

def uci_network():
    """
    Build a network for uci housing.

    """
    x = fluid.data(name='x', shape=[UCI_BATCH_SIZE, 13], dtype='float32')
    y = fluid.data(name='y', shape=[UCI_BATCH_SIZE, 1], dtype='float32')
    param_attr = paddle.fluid.param_attr.ParamAttr(name="fc_0.w_0",
                                                   initializer=fluid.initializer.ConstantInitializer(0.0))
    bias_attr = paddle.fluid.param_attr.ParamAttr(name="fc_0.b_0")
    y_pre = fluid.layers.fc(input=x, size=1, param_attr=param_attr, bias_attr=bias_attr)
    # add infer_program
    infer_program = fluid.default_main_program().clone(for_test=False)
    cost = fluid.layers.square_error_cost(input=y_pre, label=y)
    avg_loss = fluid.layers.mean(cost)
    optimizer = fluid.optimizer.SGD(learning_rate=0.001)
    optimizer.minimize(avg_loss)
    return_list = [x, y, y_pre, avg_loss]
    return return_list
