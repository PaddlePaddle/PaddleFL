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

import env_set
import numpy as np
import paddle.fluid as fluid
import paddle_encrypted as paddle_enc

role, server, port = env_set.TestOptions().values()

paddle_enc.init("aby3", int(role), "localhost", server, int(port))

batch_size = 3

# x is in cypher text type
x = paddle_enc.data(name='x', shape=[batch_size, 8], dtype='int64')
# y is in cypher text type
y = paddle_enc.data(name='y', shape=[batch_size, 1], dtype='int64')

y_pre = paddle_enc.layers.fc(input=x, size=1, act=None)
y_relu = paddle_enc.layers.relu(input=y_pre)
cost = paddle_enc.layers.square_error_cost(input=y_relu, label=y)
avg_loss = paddle_enc.layers.mean(cost)

optimizer = paddle_enc.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss)

exe = fluid.Executor(place=fluid.CPUPlace())
exe.run(fluid.default_startup_program())

iters = 1
for _ in range(iters):
    d_1 = np.ones(shape=(2, batch_size, 8), dtype='int64')
    d_2 = np.zeros(shape=(2, batch_size, 1), dtype='int64')
    loss = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[avg_loss])
    print(loss)
