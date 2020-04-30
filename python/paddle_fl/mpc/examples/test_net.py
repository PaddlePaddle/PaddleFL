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
import paddle_fl.mpc as pfl_mpc

role, server, port = env_set.TestOptions().values()

pfl_mpc.init("aby3", int(role), "localhost", server, int(port))

batch_size = 3

# x is in cypher text type
x = pfl_mpc.data(name='x', shape=[batch_size, 8], dtype='int64')
# y is in cypher text type
y = pfl_mpc.data(name='y', shape=[batch_size, 1], dtype='int64')

y_pre = pfl_mpc.layers.fc(input=x, size=1, act=None)
y_relu = pfl_mpc.layers.relu(input=y_pre)
cost = pfl_mpc.layers.square_error_cost(input=y_relu, label=y)
avg_loss = pfl_mpc.layers.mean(cost)

optimizer = pfl_mpc.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss)

exe = fluid.Executor(place=fluid.CPUPlace())
exe.run(fluid.default_startup_program())

iters = 1
for _ in range(iters):
    d_1 = np.ones(shape=(2, batch_size, 8), dtype='int64')
    d_2 = np.zeros(shape=(2, batch_size, 1), dtype='int64')
    loss = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[avg_loss])
    print(loss)
