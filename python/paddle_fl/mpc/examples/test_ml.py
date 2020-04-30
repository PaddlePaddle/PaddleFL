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
test mixed mpc ops
"""

# set proper path for fluid_encrypted without install, should be first line
import env_set

import time
import sys
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

role, server, port = env_set.TestOptions().values()

pfl_mpc.init("aby3", int(role), "localhost", server, int(port))

data_1 = pfl_mpc.data(name='data_1', shape=[2, 2], dtype='int64')
data_2 = pfl_mpc.data(name='data_2', shape=[2, 2], dtype='int64')

out_sec = pfl_mpc.layers.square_error_cost(input=data_1, label=data_2)
out_fc = pfl_mpc.layers.fc(input=data_1, size=1, act=None)
out_relu = pfl_mpc.layers.relu(data_1)

d_1 = np.array([[[10, 10], [10, 10]], [[10, 10], [10, 10]]]).astype('int64')
d_2 = np.array([[[5, 5], [5, 5]], [[5, 5], [5, 5]]]).astype('int64')

exe = fluid.Executor(place=fluid.CPUPlace())
exe.run(fluid.default_startup_program())

out_sec, out_fc, out_relu = exe.run(feed={'data_1': d_1,
                                          'data_2': d_2},
                                    fetch_list=[out_sec, out_fc, out_relu])
print("square_error_cost: {},\n out_fc:{},\n out_relu:{}\n".format(
    out_sec, out_fc, out_relu))
