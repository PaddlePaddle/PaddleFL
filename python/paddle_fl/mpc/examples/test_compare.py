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
data_2 = fluid.data(name='data_2', shape=[1, 2, 2], dtype='float32')

out_gt = data_1 > data_2
out_ge = data_1 >= data_2
out_lt = data_1 < data_2
out_le = data_1 <= data_2
out_eq = data_1 == data_2
out_neq = data_1 != data_2

d_1 = np.array([[[65536, 65536], [65536, 65536]],
                [[65536, 65536], [65536, 65536]]]).astype('int64')
d_2 = np.array([[[10, 3], [0, -3]]]).astype('float32')

exe = fluid.Executor(place=fluid.CPUPlace())
exe.run(fluid.default_startup_program())

out_gt, out_ge, out_lt, out_le, out_eq, out_neq = exe.run(
    feed={'data_1': d_1,
          'data_2': d_2},
    fetch_list=[out_gt, out_ge, out_lt, out_le, out_eq, out_neq])

print("Input: \n d_1: {}\n d_2: {}\n".format(d_1, d_2))

print(
    "greater_than:{}\n greater_equal:{} \n less_than:{} \n less_equal:{} \n equal:{} \n not_equal:{} \n"
    .format(out_gt, out_ge, out_lt, out_le, out_eq, out_neq))
