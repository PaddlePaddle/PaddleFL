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
test mpc add op
"""

# set proper path for fluid_encrypted without install, should be first line
import env_set

import sys
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

role, server, port = env_set.TestOptions().values()

# call mpc add
pfl_mpc.init("aby3", int(role), "localhost", server, int(port))

data_1 = pfl_mpc.data(name='data_1', shape=[8], dtype='int64')
data_2 = pfl_mpc.data(name='data_2', shape=[8], dtype='int64')

d_1 = np.array(
    [[0, 1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]).astype('int64')
d_2 = np.array(
    [[7, 6, 5, 4, 3, 2, 1, 0], [7, 6, 5, 4, 3, 2, 1, 0]]).astype('int64')

out_add = data_1 + data_2

exe = fluid.Executor(place=fluid.CPUPlace())
out_add = exe.run(feed={
    'data_1': d_1,
    'data_2': d_2,
}, fetch_list=[out_add])

print(out_add)
