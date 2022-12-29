# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class TestOpMul3_C2(test_op_base.TestOpBase):

    def mul3_C2(self, **kwargs):

        role = 2
        num = 100
        d_1 = np.load('data_C2_P0.npy',allow_pickle=True)
        d_2 = np.load('data_C2_P1.npy',allow_pickle=True)
        d_3 = np.load('data_C2_P2.npy', allow_pickle=True)

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[num], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[num], dtype='int64')
        # math_mul = x * y
        math_mul = pfl_mpc.layers.elementwise_mul(x, y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[math_mul])
        results = exe.run(feed={'x': d_3, 'y': results[0]}, fetch_list=[math_mul])
        np.save('result_C2.npy', results[0])

    def test_mul3_C2(self):
        ret = self.multi_party_run2(target=self.mul3_C2)

if __name__ == '__main__':
    unittest.main()
