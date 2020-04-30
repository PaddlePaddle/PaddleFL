#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module test sum op.

"""
import unittest

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

import test_op_base


class TestOpSum(test_op_base.TestOpBase):

    def sum(self, **kwargs):
        """
        Test normal case.
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        d_3 = kwargs['data_3'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        data_1 = pfl_mpc.data(name='data_1', shape=[4], dtype='int64')
        data_2 = pfl_mpc.data(name='data_2', shape=[4], dtype='int64')
        data_3 = pfl_mpc.data(name='data_3', shape=[4], dtype='int64')
        op_sum = pfl_mpc.layers.sum([data_1, data_2, data_3])
        math_sum = data_1 + data_2 + data_3
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'data_1': d_1, 'data_2': d_2, 'data_3': d_3}, fetch_list=[op_sum, math_sum])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (2, 4))
        self.assertTrue(np.allclose(results[0], expected_out))

    def test_sum(self):
        data_1 = [np.array([[1, 1, 1, 1], [1, 1, 1, 1]]).astype('int64')] * self.party_num
        data_2 = [np.array([[2, 2, 2, 2], [2, 2, 2, 2]]).astype('int64')] * self.party_num
        data_3 = [np.array([[3, 3, 3, 3], [3, 3, 3, 3]]).astype('int64')] * self.party_num
        expect_results = [np.array([[6, 6, 6, 6], [6, 6, 6, 6]])] * self.party_num
        ret = self.multi_party_run(target=self.sum,
                                   data_1=data_1,
                                   data_2=data_2,
                                   data_3=data_3,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)


if __name__ == '__main__':
    unittest.main()
