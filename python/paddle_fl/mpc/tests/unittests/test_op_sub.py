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
This module test sub op.

"""
import unittest

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

import test_op_base


class TestOpSub(test_op_base.TestOpBase):

    def elementwise_sub(self, **kwargs):
        """
        Normal case.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[5], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[5], dtype='int64')
        op_sub = pfl_mpc.layers.elementwise_sub(x=x, y=y)
        math_sub = x - y
        exe = fluid.Executor(place=fluid.CPUPlace())
        sub_results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_sub, math_sub])

        self.assertTrue(np.allclose(sub_results[0], sub_results[1]))
        self.assertEqual(sub_results[0].shape, (2, 5))
        self.assertTrue(np.allclose(sub_results[0], expected_out))

    def mul_dim_sub(self, **kwargs):
        """
        Add two variables with multi dimensions.
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[2, 2], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[2, 2], dtype='int64')
        sub = x - y
        exe = fluid.Executor(place=fluid.CPUPlace())
        sub_results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[sub])

        self.assertTrue(np.allclose(sub_results[0], expected_out))

    def test_elementwise_sub(self):
        data_1 = [np.array([[1, 2, 3, 4, 5],
                            [1, 2, 3, 4, 5]]).astype('int64')] * self.party_num
        data_2 = [np.array([[1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1]]).astype('int64')] * self.party_num
        expect_results = [np.array([[0, 1, 2, 3, 4],
                                    [0, 1, 2, 3, 4]])] * self.party_num
        ret = self.multi_party_run(target=self.elementwise_sub,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)

    def test_multi_dim_sub(self):
        data_1 = [np.array([[[0, 0], [1, 1]],
                            [[0, 0], [1, 1]]]).astype('int64')] * self.party_num
        data_2 = [np.array([[[0, 0], [-1, -1]],
                            [[0, 0], [-1, -1]]]).astype('int64')] * self.party_num
        expect_results = [np.array([[[0, 0], [2, 2]],
                                    [[0, 0], [2, 2]]])] * self.party_num
        ret = self.multi_party_run(target=self.mul_dim_sub,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)


if __name__ == '__main__':
    unittest.main()
