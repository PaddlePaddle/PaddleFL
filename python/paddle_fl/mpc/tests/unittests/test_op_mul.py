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
This module test mul op.

"""
import unittest
from multiprocessing import Manager
import test_op_base
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


class TestOpMul(test_op_base.TestOpBase):

    def mul(self, **kwargs):
        """
        Mul.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[2, 2], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[2, 2], dtype='int64')
        op_mul = pfl_mpc.layers.mul(x=x, y=y)
        # math_mul = data_1 * data_2
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_mul])

        self.assertEqual(results[0].shape, (2, 2, 2))
        return_results.append(results[0])

    def diff_dim_mul(self, **kwargs):
        """
        Mul with different dimensions.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3, 4], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[4, 5], dtype='int64')
        op_mul = pfl_mpc.layers.mul(x=x, y=y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_mul])

        self.assertEqual(results[0].shape, (2, 3, 5))
        return_results.append(results[0])

    def test_mul(self):
        """
        Test normal case.
        :return:
        """
        data_1 = np.arange(0, 4).reshape((2, 2))
        data_2 = np.full(shape=(2, 2), fill_value=2)
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.mul,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = np.array([[2, 2], [10, 10]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))

    def test_diff_dim_mul(self):
        data_1 = np.arange(0, 12).reshape((3, 4))
        data_2 = np.full(shape=(4, 5), fill_value=2)
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.diff_dim_mul,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = data_1.dot(data_2)
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
