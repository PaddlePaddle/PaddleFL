#   Copyright (c) 2020 PmullePmulle Authors. All Rights Reserved.
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

    def elementwise_mul(self, **kwargs):
        """
        Add two variables with one dimension.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[4], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[4], dtype='int64')
        op_mul = pfl_mpc.layers.elementwise_mul(x=x, y=y)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_mul])

        self.assertEqual(results[0].shape, (2, 4))
        return_results.append(results[0])

    def multi_dim_mul(self, **kwargs):
        """
        Add two variables with multi dimensions.
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[2, 2], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[2, 2], dtype='int64')
        math_mul = x * y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[math_mul])

        self.assertEqual(results[0].shape, (2, 2, 2))
        return_results.append(results[0])


    def diff_dim_mul(self, **kwargs):
        """
        Add with different dimensions.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3, 4], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[4], dtype='int64')
        math_mul = x * y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[math_mul])

        self.assertEqual(results[0].shape, (2, 3, 4))
        return_results.append(results[0])

    def diff_dim_mul_mid(self, **kwargs):
        """
        Add with different dimensions.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3, 4, 2], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[3, 4], dtype='int64')
        math_mul = pfl_mpc.layers.elementwise_mul(x, y, axis=0)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[math_mul])

        self.assertEqual(results[0].shape, (2, 3, 4, 2))
        return_results.append(results[0])


    def test_elementwise_mul(self):
        data_1 = np.full(shape=(4), fill_value=2)
        data_2 = np.full(shape=(4), fill_value=3)
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.elementwise_mul,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = np.array([[6, 6, 6, 6]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))

    
    def test_multi_dim_mul(self):
        data_1 = np.full(shape=(2, 2), fill_value=2)
        data_2 = np.full(shape=(2, 2), fill_value=3)
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.multi_dim_mul,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = np.array([[6, 6], [6, 6]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))


    
    def test_diff_dim_mul(self):
        data_1 = np.full((3, 4), fill_value=2)
        data_2 = np.full((4), fill_value=2)
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
        expected_out = np.array([[4, 4, 4, 4], [4, 4, 4, 4], [4, 4, 4, 4]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))
  
     
    def test_diff_dim_mul_mid(self):
        data_1 = np.full((3, 4, 2), fill_value=2)
        data_2 = np.full((3, 4,), fill_value=1.5)
        #data_2 = np.ones((4,))
        # print(data_1)
        # print(data_2)
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.diff_dim_mul_mid,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = np.array([[[3, 3], [3, 3], [3, 3], [3, 3]], 
            [[3, 3], [3, 3], [3, 3], [3, 3]], 
            [[3, 3], [3, 3], [3, 3], [3, 3]]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))
   

if __name__ == '__main__':
    unittest.main()
