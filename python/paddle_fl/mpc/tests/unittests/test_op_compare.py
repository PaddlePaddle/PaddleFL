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
This module test compare op.

"""
import unittest

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

import test_op_base


class TestOpCompare(test_op_base.TestOpBase):

    def gt(self, **kwargs):
        """
        Greater than.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3], dtype='int64')
        y = fluid.data(name='y', shape=[3], dtype='float32')
        # todo: reshape y to [3]
        op_gt = pfl_mpc.layers.greater_than(x=x, y=y)
        math_gt = x > y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_gt, math_gt])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (3, ))
        self.assertTrue(np.allclose(results[0], expected_out))

    def ge(self, **kwargs):
        """
        Greater equal.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3], dtype='int64')
        y = fluid.data(name='y', shape=[3], dtype='float32')
        op_ge = pfl_mpc.layers.greater_equal(x=x, y=y)
        math_ge = x >= y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_ge, math_ge])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (3, ))
        self.assertTrue(np.allclose(results[0], expected_out))

    def lt(self, **kwargs):
        """
        Less than.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3], dtype='int64')
        y = fluid.data(name='y', shape=[3], dtype='float32')
        op_lt = pfl_mpc.layers.less_than(x=x, y=y)
        math_lt = x < y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_lt, math_lt])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (3, ))
        self.assertTrue(np.allclose(results[0], expected_out))

    def le(self, **kwargs):
        """
        Less equal.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3], dtype='int64')
        y = fluid.data(name='y', shape=[3], dtype='float32')
        op_le = pfl_mpc.layers.less_equal(x=x, y=y)
        math_le = x <= y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_le, math_le])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (3, ))
        self.assertTrue(np.allclose(results[0], expected_out))

    def equal(self, **kwargs):
        """
        Equal.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3], dtype='int64')
        y = fluid.data(name='y', shape=[3], dtype='float32')
        op_eq = pfl_mpc.layers.equal(x=x, y=y)
        math_eq = x == y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_eq, math_eq])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (3, ))
        self.assertTrue(np.allclose(results[0], expected_out))

    def not_equal(self, **kwargs):
        """
        Not equal.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        expected_out = kwargs['expect_results'][role]

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[3], dtype='int64')
        y = fluid.data(name='y', shape=[3], dtype='float32')
        op_ne = pfl_mpc.layers.not_equal(x=x, y=y)
        math_ne = x != y
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[op_ne, math_ne])

        self.assertTrue(np.allclose(results[0], results[1]))
        self.assertEqual(results[0].shape, (3, ))
        self.assertTrue(np.allclose(results[0], expected_out))

    def test_gt(self):
        data_1 = [np.array([[65536, 65536, 65536],
                            [65536, 65536, 65536]]).astype('int64')] * self.party_num
        data_2 = [np.array([5, 3, 2]).astype('float32')] * self.party_num
        expect_results = [np.array([0, 0, 1])] * self.party_num
        ret = self.multi_party_run(target=self.gt,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)

    def test_ge(self):
        data_1 = [np.array([[65536, 65536, 65536],
                            [65536, 65536, 65536]]).astype('int64')] * self.party_num
        data_2 = [np.array([5, 3, 2]).astype('float32')] * self.party_num
        expect_results = [np.array([0, 1, 1])] * self.party_num
        ret = self.multi_party_run(target=self.ge,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)

    def test_lt(self):
        data_1 = [np.array([[65536, 65536, 65536],
                            [65536, 65536, 65536]]).astype('int64')] * self.party_num
        data_2 = [np.array([5, 3, 2]).astype('float32')] * self.party_num
        expect_results = [np.array([1, 0, 0])] * self.party_num
        ret = self.multi_party_run(target=self.lt,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)

    def test_le(self):
        data_1 = [np.array([[65536, 65536, 65536],
                            [65536, 65536, 65536]]).astype('int64')] * self.party_num
        data_2 = [np.array([5, 3, 2]).astype('float32')] * self.party_num
        expect_results = [np.array([1, 1, 0])] * self.party_num
        ret = self.multi_party_run(target=self.le,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)

    def test_equal(self):
        data_1 = [np.array([[65536, 65536, 65536],
                            [65536, 65536, 65536]]).astype('int64')] * self.party_num
        data_2 = [np.array([5, 3, 2]).astype('float32')] * self.party_num
        expect_results = [np.array([0, 1, 0])] * self.party_num
        ret = self.multi_party_run(target=self.equal,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)

    def test_not_equal(self):
        data_1 = [np.array([[65536, 65536, 65536],
                            [65536, 65536, 65536]]).astype('int64')] * self.party_num
        data_2 = [np.array([5, 3, 2]).astype('float32')] * self.party_num
        expect_results = [np.array([1, 0, 1])] * self.party_num
        ret = self.multi_party_run(target=self.not_equal,
                                   data_1=data_1,
                                   data_2=data_2,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)


if __name__ == '__main__':
    unittest.main()
