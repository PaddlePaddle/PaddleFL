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
This module test square_error_cost op.

"""
import unittest
from multiprocessing import Manager
import test_op_base
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


class TestOpSquareErrorCost(test_op_base.TestOpBase):

    def square_error_cost(self, **kwargs):
        """
        Normal case.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        data_1 = pfl_mpc.data(name='data_1', shape=[2, 2], dtype='int64')
        data_2 = pfl_mpc.data(name='data_2', shape=[2, 2], dtype='int64')
        cost = pfl_mpc.layers.square_error_cost(input=data_1, label=data_2)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'data_1': d_1, 'data_2': d_2}, fetch_list=[cost])

        self.assertEqual(results[0].shape, (2, 2, 2))
        return_results.append(results[0])

    def test_square_error_cost(self):
        data_1 = np.arange(0, 4).reshape((2, 2))
        data_2 = np.full(shape=(2, 2), fill_value=2)
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.square_error_cost,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = np.array([[4, 1], [0, 1]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
