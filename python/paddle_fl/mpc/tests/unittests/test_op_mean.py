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
This module test mean op.

"""
import unittest
from multiprocessing import Manager
import test_op_base
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


class TestOpMean(test_op_base.TestOpBase):

    def mean(self, **kwargs):
        """
        Mean.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        data_1 = pfl_mpc.data(name='data_1', shape=[2, 4], dtype='int64')
        op_mean = pfl_mpc.layers.mean(data_1)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'data_1': d_1}, fetch_list=[op_mean])

        self.assertEqual(results[0].shape, (2, 1))
        return_results.append(results[0])

    def test_mean(self):
        """
        Test normal case.
        :return:
        """
        data_1 = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8]])
        data_1_shares = aby3.make_shares(data_1)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        return_results = Manager().list()
        ret = self.multi_party_run(target=self.mean,
                                   data_1=data_1_all3shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        expected_out = np.array([4.5])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
