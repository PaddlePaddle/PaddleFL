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
This module test add op.

"""
import unittest
from multiprocessing import Manager
import test_op_base
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


class TestOpSoftmaxWithCrossEntropy(test_op_base.TestOpBase):

    def softmax_with_cross_entropy(self, **kwargs):
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
        x = pfl_mpc.data(name='x', shape=[2], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[2], dtype='int64')
        cost, softmax = pfl_mpc.layers.softmax_with_cross_entropy(x, y, soft_label=True, return_softmax=True)
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'x': d_1, 'y': d_2}, fetch_list=[softmax])

        self.assertEqual(results[0].shape, (2, 2))
        return_results.append(results[0])


    def test_softmax_with_cross_entropy(self):

        data_1 = np.array(
            [1, 1]).astype('float32')
        data_2 = np.array(
            [1, 0]).astype('float32')
        
        expected_out = np.array(
            [0.5, 0.5]).astype('float32')
        #print("input data_1: {} \n".format(data_1))

        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)

        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.softmax_with_cross_entropy,
                                   data_1=data_1_all3shares,
                                   data_2=data_2_all3shares,
                                   return_results=return_results)

        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        #print("revealed: {} \n".format(revealed))
        #print("expected: {} \n".format(expected_out))
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
