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

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')

class TestOpBatchNorm(test_op_base.TestOpBase):

    def batch_norm(self, **kwargs):
        """
        Add two variables with one dimension.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[2, 3], dtype='int64')

        param_attr = fluid.ParamAttr(name='batch_norm_w', initializer=fluid.initializer.ConstantInitializer(value=21845))
        bias_attr = fluid.ParamAttr(name='batch_norm_b', initializer=fluid.initializer.ConstantInitializer(value=0))
        bn_out = pfl_mpc.layers.batch_norm(input=x, param_attr = param_attr, bias_attr = bias_attr)

        exe = fluid.Executor(place=fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        results = exe.run(feed={'x': d_1}, fetch_list=[bn_out])

        self.assertEqual(results[0].shape, (2, 2, 3))
        return_results.append(results[0])


    def test_batch_norm(self):

        data_1 = np.array(
            [[1, 1, 1], [5, 5, 5]]).astype('float32')
        
        expected_out = np.array(
            [[-1, -1, -1], [1, 1, 1]]).astype('float32')
        # print("input data_1: {} \n".format(data_1))

        data_1_shares = aby3.make_shares(data_1)

        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.batch_norm,
                                   data_1=data_1_all3shares,
                                   return_results=return_results)

        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        # print("revealed: {} \n".format(revealed))
        # print("expected: {} \n".format(expected_out))
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
