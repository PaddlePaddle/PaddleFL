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


class TestOpPool2d(test_op_base.TestOpBase):

    def pool2d(self, **kwargs):
        """
        Add two variables with one dimension.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[1, 1, 4, 6], dtype='int64')

        pool_out = pfl_mpc.layers.pool2d(input=x, pool_size=2, pool_stride=2)

        exe = fluid.Executor(place=fluid.CPUPlace())
        #exe.run(fluid.default_startup_program())
        results = exe.run(feed={'x': d_1}, fetch_list=[pool_out])

        self.assertEqual(results[0].shape, (2, 1, 1, 2, 3))
        return_results.append(results[0])


    def test_pool2d(self):

        data_1 = np.array(
            [[[[1, 2, 3, 4, 0, 100],
               [5, 6, 7, 8, 0, 100],
               [9, 10, 11, 12, 0, 200],
               [13, 14, 15, 16, 0, 200]]]]).astype('float32')

        expected_out = np.array(
            [[[[6, 8, 100],
               [14, 16, 200]]]]).astype('float32')
        # print("input data_1: {} \n".format(data_1))

        data_1_shares = aby3.make_shares(data_1)

        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.pool2d,
                                   data_1=data_1_all3shares,
                                   return_results=return_results)

        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        #print("revealed: {} \n".format(revealed))
        #print("expected: {} \n".format(expected_out))
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-2))


if __name__ == '__main__':
    unittest.main()
