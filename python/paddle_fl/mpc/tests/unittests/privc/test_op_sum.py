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
This module test fc op.

"""
import unittest
from multiprocessing import Manager
import numpy as np

import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle_fl.mpc as pfl_mpc
import test_op_base

scaling_factor = 32

class TestOpSum(test_op_base.TestOpBase):

    def sum(self, **kwargs):
        """
        Normal case.
        :param kwargs:
        :return:
        """
        role = kwargs['role']
        d_1 = kwargs['data_1'][role]
        d_2 = kwargs['data_2'][role]
        return_results = kwargs['return_results']

        pfl_mpc.init("privc", role, "localhost", self.server, int(self.port))
        data_1 = pfl_mpc.data(name='data_1', shape=[3, 2], dtype='int64')
        data_2 = pfl_mpc.data(name='data_2', shape=[3, 2], dtype='int64')
        out = pfl_mpc.layers.sum([data_1, data_2])
        exe = fluid.Executor(place=fluid.CPUPlace())
        results = exe.run(feed={'data_1': d_1, 'data_2': d_2}, fetch_list=[out])

        self.assertEqual(results[0].shape, (3, 2))
        return_results.append(results[0])

    def reconstruct(self, shares, type=np.float):
        return (shares[0] + shares[1]) * 2**-scaling_factor

    def share(self, plain):
        return np.array(plain * 2**scaling_factor / 2).astype("int64")

    def test_sum(self):
        data_1 = np.arange(0, 6).reshape((3, 2))
        data_1_shares = self.share(data_1)
        data_1_all2shares = [data_1_shares, data_1_shares]

        data_2 = np.arange(0, 6).reshape((3, 2))
        data_2_shares = self.share(data_2)
        data_2_all2shares = [data_2_shares, data_2_shares]

        return_results = Manager().list()
        ret = self.multi_party_run(target=self.sum,
                                   data_1=data_1_all2shares,
                                   data_2=data_2_all2shares,
                                   return_results=return_results)
        self.assertEqual(ret[0], True)
        revealed = self.reconstruct(np.array(return_results))
        expected_out = np.array([[0, 2], [4, 6], [8, 10]])
        self.assertTrue(np.allclose(revealed, expected_out, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
