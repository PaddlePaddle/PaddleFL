
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
This module test data preprocessing.

"""
import unittest
from multiprocessing import Manager
import test_op_base
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import mpc_data_utils as mdu
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


def mean_norm_naive(f_mat):
    ma = np.amax(f_mat, axis=0)
    mi = np.amin(f_mat, axis=0)

    return ma - mi, np.mean(f_mat, axis=0)


def gen_data(f_num, sample_nums):
    f_mat = np.random.rand(np.sum(sample_nums), f_num)

    f_min, f_max, f_mean = [], [], []

    prev_idx = 0

    for n in sample_nums:

        i = prev_idx
        j = i + n

        ma = np.amax(f_mat[i:j], axis=0)
        mi = np.amin(f_mat[i:j], axis=0)
        me = np.mean(f_mat[i:j], axis=0)

        f_min.append(mi)
        f_max.append(ma)
        f_mean.append(me)

        prev_idx += n

    f_min = np.array(f_min).reshape(sample_nums.size, f_num)
    f_max = np.array(f_max).reshape(sample_nums.size, f_num)
    f_mean = np.array(f_mean).reshape(sample_nums.size, f_num)

    return f_mat, f_min, f_max, f_mean

class TestOpMeanNormalize(test_op_base.TestOpBase):

    def mean_normalize(self, **kwargs):
        """
        mean_normalize op ut
        :param kwargs:
        :return:
        """
        role = kwargs['role']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))

        mi = pfl_mpc.data(name='mi', shape=self.input_size, dtype='int64')
        ma = pfl_mpc.data(name='ma', shape=self.input_size, dtype='int64')
        me = pfl_mpc.data(name='me', shape=self.input_size, dtype='int64')
        sn = pfl_mpc.data(name='sn', shape=self.input_size[:-1], dtype='int64')

        out0, out1 = pfl_mpc.layers.mean_normalize(f_min=mi,
                f_max=ma, f_mean=me, sample_num=sn)

        exe = fluid.Executor(place=fluid.CPUPlace())

        f_range, f_mean = exe.run(feed={'mi': kwargs['min'],
            'ma': kwargs['max'], 'me': kwargs['mean'], 'sn': kwargs['sample_num']},fetch_list=[out0, out1])

        self.f_range_list.append(f_range)
        self.f_mean_list.append(f_mean)

    def test_mean_normalize(self):

        f_nums = 100
        sample_nums = np.array(range(2, 10, 2))
        mat, mi, ma, me = gen_data(f_nums, sample_nums)

        self.input_size = [len(sample_nums), f_nums]

        share = lambda x: np.array([x * aby3.MPC_ONE_SHARE] * 2).astype('int64').reshape(
                [2] + list(x.shape))

        self.f_range_list = Manager().list()
        self.f_mean_list = Manager().list()

        ret = self.multi_party_run(target=self.mean_normalize,
                min=share(mi), max=share(ma), mean=share(me), sample_num=share(sample_nums))

        self.assertEqual(ret[0], True)

        f_r = aby3.reconstruct(np.array(self.f_range_list))
        f_m = aby3.reconstruct(np.array(self.f_mean_list))

        plain_r, plain_m = mean_norm_naive(mat)
        self.assertTrue(np.allclose(f_r, plain_r, atol=1e-4))
        self.assertTrue(np.allclose(f_m, plain_m, atol=1e-4))


if __name__ == '__main__':
    unittest.main()

