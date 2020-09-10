# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
mpc data preprocessing op layers.
"""
from paddle.fluid.data_feeder import check_type, check_dtype
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper

__all__ = ['mean_normalize']

def mean_normalize(f_min, f_max, f_mean, sample_num, total_sample_num):
    '''
    Mean normalization is a method used to normalize the range of independent
    variables or features of data.
    Refer to:
    https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization

    Args:
        f_min (Variable): A 2-D tensor with shape [P, N], where P is the party
                          num and N is the feature num. Each row contains the
                          local min feature val of N features.
        f_max (Variable): A 2-D tensor with shape [P, N], where P is the party
                          num and N is the feature num. Each row contains the
                          local max feature val of N features.
        f_mean (Variable): A 2-D tensor with shape [P, N], where P is the party
                           num and N is the feature num. Each row contains the
                           local min feature val of N features.
        sample_num (Variable): A 1-D tensor with shape [P], where P is the
                               party num. Each element contains sample num
                               of party_i.
        total_sample_num (int): Sum of sample nums from all party.

    Returns:
        f_range (Variable): A 1-D tensor with shape [N], where N is the
                            feature num. Each element contains global
                            range of feature_i.
        f_mean_out (Variable): A 1-D tensor with shape [N], where N is the
                               feature num. Each element contains global
                               range of feature_i.
    Examples:
        .. code-block:: python
            from multiprocessing import Manager
            from multiprocessing import Process
            import numpy as np
            import paddle.fluid as fluid
            import paddle_fl.mpc as pfl_mpc
            import mpc_data_utils as mdu
            import paddle_fl.mpc.data_utils.aby3 as aby3


            redis_server = "127.0.0.1"
            redis_port = 9937
            test_f_num = 100
            # party i owns 2 + 2*i rows of data
            test_row_split = range(2, 10, 2)


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


            class MeanNormDemo:

                def mean_normalize(self, **kwargs):
                    """
                    mean_normalize op ut
                    :param kwargs:
                    :return:
                    """
                    role = kwargs['role']

                    pfl_mpc.init("aby3", role, "localhost", redis_server, redis_port)

                    mi = pfl_mpc.data(name='mi', shape=self.input_size, dtype='int64')
                    ma = pfl_mpc.data(name='ma', shape=self.input_size, dtype='int64')
                    me = pfl_mpc.data(name='me', shape=self.input_size, dtype='int64')
                    sn = pfl_mpc.data(name='sn', shape=self.input_size, dtype='int64')

                    out0, out1 = pfl_mpc.layers.mean_normalize(f_min=mi, f_max=ma,
                            f_mean=me, sample_num=sn, total_sample_num=self.total_num)

                    exe = fluid.Executor(place=fluid.CPUPlace())

                    f_range, f_mean = exe.run(feed={'mi': kwargs['min'],
                        'ma': kwargs['max'], 'me': kwargs['mean'],
                        'sn': kwargs['sample_num']},fetch_list=[out0, out1])

                    self.f_range_list.append(f_range)
                    self.f_mean_list.append(f_mean)

                def run(self):
                    f_nums = test_f_num
                    sample_nums = np.array(test_row_split)
                    mat, mi, ma, me = gen_data(f_nums, sample_nums)

                    self.input_size = [len(sample_nums), f_nums]
                    self.total_num = mat.shape[0]

                    # simulating encrypting data
                    share = lambda x: np.array([x * mdu.mpc_one_share] * 2).astype('int64').reshape(
                            [2] + list(x.shape))

                    self.f_range_list = Manager().list()
                    self.f_mean_list = Manager().list()

                    proc = list()
                    for role in range(3):
                        args = {'role': role, 'min': share(mi), 'max': share(ma),
                                'mean': share(me), 'sample_num': share(sample_nums)}
                        p = Process(target=self.mean_normalize, kwargs=args)

                        proc.append(p)
                        p.start()

                    for p in proc:
                        p.join()

                    f_r = aby3.reconstruct(np.array(self.f_range_list))
                    f_m = aby3.reconstruct(np.array(self.f_mean_list))

                    plain_r, plain_m = mean_norm_naive(mat)
                    print("max error in featrue range:", np.max(np.abs(f_r - plain_r)))
                    print("max error in featrue mean:", np.max(np.abs(f_m - plain_m)))


            MeanNormDemo().run()
    '''
    helper = MpcLayerHelper("mean_normalize", **locals())

    # dtype = helper.input_dtype()
    dtype = 'int64'

    check_dtype(dtype, 'f_min', ['int64'], 'mean_normalize')
    check_dtype(dtype, 'f_max', ['int64'], 'mean_normalize')
    check_dtype(dtype, 'f_mean', ['int64'], 'mean_normalize')
    check_dtype(dtype, 'sample_num', ['int64'], 'mean_normalize')

    f_range = helper.create_mpc_variable_for_type_inference(dtype=f_min.dtype)
    f_mean_out= helper.create_mpc_variable_for_type_inference(dtype=f_min.dtype)

    op_type = 'mean_normalize'

    helper.append_op(
        type='mpc_' + op_type,
        inputs={
            "Min": f_min,
            "Max": f_max,
            "Mean": f_mean,
            "SampleNum": sample_num,
            },
        outputs={
            "Range": f_range,
            "MeanOut": f_mean_out,
             },
        attrs={
            # TODO: remove attr total_sample_num, reducing sample_num instead
            "total_sample_num": total_sample_num,
        })

    return f_range, f_mean_out
