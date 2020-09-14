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
from .math import reduce_sum

__all__ = ['mean_normalize']

def mean_normalize(f_min, f_max, f_mean, sample_num):
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

    Returns:
        f_range (Variable): A 1-D tensor with shape [N], where N is the
                            feature num. Each element contains global
                            range of feature_i.
        f_mean_out (Variable): A 1-D tensor with shape [N], where N is the
                               feature num. Each element contains global
                               range of feature_i.
    Examples:
        .. code-block:: python
            import paddle_fl.mpc as pfl_mpc

            pfl_mpc.init("aby3", role, "localhost", redis_server, redis_port)

            # 2 for share, 4 for 4 party, 100 for feat_num
            input_size = [2, 4, 100]

            mi = pfl_mpc.data(name='mi', shape=input_size, dtype='int64')
            ma = pfl_mpc.data(name='ma', shape=input_size, dtype='int64')
            me = pfl_mpc.data(name='me', shape=input_size, dtype='int64')
            sn = pfl_mpc.data(name='sn', shape=input_size[:-1], dtype='int64')

            out0, out1 = pfl_mpc.layers.mean_normalize(f_min=mi, f_max=ma,
                    f_mean=me, sample_num=sn)

            exe = fluid.Executor(place=fluid.CPUPlace())

            # feed encrypted data
            f_range, f_mean = exe.run(feed={'mi': f_min, 'ma': f_max,
            'me': f_mean, 'sn': sample_num}, fetch_list=[out0, out1])
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

    total_num = reduce_sum(sample_num)

    op_type = 'mean_normalize'

    helper.append_op(
        type='mpc_' + op_type,
        inputs={
            "Min": f_min,
            "Max": f_max,
            "Mean": f_mean,
            "SampleNum": sample_num,
            "TotalNum": total_num,
            },
        outputs={
            "Range": f_range,
            "MeanOut": f_mean_out,
             },
        )

    return f_range, f_mean_out
