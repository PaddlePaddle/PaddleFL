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
Mean normalize demo.
"""
import sys
import numpy as np

import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils
import prepare
import process_data

mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

role, server, port = sys.argv[1], sys.argv[2], sys.argv[3]
role, port = int(role), int(port)

share_num = 2

party_num = len(prepare.sample_nums)

feat_num = prepare.feat_width

data_path = prepare.data_path


def get_shares(path):
    """
    collect encrypted feature stats from all data owners
    """
    data = []
    for i in range(party_num):
        reader = mpc_du.load_shares(
            path + '.' + str(i), id=role, shape=(feat_num, ))
        data.append([x for x in reader()])
    data = np.array(data).reshape([party_num, share_num, feat_num])
    return np.transpose(data, axes=[1, 0, 2])


def get_sample_num(path):
    """
    get encrypted sample nums
    """
    reader = mpc_du.load_shares(path, id=role, shape=(party_num, ))
    for n in reader():
        return n


f_max = get_shares(data_path + 'feature_max')
f_min = get_shares(data_path + 'feature_min')
f_mean = get_shares(data_path + 'feature_mean')
sample_num = get_sample_num(data_path + 'sample_num')

pfl_mpc.init(mpc_protocol_name, int(role), "localhost", server, int(port))

shape = [party_num, feat_num]

mi = pfl_mpc.data(name='mi', shape=shape, dtype='int64')
ma = pfl_mpc.data(name='ma', shape=shape, dtype='int64')
me = pfl_mpc.data(name='me', shape=shape, dtype='int64')
sn = pfl_mpc.data(name='sn', shape=shape[:-1], dtype='int64')

out0, out1 = pfl_mpc.layers.mean_normalize(
    f_min=mi, f_max=ma, f_mean=me, sample_num=sn)

exe = fluid.Executor(place=fluid.CPUPlace())

f_range, f_mean = exe.run(
    feed={'mi': f_min,
          'ma': f_max,
          'me': f_mean,
          'sn': sample_num},
    fetch_list=[out0, out1])
result = np.transpose(np.array([f_range, f_mean]), axes=[1, 0, 2])

result_file = data_path + "result.part{}".format(role)
with open(result_file, 'wb') as f:
    f.write(result.tostring())
