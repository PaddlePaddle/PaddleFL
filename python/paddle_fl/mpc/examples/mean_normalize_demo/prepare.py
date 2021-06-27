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
Prepare data for mean normalize demo.
"""
import numpy as np
import process_data
from paddle_fl.mpc.data_utils.data_utils import get_datautils


mpc_du = get_datautils('aby3')
data_path = process_data.data_path

feat_width = 100
# assume data owner i has sample_nums[i] samples
sample_nums = [1, 2, 3, 4]

def gen_random_data():

    for i, num in enumerate(sample_nums):
        suffix = '.' + str(i)

        f_mat = np.random.rand(num, feat_width)
        np.save(data_path + 'feature_data' + suffix, f_mat)

        process_data.generate_encrypted_data(i, f_mat)

    mpc_du.save_shares(process_data.encrypted_data(np.array(sample_nums)),
            data_path + 'sample_num')

if __name__ == "__main__":
    gen_random_data()
