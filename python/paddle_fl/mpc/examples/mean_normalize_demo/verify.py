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
Verification for mean normalize demo.
"""
import prepare
import process_data
import numpy as np

# 0 for f_range, 1 for f_mean
# use decrypted global f_range and f_mean to rescaling local feature data
res = process_data.decrypt_data(prepare.data_path + 'result', (
    2,
    prepare.feat_width, ))

# reconstruct plaintext global data to verify
row, col = sum(prepare.sample_nums), prepare.feat_width
plain_mat = np.empty((row, col))

row = 0
for i, num in enumerate(prepare.sample_nums):
    m = np.load(prepare.data_path + 'feature_data.' + str(i) + '.npy')
    plain_mat[row:row + num] = m
    row += num


def mean_normalize(f_mat):
    """
    get plain text f_range & f_mean
    """
    ma = np.amax(f_mat, axis=0)
    mi = np.amin(f_mat, axis=0)

    return ma - mi, np.mean(f_mat, axis=0)


plain_range, plain_mean = mean_normalize(plain_mat)

print("max error in featrue range:", np.max(np.abs(res[0] - plain_range)))
print("max error in featrue mean:", np.max(np.abs(res[1] - plain_mean)))
