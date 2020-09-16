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
Decrypt and rescale for mean normalize demo.
"""
import sys
import numpy as np
import process_data
import prepare

data_path = prepare.data_path
# 0 for f_range, 1 for f_mean
# use decrypted global f_range and f_mean to rescaling local feature data
res = process_data.decrypt_data(data_path + 'result', (2, prepare.feat_width, ))

party = sys.argv[1]

input = np.load(data_path + 'feature_data.' + party + '.npy')

output = (input - res[1]) / res[0]

np.save(data_path + 'normalized_data.' + party, output)


