# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

if __name__ == '__main__':
        return_results = list()
        result_C0 = np.load('result_C0.npy', allow_pickle=True)
        result_C1 = np.load('result_C1.npy',allow_pickle=True)
        result_C2 = np.load('result_C2.npy',allow_pickle=True)
        return_results.append(result_C0)
        return_results.append(result_C1)
        return_results.append(result_C2)
        revealed = aby3.reconstruct(np.array(return_results))
        print(revealed) // Maximum obtained after ciphertext calculation and decryption
