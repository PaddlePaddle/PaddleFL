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
        with open(r'Input-P0.list', 'r') as file:
            content_list_0 = file.readlines()
        contentall_0 = np.array([float(x) for x in content_list_0])

        data_1 = np.full((1), fill_value=np.max(contentall_0))
        data_1_shares = aby3.make_shares(data_1)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        np.save('data_C0_P0.npy', data_1_all3shares[0])
        np.save('data_C1_P0.npy', data_1_all3shares[1])
        np.save('data_C2_P0.npy', data_1_all3shares[2])

        with open(r'Input-P1.list', 'r') as file:
            content_list_1 = file.readlines()
        contentall_1 = np.array([float(x) for x in content_list_1])

        data_2 = np.full((1), fill_value=np.max(contentall_1))
        data_2_shares = aby3.make_shares(data_2)
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])
        np.save('data_C0_P1.npy', data_2_all3shares[0])
        np.save('data_C1_P1.npy', data_2_all3shares[1])
        np.save('data_C2_P1.npy', data_2_all3shares[2])

        contentall = np.append(contentall_0,contentall_1)
        print(np.max(contentall))

