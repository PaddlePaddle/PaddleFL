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
            content_list0 = file.readlines()
        contentall_0 = np.array([float(x) for x in content_list0])
        with open(r'Input-P1.list', 'r') as file:
            content_list1 = file.readlines()
        contentall_1 = np.array([float(x) for x in content_list1])
        with open(r'Input-P2.list', 'r') as file:
            content_list2 = file.readlines()
        contentall_2 = np.array([float(x) for x in content_list2])

        data_1 = np.sort(contentall_0)
        data_2 = np.sort(contentall_1)
        data_3 = np.sort(contentall_2)
        len_1, len_2, len_3 = data_1.shape[0], data_2.shape[0], data_3.shape[0]
        total = len_1 + len_2 + len_3

        if total % 2 == 1:
            total_data = np.append(data_1,data_2)
            total_data = np.append(total_data,data_3)
            print(np.sort(total_data)[total // 2:total // 2 + 1]) // Median calculated in clear text
            #print(np.sort(total_data))
        else:
            total_data = np.append(data_1,data_2)
            total_data = np.append(total_data,data_3)
            print((np.sort(total_data)[total // 2 - 1:total // 2] + np.sort(total_data)[total // 2:total // 2 + 1]) * 0.5) // Median calculated in clear text
            #print(np.sort(total_data))
            data_tmp = np.array([0.5])
            data_tmp_shares = aby3.make_shares(data_tmp)
            data_tmp_all3shares = np.array([aby3.get_shares(data_tmp_shares, i) for i in range(3)])
            np.save('data_C0_tmp.npy', data_tmp_all3shares[0])
            np.save('data_C1_tmp.npy', data_tmp_all3shares[1])
            np.save('data_C2_tmp.npy', data_tmp_all3shares[2])
        data_1_shares = aby3.make_shares(data_1)
        data_2_shares = aby3.make_shares(data_2)
        data_3_shares = aby3.make_shares(data_3)
        data_1_all3shares = np.array([aby3.get_shares(data_1_shares, i) for i in range(3)])
        data_2_all3shares = np.array([aby3.get_shares(data_2_shares, i) for i in range(3)])
        data_3_all3shares = np.array([aby3.get_shares(data_3_shares, i) for i in range(3)])
        #print(data_1_all3shares[0].shape)
        np.save('data_C0_P1.npy', data_1_all3shares[0])
        np.save('data_C0_P2.npy', data_2_all3shares[0])
        np.save('data_C0_P3.npy', data_3_all3shares[0])
        np.save('data_C1_P1.npy', data_1_all3shares[1])
        np.save('data_C1_P2.npy', data_2_all3shares[1])
        np.save('data_C1_P3.npy', data_3_all3shares[1])
        np.save('data_C2_P1.npy', data_1_all3shares[2])
        np.save('data_C2_P2.npy', data_2_all3shares[2])
        np.save('data_C2_P3.npy', data_3_all3shares[2])

