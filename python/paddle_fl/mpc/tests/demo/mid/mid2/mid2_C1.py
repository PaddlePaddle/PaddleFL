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

import unittest
from multiprocessing import Manager
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import test_op_base
from paddle_fl.mpc.data_utils.data_utils import get_datautils

aby3 = get_datautils('aby3')

class Solution(test_op_base.TestOpBase):
    def findMedianSortedArrays(self, **kwargs):
        def getKthElement(k):
            index1, index2 = 0, 0
            op_sub = pfl_mpc.layers.elementwise_sub(x=x, y=y)
            op_gt = pfl_mpc.layers.greater_than(x=x, y=zero)
            while True:
                if index1 ==  d_1_length:
                    return d_2[:,index2 + k - 1:index2 + k],index1, index2 + k,1
                if index2 ==  d_2_length:
                    return d_1[:,index1 + k - 1:index1 + k],index1 + k, index2,0
                if k == 1:
                    d_tmp = exe.run(feed={'x': d_1[:,index1:index1+1], 'y': d_2[:,index2:index2+1],'zero': d_zero}, fetch_list=[op_sub])
                    results = exe.run(feed={'x': d_tmp[0],'y':d_2[:,index2:index2+1], 'zero': d_zero}, fetch_list=[op_gt])
                    if results[0] == 1:
                        return d_2[:,index2:index2+1],index1, index2 + 1,1
                    else:
                        return d_1[:,index1:index1+1],index1 + 1, index2,0

                newIndex1 = min(index1 + k // 2 - 1,  d_1_length - 1)
                newIndex2 = min(index2 + k // 2 - 1,  d_2_length - 1)
                d_tmp = exe.run(feed={'x': d_1[:,newIndex1:newIndex1+1], 'y': d_2[:,newIndex2:newIndex2+1], 'zero': d_zero}, fetch_list=[op_sub])
                results = exe.run(feed={'x': d_tmp[0],'y': d_2[:,newIndex2:newIndex2+1], 'zero': d_zero}, fetch_list=[op_gt])
                if results[0] == 0:
                    k -= newIndex1 - index1 + 1
                    index1 = newIndex1 + 1
                else:
                    k -= newIndex2 - index2 + 1
                    index2 = newIndex2 + 1

        def getNextElement(index1,index2,num):
            op_sub = pfl_mpc.layers.elementwise_sub(x=x, y=y)
            op_gt = pfl_mpc.layers.greater_than(x=x, y=zero)
            if(num == 0 and index1 ==  d_1_length):
                return d_2[:,index2:index2+1]     
            if(num == 1 and index2 ==  d_2_length):
                return d_1[:,index1:index1+1]
            else:
                d_tmp = exe.run(feed={'x': d_1[:,index1:index1+1], 'y': d_2[:,index2:index2+1],'zero': d_zero}, fetch_list=[op_sub])
                results = exe.run(feed={'x': d_tmp[0],'y':d_2[:,index2:index2+1], 'zero': d_zero}, fetch_list=[op_gt])
                if results[0] == 1:
                    return d_2[:,index2:index2+1]
                else:
                    return d_1[:,index1:index1+1]

        role = 1
        d_1 = np.load('data_C1_P1.npy',allow_pickle=True)
        d_2 = np.load('data_C1_P2.npy',allow_pickle=True)
        d_zero = np.full((1), fill_value=0).astype('float32')
        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))
        x = pfl_mpc.data(name='x', shape=[1], dtype='int64')
        y = pfl_mpc.data(name='y', shape=[1], dtype='int64')
        zero = fluid.data(name='zero', shape=[1], dtype='float32')
        op_add = pfl_mpc.layers.elementwise_add(x=x, y=y)
        math_mul = pfl_mpc.layers.elementwise_mul(x, y)
        exe = fluid.Executor(place=fluid.CPUPlace())

        d_1_length, d_2_length = d_1.shape[1], d_2.shape[1]
        totalLength = d_1_length + d_2_length
        if totalLength % 2 == 1:
            results = getKthElement((totalLength + 1) // 2)
            np.save('result_C1.npy', results[0])
        else:
            op_add = pfl_mpc.layers.elementwise_add(x=x, y=y)
            mid_pre = getKthElement(totalLength // 2)
            mid_post = getNextElement(mid_pre[1],mid_pre[2],mid_pre[3])
            d_tmp = np.load('data_C1_tmp.npy',allow_pickle=True)
            tmp = exe.run(feed={'x': mid_pre[0], 'y': mid_post, 'zero': d_zero}, fetch_list=[op_add])
            results = exe.run(feed={'x': tmp[0],'y':d_tmp,'zero': d_zero}, fetch_list=[math_mul])
            np.save('result_C1.npy', results[0])

    def test_mid2_C1(self):
        ret = self.multi_party_run1(target=self.findMedianSortedArrays)

if __name__ == '__main__':
    unittest.main()
