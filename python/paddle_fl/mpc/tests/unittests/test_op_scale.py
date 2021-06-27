#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module test scale op.

"""
import unittest
from multiprocessing import Manager
import numpy as np


import test_op_base
from op_test import OpTest
import paddle.fluid as fluid
import paddle.fluid.core as core
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')

class TestScaleOp(OpTest):
    def setUp(self):
        self.op_type = "mpc_scale"
        self.dtype = np.int64
        self.init_dtype_type()
        input_p = np.random.random((10, 10))
        self.inputs = {'X': self.lazy_share(input_p).astype(self.dtype)}
        self.attrs = {'scale': -2.3}
        self.outputs = {
            'Out': self.lazy_share(input_p * self.attrs['scale'])
        }

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3,)

    def test_check_grad(self):
        place =  core.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.05)


class TestScaleOpScaleVariable(OpTest):
    def setUp(self):
        self.op_type = "mpc_scale"
        self.dtype = np.int64
        self.init_dtype_type()
        self.scale = -2.3
        input_p = np.random.random((10, 10))
        self.inputs = {
            'X': self.lazy_share(input_p),
            'ScaleTensor': np.array([self.scale]).astype('float')
        }
        self.attrs = {}
        self.outputs = {'Out': self.lazy_share(input_p * self.scale)}

    def init_dtype_type(self):
        pass

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)

    def test_check_grad(self):
        place =  core.CPUPlace()
        self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.05)

if __name__ == "__main__":
    unittest.main()
