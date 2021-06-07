#  Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module test matrix mul op.

"""
import unittest
from multiprocessing import Manager
import numpy as np

from op_test import OpTest

import paddle.fluid as fluid
import paddle.fluid.core as core


class TestMulOp(OpTest):
    def setUp(self):
        self.op_type = "mpc_mul"
        self.dtype = np.int64
        self.init_dtype_type()
        self.init_input_output()
        self.inputs = {
            'X': self.lazy_share(self.x),
            'Y': self.lazy_share(self.y)
        }
        self.outputs = {'Out': self.lazy_share(self.out)}

    def init_dtype_type(self):
        pass

    def init_input_output(self):
        self.x = np.random.random((3, 4)).astype('float')
        self.y = np.random.random((4, 3)).astype('float')
        self.out = np.dot(self.x, self.y)

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)
    
    def test_check_grad_normal(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['X', 'Y'], 'Out')
    
    def test_check_grad_ingore_x(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))
    
    def test_check_grad_ingore_y(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


class TestMulOp2(OpTest):
    def setUp(self):
        self.op_type = "mpc_mul"
        self.dtype = np.int64
        self.init_dtype_type()
        self.init_input_output()
        self.inputs = {
            'X': self.lazy_share(self.x),
            'Y': self.lazy_share(self.y)
        }
        self.attrs = {
            'x_num_col_dims': 2,
            'y_num_col_dims': 2,
        }

        result = np.dot(self.x.reshape(2 * 3, 2 * 3),
                        self.y.reshape(3 * 2, 1 * 2 * 3))
        result = result.reshape(2, 3, 1, 2, 3)

        self.outputs = {'Out': self.lazy_share(result)}

    def init_dtype_type(self):
        pass

    def init_input_output(self):
        self.x = np.random.random((2, 3, 2, 3)).astype(np.float64)
        self.y = np.random.random((3, 2, 1, 2, 3)).astype(np.float64)

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)

    def test_check_grad_normal(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['X', 'Y'], 'Out')
    
    def test_check_grad_ingore_x(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['Y'], 'Out', max_relative_error=0.5, no_grad_set=set("X"))

    def test_check_grad_ingore_y(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['X'], 'Out', max_relative_error=0.5, no_grad_set=set('Y'))


if __name__ == "__main__":
    unittest.main()