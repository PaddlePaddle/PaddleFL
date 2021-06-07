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
This module test elementwise add op.

"""
import unittest
from multiprocessing import Manager
import numpy as np

from op_test import OpTest

import paddle.fluid as fluid
import paddle.fluid.core as core


class TestElementwiseAddOp(OpTest):
    def init_kernel_type(self):
        self.use_mkldnn = False

    def setUp(self):
        self.op_type = "mpc_elementwise_add"
        self.init_dtype()
        self.init_input_output()
        self.init_kernel_type()
        self.init_axis()

        self.inputs = {
            'X': self.lazy_share(self.x),
            'Y': self.lazy_share(self.y)
        }
        self.attrs = {'axis': self.axis, 'use_mkldnn': self.use_mkldnn}
        self.outputs = {'Out': self.lazy_share(self.out)}

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)

    def test_check_grad_normal(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place, ['X', 'Y'], 'Out',  max_relative_error=0.5)
    
    def test_check_grad_ingore_x(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['Y'],
            'Out',
            no_grad_set=set("X"),
             max_relative_error=0.5)

    def test_check_grad_ingore_y(self):
        place = core.CPUPlace()
        self.check_grad_with_place(
            place,
            ['X'],
            'Out',
            no_grad_set=set('Y'),
            max_relative_error=2.0)
    
    def init_input_output(self):
        self.x = np.random.uniform(0.1, 1, [3, 2]).astype(np.float64)
        self.y = np.random.uniform(0.1, 1, [3, 2]).astype(np.float64)
        self.out = np.add(self.x, self.y)

    def init_dtype(self):
        self.dtype = np.int64

    def init_axis(self):
        self.axis = -1


class TestElementwiseAddOp_scalar(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(np.float64)
        self.y = np.random.rand(4).astype(np.float64)
        self.out = self.x + self.y


class TestElementwiseAddOp_Vector(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.random((4, )).astype(np.float64)
        self.y = np.random.random((4, )).astype(np.float64)
        self.out = np.add(self.x, self.y)


class TestElementwiseAddOp_broadcast_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(4, 2, 3).astype(np.float64)
        self.y = np.random.rand(4).astype(np.float64)
        self.out = self.x + self.y.reshape(4, 1, 1)

    def init_axis(self):
        self.axis = 0


class TestElementwiseAddOp_broadcast_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 4, 3).astype(np.float64)
        self.y = np.random.rand(4).astype(np.float64)
        self.out = self.x + self.y.reshape(1, 4, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_broadcast_2(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(np.float64)
        self.y = np.random.rand(4).astype(np.float64)
        self.out = self.x + self.y.reshape(1, 1, 4)


class TestElementwiseAddOp_broadcast_3(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4, 3).astype(np.float64)
        self.y = np.random.rand(3, 4).astype(np.float64)
        self.out = self.x + self.y.reshape(1, 3, 4, 1)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_rowwise_add_0(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(2, 3, 4).astype(np.float64)
        self.y = np.random.rand(3, 4).astype(np.float64)
        self.out = self.x + self.y.reshape(1, 3, 4)

    def init_axis(self):
        self.axis = 1


class TestElementwiseAddOp_rowwise_add_1(TestElementwiseAddOp):
    def init_input_output(self):
        self.x = np.random.rand(4, 1).astype(np.float64)
        self.y = np.random.rand(1).astype(np.float64)
        self.out = self.x + self.y.reshape(1, 1)

    def init_axis(self):
        self.axis = 1


if __name__ == '__main__':
    unittest.main()
