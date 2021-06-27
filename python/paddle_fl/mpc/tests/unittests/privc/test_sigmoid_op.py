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
This module test relu op.

"""
import unittest
from multiprocessing import Manager
import numpy as np

from op_test import OpTest

import paddle.fluid as fluid
import paddle.fluid.core as core
from scipy.special import logit
from scipy.special import expit

class TestSigmoidCrossEntropyWithLogitsOp1(OpTest):
    """Test sigmoid_cross_entropy_with_logit_op with binary label
    """

    def setUp(self):
        self.op_type = "mpc_sigmoid_cross_entropy_with_logits"

        self.init_input_output()
        self.inputs = {
            'X': self.lazy_share(self.x),
            'Label': self.lazy_share(self.label)
        }
        self.outputs = {'Out': self.lazy_share(self.out)}

    def init_input_output(self):
        batch_size = 10
        num_classes = 4
        self.x = logit(
                np.random.uniform(0, 1, (batch_size, num_classes))
                .astype("float64"))
        self.label = np.random.randint(0, 2, (batch_size, num_classes))

        # approximate sigmoid with f(x) = {=0, x < -0.5;  x + 0.5, -0.5 <= x <= 0.5; 1, x> 0.5}
        self.out = np.minimum(np.maximum(0, self.x + 0.5), 1)

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)
    def test_check_grad(self):
        place = core.CPUPlace()
        # TODO max_relative_error is too large, find reason
        self.check_grad_with_place(place, ['X'], "Out", max_relative_error = 50)
    

if __name__ == '__main__':
    unittest.main()
