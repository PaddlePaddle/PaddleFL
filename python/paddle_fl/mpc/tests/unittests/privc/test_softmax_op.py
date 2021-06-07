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
This module test softmax(argmax) op.

"""
import unittest
from multiprocessing import Manager
import numpy as np

from op_test import OpTest

import paddle.fluid as fluid
import paddle.fluid.core as core

def stable_softmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    # clip to shiftx, otherwise, when calc loss with
    # log(exp(shiftx)), may get log(0)=INF
    shiftx = (x - np.max(x)).clip(-64.)
    exps = np.exp(shiftx)
    return exps / np.sum(exps)


def cross_entropy(softmax, label, soft_label, axis, ignore_index=-1):
    if soft_label:
        return (-label * np.log(softmax)).sum(axis=axis, keepdims=True)

    shape = softmax.shape
    axis %= len(shape)
    n = int(np.prod(shape[:axis]))
    axis_dim = shape[axis]
    remain = int(np.prod(shape[axis + 1:]))
    softmax_reshape = softmax.reshape((n, axis_dim, remain))
    label_reshape = label.reshape((n, 1, remain))
    result = np.zeros_like(label_reshape, dtype=softmax.dtype)
    for i in range(n):
        for j in range(remain):
            lbl = label_reshape[i, 0, j]
            if lbl != ignore_index:
                result[i, 0, j] -= np.log(softmax_reshape[i, lbl, j])
    return result.reshape(label.shape)

"""
class TestSoftmaxOp(OpTest):
    def setUp(self):
        self.op_type = "mpc_softmax_with_cross_entropy"
        self.dtype = np.int64
        self.init_input_output()
        self.inputs = {
            'Logits': self.lazy_share(self.logit),
            'Label': self.lazy_share(self.label)
        }
        self.attrs = {'is_test': True, 'axis': -1, 'soft_label': True}
        self.outputs = {'Softmax': self.lazy_share(self.out), 'Loss': np.array([]).astype("int64")}

    def to_one_hot(self, x, depth):
        out = np.zeros(shape=(np.product(x.shape), depth)).astype('float')
        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0
        return out

    def init_input_output(self):
        # x in range (-100, 100)
        self.logit = 200 * np.random.random((10, 20)).astype('float') - 100
        self.label = self.logit
        arg_max = np.argmax(self.logit, axis=1)
        self.out = self.to_one_hot(arg_max, self.logit.shape[1])

    def test_check_output(self):
        place =  core.CPUPlace()
        self.check_output_with_place(place, no_check_set=("Loss"))

    def test_check_grad(self):
        place =  core.CPUPlace()
        self.check_grad_with_place(place, ['Logits'], "Loss", max_relative_error=0.1)
"""
class TestSoftmaxWithCrossEntropyOp(OpTest):
    """
    Test softmax with cross entropy operator with discreate one-hot labels.
    """

    def initParams(self):
        self.op_type = "mpc_softmax_with_cross_entropy"
        self.numeric_stable_mode = False
        self.soft_label = True
        self.dtype = np.int64
        self.axis = -1
        self.ignore_index = -1
        self.shape = [5, 4]

    def setUp(self):
        self.initParams()

        logits = getattr(
            self, "logits",
            np.random.uniform(0.1, 1.0, self.shape).astype(np.float64))
        softmax = np.apply_along_axis(stable_softmax, self.axis, logits)

        if self.soft_label:
            labels = np.random.uniform(0.1, 1.0, self.shape).astype(np.float64)
            labels /= np.sum(labels, axis=self.axis, keepdims=True)
        else:
            axis_dim = self.shape[self.axis]
            self.shape[self.axis] = 1
            labels = np.random.randint(0, axis_dim, self.shape, dtype="int64")

        loss = cross_entropy(softmax, labels, self.soft_label, self.axis,
                             self.ignore_index)

        self.inputs = {"Logits": self.lazy_share(logits), "Label": self.lazy_share(labels)}
        self.outputs = {
            "Softmax": self.lazy_share(softmax.astype(np.float64)),
            "Loss": self.lazy_share(loss.astype(np.float64))
        }
        self.attrs = {
            "numeric_stable_mode": self.numeric_stable_mode,
            "soft_label": self.soft_label,
            "use_relu": False,
        }
        if self.ignore_index >= 0:
            self.attrs['ignore_index'] = self.ignore_index
        if self.axis != -1:
            self.attrs['axis'] = self.axis

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, no_check_set=("Loss"),atol=1e-3)

    def test_check_grad(self):
        place = core.CPUPlace()
        self.check_grad_with_place(place, ["Logits"], "Loss", max_relative_error=0.05)


class TestSoftmaxWithCrossEntropyOpUseRELU(TestSoftmaxWithCrossEntropyOp):
    def setUp(self):
        super().setUp()
        self.attrs["use_relu"] = True

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, no_check_set=("Loss"),atol=5)

    def test_check_grad(self):
        place = core.CPUPlace()
        self.check_grad_with_place(place, ["Logits"], "Loss", max_relative_error=0.5)


if __name__ == "__main__":
    unittest.main()