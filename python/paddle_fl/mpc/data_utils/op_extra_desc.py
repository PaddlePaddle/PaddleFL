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
This module add extra input/output variable for some ops, including
relu, relu_grad, pool2d, pool2d_grad.
"""
import abc
import six

ops_to_add_extra = ['relu', 'relu_grad', 'pool2d', 'pool2d_grad']

def add_extra_desc(op, block):
    if op.type not in ops_to_add_extra:
        return
    if op.type == 'relu':
        instance = ReluExtraDesc()
    if op.type == 'relu_grad':
        instance = ReluGradExtraDesc()
    if op.type == 'pool2d':
        instance = Pool2dExtraDesc()
    if op.type == 'pool2d_grad':
        instance = Pool2dGradExtraDesc()
    instance.add_extra_desc(op, block)


@six.add_metaclass(abc.ABCMeta)
class OpExtraDesc(object):
    @abc.abstractmethod
    def add_extra_desc(self, op, block):
        pass


class ReluExtraDesc(OpExtraDesc):
    def add_extra_desc(self, op, block):
        derivative = block.create_var(
            name=op.output_arg_names[0] + ".relu_derivative",
            shape=block.var(op.output_arg_names[0]).shape,
            dtype="int64",
            lod_level=0)
        op.desc.set_output('Derivative', [op.output_arg_names[0] + ".relu_derivative"])


class ReluGradExtraDesc(OpExtraDesc):
    def add_extra_desc(self, op, block):
        op.desc.set_input('Derivative', [op.input_arg_names[0] + ".relu_derivative"])


class Pool2dExtraDesc(OpExtraDesc):
    def add_extra_desc(self, op, block):
        tensor_shape = []
        tensor_shape = list(block.var(op.input_arg_names[0]).shape) # tensor_shape[0-2]
        ksize_shape = list(op.attr('ksize'))
        tensor_shape[3] = ksize_shape[0] * ksize_shape[1]
        tensor_shape[4] = block.var(op.output_arg_names[0]).shape[3] * block.var(op.output_arg_names[0]).shape[4]
        one_hot_tensor = block.create_var(
        name=op.output_arg_names[0] + ".one_hot_tensor",
        shape=tensor_shape,
        dtype="int64",
        lod_level=0)
        op.desc.set_output('One_hot_tensor', [op.output_arg_names[0] + ".one_hot_tensor"])


class Pool2dGradExtraDesc(OpExtraDesc):
    def add_extra_desc(self, op, block):
        op.desc.set_input('One_hot_tensor', [op.input_arg_names[0] + '.one_hot_tensor'])
