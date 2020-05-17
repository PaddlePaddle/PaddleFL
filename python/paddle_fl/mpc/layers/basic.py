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
basic mpc op layers.
"""
from paddle.fluid.data_feeder import check_variable_and_dtype

from ..framework import MpcVariable
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper

__all__ = [
    'elementwise_add',
    'elementwise_sub',
]


def _elementwise_op(helper):
    op_type = helper.layer_type
    x = helper.kwargs.get('x', None)
    y = helper.kwargs.get('y', None)

    assert x is not None, 'x cannot be None in {}'.format(op_type)
    assert y is not None, 'y cannot be None in {}'.format(op_type)
    check_mpc_variable_and_dtype(x, 'x', ['int64'], op_type)
    check_mpc_variable_and_dtype(y, 'y', ['int64'], op_type)

    axis = helper.kwargs.get('axis', -1)
    use_mkldnn = helper.kwargs.get('use_mkldnn', False)
    name = helper.kwargs.get('name', None)
    if name is None:
        out = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_mpc_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mpc_" + op_type,
        inputs={'X': x,
                'Y': y},
        outputs={'Out': out},
        attrs={'axis': axis,
               'use_mkldnn': use_mkldnn})
    return helper.append_mpc_activation(out)


def elementwise_add(x, y, axis=-1, act=None, name=None):
    """
    elementwise_add Operator.
    This operator is used to perform addition for input $x$ and $y$.
    The equation is:
    ..  math::
        Out = x + y
    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not.
    But the output only shares the LoD information with input $x$.
    Args:
        x (MpcVariable): The first input Tensor/LoDTensor of elementwise_add_op.
        y (MpcVariable): The second input Tensor/LoDTensor of elementwise_add_op.
        The dimensions of must be less than or equal to the dimensions of x.
        axis: If X.dimension != Y.dimension, Y.dimension must be a subsequence of x.dimension.
        And axis is the start dimension index for broadcasting Y onto X.
        act (string, optional): Activation applied to the output. Default is None.
        name (string, optional): Name of the output. Default is None. It is used to print debug info for developers. 
    Returns:
       MpcVariable(Tensor/LoDTensor): The output Tensor/LoDTensor of elementwise add op.

    Examples: todo
    """
    return _elementwise_op(MpcLayerHelper('elementwise_add', **locals()))


def elementwise_sub(x, y, axis=-1, act=None, name=None):
    """
    elementwise_sub Operator.
    This operator is used to perform subtraction for input $x$ and $y$.
    The equation is:
    ..  math::
        Out = x - y
    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not. 
    But the output only shares the LoD information with input $x$.
    Args:
        x (MpcVariable): The first input Tensor/LoDTensor of elementwise_sub_op.
        y (MpcVariable): The second input Tensor/LoDTensor of elementwise_add_op. 
                         The dimensions of must be less than or equal to the dimensions of x.
        axis: If X.dimension != Y.dimension, Y.dimension must be a subsequence of x.dimension. 
              And axis is the start dimension index for broadcasting Y onto X.
        act (string, optional): Activation applied to the output. Default is None.
        name (string, optional): Name of the output. Default is None. It is used to print debug info for developers.
    Returns:
       MpcVariable(Tensor/LoDTensor): The output Tensor/LoDTensor of elementwise add op.

    Examples: todo
    """
    return _elementwise_op(MpcLayerHelper('elementwise_sub', **locals()))
