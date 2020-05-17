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
mpc math op layers.
"""

from ..framework import MpcVariable
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper

__all__ = [
    'mean',
    'square',
    'sum',
    'square_error_cost',
]


def mean(x, name=None):
    """
    Mean Operator calculates the mean of all elements in X.

    Args:
        x(MpcVariable): (Tensor) The input of mean op
        name(basestring|None): Name of the output.
    Returns:
        out(MpcVariable): (Tensor) The output of mean op
    Examples: todo
    """
    helper = MpcLayerHelper("mean", **locals())
    check_mpc_variable_and_dtype(x, 'x', ['int64'], 'mean')
    if name is None:
        out = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_mpc_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mpc_mean", inputs={"X": x}, attrs={}, outputs={"Out": out})

    return out


def square(x, name=None):
    """
    square Operator calculates the square of each element in X.

    Args:
        x(MpcVariable): (Tensor) The input of square op
        name(basestring|None): Name of the output.
    Returns:
        out(MpcVariable): (Tensor) The output of square op
    Examples: todo
    """
    helper = MpcLayerHelper("square", **locals())
    check_mpc_variable_and_dtype(x, 'x', ['int64'], 'square')
    if name is None:
        out = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_mpc_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mpc_square", inputs={"X": x}, attrs={}, outputs={"Out": out})

    return out


def sum(x):
    """
    Sum Operator calculates the sum of all elements in X.

    Args:
        x (MpcVariable|list(MpcVariable)) The input of sum op
        name(basestring|None): Name of the output.
    Returns:
        out(MpcVariable): (Tensor) The output of mean op
    Examples: todo
    """
    helper = MpcLayerHelper("sum", **locals())
    out = helper.create_mpc_variable_for_type_inference(dtype=helper.input_dtype('x'))
    helper.append_op(
        type="mpc_sum",
        inputs={"X": x},
        outputs={"Out": out},
        attrs={'use_mkldnn': False})
    return out


def square_error_cost(input, label):
    """
    This op accepts input predictions and target label and returns the
    squared error cost.
    For predictions label, and target label, the equation is:
    .. math::
        Out = (input - label)^2
    Parameters:
        input (MpcVariable): Input tensor, the data type should be float32.
        label (MpcVariable): Label tensor, the data type should be float32.
    Returns:
        The tensor variable storing the element-wise squared error \
                  difference between input and label.
    Return type: MpcVariable.
    Examples: todo
    """
    helper = MpcLayerHelper('square_error_cost', **locals())
    minus_out = helper.create_mpc_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='mpc_elementwise_sub',
        inputs={'X': [input],
                'Y': [label]},
        outputs={'Out': [minus_out]})

    square_out = helper.create_mpc_variable_for_type_inference(dtype=input.dtype)
    helper.append_op(
        type='mpc_square', 
        inputs={'X': [minus_out]},
        outputs={'Out': [square_out]})
    return square_out
