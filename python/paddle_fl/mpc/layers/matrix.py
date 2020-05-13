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
mpc matrix op layers.
"""

from ..framework import MpcVariable
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper

__all__ = [
    'mul',
]


def mul(x, y, x_num_col_dims=1, y_num_col_dims=1, name=None):
    """
    Mul Operator.
    This operator is used to perform matrix multiplication for input $x$ and $y$.
    The equation is:
    ..  math::
        Out = x * y
    Both the input $x$ and $y$ can carry the LoD (Level of Details) information, or not.
    But the output only shares the LoD information with input $x$.
    Args:
        x (MpcVariable): The first input Tensor/LoDTensor of mul_op.
        y (MpcVariable): The second input Tensor/LoDTensor of mul_op.
        x_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs.
        If the input $x$ is a tensor with more than two dimensions, $x$ will be flattened into a
        two-dimensional matrix first.
        The flattening rule is: the first `num_col_dims` will be flattened to form the first dimension of the
        final matrix (the height of the matrix), and the rest `rank(x) - num_col_dims` dimensions are flattened to
        form the second dimension of the final matrix (the width of the matrix).
        As a result, height of the flattened matrix is equal to the product of $x$'s
        first `x_num_col_dims` dimensions' sizes,
        and width of the flattened matrix is equal to the product of $x$'s
        last `rank(x) - num_col_dims` dimensions' size.
        For example, suppose $x$ is a 6-dimensional tensor with the shape [2, 3, 4, 5, 6], and `x_num_col_dims` = 3.
        Thus, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default is 1.
        y_num_col_dims (int, optional): The mul_op can take tensors with more than two dimensions as its inputs.
        If the input $y$ is a tensor with more than two dimensions, $y$ will be flattened into a
        two-dimensional matrix first.
        The attribute `y_num_col_dims` determines how $y$ is flattened. See comments
        of `x_num_col_dims` for more details. Default is 1.
        name (str, optional): Name of the output. Normally there is no need for user to set this property.
        For more information, please refer to :ref:`api_guide_Name`. Default is None.
    Returns:
       MpcVariable(Tensor/LoDTensor): The output Tensor/LoDTensor of mul op.

    Example: todo
    """

    inputs = {"X": [x], "Y": [y]}
    attrs = {
        "x_num_col_dims": x_num_col_dims, 
        "y_num_col_dims": y_num_col_dims
    }

    helper = MpcLayerHelper("mul", **locals())
    check_mpc_variable_and_dtype(x, 'x', ['int64'], 'mul')
    check_mpc_variable_and_dtype(y, 'y', ['int64'], 'mul')
    if name is None:
        out = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_mpc_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mpc_mul", 
        inputs={"X": x, 
                "Y": y}, 
        attrs=attrs, 
        outputs={"Out": out})
    return out
