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
import numpy
import paddle.fluid as fluid
from ..framework import MpcVariable
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper
from .ml import reshape

__all__ = [
    'mean',
    'square',
    'sum',
    'square_error_cost',
    'reduce_sum',
    'scale'
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
        type="mpc_mean", inputs={"X": x}, outputs={"Out": out})

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



def reduce_sum(input, dim=None, keep_dim=False, name=None):
    """
    Computes the sum of tensor elements over the given dimension.

    Args:
        input (MpcVariable) The input of sum op name(basestring|None): Name of the output.
        dim (list|int, optional): The dimensions along which the sum is performed. If
            :attr:`None`, sum all elements of :attr:`input` and return a
            Tensor variable with a single element, otherwise must be in the
            range :math:`[-rank(input), rank(input))`. If :math:`dim[i] < 0`,
            the dimension to reduce is :math:`rank + dim[i]`.
            NOTE: 'dim' should not contain 0, becausedims[0] is share number.
        keep_dim (bool, optional): Whether to reserve the reduced dimension in the
            output Tensor. The result tensor will have one fewer dimension
            than the :attr:`input` unless :attr:`keep_dim` is true, default
            value is False.
        name(str, optional): The default value is None.  Normally there is no need for
            user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable: Tensor, results of summation operation on the specified dim of input tensor,
        it's data type is the same as input's Tensor.
    Raises:
        TypeError, if out data type is different with the input data type.

    Returns:
        out(MpcVariable): (Tensor) The output of mean op
    Examples:
        .. code-block:: python

            import paddle_fl.mpc as pfl_mpc

            pfl_mpc.init("aby3", int(args.role), "localhost", args.server, int(args.port))
            data_1 = pfl_mpc.data(name='x', shape=[3, 3], dtype='int64')
            pfl_mpc.layers.reshape(data_1, [1, 2])  # shape: [2, 1, 1]
            # data_1 = np.full(shape=(3, 4), fill_value=2)
            # reduce_sum: 24
    """
    if dim is not None and not isinstance(dim, list):
        dim = [dim]

    if dim != None and dim != []:
        if 0 in dim:
            raise ValueError(
                "'dim' should not contain 0, because dim[0] is share number."
            )
    else:
        dim = [i for i in range(len(input.shape))][1:]

    attrs = {
        'dim': dim,
        'keep_dim': keep_dim,
        'reduce_all': False
    }
    check_mpc_variable_and_dtype(
        input, 'input', ['int64'], 'reduce_sum')
    helper = MpcLayerHelper('reduce_sum', **locals())
    out = helper.create_mpc_variable_for_type_inference(dtype=helper.input_dtype())
    helper.append_op(
        type='reduce_sum',
        inputs={'X': input},
        outputs={'Out': out},
        attrs=attrs)
    if out.shape  == (2,):
        out = reshape(out, list(out.shape) + [1])
    return out


def scale(x, scale=1.0, bias=0.0, bias_after_scale=True, act=None, name=None):
    """
    Scale operator.
    Putting scale and bias to the input Tensor as following:
    ``bias_after_scale`` is True:
    .. math::
                            Out=scale*X+bias
    ``bias_after_scale`` is False:
    .. math::
                            Out=scale*(X+bias)
    Args:
        x(MpcVariable): Input N-D Tensor of scale operator. Data type should be int64.
        scale(float|Variable): The scale factor of the input, it should be a float number or a Variable with shape [1] and data type as float32.
        bias(float): The bias to be put on the input.
        bias_after_scale(bool): Apply bias addition after or before scaling. It is useful for numeric stability in some circumstances.
        act(str, optional): Activation applied to the output such as tanh, softmax, sigmoid, relu.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.  For more information, please refer to :ref:`api_guide_Name`
    Returns:
        Variable(Tensor|LoDTensor): Output tensor of scale operator, with shape and data type same as input.
    Examples:
        .. code-block:: python
            import paddle_fl.mpc as pfl_mpc
            pfl_mpc.init("aby3", int(args.role), "localhost", args.server, int(args.port))
            data_1 = pfl_mpc.data(name='x', shape=[3, 3], dtype='int64')
            pfl_mpc.layers.scale(data_1, 0.5)
    """

    check_mpc_variable_and_dtype(x, "x", ['int64'], "scale")
    inputs = {'X': [x]}
    attrs = {
        'bias': float(bias),
        'bias_after_scale': bias_after_scale,
    }
    if isinstance(scale, MpcVariable):
        inputs['ScaleTensor'] = [scale]
    else:
        attrs['scale'] = float(scale)
    helper = MpcLayerHelper('scale', **locals())
    out = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)

    helper.append_op(
        type='mpc_scale', inputs=inputs, outputs={'Out': out}, attrs=attrs)
    return helper.append_activation(out)

