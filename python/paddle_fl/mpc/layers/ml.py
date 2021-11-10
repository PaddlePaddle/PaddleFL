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
mpc ml op layers.
"""
import os
import numpy
from functools import reduce
import mpc_data_utils as mdu
import paddle.fluid as fluid
from paddle.fluid.data_feeder import check_type, check_dtype
import paddle.fluid.layers.utils as utils
from paddle.fluid.initializer import Constant
from paddle.fluid.layers.tensor import fill_constant
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.framework import Variable
from ..framework import MpcVariable, MpcProtocols
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper

__all__ = [
    'fc',
    'relu',
    'softmax',
    'sigmoid_cross_entropy_with_logits',
    'softmax_with_cross_entropy',
    'pool2d',
    'batch_norm',
    'reshape',
    'mean_normalize',
]


def fc(input,
       size,
       num_flatten_dims=1,
       param_attr=None,
       bias_attr=None,
       act=None,
       name=None):
    """
    **Fully Connected Layer**
    This operator creates a fully connected layer in the network. It can take
    a Tensor(or LoDTensor) or a list of Tensor(or LoDTensor) as its inputs(see
    Args in detail). It creates a variable called weight for each input Tensor,
    which represents a fully connected weight matrix from each input unit to
    each output unit. The fully connected layer multiplies each input Tensor
    with its corresponding weight to produce an output Tensor with shape :math:`[M, size]` ,
    where M is batch size. If a list of Tensor is given, the results of
    multiple output Tensors with shape :math:`[M, size]` will be summed up. If :attr:`bias_attr`
    is not None, a bias variable will be created and added to the output.
    Finally, if :attr:`act` is not None, it will be applied to the output as well.
    When the input is a single Tensor(or LoDTensor):
    .. math::
        Out = Act({XW + b})
    When the input is a list of Tensor(or LoDTensor):
    .. math::
        Out = Act({\sum_{i=0}^{N-1}X_iW_i + b})
    In the above equation:
    * :math:`N`: Number of the input. N equals to len(input) if input is list of Variable.
    * :math:`X_i`: The i-th input tensor.
    * :math:`W_i`: The i-th weights matrix corresponding i-th input tensor.
    * :math:`b`: The bias parameter created by this layer (if needed).
    * :math:`Act`: The activation function.
    * :math:`Out`: The output Tensor.
    .. code-block:: text
        Case 1:
        Given a single Tensor data_1, and num_flatten_dims = 2:
            data_1.data = [[[0.1, 0.2],
                            [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size
            out = fluid.layers.fc(input=data_1, size=1, num_flatten_dims=2)
        Then output is:
            out.data = [[0.83234344], [0.34936576]]
            out.shape = (1, 2, 1)
        Case 2:
        Given a list of Tensor:
            data_1.data = [[[0.1, 0.2],
                           [0.3, 0.4]]]
            data_1.shape = (1, 2, 2) # 1 is batch_size
            data_2 = [[[0.1, 0.2, 0.3]]]
            data_2.shape = (1, 1, 3)
            out = fluid.layers.fc(input=[data_1, data_2], size=2)
        Then:
            out.data = [[0.18669507, 0.1893476]]
            out.shape = (1, 2)
    Args:
        input (MpcVariable|list of MpcVariable): A Tensor(or LoDTensor) with shape :math:`[N_1, N_2,..., N_k]` or
            a list of Tensor(or LoDTensor). The dimensions of the input Tensor is at least 2 and the data
            type should be float32 or float64.
        size(int): The number of output units in this layer, which also means the feature size of output
            Tensor(or LoDTensor).
        num_flatten_dims (int): The fc layer can accept an input Tensor with more than
            two dimensions. If this happens, the multidimensional tensor will first be flattened
            into a 2-D matrix. The parameter :attr:`num_flatten_dims` determines how the input
            Tensor is flattened: the first :attr:`num_flatten_dims` (inclusive, index starts from 1)
            dimensions will be flatten to form the first dimension of the final matrix (height of
            the matrix), and the rest :math:`rank(X) - num\_flatten\_dims` dimensions are flattened to
            form the second dimension of the final matrix (width of the matrix). For example, assuming that
            X is a 5-dimensional Tensor with a shape [2, 3, 4, 5, 6], and :attr:`num_flatten_dims` = 3.
            Then, the flattened matrix will have a shape [2 x 3 x 4, 5 x 6] = [24, 30]. Default: 1.
        param_attr (ParamAttr): To specify the weight parameter property. Default: None, which means the
            default weight parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr): To specify the bias parameter property. Default: None, which means the
            default bias parameter property is used. See usage for details in :ref:`api_fluid_ParamAttr` .
        act (str): Activation to be applied to the output of this layer, such as tanh, softmax,
            sigmoid, relu. For more information, please refer to :ref:`api_guide_activations_en` . Default: None.
        name (str, optional): The default value is None.  Normally there is no need for user to set this property.
            For more information, please refer to :ref:`api_guide_Name` .
    Returns:
        MpcVariable: Tensor or LoDTensor calculated by fc layer. The data type is same with input.
    Raises:
        ValueError: If dimensions of the input Tensor is less than 2.
    Examples: todo
    """

    helper = MpcLayerHelper("fc", **locals())
    check_type(input, 'input', (list, tuple, MpcVariable), 'fc')
    if isinstance(input, (list, tuple)):
        for i, input_x in enumerate(input):
            check_type(input_x, 'input[' + str(i) + ']', MpcVariable, 'fc')
    dtype = helper.input_dtype()
    check_dtype(dtype, 'input', ['int64'], 'fc')

    mpc_protocol_index = numpy.array(fluid.global_scope().find_var("mpc_protocol_index").get_tensor())

    mul_results = []
    for input_var, param_attr in helper.iter_inputs_and_params():
        input_shape = input_var.shape
        if num_flatten_dims == -1:
            num_flatten_dims = len(input_shape) - 1
            param_num_flatten_dims = num_flatten_dims
        else:
            if MpcProtocols(mpc_protocol_index) is MpcProtocols.ABY3:
                param_num_flatten_dims = num_flatten_dims + 1  # The first dimension '2' of input is share number.
            else: # privc
                param_num_flatten_dims = num_flatten_dims
        param_shape = [
            reduce(lambda a, b: a * b, input_shape[param_num_flatten_dims:], 1)
        ] + [size]
        w = helper.create_mpc_parameter(
            attr=param_attr, shape=param_shape, dtype=dtype, is_bias=False)
        tmp = helper.create_mpc_variable_for_type_inference(dtype)
        helper.append_op(
            type="mpc_mul",
            inputs={"X": input_var,
                    "Y": w},
            outputs={"Out": tmp},
            attrs={"x_num_col_dims": num_flatten_dims,
                   "y_num_col_dims": 1})
        mul_results.append(tmp)

    if len(mul_results) == 1:
        pre_bias = mul_results[0]
    else:
        pre_bias = helper.create_mpc_variable_for_type_inference(dtype)
        helper.append_op(
            type="mpc_sum",
            inputs={"X": mul_results},
            outputs={"Out": pre_bias},
            attrs={"use_mkldnn": False})

    # add bias
    pre_activation = helper.append_mpc_bias_op(
        pre_bias, dim_start=num_flatten_dims)
    # add activation
    return helper.append_mpc_activation(pre_activation)


def softmax(input, use_cudnn=False, name=None, axis=-1):
    """
    This operator implements the softmax layer. The calculation process is as follows:
    1. The dimension :attr:`axis` of the ``input`` will be permuted to the last.

    2. Then the input tensor will be logically flattened to a 2-D matrix. The matrix's
    second dimension(row length) is the same as the dimension :attr:`axis` of the input
    tensor, and the first dimension(column length) is the product of all other
    dimensions of the input tensor. For each row of the matrix, the softmax operator
    squashes the K-dimensional(K is the width of the matrix, which is also the size
    of the input tensor's dimension :attr:`axis`) vector of arbitrary real values to a
    K-dimensional vector of real values in the range [0, 1] that add up to 1.
    3. After the softmax operation is completed, the inverse operations of steps 1 and 2
    are performed to restore the two-dimensional matrix to the same dimension as the ``input``.
    It computes the exponential of the given dimension and the sum of exponential
    values of all the other dimensions in the K-dimensional vector input.
    Then the ratio of the exponential of the given dimension and the sum of
    exponential values of all the other dimensions is the output of the softmax
    operator.
    For each row :math:`i` and each column :math:`j` in the matrix, we have:
    .. math::
        Out[i, j] = \\frac{\exp(X[i, j])}{\sum_j(exp(X[i, j])}
    """
    attrs = {"axis": axis, "use_cudnn": use_cudnn}
    helper = MpcLayerHelper('softmax', **locals())
    check_mpc_variable_and_dtype(input, 'input', ['int64'], 'softmax')

    dtype = helper.input_dtype()
    mpc_softmax_out = helper.create_mpc_variable_for_type_inference(dtype)
    helper.append_op(
        type="mpc_softmax",
        inputs={"X": input},
        outputs={"Out": mpc_softmax_out},
        attrs=attrs)
    return mpc_softmax_out


def relu(input, name=None):
    """
    Args:
        x(Variable): ${x_comment}
        name(str, optional): The default value is None. Normally there is no
            need for user to set this property. For more information, please
            refer to :ref:`api_guide_Name`.
    Returns:
        Variable: ${out_comment}
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            import numpy as np
            in1 = np.array([[-1,0],[1,2.6]])
            with fluid.dygraph.guard():
                x1 = fluid.dygraph.to_variable(in1)
                out1 = fluid.layers.relu(x1)
                print(out1.numpy())
                # [[0.  0. ]
                #  [1.  2.6]]

    """
    inputs = {'X': [input]}
    helper = MpcLayerHelper('relu', **locals())
    dtype = helper.input_dtype(input_param_name='input')
    out = helper.create_mpc_variable_for_type_inference(dtype)
    derivative = helper.create_mpc_variable_for_type_inference(dtype)
    helper.append_op(
        type="mpc_relu",
        inputs={"X": input},
        outputs={"Out": out,
                 "Derivative": derivative})
    return out


def sigmoid_cross_entropy_with_logits(x, label, name=None):
    """
    sigmoid_cross_entropy_with_logits
        forward: out = sigmoid(x). todo: add cross_entropy
        backward: dx = sigmoid(x) - label
    Args:
        x(MpcVariable): input
        label(MpcVariable): labels
        name(str|None): The default value is None.  Normally there is
            no need for user to set this property.  For more information,
            please refer to :ref:`api_guide_Name`
    Returns:
        out(MpcVariable): out = sigmoid(x)
    """

    helper = MpcLayerHelper("sigmoid_cross_entropy_with_logits", **locals())

    if name is None:
        out = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)
    else:
        out = helper.create_mpc_variable(
            name=name, dtype=x.dtype, persistable=False)

    helper.append_op(
        type="mpc_sigmoid_cross_entropy_with_logits",
        inputs={"X": x,
                "Label": label},
        outputs={"Out": out})
    return out


def softmax_with_cross_entropy(logits,
                               label,
                               soft_label=False,
                               return_softmax=False,
                               axis=-1,
                               use_relu=False,
                               use_long_div=True):
    """
    forward: out = softmax(x). todo: add cross_entropy
    backward: dx = dout.expand * (softmax(x) - label)

    use_relu: False(default): output = exp(x_i) / sum(exp(x_i))
              True: output = relu(x_i) / sum(relu(x_i))
    use_long_div: True(default): long division implemented by boolean circuit.
                                 slow with high precision.
                                 range of quotient: [0, 2^20).
                  False: find inverse of divisor by Newton's method.
                         fast with low precision.
                         range of divisor: (0, 2^15).
    """

    attrs = {
        'soft_label': soft_label,
        'axis': axis,
        'use_relu': use_relu,
        'use_long_div': use_long_div
    }

    helper = MpcLayerHelper('softmax_with_cross_entropy', **locals())
    softmax = helper.create_variable_for_type_inference(dtype=logits.dtype)
    loss = helper.create_variable_for_type_inference(dtype=logits.dtype)
    helper.append_op(
        type='mpc_softmax_with_cross_entropy',
        inputs={'Logits': logits,
                'Label': label},
        outputs={'Softmax': softmax,
                 'Loss': loss},
        attrs=attrs)
    if return_softmax:
        return loss, softmax
    else:
        raise NotImplementedError(
            "'return_softmax' should be true. Loss is NULL, only for backward.")


def pool2d(input,
           pool_size=-1,
           pool_type="max",
           pool_stride=1,
           pool_padding=0,
           global_pooling=False,
           ceil_mode=False,
           name=None,
           exclusive=True,
           data_format="NCHW"):
    """
    pool2d
    """
    if pool_type not in ["max", "avg"]:
        raise ValueError(
            "Unknown Attr(pool_type): '%s'. It can only be 'max'.",
            str(pool_type))

    if global_pooling is False and pool_size == -1:
        raise ValueError(
            "When Attr(global_pooling) is False, Attr(pool_size) must be passed "
            "and be a valid value. Received pool_size: %s." % str(pool_size))

    if data_format not in ["NCHW"]:
        raise ValueError("Attr(data_format) should be 'NCHW'. Received "
                         "Attr(data_format): %s." % str(data_format))

    pool_size = utils.convert_to_list(pool_size, 2, 'pool_size')
    pool_stride = utils.convert_to_list(pool_stride, 2, 'pool_stride')

    def update_padding(padding, data_format):
        """
        update_padding: convert to 2-dimension padding
        """

        def is_list_or_tuple(ele):
            """
            return true if ele is list or tuple.
            """
            if isinstance(ele, list) or isinstance(ele, tuple):
                return True
            return False

        # covert padding size to 2 (H, W)
        if is_list_or_tuple(padding) and len(padding) == 4:
            if is_list_or_tuple(padding[0]):
                if not (padding[0] == [0, 0] and padding[1] == [0, 0]):
                    raise ValueError(
                        "Non-zero pool_padding(%s) in the batch or channel dimensions "
                        "is not supported." % str(padding))
                padding = padding[2:4]  # data_format == "NCHW":
                #padding = [ele for a_list in padding for ele in a_list]
                for a_list in padding:
                    for ele in a_list:
                        padding.append(ele)
            padding = utils.convert_to_list(padding, 4, 'padding')

            if utils._is_symmetric_padding(padding, 2):
                padding = [padding[0], padding[2]]
        else:
            padding = utils.convert_to_list(padding, 2, 'padding')

        return padding

    padding_algorithm = "EXPLICIT"
    if isinstance(pool_padding, str):
        pool_padding = pool_padding.upper()
        if pool_padding not in ["SAME", "VALID"]:
            raise ValueError(
                "Unknown Attr(pool_padding): '%s'. It can only be 'SAME' or 'VALID'."
                % str(pool_padding))
        if pool_padding == "VALID":
            padding_algorithm = "VALID"
            pool_padding = [0, 0]
            if ceil_mode != False:
                raise ValueError(
                    "When Attr(pool_padding) is \"VALID\", Attr(ceil_mode) must be False. "
                    "Received ceil_mode: True.")
        elif pool_padding == "SAME":
            padding_algorithm = "SAME"
            pool_padding = [0, 0]

    pool_padding = update_padding(pool_padding, data_format)  # [h, w]

    op_type = 'pool2d'
    helper = MpcLayerHelper(op_type, **locals())
    dtype = helper.input_dtype()
    pool_out = helper.create_mpc_variable_for_type_inference(dtype)
    one_hot_tensor = helper.create_variable_for_type_inference(
        dtype=input.dtype)

    helper.append_op(
        type='mpc_' + op_type,
        inputs={"X": input},
        outputs={"Out": pool_out,
                 "One_hot_tensor": one_hot_tensor},
        attrs={
            "pooling_type": pool_type,
            "ksize": pool_size,
            "global_pooling": global_pooling,
            "strides": pool_stride,
            "paddings": pool_padding,
            "padding_algorithm": padding_algorithm,
            "ceil_mode": ceil_mode,
            "exclusive": exclusive,
            "data_format": data_format,
        })

    return pool_out


def batch_norm(input,
               act=None,
               is_test=False,
               momentum=0.9,
               epsilon=1e-04,
               param_attr=None,
               bias_attr=None,
               data_layout='NCHW',
               in_place=False,
               name=None,
               moving_mean_name=None,
               moving_variance_name=None,
               do_model_average_for_mean_and_var=True,
               use_global_stats=False):
    """
    **Batch Normalization Layer**
    """
    assert bias_attr is not False, "bias_attr should not be False in batch_norm."
    helper = MpcLayerHelper('batch_norm', **locals())

    check_mpc_variable_and_dtype(input, 'input', ['int64'], 'batch_norm')
    dtype = helper.input_dtype()

    has_reserve_space = False
    if data_layout == 'NHWC':
        flag = os.environ.get('FLAGS_cudnn_batchnorm_spatial_persistent')
        if flag is not None and flag.lower() in ['true', '1']:
            has_reserve_space = True

        # plaintext_dtype = core.VarDesc.VarType.FP32

    input_shape = input.shape
    if data_layout == 'NCHW':
        channel_num = input_shape[2]
    else:
        if data_layout == 'NHWC':
            channel_num = input_shape[-1]
        else:
            raise ValueError("unsupported data layout:" + data_layout)

    param_shape = [channel_num]
    mpc_param_shape = [2, channel_num]

    # create parameter
    scale = helper.create_mpc_parameter(
        attr=helper.param_attr,
        shape=param_shape,
        dtype=dtype,
        default_initializer=Constant(mdu.mpc_one_share))
    bias = helper.create_mpc_parameter(
        attr=helper.bias_attr, shape=param_shape, dtype=dtype, is_bias=True)

    mean = helper.create_mpc_parameter(
        attr=ParamAttr(
            name=moving_mean_name,
            initializer=Constant(0),
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    mean.stop_gradient = True

    variance = helper.create_mpc_parameter(
        attr=ParamAttr(
            name=moving_variance_name,
            initializer=Constant(mdu.mpc_one_share),  # plaintext: 1
            trainable=False,
            do_model_average=do_model_average_for_mean_and_var),
        shape=param_shape,
        dtype=dtype)
    variance.stop_gradient = True

    # create output
    # mean and mean_out share the same memory
    mean_out = mean
    # variance and variance out share the same memory
    variance_out = variance
    saved_mean = helper.create_mpc_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)
    saved_variance = helper.create_mpc_variable_for_type_inference(
        dtype=dtype, stop_gradient=True)

    #reserve_space = None
    #if has_reserve_space:
    #    reserve_space = helper.create_variable_for_type_inference(
    #        dtype=core.VarDesc.VarType.FP16, stop_gradient=True)

    batch_norm_out = input if in_place else \
            helper.create_mpc_variable_for_type_inference(dtype)

    inputs = {
        "X": input,
        "Scale": scale,
        "Bias": bias,
        "Mean": mean,
        "Variance": variance
    }
    attrs = {
        "epsilon": epsilon,
        "is_test": is_test,
        "data_layout": data_layout,
        "use_mkldnn": False,
        "fuse_with_relu": False,
        "use_global_stats": use_global_stats
    }
    if isinstance(momentum, Variable):
        inputs['MomemtumTensor'] = momentum
    else:
        attrs['momentum'] = momentum

    outputs = {
        "Y": batch_norm_out,
        "MeanOut": mean_out,
        "VarianceOut": variance_out,
        "SavedMean": saved_mean,
        "SavedVariance": saved_variance
    }
    #if reserve_space is not None:
    #    outputs["ReserveSpace"] = reserve_space

    helper.append_op(
        type="mpc_batch_norm", inputs=inputs, outputs=outputs, attrs=attrs)

    return helper.append_mpc_activation(batch_norm_out)


def reshape(x, shape, actual_shape=None, act=None, inplace=False, name=None):
    """
    This operator changes the shape of ``x`` without changing its data.

    The target shape can be given by ``shape`` or ``actual_shape``.
    When ``shape`` and ``actual_shape`` are set at the same time,
    ``actual_shape`` has a higher priority than ``shape``
    but at this time ``shape`` can only be an integer list or tuple, and ``shape`` still should be set correctly to
    guarantee shape inference in compile-time.

    Some tricks exist when specifying the target shape.

    1. -1 means the value of this dimension is inferred from the total element
    number of x and remaining dimensions. Thus one and only one dimension can
    be set -1.

    2. 0 means the actual dimension value is going to be copied from the
    corresponding dimension of x. The index of 0s in shape can not exceed
    the dimension of x.

    Args:
        x(Variable): A ``Tensor`` or ``LoDTensor`` . The data type is ``int64``.
        shape(list|tuple|Variable): Define the target shape. At most one dimension of the target shape can be -1.
                        The data type is ``int32`` . If ``shape`` is a list or tuple, the elements of it should be integers or Tensors with shape [1].
                        If ``shape`` is an Variable, it should be an 1-D Tensor .
        actual_shape(variable, optional): An 1-D ``Tensor`` or ``LoDTensor`` . The data type is ``int32`` . If provided, reshape
                                according to this given shape rather than ``shape`` specifying shape.
                                That is to say ``actual_shape`` has a higher priority
                                than ``shape(list|tuple)`` but not ``shape(Variable)``. \
                                This argument ``actual_shape`` will be removed in a future version. \
        act (str, optional): The non-linear activation to be applied to the reshaped input. Default None.
        inplace(bool, optional): If ``inplace`` is True, the input and output of ``layers.reshape``
                       are the same variable. Otherwise, the input and output of
                       ``layers.reshape`` are different variable. Default False. Note that if ``x``
                       is more than one OPs' input, ``inplace`` must be False.
        name(str, optional): The default value is None. Normally there is no need for user to set this property.
                            For more information, please refer to :ref:`api_guide_Name` .

    Returns:
        Variable: A ``Tensor`` or ``LoDTensor``. The data type is same as ``x``. It is a new tensor variable if ``inplace`` is ``False``, otherwise it is ``x``. If ``act`` is None, return the reshaped tensor variable, otherwise return the activated tensor variable.


    Examples:
        .. code-block:: python
            import paddle_fl.mpc as pfl_mpc

            pfl_mpc.init("aby3", int(args.role), "localhost", args.server, int(args.port))
            data_1 = pfl_mpc.data(name='x', shape=[3, 3], dtype='int64')
            op_reshape = pfl_mpc.layers.reshape(data_1, [2, 1, 9])
    """

    check_mpc_variable_and_dtype(x, 'x', ['int64'], 'reshape')
    check_type(shape, 'shape', (list, tuple, Variable), 'reshape')
    check_type(actual_shape, 'actual_shape', (Variable, type(None)), 'reshape')

    helper = MpcLayerHelper("reshape2", **locals())
    _helper = LayerHelper("reshape2", **locals())

    def get_new_shape_tensor(list_shape):
        """
        get_new_shape_tensor
        """
        new_shape_tensor = []
        for dim in list_shape:
            if isinstance(dim, Variable):
                dim.stop_gradient = True
                new_shape_tensor.append(dim)
            else:
                assert (isinstance(dim, int))
                temp_out = _helper.create_variable_for_type_inference('int32')
                fill_constant([1], 'int32', dim, force_cpu=True, out=temp_out)
                new_shape_tensor.append(temp_out)
        return new_shape_tensor

    def get_attr_shape(list_shape):
        """
        get_attr_shape
        """
        unk_dim_idx = -1
        attrs_shape = []
        for dim_idx, dim_size in enumerate(list_shape):
            if isinstance(dim_size, Variable):
                attrs_shape.append(-1)
            else:
                attrs_shape.append(dim_size)
                if dim_size == -1:
                    assert unk_dim_idx == -1, (
                        "Only one dimension value of 'shape' in reshape can "
                        "be -1. But received shape[%d] is also -1." % dim_idx)
                    unk_dim_idx = dim_idx
                elif dim_size == 0:
                    assert dim_idx < len(x.shape), (
                        "The index of 0 in `shape` must be less than "
                        "the input tensor X's dimensions. "
                        "But received shape[%d] = 0, X's dimensions = %d." %
                        (dim_idx, len(x.shape)))
                else:
                    assert dim_size > 0, (
                        "Each dimension value of 'shape' in reshape must not "
                        "be negative except one unknown dimension. "
                        "But received shape[%d] = %s." %
                        (dim_idx, str(dim_size)))
        return attrs_shape

    inputs = {"X": x}
    attrs = {}
    if isinstance(shape, Variable):
        shape.stop_gradient = True
        inputs["Shape"] = shape
    elif isinstance(shape, (list, tuple)):
        assert len(shape) > 0, (
            "The size of 'shape' in reshape can't be zero, "
            "but received %s." % len(shape))
        attrs["shape"] = get_attr_shape(shape)

        if utils._contain_var(shape):
            inputs['ShapeTensor'] = get_new_shape_tensor(shape)
        elif isinstance(actual_shape, Variable):
            actual_shape.stop_gradient = True
            inputs["Shape"] = actual_shape

    out = x if inplace else helper.create_mpc_variable_for_type_inference(
        dtype=x.dtype)
    x_shape = helper.create_mpc_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(
        type="reshape2",
        inputs=inputs,
        attrs=attrs,
        outputs={"Out": out,
                 "XShape": x_shape})

    return helper.append_mpc_activation(out)


def mean_normalize(f_min, f_max, f_mean, sample_num):
    """
    Mean normalization is a method used to normalize the range of independent
    variables or features of data.
    Refer to:
    https://en.wikipedia.org/wiki/Feature_scaling#Mean_normalization

    Args:
        f_min (Variable): A 2-D tensor with shape [P, N], where P is the party
                          num and N is the feature num. Each row contains the
                          local min feature val of N features.
        f_max (Variable): A 2-D tensor with shape [P, N], where P is the party
                          num and N is the feature num. Each row contains the
                          local max feature val of N features.
        f_mean (Variable): A 2-D tensor with shape [P, N], where P is the party
                           num and N is the feature num. Each row contains the
                           local min feature val of N features.
        sample_num (Variable): A 1-D tensor with shape [P], where P is the
                               party num. Each element contains sample num
                               of party_i.

    Returns:
        f_range (Variable): A 1-D tensor with shape [N], where N is the
                            feature num. Each element contains global
                            range of feature_i.
        f_mean_out (Variable): A 1-D tensor with shape [N], where N is the
                               feature num. Each element contains global
                               range of feature_i.
    Examples:
        .. code-block:: python
            import paddle_fl.mpc as pfl_mpc

            pfl_mpc.init("aby3", role, "localhost", redis_server, redis_port)

            # 2 for share, 4 for 4 party, 100 for feat_num
            input_size = [2, 4, 100]

            mi = pfl_mpc.data(name='mi', shape=input_size, dtype='int64')
            ma = pfl_mpc.data(name='ma', shape=input_size, dtype='int64')
            me = pfl_mpc.data(name='me', shape=input_size, dtype='int64')
            sn = pfl_mpc.data(name='sn', shape=input_size[:-1], dtype='int64')

            out0, out1 = pfl_mpc.layers.mean_normalize(f_min=mi, f_max=ma,
                    f_mean=me, sample_num=sn)

            exe = fluid.Executor(place=fluid.CPUPlace())

            # feed encrypted data
            f_range, f_mean = exe.run(feed={'mi': f_min, 'ma': f_max,
            'me': f_mean, 'sn': sample_num}, fetch_list=[out0, out1])
    """
    helper = MpcLayerHelper("mean_normalize", **locals())

    # dtype = helper.input_dtype()
    dtype = 'int64'

    check_dtype(dtype, 'f_min', ['int64'], 'mean_normalize')
    check_dtype(dtype, 'f_max', ['int64'], 'mean_normalize')
    check_dtype(dtype, 'f_mean', ['int64'], 'mean_normalize')
    check_dtype(dtype, 'sample_num', ['int64'], 'mean_normalize')

    f_range = helper.create_mpc_variable_for_type_inference(dtype=f_min.dtype)
    f_mean_out = helper.create_mpc_variable_for_type_inference(
        dtype=f_min.dtype)

    # to avoid circular dependencies
    from .math import reduce_sum

    total_num = reduce_sum(sample_num)

    op_type = 'mean_normalize'

    helper.append_op(
        type='mpc_' + op_type,
        inputs={
            "Min": f_min,
            "Max": f_max,
            "Mean": f_mean,
            "SampleNum": sample_num,
            "TotalNum": total_num,
        },
        outputs={
            "Range": f_range,
            "MeanOut": f_mean_out,
        }, )

    return f_range, f_mean_out
