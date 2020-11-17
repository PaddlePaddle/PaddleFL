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
basic tensor layers.
"""
import six
import numpy
from paddle.fluid.data_feeder import check_type, check_dtype
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Initializer

from ..framework import MpcVariable
from ..framework import check_mpc_variable_and_dtype
from ..mpc_layer_helper import MpcLayerHelper


__all__ = [
    'create_mpc_parameter',
]


def create_mpc_parameter(shape,
                         dtype,
                         name=None,
                         attr=None,
                         is_bias=False,
                         default_initializer=None):
    """
    :api_attr: Static Graph
    This function creates a mpc parameter. The parameter is a learnable variable, which can have
    gradient, and can be optimized.
    NOTE: this is a very low-level API. This API is useful when you create
    operator by your self. instead of using layers.
    Parameters:
        shape (list of int): Shape of the parameter
        dtype (str): Data type of the parameter
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.
        attr (ParamAttr, optional): Attributes of the parameter
        is_bias (bool, optional): This can affect which default initializer is chosen
                       when default_initializer is None. If is_bias,
                       initializer.Constant(0.0) will be used. Otherwise,
                       Xavier() will be used.
        default_initializer (Initializer, optional): Initializer for the parameter
    Returns:
        The created parameter.
    Examples:
        .. code-block:: python
            import paddle_fl.mpc as pfl_mpc
            pfl_mpc.init("aby3", role, "localhost", redis_server, redis_port)
            W = pfl_mpc.layers.create_mpc_parameter(shape=[784, 200], dtype='int64')
    """
    check_type(shape, 'shape', (list, tuple, numpy.ndarray), 'create_mpc_parameter')
    for item in shape:
        if six.PY2:
            check_type(item, 'item of shape',
                       (int, long, numpy.uint8, numpy.int8, numpy.int16,
                        numpy.int32, numpy.int64), 'create_mpc_parameter')
        else:
            check_type(item, 'item of shape',
                       (int, numpy.uint8, numpy.int8, numpy.int16, numpy.int32,
                        numpy.int64), 'create_mpc_parameter')

    check_dtype(dtype, 'dtype', ['int64'], 'create_mpc_parameter')
    check_type(attr, 'attr', (type(None), ParamAttr), 'create_mpc_parameter')
    check_type(default_initializer, 'default_initializer',
               (type(None), Initializer), 'create_mpc_parameter')

    helper = MpcLayerHelper("create_mpc_parameter", **locals())
    if attr is None:
        attr = ParamAttr(name=name)
    return helper.create_mpc_parameter(attr, shape,
                                       dtype, is_bias,
                                       default_initializer)

