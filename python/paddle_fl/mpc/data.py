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
This module provides data creation operation for paddle_mpc.
"""

import six

from paddle.fluid import core

from .mpc_layer_helper import MpcLayerHelper

__all__ = ['data']


def data(name, shape, dtype='int64', lod_level=0):
    """
    Refer to fluid.data in Paddle 1.7. This is the data layer in paddle_mpc.
    This function creates a mpc variable on the global block.
    :param name: the name of mpc variable
    :param shape: the shape of mpc variable. Note that the shape
    will be resized when constructed.
    :param dtype: the type of data which actually is in cypher text type.
    :param lod_level: The LoD level of the LoDTensor.
    :return: the global mpc variable that gives access to the data.
    """
    if dtype != 'int64':
        raise TypeError("Data type of {} should be int64.".format(name))
    mpc_helper = MpcLayerHelper('data', **locals())
    shape = list(shape)
    for i in six.moves.range(len(shape)):
        if shape[i] is None:
            shape[i] = -1

    return mpc_helper.create_global_mpc_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        type=core.VarDesc.VarType.LOD_TENSOR,
        stop_gradient=True,
        lod_level=lod_level,
        is_data=True,
        need_check_feed=True)
