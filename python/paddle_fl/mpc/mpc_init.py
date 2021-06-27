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
mpc instance initialized..
"""

import numpy
import paddle.fluid as fluid
from .mpc_layer_helper import MpcLayerHelper
from .framework import MpcProtocols

__all__ = ['init', ]

def init(protocol_name,
         role,
         local_addr=None,
         net_server_addr=None,
         net_server_port=None,
         endpoints=None,
         network_mode="gloo",
         name=None):
    """
    init operator.
    This operator is used to initializ MPC instance, which can be used in other MPC operators.
    protocol_name (string):
    role (int):
    local_addr (string):
    net_server_addr (string):
    net_server_port (int):
    endpoints (string):
    """
    mpc_protocol_index = MpcProtocols[protocol_name.upper()].value
    fluid.global_scope().var("mpc_protocol_index").get_tensor().set(
        numpy.array((mpc_protocol_index)), fluid.CPUPlace())

    helper = MpcLayerHelper("mpc_init", **locals())
    helper.append_op(
        type="mpc_init",
        attrs={
            "protocol_name": protocol_name,
            "role": role,
            "local_addr": local_addr,
            "net_server_addr": net_server_addr,
            "net_server_port": net_server_port,
            "endpoints": endpoints,
            "network_mode": network_mode
        })
