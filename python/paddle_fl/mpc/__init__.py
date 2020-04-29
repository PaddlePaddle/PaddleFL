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
Import modules.
"""

import os
import sys
import sysconfig
import paddle.fluid as fluid

_paddle_enc_root = sysconfig.get_paths()["purelib"] + "/paddle_fl/mpc"
_paddle_enc_lib_path = _paddle_enc_root + '/libs'
_paddle_enc_lib = _paddle_enc_lib_path + '/libpaddle_enc.so'


def set_rpath_to_all_libs():
    """
    Set proper rpath to all libs
    """

    def set_rpath(lib, rpath):
        """
        patchelf util
        """
        os.system('patchelf --set-rpath {} {}'.format(rpath, lib))

    import paddle
    paddle_lib_path = os.path.dirname(paddle.__file__)
    paddle_lib_path = os.path.join(paddle_lib_path, 'libs')

    third_party_lib_path = _paddle_enc_lib_path + '/third_party'
    openssl_lib_path = third_party_lib_path + '/openssl'
    mpc_data_utils_lib = _paddle_enc_lib_path + '/mpc_data_utils.so'
    psi_lib = _paddle_enc_lib_path + '/libpsi.so'

    set_rpath(_paddle_enc_lib, paddle_lib_path + ":" + _paddle_enc_lib_path +
              ":" + third_party_lib_path)
    set_rpath(
        mpc_data_utils_lib,
        paddle_lib_path + ":" + _paddle_enc_lib_path + ":" + openssl_lib_path)
    if os.path.exists(psi_lib):
        set_rpath(psi_lib, openssl_lib_path)


sys.path.append(_paddle_enc_lib_path)
set_rpath_to_all_libs()
fluid.load_op_library(_paddle_enc_lib)

from .data import *
from . import layers
from .optimizer import *
from .mpc_init import *

from . import data_utils
from .io import *
from .version import version
from .layers import mpc_math_op_patch

mpc_math_op_patch.monkey_patch_mpc_variable()
