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
import sysconfig
import sys

he_utils_path = sysconfig.get_paths()["purelib"] + "/paddle_fl/feature_engineering/libs"
he_utils_lib = he_utils_path + '/he_utils.so'
sys.path.append(he_utils_path)
os.system('patchelf --set-rpath {} {}'.format(he_utils_path, he_utils_lib))

from . import core

