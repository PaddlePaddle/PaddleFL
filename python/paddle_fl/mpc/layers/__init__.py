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
mpc layers:
    basic:  'elementwise_add', 'elementwise_sub'
    math:   'mean', 'square', 'sum', 'square_error_cost'
    matrix: 'mul'
    ml:     'fc', 'relu', 'softmax'(todo)
    compare:'greater_than', 'greater_equal', 'less_than', 'less_equal', 'equal', 'not_equal'
"""

from . import basic
from .basic import *
from . import math
from .math import *
from . import matrix
from .matrix import *
from . import ml
from .ml import *
from . import compare
from .compare import *

__all__ = []
__all__ += basic.__all__
__all__ += math.__all__
__all__ += matrix.__all__
__all__ += ml.__all__
__all__ += compare.__all__
