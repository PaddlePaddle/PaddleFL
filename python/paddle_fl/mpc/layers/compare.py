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
mpc math compare layers.
"""

from ..framework import MpcVariable
from ..mpc_layer_helper import MpcLayerHelper

__all__ = [
    'greater_than',
    'greater_equal',
    'less_than',
    'less_equal',
    'equal',
    'not_equal',
]


def greater_than(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x > y` elementwise, 
    which is equivalent function to the overloaded operator `>`.
    """
    helper = MpcLayerHelper("greater_than", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='mpc_greater_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def greater_equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x >= y` elementwise, 
    which is equivalent function to the overloaded operator `>=`.
    """
    helper = MpcLayerHelper("greater_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='mpc_greater_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def less_than(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x < y` elementwise, 
    which is equivalent function to the overloaded operator `<`.
    """
    helper = MpcLayerHelper("less_than", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='mpc_less_than',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def less_equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x <= y` elementwise, 
    which is equivalent function to the overloaded operator `<=`.
    """
    helper = MpcLayerHelper("less_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='mpc_less_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x = y` elementwise, 
    which is equivalent function to the overloaded operator `=`.
    """
    helper = MpcLayerHelper("equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='mpc_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond


def not_equal(x, y, cond=None):
    """
    This OP returns the truth value of :math:`x != y` elementwise, 
    which is equivalent function to the overloaded operator `!=`.
    """
    helper = MpcLayerHelper("not_equal", **locals())
    if cond is None:
        cond = helper.create_variable_for_type_inference(dtype='bool')
        cond.stop_gradient = True

    attrs = dict()

    helper.append_op(
        type='mpc_not_equal',
        inputs={'X': [x],
                'Y': [y]},
        outputs={'Out': [cond]},
        attrs=attrs)
    return cond
