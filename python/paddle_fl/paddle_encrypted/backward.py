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
Backward operations
"""

from __future__ import print_function

from paddle.fluid import framework as framework
from paddle.fluid import core
import collections
import copy
import six
import logging
from paddle import compat as cpt
from paddle.fluid import unique_name
from paddle.fluid import log_helper

import paddle.fluid
import paddle.fluid.backward as backward
from .framework import is_mpc_parameter

_logger = log_helper.get_logger(
    __name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')


def _create_loss_op_desc_(loss):
    op_desc = backward._create_op_desc_(
        "fill_constant", {},
        {"Out": [backward._append_grad_suffix_(loss.name)]}, {
            "shape": [2, 1],
            "value": 21845,
            "dtype": loss.dtype,
            "force_cpu": False,
            core.op_proto_and_checker_maker.kOpRoleAttrName():
            int(core.op_proto_and_checker_maker.OpRole.Backward) |
            int(core.op_proto_and_checker_maker.OpRole.Loss),
        })
    return op_desc


def append_backward(loss,
                    parameter_list=None,
                    no_grad_set=None,
                    callbacks=None,
                    checkpoints=None):
    """
    This function appends backward part to main_program.
    A complete neural network training is made up of forward and backward
    propagation. However, when we configure a network, we only need to
    specify its forward part. This function uses the chain rule to automatically
    generate the backward part according to the forward part.
    In most cases, users do not need to invoke this function manually.
    It will be automatically invoked by the optimizer's `minimize` function.
    Parameters:
        loss( :ref:`api_guide_Variable_en` ): The loss variable of the network.
        parameter_list(list of str, optional): Names of parameters that need
                                           to be updated by optimizers.
                                           If it is None, all parameters
                                           will be updated.
                                           Default: None.
        no_grad_set(set of str, optional): Variable names in the :ref:`api_guide_Block_en` 0 whose gradients
                               should be ignored. All variables with
                               `stop_gradient=True` from all blocks will
                               be automatically added into this set.
                               If this parameter is not None, the names in this set will be added to the default set.
                               Default: None.
       callbacks(list of callable object, optional): List of callback functions.
                                               The callbacks are used for
                                               doing some custom jobs during
                                               backward part building. All
                                               callable objects in it will
                                               be invoked once each time a
                                               new gradient operator is added
                                               into the program. The callable
                                               object must has two input
                                               parameters: 'block' and 'context'.
                                               The 'block' is the :ref:`api_guide_Block_en` which
                                               the new gradient operator will
                                               be added to. The 'context' is a
                                               map, whose keys are gradient
                                               variable names and values are
                                               corresponding original :ref:`api_guide_Variable_en` .
                                               In addition to this, the 'context'
                                               has another special key-value pair:
                                               the key is string '__current_op_desc__'
                                               and the value is the op_desc of the
                                               gradient operator who has just
                                               triggered the callable object.
                                               Default: None.
    Returns:
        list of tuple ( :ref:`api_guide_Variable_en` , :ref:`api_guide_Variable_en` ): Pairs of parameter and its corresponding gradients.
        The key is the parameter and the value is gradient variable.
    Raises:
        AssertionError: If `loss` is not an instance of Variable.
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            x = fluid.data(name='x', shape=[None, 13], dtype='float32')
            y = fluid.data(name='y', shape=[None, 1], dtype='float32')
            y_predict = fluid.layers.fc(input=x, size=1, act=None)
            loss = fluid.layers.square_error_cost(input=y_predict, label=y)
            avg_loss = fluid.layers.mean(loss)
            param_grad_list = fluid.backward.append_backward(loss=avg_loss)
            p_g_list1 = fluid.backward.append_backward(loss=avg_loss)  # len(p_g_list1) == 2
            p_g_list2 = fluid.backward.append_backward(loss=avg_loss, parameter_list=[p_g_list1[0][0].name])  # len(p_g_list1) == 1
            p_g_list3 = fluid.backward.append_backward(loss=avg_loss, no_grad_set=set([p_g_list1[0][0].name]))  # len(p_g_list1) == 1
            p_g_list4 = fluid.backward.append_backward(loss=avg_loss, parameter_list=[p_g_list1[0][0].name], no_grad_set=set([p_g_list1[0][0].name]))  # len(p_g_list1) == 0
    """

    assert isinstance(loss, framework.Variable)

    if loss.op is None:
        # the loss is from a cloned program. Find loss op manually.
        backward._find_loss_op_(loss)

    loss.op._set_attr(core.op_proto_and_checker_maker.kOpRoleAttrName(),
                      int(core.op_proto_and_checker_maker.OpRole.Forward) |
                      int(core.op_proto_and_checker_maker.OpRole.Loss))

    if callbacks is not None:
        isinstance(callbacks, list)

    program = loss.block.program
    program._appending_grad_times += 1

    if no_grad_set is None:
        no_grad_set = set()
    no_grad_set = copy.copy(no_grad_set)
    no_grad_dict = backward._get_stop_gradients_(program)
    no_grad_dict[0].update(
        list(map(backward._append_grad_suffix_, no_grad_set)))

    grad_info_map = dict()
    root_block = program.block(0)

    fwd_op_num = root_block.desc.op_size()
    current_block_idx = program.current_block_idx
    grad_to_var = dict()

    op_desc = _create_loss_op_desc_(loss)
    root_block.desc.append_op().copy_from(op_desc)

    block_no_grad_set = set(map(backward._strip_grad_suffix_, no_grad_dict[0]))
    op_path = backward._find_op_path_(root_block, [loss], [],
                                      block_no_grad_set)
    no_grad_vars = backward._find_no_grad_vars(root_block, op_path, [loss],
                                               block_no_grad_set)
    block_no_grad_set.update(no_grad_vars)
    no_grad_dict[0].update(
        list(map(backward._append_grad_suffix_, block_no_grad_set)))

    input_grad_names_set = None
    # For double backward, input_grad_names is used for filter
    # some non-used gradients op.
    if program._appending_grad_times > 1:
        input_grad_names_set = set([backward._append_grad_suffix_(loss.name)])

    backward._append_backward_ops_(
        root_block,
        op_path,
        root_block,
        no_grad_dict,
        grad_to_var,
        callbacks,
        input_grad_names_set=input_grad_names_set)

    # Because calc_gradient may be called multiple times,
    # we need rename the internal gradient variables so that they have
    # different names.
    backward._rename_grad_(root_block, fwd_op_num, grad_to_var, {})

    backward._append_backward_vars_(root_block, fwd_op_num, grad_to_var,
                                    grad_info_map)

    program.current_block_idx = current_block_idx
    program._sync_with_cpp()

    if parameter_list is not None:
        parameters = parameter_list
    else:
        params = list(filter(is_mpc_parameter, program.list_vars()))
        parameters = [param.name for param in params if param.trainable]

    params_and_grads = []
    for param in parameters:
        if cpt.to_text(param) not in grad_info_map:
            continue
        grad_info = grad_info_map[param]
        grad_block = grad_info[1]
        if not grad_block.has_var(grad_info[0]):
            raise ValueError("grad block[{0}] did not have grad var {1}".
                             format(grad_info[1], grad_info[0]))
        # Get the param var from the global block
        param_var = program.global_block().var(param)
        grad_var = grad_block.var(grad_info[0])
        if loss.block.has_var(grad_info[0]):
            params_and_grads.append((param_var, grad_var))
        else:
            params_and_grads.append((param_var, None))

    op_role_var_attr_name = core.op_proto_and_checker_maker.kOpRoleVarAttrName(
    )
    for p, g in params_and_grads:
        if g is None:
            continue
        for op in reversed(program.global_block().ops):
            assert isinstance(op, framework.Operator)
            if g.name in op.output_arg_names:
                g.op = op
                break

        if g.op is None:
            raise ValueError("Unexpected branch")
        attr_val = [p.name, g.name]
        if g.op.has_attr(op_role_var_attr_name):
            attr_val.extend(g.op.attr(op_role_var_attr_name))
        g.op._set_attr(op_role_var_attr_name, attr_val)

    return params_and_grads
