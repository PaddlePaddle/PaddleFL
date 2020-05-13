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
This module provide basic data structure and related methods
for paddle_mpc, namely MpcVariable and MpcParameter, which
are similar to Variable and Parameter in PaddlePaddle.
"""
from paddle import compat as cpt
from paddle.fluid import core
from paddle.fluid import unique_name
from paddle.fluid.framework import Variable
from paddle.fluid.framework import convert_np_dtype_to_dtype_
from paddle.fluid.data_feeder import check_type, check_dtype

class MpcVariable(Variable):
    """
    Extends from paddle.fluid.framework.Variable and rewrite
    the __init__ method where the shape is resized.
    """
    def __init__(self,
                 block,
                 type=core.VarDesc.VarType.LOD_TENSOR,
                 name=None,
                 shape=None,
                 dtype=None,
                 lod_level=None,
                 capacity=None,
                 persistable=None,
                 error_clip=None,
                 stop_gradient=False,
                 is_data=False,
                 need_check_feed=False,
                 belong_to_optimizer=False,
                 **kwargs):
        self.block = block
        if name is None:
            name = unique_name.generate('_generated_var')

        if dtype is not None:
            if not isinstance(dtype, core.VarDesc.VarType):
                dtype = convert_np_dtype_to_dtype_(dtype)

        self.belong_to_optimizer = belong_to_optimizer

        self.error_clip = error_clip

        is_new_var = False
        name = cpt.to_text(name)
        self.desc = self.block.desc.find_var(cpt.to_bytes(name))

        if self.desc is None:
            self.desc = self.block.desc.var(cpt.to_bytes(name))
            is_new_var = True

        if is_new_var:
            self.desc.set_type(type)
        elif self.desc.type() != type:
            raise ValueError("MpcVariable {0} has been created before. The "
                             "previous type is {1}; the new type is {2}. They"
                             " are not matched".format(self.name,
                                                       self.desc.type(), type))
        if shape is not None:
            if is_new_var:
                # resize the shape for MpcVariable
                mpc_shape = list(shape)
                mpc_shape.insert(0, 2)
                self.desc.set_shape(mpc_shape)
            else:
                old_shape = self.shape
                shape = tuple(shape)
                if shape != old_shape:
                    raise ValueError(
                        "MpcVariable {0} has been created before. the previous "
                        "shape is {1}; the new shape is {2}. They are not "
                        "matched.".format(self.name, old_shape, shape))
        if dtype is not None:
            if is_new_var:
                self.desc.set_dtype(dtype)
            else:
                old_dtype = self.dtype
                if dtype != old_dtype:
                    raise ValueError("MpcVariable {0} has been created before. "
                                     "The previous data type is {1}; the new "
                                     "data type is {2}. They are not "
                                     "matched.".format(self.name, old_dtype,
                                                       dtype))

        if lod_level is not None:
            if is_new_var:
                self.desc.set_lod_level(lod_level)
            else:
                if lod_level != self.lod_level:
                    raise ValueError("MpcVariable {0} has been created before. "
                                     "The previous lod_level is {1}; the new "
                                     "lod_level is {2}. They are not "
                                     "matched".format(self.name, self.lod_level,
                                                      lod_level))
        if persistable is not None:
            if is_new_var:
                self.desc.set_persistable(persistable)
            else:
                if persistable != self.persistable:
                    raise ValueError(
                        "MpcVariable {0} has been created before."
                        "The previous persistable is {1}; the new "
                        "persistable is {2}. They are not matched".format(
                            self.name, self.persistable, persistable))

        if need_check_feed and is_new_var:
            self.desc.set_need_check_feed(need_check_feed)

        if capacity is not None:
            if is_new_var:
                self.desc.set_capacity(capacity)
            else:
                # TODO(abhinavarora) by Paddle 1.7: Compare with set capacity once,
                # get_capacity is implemented
                pass

        self.block.vars[name] = self
        self.op = None
        self._stop_gradient = stop_gradient
        self.is_data = is_data


class MpcParameter(MpcVariable):
    """
    Extends from MpcVariable, which follows the design in Paddle 1.7.
    Note that MpcParameter is the same with paddle.fluid.framework.Parameter
    except that Variable is changed into MpcVariable.

    """

    def __init__(self,
                 block,
                 shape,
                 dtype,
                 type=core.VarDesc.VarType.LOD_TENSOR,
                 **kwargs):
        if shape is None:
            raise ValueError("The shape of MpcParameter should not be None")
        if dtype is None:
            raise ValueError("The dtype of MpcParameter should not be None")

        if len(shape) == 0:
            raise ValueError(
                "The dimensions of shape for MpcParameter must be greater than 0")

        for each in shape:
            if each < 0:
                raise ValueError(
                    "Each dimension of shape for MpcParameter must be greater than 0,"
                    " but received %s" % list(shape))
        # change Variable into MpcVariable
        MpcVariable.__init__(
            self,
            block,
            persistable=True,
            shape=shape,
            dtype=dtype,
            type=type,
            **kwargs)
        self.trainable = kwargs.get('trainable', True)

        self.optimize_attr = kwargs.get('optimize_attr', {'learning_rate': 1.0})

        self.regularizer = kwargs.get('regularizer', None)

        self.gradient_clip_attr = kwargs.get('gradient_clip_attr', None)

        self.do_model_average = kwargs.get('do_model_average', None)

        self.is_distributed = False

    def __str__(self):
        return self.to_string(True)

    def to_string(self, throw_on_error, with_details=False):
        """
        To debug string.
        :param throw_on_error:
        :param with_details:
        :return:
        """
        assert isinstance(throw_on_error, bool) and isinstance(with_details,
                                                               bool)
        if with_details:
            res_str = MpcVariable.to_string(self, throw_on_error, True)
            additional_attr = ("trainable", "optimize_attr", "regularizer",
                               "gradient_clip_attr", "do_model_average")
            for attr_name in additional_attr:
                res_str += "%s: %s\n" % (attr_name,
                                         cpt.to_text(getattr(self, attr_name)))
        else:
            res_str = MpcVariable.to_string(self, throw_on_error, False)
        return res_str

    __repr__ = __str__


def create_mpc_parameter(block, *args, **kwargs):
    """
    This method should be in fluid.framework.Block in Paddle1.7.
    There is no (new) Block in paddle_mpc, so we put it in framework
    and change the first param from self into block.
    Refer to paddle.fluid.framework.Block.create_parameter().
    :param block:
    :param args:
    :param kwargs:
    :return:
    """
    global_block = block.program.global_block()
    mpc_param = MpcParameter(global_block, *args, **kwargs)

    if 'initializer' in kwargs:

        def _is_inited_by(block, var):
            init_ops = []
            for op in block.ops:
                if var.name in op.output_arg_names:
                    # In startup_program, "c_broadcast" and "c_sync_comm_stream"
                    # are treated as initialization ops that cause error.
                    # Think of "c_broadcast" and "c_sync_comm_stream" as a special case here.
                    if op.type in ["c_broadcast", "c_sync_comm_stream"]:
                        continue
                    init_ops.append(op)
            return init_ops

        initializer = kwargs['initializer']
        init_ops = _is_inited_by(global_block, mpc_param)
        init_ops_len = len(init_ops)
        if init_ops_len > 1:
            raise RuntimeError("mpc_param " + mpc_param.name +
                               " is inited by multiple init ops " + str(init_ops))
        elif init_ops_len == 1:
            # TODO(Paddle 1.7): already inited, do nothing, should log a warning
            pass
        else:
            initializer(mpc_param, block)
    mpc_param.stop_gradient = False
    return mpc_param


def create_mpc_var(block, *args, **kwargs):
    """
    Refer to fluid.framework.Block.create_var() in Paddle1.7.
    There is no (new) Block in paddle_mpc, so we put it in framework
    and change the first param from self into block.
    :param block:
    :param args:
    :param kwargs:
    :return:
    """

    var = MpcVariable(block=block, *args, **kwargs)
    if 'initializer' in kwargs:
        kwargs['initializer'](var, block)
    return var

def is_mpc_parameter(var):
    """
    Check whether the given variable is an instance of MpcParameter.
    Args:
        var(Variable): The variable to be checked.
    Returns:
        bool: True if the given `var` is an instance of Parameter,
        False if not.
    """
    return type(var) == MpcParameter

def check_mpc_variable_and_dtype(input,
                                 input_name,
                                 expected_dtype,
                                 op_name,
                                 extra_message=''):
    check_type(input, input_name, MpcVariable, op_name, extra_message)
    check_dtype(input.dtype, input_name, expected_dtype, op_name, extra_message)

