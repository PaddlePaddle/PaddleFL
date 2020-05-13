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
Monkey patch for mpc operator overloading.
"""

from __future__ import print_function

from paddle.fluid.framework import unique_name, Variable
from paddle.fluid.layers.layer_function_generator import OpProtoHolder

from ..framework import MpcVariable, create_mpc_var

supported_mpc_ops = ['__add__', '__radd__', '__sub__', '__rsub__']
compare_ops = ['__gt__', '__ge__', '__lt__', '__le__', '__eq__', '__ne__']
supported_mpc_ops.extend(compare_ops)


def monkey_patch_mpc_variable():
    """
    Monkey patch for operator overloading.
    :return:
    """
    def unique_tmp_name():
        """
        Generate temp name for variable.
        :return:
        """
        return unique_name.generate("tmp")

    def safe_get_dtype(var):
        """
        Get data type.
        :param var:
        :return:
        """
        try:
            dtype = var.dtype
        except:
            raise ValueError("Cannot get data type from %s", var.name)
        return dtype

    def current_block(var):
        """
        Get current block.
        :param var:
        :return:
        """
        return var.block

    def create_new_tmp_mpc_var(block, dtype):
        """
        Create new temp mpc variable
        :param block:
        :param dtype:
        :return:
        """
        tmp_name = unique_tmp_name()
        return create_mpc_var(block=block, name=tmp_name, dtype=dtype)

    def create_new_tmp_var(block, dtype):
        """
        Create new temp variable.
        :param block:
        :param dtype:
        :return:
        """
        tmp_name = unique_tmp_name()
        return block.create_var(name=tmp_name, dtype=dtype)

    def _elemwise_method_creator_(method_name,
                                  op_type,
                                  reverse=False):
        """
        Operator overloading for different method.
        :param method_name: the name of operator which is overloaded.
        :param op_type:
        :param reverse:
        :return:
        """
        def __impl__(self, other_var):
            lhs_dtype = safe_get_dtype(self)

            if method_name in compare_ops:
                if not isinstance(other_var, Variable):
                    raise NotImplementedError("Unsupported data type of {} for compare operations."
                                              .format(other_var.name))
            else:
                if not isinstance(other_var, MpcVariable):
                    raise NotImplementedError("Unsupported data type of {}.".format(other_var.name))

            rhs_dtype = safe_get_dtype(other_var)
            if reverse:
                tmp = self
                self = other_var
                other_var = tmp

            if method_name in compare_ops:
                out = create_new_tmp_var(current_block(self), dtype=rhs_dtype)
            else:
                out = create_new_tmp_mpc_var(current_block(self), dtype=lhs_dtype)

            # out = create_new_tmp_mpc_var(current_block(self), dtype=lhs_dtype)

            axis = -1
            if other_var.shape[0] == -1:
                axis = 0
            assert len(self.shape) >= len(other_var.shape), (
                    "The rank of the first argument of an binary operator cannot "
                    "be smaller than the rank of its second argument: %s vs %s" %
                    (len(self.shape), len(other_var.shape)))

            current_block(self).append_op(
                type=op_type,
                inputs={'X': [self],
                        'Y': [other_var]},
                outputs={'Out': out},
                attrs={'axis': axis})
            return out

        comment = OpProtoHolder.instance().get_op_proto(op_type).comment

        __impl__.__doc__ = """
        {0}
        Args:
            self(MpcVariable): left hand variable
            other_var(MpcVariable): right hand variable

        Returns:
            MpcVariable
        """.format(comment)
        __impl__.__name__ = method_name
        return __impl__

    def announce_not_impl(self, other_var):
        """
        Raise NotImplementedError for unsupported operation.
        :param self:
        :param other_var:
        :return:
        """
        raise NotImplementedError('Unsupported mpc operation.')

    # inject methods
    for method_name, op_type, reverse in (
            ("__add__", "mpc_elementwise_add", False),
            # a+b == b+a. Do not need to reverse explicitly
            ("__radd__", "mpc_elementwise_add", False),
            ("__sub__", "mpc_elementwise_sub", False),
            ("__rsub__", "mpc_elementwise_sub", True),
            ("__mul__", "mpc_elementwise_mul", False),
            # a*b == b*a. Do not need to reverse explicitly
            ("__rmul__", "mpc_elementwise_mul", False),
            ("__div__", "mpc_elementwise_div", False),
            ("__truediv__", "mpc_elementwise_div", False),
            ("__rdiv__", "mpc_elementwise_div", True),
            ("__rtruediv__", "mpc_elementwise_div", True),
            ("__pow__", "mpc_elementwise_pow", False),
            ("__rpow__", "mpc_elementwise_pow", True),
            ("__floordiv__", "mpc_elementwise_floordiv", False),
            ("__mod__", "mpc_elementwise_mod", False),
            # for logical compare
            ("__eq__", "mpc_equal", False),
            ("__ne__", "mpc_not_equal", False),
            ("__lt__", "mpc_less_than", False),
            ("__le__", "mpc_less_equal", False),
            ("__gt__", "mpc_greater_than", False),
            ("__ge__", "mpc_greater_equal", False)
    ):
        # Not support computation between MpcVariable and scalar.
        setattr(MpcVariable,
                method_name,
                _elemwise_method_creator_(method_name, op_type, reverse)
                if method_name in supported_mpc_ops else announce_not_impl)

