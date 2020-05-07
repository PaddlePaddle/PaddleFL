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
This module provide MpcLayerHelper, which are similar to
LayerHelper in PaddlePaddle.
"""

# system module
import copy
import six

# paddle module
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid import core
from paddle.fluid import unique_name
from paddle.fluid.param_attr import ParamAttr, WeightNormParamAttr
from paddle.fluid.initializer import ConstantInitializer

# mpc_paddle module
from .framework import MpcVariable, MpcParameter, create_mpc_parameter, create_mpc_var


class MpcLayerHelper(LayerHelper):
    """
    Refer to paddle.fluid.LayerHelper.
    Add new methods for MpcVariable and MpcOps.
    """

    def create_global_mpc_variable(self, persistable=False, *args, **kwargs):
        """
        Refer to paddle.fluid.layer_helper_base.create_global_variable().
        Create global mpc variable.
        :param persistable:
        :param args:
        :param kwargs:
        :return:
        """
        mpc_block = self.main_program.global_block()
        mpc_var = MpcVariable(
            block=mpc_block, *args, persistable=persistable, **kwargs)
        if 'initializer' in kwargs:
            kwargs['initializer'](mpc_var, self)
        return mpc_var

    # TODO(Paddle1.7): hide the func after we move the layers to Layers
    def create_mpc_parameter(self,
                             attr,
                             shape,
                             dtype,
                             is_bias=False,
                             default_initializer=None,
                             stop_gradient=False,
                             type=core.VarDesc.VarType.LOD_TENSOR):
        """
        Create mpc parameters for this layers.
        Refer to LayerHelper.create_parameter in Paddle 1.7.
        :param attr:
        :param shape:
        :param dtype:
        :param is_bias:
        :param default_initializer:
        :param stop_gradient:
        :param type:
        :return:
        """
        # Deepcopy the attr so that parameters can be shared in program
        attr = copy.deepcopy(attr)
        attr = ParamAttr._to_attr(attr)
        if not attr:
            return None
        assert isinstance(attr, ParamAttr)
        suffix = 'b' if is_bias else 'w'
        if attr.name is None:
            attr.name = unique_name.generate(".".join([self.name, suffix]))

        if default_initializer is None and attr.initializer is None:
            if isinstance(dtype, core.VarDesc.VarType):
                if dtype != core.VarDesc.VarType.INT64:
                    raise TypeError(
                        "Can not create mpc parameter with default initializer "
                        "when dtype is not int64 type. Set default_initializer "
                        "to fit the parameter dtype!")
            else:
                if not dtype == "int64":
                    raise TypeError(
                        "Can not create mpc parameter with default initializer when "
                        "dtype is not int64 type. Set default_initializer to "
                        "fit the parameter dtype!")
            if is_bias:
                attr._set_default_bias_initializer()
            else:
                attr._set_default_initializer(ConstantInitializer(0))
        else:
            attr._set_default_initializer(default_initializer)

        # TODO(xukun07): not support WeightNormParamAttr in this first version
        # Paddle1.7: If weight normalization is set, insert extra parameters and ops.
        # Refer to https://arxiv.org/pdf/1602.07868.pdf
        if isinstance(attr, WeightNormParamAttr):
            # param = self._create_weight_normalize(attr, shape, dtype)
            # WeightNormParamAttr.params_with_weight_norm.append(param)
            # return param
            raise NotImplementedError(
                "The WeightNormParamAttr for attr is not "
                "supported in this version")

        startup_program_global_block = self.startup_program.global_block()
        create_mpc_parameter(
            block=startup_program_global_block,
            dtype=dtype,
            shape=shape,
            type=type,
            **attr._to_kwargs(with_initializer=True))
        main_program_global_block = self.main_program.global_block()
        return create_mpc_parameter(
            block=main_program_global_block,
            dtype=dtype,
            shape=shape,
            type=type,
            **attr._to_kwargs())

    # Note : not sure if this rewrite is needed
    def get_mpc_parameter(self, name):
        """
        Refer to LayerHelper.get_parameter in Paddle.
        :param name:
        :return:
        """
        param = self.main_program.global_block().var(name)
        if not isinstance(param, MpcParameter):
            raise ValueError("no MpcParameter name %s found" % name)
        return param

    def create_mpc_variable_for_type_inference(self,
                                               dtype,
                                               stop_gradient=False):
        """
        Create a temporary mpc variable that should be type inferred layer.
        Refer to LayerHelperBase.create_variable_for_type_inference in Paddle 1.7.
        :param dtype:
        :param stop_gradient:
        :return:
        """
        main_program_current_block = self.main_program.current_block()
        return create_mpc_var(
            block=main_program_current_block,
            name=unique_name.generate_with_ignorable_key(".".join(
                [self.name, 'tmp'])),
            dtype=dtype,
            type=core.VarDesc.VarType.LOD_TENSOR,
            persistable=False,
            stop_gradient=stop_gradient)

    def append_mpc_bias_op(self, input_var, dim_start=1, dim_end=None):
        """
        Append bias operator and return its mpc output. If the user does not set
        bias_attr, append_bias_op will return mpc input_var
        Refer to LayerHelper.append_bias_op in Paddle 1.7. Return mpc var.
        :param input_var:
        :param dim_start:
        :param dim_end:
        :return:
        """
        size = list(input_var.shape[
            dim_start + 1:dim_end])  # dims[0]: share_num; dims[1]: batch_size
        bias_attr = self.bias_attr
        if not bias_attr:
            return input_var

        b = self.create_mpc_parameter(
            attr=bias_attr, shape=size, dtype=input_var.dtype, is_bias=True)
        tmp = self.create_mpc_variable_for_type_inference(
            dtype=input_var.dtype)
        # Note: the type of mpc op = "mpc_" + paddle op
        self.append_op(
            type='mpc_elementwise_add',
            inputs={'X': [input_var],
                    'Y': [b]},
            outputs={'Out': [tmp]},
            attrs={'axis': dim_start})
        return tmp

    def append_mpc_activation(self, input_var):
        """
        Append mpc activation for this layer.
        Refer to LayerHelper.append_activation in Paddle 1.7.
        Return mpc ver.
        :param input_var:
        :return:
        """
        act = self.kwargs.get('act', None)
        if act is None:
            return input_var
        if isinstance(act, six.string_types):
            act = {'type': act}
        else:
            raise TypeError(str(act) + " should be unicode or str")

        if 'use_cudnn' in self.kwargs and self.kwargs.get('use_cudnn'):
            act['use_cudnn'] = self.kwargs.get('use_cudnn')
        if 'use_mkldnn' in self.kwargs:
            act['use_mkldnn'] = self.kwargs.get('use_mkldnn')
        act_type = act.pop('type')

        tmp = self.create_mpc_variable_for_type_inference(
            dtype=input_var.dtype)
        # add "mpc_" as prefix of mpc activation
        self.append_op(
            type="mpc_" + act_type,
            inputs={"X": [input_var]},
            outputs={"Out": [tmp]},
            attrs=act)
        return tmp

    def create_mpc_variable(self, *args, **kwargs):
        """
        Create MpcVariable for this layers.
        Refer to LayerHelperBase.create_variable in Paddle 1.7.
        Return created MpcVariable.
        :param args:
        :param kwargs:
        :return:
        """
        main_program_current_block = self.main_program.current_block()
        return create_mpc_var(
            block=main_program_current_block, *args, **kwargs)
