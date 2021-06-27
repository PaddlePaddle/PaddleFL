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
MPC('int64') Initializer
"""
import numpy as np
import mpc_data_utils as mdu
import paddle.fluid as fluid
from paddle.fluid.initializer import Initializer
import paddle.fluid.framework as framework
from paddle.fluid.core import VarDesc
from paddle.fluid import unique_name
from paddle.fluid.data_feeder import check_variable_and_dtype, check_type, check_dtype
from paddle_fl.mpc.framework import MpcProtocols


class NumpyArrayInitializer(Initializer):
    """Init a mpc parameter with an numpy array (astype('int64'))
    This op initialize the variable by numpy array.

    Args:
        value (numpy): numpy array to initialize the variable

    Returns:
        A Tensor variable initialized by numpy.

    Examples:
        .. code-block:: python

            import paddle_fl.mpc as pfl
            import numpy
            weight_share = numpy.array([1,2]).astype('int64')
            w_param_attrs = fluid.ParamAttr(name='emb_weight',
                                        learning_rate=0.5,
                                        initializer=pfl_mpc.initializer.NumpyArrayInitializer(weight_share),
                                        trainable=True)
    """

    def __init__(self, value):
        import numpy
        assert isinstance(value, numpy.ndarray)
        super(NumpyArrayInitializer, self).__init__()
        self._value = value

    def __call__(self, var, block):
        """Add constant initialization ops for a variable

        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added

        Returns:
            the initialization op
        """
        assert isinstance(var, framework.Variable)
        assert isinstance(block, framework.Block)

        out_var = var
        out_dtype = var.dtype
        np_value = self._value

        value_name = "int64_values"
        if (out_dtype != VarDesc.VarType.INT64):
            raise ValueError(
                "Only 'int64' dtype is supported in paddlefl's initializer, "
                "Use paddle.fluid.initializer for other dtype.")
        values = [int(v) for v in np_value.flat]

        if self._value.size > 1024 * 1024 * 1024:
            raise ValueError("The size of input is too big. Please consider "
                             "saving it to file and 'load_op' to load it")
        op = block._prepend_op(
            type='assign_value',
            outputs={'Out': out_var},
            attrs={
                'dtype': out_dtype,
                'shape': list(self._value.shape),
                value_name: values
            },
            stop_gradient=True)

        if not framework.in_dygraph_mode():
            var.op = op
        return op


class XavierInitializer(Initializer):
    """
    This class implements the Xavier weight initializer from the paper
    `Understanding the difficulty of training deep feedforward neural
    networks <http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf>`_
    by Xavier Glorot and Yoshua Bengio.
    This initializer is designed to keep the scale of the gradients
    approximately same in all the layers. In case of Uniform distribution,
    the range is [-x, x], where
    .. math::
        x = \sqrt{\\frac{6.0}{fan\_in + fan\_out}}
    In case of Normal distribution, the mean is 0 and the standard deviation
    is
    .. math::
        \sqrt{\\frac{2.0}{fan\_in + fan\_out}}
    Args:
        uniform (bool,default True): whether to use uniform ,if False use normal distribution
        fan_in (float,default None): fan_in for Xavier initialization. If None, it is
                inferred from the variable.
        fan_out (float,default None): fan_out for Xavier initialization. If None, it is
                 inferred from the variable.
        seed (int): random seed
    Note:
        It is recommended to set fan_in and fan_out to None for most cases.
        Share of the distribution will be returned.
        The seeds of three parties should be same.
    Examples:
        .. code-block:: python
            import paddle_fl.mpc as pfl_mpc
            queries = pfl_mpc.data(name='x', shape=[2,1], dtype='int64')
            fc = pfl_mpc.layers.fc(
                input=queries, size=10,
                param_attr=pfl_mpc.initializer.Xavier(uniform=False))
    """


    class XavierInitImpl(Initializer):

        def get_mpc_one_share(self):
            """
            get ONE share of mpc protocol
            """
            pass

        def get_plain_var_shape(self, mpc_var):
            """
            get the shape of the plain form of mpc var
            """ 
            pass

        def append_cast_op(self, input_var, output_var, block): 
            """
            expand input_var for some mpc protocols, and cast input_var into output_var
            """
            pass


    class Aby3XavierInitImpl(XavierInitImpl):
        """
        XavierInitialier implementation for aby3 protocol
        """
        def _compute_fans(self, var):
            """Compute the fan_in and the fan_out for layers
               Note: var.shape[0] is share_num.
            Args:
                var: variable for which fan_in and fan_out have to be computed
            Returns:
                tuple of two integers (fan_in, fan_out)
            """
            shape = var.shape
            if not shape or len(shape) == 0:
                raise ValueError(
                    "Shape should be larger than 0 in paddlefl's initializer.")
            elif len(shape) == 1:
                fan_in = fan_out = 1
            elif len(shape) == 2:
                fan_in = fan_out = shape[1]
            elif len(shape) == 3:
                # This is the case for simple matrix multiply
                fan_in = shape[1]
                fan_out = shape[2]
            else:
                # Assume this to be a convolutional kernel
                # In PaddlePaddle, the shape of the kernel is like:
                # [num_filters, num_filter_channels, ...] where the remaining
                # dimensions are the filter_size
                receptive_field_size = np.prod(shape[3:])
                fan_in = shape[2] * receptive_field_size
                fan_out = shape[1] * receptive_field_size

            return (fan_in, fan_out)

        def get_mpc_one_share(self):
            """
            get ONE share of aby3 protocol
            """
            return mdu.aby3_one_share

        def get_plain_var_shape(self, mpc_var):
            """
            get the shape of the plain form of mpc var
            """ 
            shape_ = list(mpc_var.shape)
            shape_[0] = 1
            return shape_

        def append_cast_op(self, input_var, output_var, block):
            """
            expand input_var from  (1, ...) to  (2, ...)
            and cast expand_input_var into output_var
            """
            # extend encrytped var
            expand_input_var = block.create_var(
                name=unique_name.generate(".".join(
                    ['gaussian_random_expand', output_var.name, 'tmp'])),
                shape=input_var.shape,
                dtype=input_var.dtype,
                type=VarDesc.VarType.LOD_TENSOR,
                persistable=False)

            block.append_op(
                type="concat",
                inputs={"X": [input_var, input_var]},
                outputs={"Out": [expand_input_var]},
                attrs={"axis": 0})

            # cast expand_input_var into output_var
            block.append_op(
                type="cast",
                inputs={"X": expand_input_var},
                outputs={"Out": output_var},
                attrs={"in_dtype": expand_input_var.dtype,
                       "out_dtype": output_var.dtype})
            

    class PrivcXavierInitImpl(XavierInitImpl):
        """
        XavierInitialier implementation for privc protocol
        """
        def get_mpc_one_share(self):
            """
            get ONE share of privc protocol
            """
            return mdu.privc_one_share

        def get_plain_var_shape(self, mpc_var):
            """
            get the shape of the plain form of mpc var
            """ 
            return mpc_var.shape

        def append_cast_op(self, input_var, output_var, block): 
            """
            cast input_var into output_var
            """
            block.append_op(
                type="cast",
                inputs={"X": input_var},
                outputs={"Out": output_var},
                attrs={"in_dtype": input_var.dtype,
                       "out_dtype": output_var.dtype})


    def __init__(self, uniform=True, fan_in=None, fan_out=None, seed=0):
        assert uniform is not None
        assert seed is not None
        super(XavierInitializer, self).__init__()
        self._uniform = uniform
        self._fan_in = fan_in
        self._fan_out = fan_out
        self._seed = seed

        mpc_protocol_index = np.array(fluid.global_scope().find_var("mpc_protocol_index").get_tensor())
        mpc_protocol = MpcProtocols(mpc_protocol_index)

        if mpc_protocol is MpcProtocols.ABY3:
            self.initializer = self.Aby3XavierInitImpl()
        if mpc_protocol is MpcProtocols.PRIVC:
            self.initializer = self.PrivcXavierInitImpl()

    def _compute_fans(self, var):
        """Compute the fan_in and the fan_out for layers
        This method computes the fan_in and the fan_out
        for neural network layers, if not specified. It is
        not possible to perfectly estimate fan_in and fan_out.
        This method will estimate it correctly for matrix multiply and
        convolutions.
        Args:
            var: variable for which fan_in and fan_out have to be computed
        Returns:
            tuple of two integers (fan_in, fan_out)
        """
        return self.initializer._compute_fans(var)

    def __call__(self, var, block):
        """Add xavier initialization ops for a variable
        Args:
            var: Variable that needs to be initialized
            block: The block in which initialization ops
                   should be added
        Returns:
            the initialization op
        """
        assert isinstance(block, framework.Block)
        check_variable_and_dtype(var, "Out", ["int64"], "xavier_init")

        if (var.dtype != VarDesc.VarType.INT64):
            raise ValueError(
                "Only 'int64' dtype is supported in paddlefl's initializer.")

        f_in, f_out = self._compute_fans(var)

        # If fan_in and fan_out are passed, use them
        fan_in = f_in if self._fan_in is None else self._fan_in
        fan_out = f_out if self._fan_out is None else self._fan_out

        if self._seed == 0:
            self._seed = block.program.random_seed

        out_dtype = VarDesc.VarType.FP32
        shape_ = list(var.shape)

        mpc_one_share = self.initializer.get_mpc_one_share()
        shape_ = self.initializer.get_plain_var_shape(var)

        # create tmp var:
        out_var = block.create_var(
            name=unique_name.generate(".".join(
                ['gaussian_random', var.name, 'tmp'])),
            shape=shape_,
            dtype=out_dtype,
            type=VarDesc.VarType.LOD_TENSOR,
            persistable=False)

        if self._uniform:
            limit = np.sqrt(6.0 / float(fan_in + fan_out))
            op = block._prepend_op(
                type="uniform_random",
                inputs={},
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "min": -limit,
                    "max": limit,
                    "seed": self._seed
                },
                stop_gradient=True)
        else:
            std = np.sqrt(2.0 / float(fan_in + fan_out))
            op = block._prepend_op(
                type="gaussian_random",
                outputs={"Out": out_var},
                attrs={
                    "shape": out_var.shape,
                    "dtype": out_dtype,
                    "mean": 0.0,
                    "std": std,
                    "seed": self._seed
                },
                stop_gradient=True)

        # convert plaintext into cyphertext
        block.append_op(
            type="scale",
            inputs={"X": out_var},
            outputs={"Out": out_var},
            attrs={"scale": float(mpc_one_share)})

        # cast var (extend var for some mpc protocols).
        self.initializer.append_cast_op(out_var, var, block)

        if not framework.in_dygraph_mode():
            var.op = op
        return op
