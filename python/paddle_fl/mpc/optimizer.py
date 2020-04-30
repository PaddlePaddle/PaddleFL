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
MPC Optimizers
"""

from paddle.fluid.optimizer import Optimizer
from paddle.fluid import framework
from paddle.fluid.framework import program_guard
from paddle.fluid.framework import Variable
from paddle.fluid.clip import error_clip_callback
from paddle.fluid import unique_name
from paddle.fluid.initializer import Constant
from .backward import append_backward
from .mpc_layer_helper import MpcLayerHelper


class MPCSGDOptimizer(Optimizer):
    """
    MPCSGDOptimizer Implementation based on Optimizer Class in Paddle
    """

    def __init__(self, learning_rate, regularization=None, name=None):
        """
        """
        assert learning_rate is not None
        super(MPCSGDOptimizer, self).__init__(
            learning_rate=learning_rate,
            regularization=regularization,
            name=name)
        self.type = "mpc_sgd"

    def _append_optimize_op(self, block, param_and_grad):
        """
        Optimizer of the stochastic gradient descent algorithm.
        .. math::
            param\_out = param - learning\_rate * grad
        Parameters:
            learning_rate (float|Variable): The learning rate used to update parameters. \
                Can be a float value or a Variable with one float value as data element.
            parameter_list (list, optional):  List of ``Variable`` names to update to minimize ``loss``. \
                This parameter is required in dygraph mode. \
                The default value is None in static mode, at this time all parameters will be updated.
            regularization: A Regularizer, such as :ref:`api_fluid_regularizer_L2DecayRegularizer`. \
                Optional, default is None.
            name (str, optional): This parameter is used by developers to print debugging information. \
                For details, please refer to :ref:`api_guide_Name`. Default is None.
        """
        assert isinstance(block, framework.Block)
        mpc_sgd_op = block.append_op(
            type=self.type,
            inputs={
                "Param": param_and_grad[0],
                "Grad": param_and_grad[1],
                "LearningRate": self._create_param_lr(param_and_grad)
            },
            outputs={"ParamOut": param_and_grad[0]},
            stop_gradient=True)

        return mpc_sgd_op

    def backward(self,
                 loss,
                 startup_program=None,
                 parameter_list=None,
                 no_grad_set=None,
                 callbacks=None):
        """
        The first part of ``minimize``, do auto-diff to append backward operations for
        the current program.
        Args:
            loss (Variable): ``loss`` variable to run optimizations.
            startup_program (Program, optional): :ref:`api_fluid_Program` for
                initializing parameters in ``parameter_list``. The default value
                is None, at this time :ref:`api_fluid_default_startup_program` will be used.
            parameter_list (list, optional): List of ``Variable`` names to update
                to minimize ``loss``. The default value is None, at this time all parameters
                will be updated.
            no_grad_set (set, optional): Set of ``Variable`` objects that don't need
                to be updated. The default value is None.
            callbacks (list, optional): list of callable objects to run when appending backward
                operator for one parameter. The default value is None.
        Return:
            list: list of (param, grad) variable pairs, param is ``Parameter``,
                grad is the gradient value corresponding to the parameter.
        Examples:
            See examples in ``apply_gradients``.
        """
        no_grad_set = self._get_no_grad_set(loss, no_grad_set)

        self._dtype = loss.dtype

        if callbacks is None:
            callbacks = [error_clip_callback]
        else:
            assert (isinstance(callbacks, list))
        program = loss.block.program
        assert len(loss.shape) == 2 and loss.shape[0] == 2 and loss.shape[1] == 1, \
                "The loss.shape should be (2L,), but the current loss.shape is {}. " \
                "Maybe that you should call fluid.layers.mean to process the current loss.".format(
                    loss.shape)
        with program_guard(program, startup_program):
            params_grads = append_backward(loss, parameter_list, no_grad_set,
                                           callbacks)
            # Note: since we can't use all_reduce_op now,
            #  dgc_op should be the last op of one grad.
            self._append_dgc_ops(params_grads)
        return params_grads

    def _create_global_learning_rate(self):
        lr = self._global_learning_rate()

        if isinstance(lr, framework.Variable):
            return
        else:
            if not isinstance(self._learning_rate, float):
                raise TypeError(
                    "learning rate variable is create outside optimizer,"
                    "can not create new learning rate variable for new program")

        # create learning rate in the current main program
        self._learning_rate_map[framework.default_main_program(
        )] = create_global_var(
            name=unique_name.generate("learning_rate"),
            shape=[1],
            value=float(self._learning_rate),
            dtype='double',
            persistable=True)

    def _create_param_lr(self, param_and_grad):
        """
        create learning rate parameter
        """
        # create learning rate variable for every parameter
        param = param_and_grad[0]
        param_lr = param.optimize_attr['learning_rate']
        if type(param_lr) == Variable:
            return param_lr
        else:
            if param_lr == 1.0:
                return self._global_learning_rate()
            else:
                with fluid.default_main_program()._lr_schedule_guard(
                        is_with_opt=True), framework.name_scope(
                            'scale_with_param_lr'):
                    return self._global_learning_rate() * param_lr

    def _global_learning_rate(self, program=None):
        """
        get global decayed learning rate
        :return:
        """
        if program is None:
            program = framework.default_main_program()
        return self._learning_rate_map.get(program, None)


def create_global_var(shape,
                      value,
                      dtype,
                      persistable=False,
                      force_cpu=False,
                      name=None):
    """
    This function creates a new tensor variable with value in the global block(block 0).
    Parameters:
        shape (list of int): Shape of the variable
        value (float): The value of the variable. The new created
                      variable will be filled with it.
        dtype (str): Data type of the variable
        persistable (bool, optional): If this variable is persistable.
                           Default: False
        force_cpu (bool, optional): Force this variable to be on CPU.
                         Default: False
        name (str, optional): For detailed information, please refer to
           :ref:`api_guide_Name` . Usually name is no need to set and None by default.
    Returns:
        Variable: The created Variable
    Examples:
        .. code-block:: python
            import paddle.fluid as fluid
            import paddle.fluid.layers as layers
            var = layers.create_global_var(shape=[2,3], value=1.0, dtype='float32',
                                           persistable=True, force_cpu=True, name='new_var')
    """
    helper = MpcLayerHelper("global_var", **locals())
    var = helper.create_global_variable(
        dtype=dtype,
        shape=shape,
        persistable=persistable,
        name=name,
        stop_gradient=True)
    helper.set_variable_initializer(
        var, initializer=Constant(
            value=float(value), force_cpu=force_cpu))

    return var


SGD = MPCSGDOptimizer
