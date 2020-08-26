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
mpc rnn op layers.
"""

from paddle.fluid.framework import in_dygraph_mode
from ..mpc_layer_helper import MpcLayerHelper
from paddle.fluid.data_feeder import check_variable_and_dtype, check_dtype

def dynamic_gru(input,
                size,
                param_attr=None,
                bias_attr=None,
                is_reverse=False,
                gate_activation='sigmoid',
                candidate_activation='relu',
                h_0=None,
                origin_mode=False):
    """
    **Note: The input type of this must be LoDTensor. If the input type to be
    processed is Tensor, use** :ref:`api_fluid_layers_StaticRNN` .

    This operator is used to perform the calculations for a single layer of
    Gated Recurrent Unit (GRU) on full sequences step by step. The calculations
    in one time step support these two modes:

    If ``origin_mode`` is True, then the formula used is from paper
    `Learning Phrase Representations using RNN Encoder Decoder for Statistical
    Machine Translation <https://arxiv.org/pdf/1406.1078.pdf>`_ .

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = u_t \odot h_{t-1} + (1-u_t) \odot \\tilde{h_t}


    if ``origin_mode`` is False, then the formula used is from paper
    `Empirical Evaluation of Gated Recurrent Neural Networks on Sequence
    Modeling  <https://arxiv.org/pdf/1412.3555.pdf>`_

    .. math::

        u_t & = act_g(W_{ux}x_{t} + W_{uh}h_{t-1} + b_u)

        r_t & = act_g(W_{rx}x_{t} + W_{rh}h_{t-1} + b_r)

        \\tilde{h_t} & = act_c(W_{cx}x_{t} + W_{ch}(r_t \odot h_{t-1}) + b_c)

        h_t & = (1-u_t) \odot h_{t-1} + u_t \odot \\tilde{h_t}

    :math:`x_t` is the input of current time step, but it is not from ``input`` .
    This operator does not include the calculations :math:`W_{ux}x_{t}, W_{rx}x_{t}, W_{cx}x_{t}` ,
    **Note** thus a fully-connect layer whose size is 3 times of ``size`` should
    be used before this operator, and the output should be used as ``input`` here.
    :math:`h_{t-1}` is the hidden state from previous time step. 
    :math:`u_t` , :math:`r_t` , :math:`\\tilde{h_t}` and :math:`h_t` stand for
    update gate, reset gate, candidate hidden and hidden output separately.
    :math:`W_{uh}, b_u` , :math:`W_{rh}, b_r` and :math:`W_{ch}, b_c` stand for
    the weight matrix and bias used in update gate, reset gate, candidate hidden
    calculations. For implementation, the three weight matrix are merged into a
    tensor shaped :math:`[D, D \\times 3]` , the three bias are concatenated as
    a tensor shaped :math:`[1, D \\times 3]` , where :math:`D` stands for the
    hidden size; The data layout of weight tensor is: :math:`W_{uh}` and :math:`W_{rh}`
    are concatenated with shape :math:`[D, D  \\times 2]` lying on the first part,
    and :math:`W_{ch}` lying on the latter part with shape :math:`[D, D]` .


    Args:
        input(Variable): A LoDTensor whose lod level is 1, representing the input
            after linear projection. Its shape should be :math:`[T, 2, D \\times 3]` ,
            which is transpose mpc input by axis {1, 0, 2}, and set mpc shares lod,
            where :math:`T` stands for the total sequence lengths in this mini-batch,
            :math:`D` for the hidden size. The data type should be int64.
        size(int): Indicate the hidden size.
        param_attr(ParamAttr, optional):  To specify the weight parameter property.
            Default: None, which means the default weight parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        bias_attr (ParamAttr, optional): To specify the bias parameter property.
            Default: None, which means the default bias parameter property is used.
            See usage for details in :ref:`api_fluid_ParamAttr` .
        is_reverse(bool, optional): Whether to compute in the reversed order of
            input sequences. Default False.
        gate_activation(str, optional): The activation function corresponding to
            :math:`act_g` in the formula. Only 'sigmoid' is supported now.
        candidate_activation(str, optional): The activation function corresponding to
            :math:`act_c` in the formula. Only "relu" is supported now.
        h_0 (Variable, optional): A Tensor representing the initial hidden state.
            It not provided, the default initial hidden state is 0. The shape is
            :math:`[2, N, D]` , where :math:`N` is the number of sequences in the
            mini-batch, :math:`D` for the hidden size. The data type should be
            same as ``input`` . Default None.

    Returns:
        Variable: A LoDTensor whose lod level is 1 and shape is :math:`[2, T, D]` , \
            where :math:`T` stands for the total sequence lengths in this mini-batch \
            :math:`D` for the hidden size. It represents GRU transformed sequence output, \
            and has the same lod and data type with ``input`` .

    Examples:

        .. code-block:: python

            import paddle.fluid as fluid

            dict_dim, emb_dim = 128, 64
            data = fluid.data(name='sequence',
                      shape=[None],
                      dtype='int64',
                      lod_level=1)
            emb = fluid.embedding(input=data, size=[dict_dim, emb_dim])
            hidden_dim = 512
            x = fluid.layers.fc(input=emb, size=hidden_dim * 3)
            hidden = fluid.layers.dynamic_gru(input=x, size=hidden_dim)
    """

    assert in_dygraph_mode(
    ) is not True, "please use gru instead of dynamic_gru in dygraph mode!"

    helper = MpcLayerHelper('mpc_gru', **locals())
    dtype = helper.input_dtype()

    check_variable_and_dtype(input, 'input', ['int64'], 'mpc_gru')
    check_dtype(dtype, 'dtype', ['int64'], 'mpc_gru')

    weight = helper.create_mpc_parameter(
        attr=helper.param_attr, shape=[size, 3 * size], dtype=dtype)
    bias = helper.create_mpc_parameter(
        attr=helper.bias_attr, shape=[1, 3 * size], dtype=dtype, is_bias=True)
    batch_size = input.shape[0]
    inputs = {'Input': input, 'Weight': weight, 'Bias': bias}
    if h_0:
        assert h_0.shape == (
            2, batch_size, size
        ), 'The shape of h0 should be(batch_size, %d)' % size
        inputs['H0'] = h_0

    hidden = helper.create_mpc_variable_for_type_inference(dtype)
    batch_gate = helper.create_mpc_variable_for_type_inference(dtype)
    batch_reset_hidden_prev = helper.create_mpc_variable_for_type_inference(dtype)
    batch_hidden = helper.create_mpc_variable_for_type_inference(dtype)

    helper.append_op(
        type='mpc_gru',
        inputs=inputs,
        outputs={
            'Hidden': hidden,
            'BatchGate': batch_gate,
            'BatchResetHiddenPrev': batch_reset_hidden_prev,
            'BatchHidden': batch_hidden
        },
        attrs={
            'is_reverse': is_reverse,
            'gate_activation': gate_activation,
            'activation': candidate_activation,
            'origin_mode': origin_mode
        })
    return hidden
