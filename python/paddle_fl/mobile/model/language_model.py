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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from model import *
from layer import *
from .model_base import ModelBase
import paddle.fluid as fluid

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np
from paddle.fluid import ParamAttr
from paddle.fluid.contrib.layers import basic_lstm


class LanguageModel(ModelBase):
    def __init__(self):
        # model args
        self.hidden_size_ = 200
        self.vocab_size_ = 10000
        self.num_layers_ = 2
        self.num_steps_ = 10  # fix
        self.init_scale_ = 0.1
        self.dropout_ = 0.0
        self.rnn_model_ = 'basic_lstm'
        self.pad_symbol_ = 0
        self.unk_symbol_ = 1

        # results
        self.correct_ = None
        self.prediction_ = None
        self.loss_ = None

        # private vars
        self.user_params_ = []
        self.program_ = None
        self.startup_program_ = None
        self.input_name_list_ = None
        self.target_var_names_ = []

    def get_model_input_names(self):
        return self.input_name_list_

    def get_model_loss(self):
        return self.loss_

    def get_model_loss_name(self):
        return self.loss_.name

    def get_model_metrics(self):
        return {"correct": self.correct_.name}

    def get_target_names(self):
        return self.target_var_names_

    def build_model(self, model_configs):
        hidden_size = self.hidden_size_
        init_scale = self.init_scale_
        dropout = self.dropout_
        num_layers = self.num_layers_
        num_steps = self.num_steps_
        pad_symbol = self.pad_symbol_
        unk_symbol = self.unk_symbol_
        vocab_size = self.vocab_size_
        rnn_model = self.rnn_model_
        x = fluid.data(name="x", shape=[None, num_steps], dtype='int64')
        y = fluid.data(name="y", shape=[None, num_steps], dtype='int64')
        x = layers.reshape(x, shape=[-1, num_steps, 1])
        y = layers.reshape(y, shape=[-1, 1])
        self.input_name_list_ = ['x', 'y']

        init_hidden = layers.fill_constant_batch_size_like(
            input=x,
            shape=[-1, num_layers, hidden_size],
            value=0,
            dtype="float32")
        init_cell = layers.fill_constant_batch_size_like(
            input=x,
            shape=[-1, num_layers, hidden_size],
            value=0,
            dtype="float32")

        init_hidden = layers.transpose(init_hidden, perm=[1, 0, 2])
        init_cell = layers.transpose(init_cell, perm=[1, 0, 2])

        init_hidden_reshape = layers.reshape(
            init_hidden, shape=[num_layers, -1, hidden_size])
        init_cell_reshape = layers.reshape(
            init_cell, shape=[num_layers, -1, hidden_size])

        x_emb = layers.embedding(
            input=x,
            size=[vocab_size, hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale)))

        x_emb = layers.reshape(
            x_emb, shape=[-1, num_steps, hidden_size], inplace=True)
        if dropout != None and dropout > 0.0:
            x_emb = layers.dropout(
                x_emb,
                dropout_prob=dropout,
                dropout_implementation='upscale_in_train')

        if rnn_model == "padding":
            rnn_out, last_hidden, last_cell = self._padding_rnn(
                x_emb,
                len=num_steps,
                init_hidden=init_hidden_reshape,
                init_cell=init_cell_reshape)
        elif rnn_model == "static":
            rnn_out, last_hidden, last_cell = self._encoder_static(
                x_emb,
                len=num_steps,
                init_hidden=init_hidden_reshape,
                init_cell=init_cell_reshape)
        elif rnn_model == "cudnn":
            x_emb = layers.transpose(x_emb, perm=[1, 0, 2])
            rnn_out, last_hidden, last_cell = layers.lstm(
                x_emb,
                init_hidden_reshape,
                init_cell_reshape,
                num_steps,
                hidden_size,
                num_layers,
                is_bidirec=False,
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            rnn_out = layers.transpose(rnn_out, perm=[1, 0, 2])
        elif rnn_model == "basic_lstm":
            rnn_out, last_hidden, last_cell = basic_lstm(
                x_emb,
                init_hidden,
                init_cell,
                hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout_prob=dropout,
                param_attr=ParamAttr(
                    initializer=fluid.initializer.UniformInitializer(
                        low=-init_scale, high=init_scale)),
                bias_attr=ParamAttr(
                    initializer=fluid.initializer.Constant(0.0)),
                forget_bias=0.0)
        else:
            raise Exception("type not support")

        rnn_out = layers.reshape(
            rnn_out, shape=[-1, num_steps, hidden_size], inplace=True)

        softmax_weight = layers.create_parameter(
            [hidden_size, vocab_size],
            dtype="float32",
            name="softmax_weight",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale))
        softmax_bias = layers.create_parameter(
            [vocab_size],
            dtype="float32",
            name='softmax_bias',
            default_initializer=fluid.initializer.UniformInitializer(
                low=-init_scale, high=init_scale))

        projection = layers.matmul(rnn_out, softmax_weight)
        projection = layers.elementwise_add(projection, softmax_bias)
        projection = layers.reshape(
            projection, shape=[-1, vocab_size], inplace=True)

        # correct predictions
        labels_reshaped = fluid.layers.reshape(y, [-1])
        pred = fluid.layers.cast(
            fluid.layers.argmax(projection, 1), dtype="int64")
        correct_pred = fluid.layers.cast(
            fluid.layers.equal(pred, labels_reshaped), dtype="int64")
        self.prediction_ = pred
        self.target_var_names_.append(pred)

        # predicting unknown is always considered wrong
        unk_tensor = fluid.layers.fill_constant(
            fluid.layers.shape(labels_reshaped),
            value=unk_symbol,
            dtype='int64')
        pred_unk = fluid.layers.cast(
            fluid.layers.equal(pred, unk_tensor), dtype="int64")
        correct_unk = fluid.layers.elementwise_mul(pred_unk, correct_pred)

        # predicting padding is always considered wrong
        pad_tensor = fluid.layers.fill_constant(
            fluid.layers.shape(labels_reshaped), value=0, dtype='int64')
        pred_pad = fluid.layers.cast(
            fluid.layers.equal(pred, pad_tensor), dtype="int64")
        correct_pad = fluid.layers.elementwise_mul(pred_pad, correct_pred)

        # acc
        correct_count = fluid.layers.reduce_sum(correct_pred) \
                - fluid.layers.reduce_sum(correct_unk) \
                - fluid.layers.reduce_sum(correct_pad)
        self.correct_ = correct_count
        self.target_var_names_.append(correct_count)

        loss = layers.softmax_with_cross_entropy(
            logits=projection, label=y, soft_label=False)

        loss = layers.reshape(loss, shape=[-1, num_steps], inplace=True)
        loss = layers.reduce_mean(loss, dim=[0])
        loss = layers.reduce_sum(loss)
        self.loss_ = loss
        self.target_var_names_.append(loss)

        loss.persistable = True

        # This will feed last_hidden, last_cell to init_hidden, init_cell, which
        # can be used directly in next batch. This can avoid the fetching of
        # last_hidden and last_cell and feeding of init_hidden and init_cell in
        # each training step.
        #last_hidden = layers.transpose(last_hidden, perm=[1, 0, 2])
        #last_cell = layers.transpose(last_cell, perm=[1, 0, 2])
        #self.input_name_list_ = ['x', 'y', 'init_hidden', 'init_cell']

        self.program_ = fluid.default_main_program()
        self.startup_program_ = fluid.default_startup_program()

    def _padding_rnn(input_embedding, len=3, init_hidden=None, init_cell=None):
        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        hidden_size = self.hidden_size_
        init_scale = self.init_scale_
        dropout = slef.dropout_
        num_layers = self.num_layers_
        num_steps = self._num_steps_
        for i in range(num_layers):
            weight_1 = layers.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            weight_1_arr.append(weight_1)
            bias_1 = layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            bias_arr.append(bias_1)

            pre_hidden = layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = layers.reshape(pre_hidden, shape=[-1, hidden_size])
            pre_cell = layers.reshape(pre_cell, shape=[-1, hidden_size])
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        input_embedding = layers.transpose(input_embedding, perm=[1, 0, 2])
        rnn = PaddingRNN()

        with rnn.step():
            input = rnn.step_input(input_embedding)
            for k in range(num_layers):
                pre_hidden = rnn.memory(init=hidden_array[k])
                pre_cell = rnn.memory(init=cell_array[k])
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]

                nn = layers.concat([input, pre_hidden], 1)
                gate_input = layers.matmul(x=nn, y=weight_1)

                gate_input = layers.elementwise_add(gate_input, bias)
                i = layers.slice(
                    gate_input, axes=[1], starts=[0], ends=[hidden_size])
                j = layers.slice(
                    gate_input,
                    axes=[1],
                    starts=[hidden_size],
                    ends=[hidden_size * 2])
                f = layers.slice(
                    gate_input,
                    axes=[1],
                    starts=[hidden_size * 2],
                    ends=[hidden_size * 3])
                o = layers.slice(
                    gate_input,
                    axes=[1],
                    starts=[hidden_size * 3],
                    ends=[hidden_size * 4])

                c = pre_cell * layers.sigmoid(f) + layers.sigmoid(
                    i) * layers.tanh(j)
                m = layers.tanh(c) * layers.sigmoid(o)

                rnn.update_memory(pre_hidden, m)
                rnn.update_memory(pre_cell, c)

                rnn.step_output(m)
                rnn.step_output(c)

                input = m

                if dropout != None and dropout > 0.0:
                    input = layers.dropout(
                        input,
                        dropout_prob=dropout,
                        dropout_implementation='upscale_in_train')

            rnn.step_output(input)
        rnnout = rnn()

        last_hidden_array = []
        last_cell_array = []
        real_res = rnnout[-1]
        for i in range(num_layers):
            m = rnnout[i * 2]
            c = rnnout[i * 2 + 1]
            m.stop_gradient = True
            c.stop_gradient = True
            last_h = layers.slice(
                m, axes=[0], starts=[num_steps - 1], ends=[num_steps])
            last_hidden_array.append(last_h)
            last_c = layers.slice(
                c, axes=[0], starts=[num_steps - 1], ends=[num_steps])
            last_cell_array.append(last_c)
        real_res = layers.transpose(x=real_res, perm=[1, 0, 2])
        last_hidden = layers.concat(last_hidden_array, 0)
        last_cell = layers.concat(last_cell_array, 0)

        return real_res, last_hidden, last_cell

    def _encoder_static(input_embedding,
                        len=3,
                        init_hidden=None,
                        init_cell=None):
        weight_1_arr = []
        weight_2_arr = []
        bias_arr = []
        hidden_array = []
        cell_array = []
        mask_array = []
        hidden_size = self.hidden_size_
        init_scale = self.init_scale_
        dropout = slef.dropout_
        num_layers = self.num_layers_
        for i in range(num_layers):
            weight_1 = layers.create_parameter(
                [hidden_size * 2, hidden_size * 4],
                dtype="float32",
                name="fc_weight1_" + str(i),
                default_initializer=fluid.initializer.UniformInitializer(
                    low=-init_scale, high=init_scale))
            weight_1_arr.append(weight_1)
            bias_1 = layers.create_parameter(
                [hidden_size * 4],
                dtype="float32",
                name="fc_bias1_" + str(i),
                default_initializer=fluid.initializer.Constant(0.0))
            bias_arr.append(bias_1)

            pre_hidden = layers.slice(
                init_hidden, axes=[0], starts=[i], ends=[i + 1])
            pre_cell = layers.slice(
                init_cell, axes=[0], starts=[i], ends=[i + 1])
            pre_hidden = layers.reshape(
                pre_hidden, shape=[-1, hidden_size], inplace=True)
            pre_cell = layers.reshape(
                pre_cell, shape=[-1, hidden_size], inplace=True)
            hidden_array.append(pre_hidden)
            cell_array.append(pre_cell)

        res = []
        sliced_inputs = layers.split(
            input_embedding, num_or_sections=len, dim=1)

        for index in range(len):
            input = sliced_inputs[index]
            input = layers.reshape(
                input, shape=[-1, hidden_size], inplace=True)
            for k in range(num_layers):
                pre_hidden = hidden_array[k]
                pre_cell = cell_array[k]
                weight_1 = weight_1_arr[k]
                bias = bias_arr[k]
                nn = layers.concat([input, pre_hidden], 1)
                gate_input = layers.matmul(x=nn, y=weight_1)

                gate_input = layers.elementwise_add(gate_input, bias)
                i, j, f, o = layers.split(
                    gate_input, num_or_sections=4, dim=-1)

                c = pre_cell * layers.sigmoid(f) + layers.sigmoid(
                    i) * layers.tanh(j)
                m = layers.tanh(c) * layers.sigmoid(o)

                hidden_array[k] = m
                cell_array[k] = c
                input = m

                if dropout != None and dropout > 0.0:
                    input = layers.dropout(
                        input,
                        dropout_prob=dropout,
                        dropout_implementation='upscale_in_train')

            res.append(input)

        last_hidden = layers.concat(hidden_array, 1)
        last_hidden = layers.reshape(
            last_hidden, shape=[-1, num_layers, hidden_size], inplace=True)
        last_hidden = layers.transpose(x=last_hidden, perm=[1, 0, 2])

        last_cell = layers.concat(cell_array, 1)
        last_cell = layers.reshape(
            last_cell, shape=[-1, num_layers, hidden_size])
        last_cell = layers.transpose(x=last_cell, perm=[1, 0, 2])

        real_res = layers.concat(res, 0)
        real_res = layers.reshape(
            real_res, shape=[len, -1, hidden_size], inplace=True)
        real_res = layers.transpose(x=real_res, perm=[1, 0, 2])

        return real_res, last_hidden, last_cell
