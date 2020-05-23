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
from paddle.fluid import ParamAttr
from paddle.fluid.contrib.layers import basic_lstm


class LanguageModel(ModelBase):
    def __init__(self):
        # model args
        self.seq_len_ = 10  # fixed
        self.n_hidden_ = 256
        self.num_layers_ = 2
        self.pad_symbol_ = 0
        self.unk_symbol_ = 1
        self.vocab_size_ = 10000
        self.init_scale_ = 0.1
        self.max_grad_norm_ = 5
        self.dropout_prob_ = 0.0

        # results
        self.correct_ = None
        self.pred_ = None
        self.loss_ = None

        # private vars
        self.user_params_ = []
        self.program_ = None
        self.startup_program_ = None
        self.input_name_list_ = None
        self.target_var_names_ = []

    def update_params(self, config):
        self.n_hidden_ = config.get("n_hidden", 256)
        self.num_layers_ = config.get("num_layers", 2)
        self.init_scale_ = config.get("init_scale", 0.1)
        self.max_grad_norm_ = config.get("max_grad_norm", 5)
        self.dropout_prob_ = config.get("dropout_prob", 0.0)

    def get_model_input_names(self):
        return self.input_name_list_

    def get_model_loss(self):
        return self.loss_

    def get_model_loss_name(self):
        return self.loss_.name

    def get_model_metrics(self):
        metrics = {
            "init_hidden": self.last_hidden_.name,
            "init_cell": self.last_cell_.name,
            "correct": self.correct_.name
        }
        return metrics

    def get_target_names(self):
        return self.target_var_names_

    def build_model(self, model_configs):
        self.update_params(model_configs)
        features = fluid.layers.data(name="features",
                                     shape=[None, self.seq_len_],
                                     dtype='int64')
        labels = fluid.layers.data(name="labels",
                                   shape=[None, self.seq_len_],
                                   dtype='int64')
        sequence_length_ph = fluid.layers.data(name="seq_len_ph",
                                               shape=[None],
                                               dtype='int64')
        sequence_mask_ph = fluid.layers.data(name="seq_mask_ph",
                                             shape=[None],
                                             dtype='float32')

        init_hidden = fluid.layers.data(
            name="init_hidden",
            shape=[None, self.num_layers_, self.n_hidden_],
            dtype='float32')
        init_cell = fluid.layers.data(
            name="init_cell",
            shape=[None, self.num_layers_, self.n_hidden_],
            dtype='float32')

        init_hidden = layers.transpose(init_hidden, perm=[1, 0, 2])
        init_cell = layers.transpose(init_cell, perm=[1, 0, 2])

        init_hidden_reshape = layers.reshape(
            init_hidden, shape=[self.num_layers_, -1, self.n_hidden_])
        init_cell_reshape = layers.reshape(
            init_cell, shape=[self.num_layers_, -1, self.n_hidden_])

        features = layers.reshape(features, shape=[-1, self.seq_len_, 1])

        # word embedding
        inputs = layers.embedding(
            input=features,
            size=[self.vocab_size_, self.n_hidden_],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='embedding_para',
                initializer=fluid.initializer.UniformInitializer(
                    low=-self.init_scale_, high=self.init_scale_)))

        # LSTM
        output, last_hidden, last_cell = self._build_rnn_graph(
            inputs, init_hidden, init_cell, sequence_length_ph)

        output = layers.reshape(output,
                                shape=[-1, self.seq_len_, self.n_hidden_],
                                inplace=True)
        self.last_hidden_ = layers.reshape(
            last_hidden, [-1, self.num_layers_, self.n_hidden_])
        self.last_cell_ = layers.reshape(
            last_cell, [-1, self.num_layers_, self.n_hidden_])

        # softmax
        softmax_w = layers.create_parameter(
            [self.n_hidden_, self.vocab_size_],
            dtype="float32",
            name="softmax_w",
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale_, high=self.init_scale_))
        softmax_b = layers.create_parameter(
            [self.vocab_size_],
            dtype="float32",
            name='softmax_b',
            default_initializer=fluid.initializer.UniformInitializer(
                low=-self.init_scale_, high=self.init_scale_))

        logits = layers.matmul(output, softmax_w)
        logits = layers.elementwise_add(logits, softmax_b)
        logits = layers.reshape(logits,
                                shape=[-1, self.vocab_size_],
                                inplace=True)

        # correct predictions
        labels_reshaped = layers.reshape(labels, [-1])
        pred = layers.cast(layers.argmax(logits, 1), dtype="int64")
        correct_pred = layers.cast(layers.equal(pred, labels_reshaped),
                                   dtype="int64")
        self.pred_ = pred

        # predicting unknown is always considered wrong
        # only in paddle 1.8
        unk_tensor = layers.fill_constant(layers.shape(labels_reshaped),
                                          value=self.unk_symbol_,
                                          dtype='int64')
        pred_unk = layers.cast(layers.equal(pred, unk_tensor), dtype="int64")
        correct_unk = layers.elementwise_mul(pred_unk, correct_pred)

        # predicting padding is always considered wrong
        pad_tensor = layers.fill_constant(layers.shape(labels_reshaped),
                                          value=self.pad_symbol_,
                                          dtype='int64')
        pred_pad = layers.cast(layers.equal(pred, pad_tensor), dtype="int64")
        correct_pad = layers.elementwise_mul(pred_pad, correct_pred)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = layers.reshape(logits, [-1, self.seq_len_, self.vocab_size_])

        labels = layers.reshape(labels, [-1, self.seq_len_, 1])
        loss = layers.softmax_with_cross_entropy(logits=logits,
                                                 label=labels,
                                                 soft_label=False,
                                                 return_softmax=False)
        sequence_mask = layers.reshape(sequence_mask_ph,
                                       [-1, self.seq_len_, 1])
        loss = layers.reduce_mean(layers.elementwise_mul(loss, sequence_mask))

        eval_metric_ops = fluid.layers.reduce_sum(correct_pred) \
                - fluid.layers.reduce_sum(correct_unk) \
                - fluid.layers.reduce_sum(correct_pad)

        self.loss_ = loss
        self.correct_ = eval_metric_ops
        self.input_name_list_ = [
            'features', 'labels', 'seq_len_ph', 'seq_mask_ph', 'init_hidden',
            'init_cell'
        ]
        self.target_var_names_ = [
            self.loss_, self.last_hidden_, self.last_cell_, self.correct_
        ]

        self.program_ = fluid.default_main_program()
        self.startup_program_ = fluid.default_startup_program()

    def _build_rnn_graph(self, inputs, init_hidden, init_cell,
                         sequence_length_ph):
        rnn_out, last_hidden, last_cell = basic_lstm(
            input=inputs,
            init_hidden=init_hidden,
            init_cell=init_cell,
            hidden_size=self.n_hidden_,
            num_layers=self.num_layers_,
            batch_first=True,
            dropout_prob=self.dropout_prob_,
            sequence_length=sequence_length_ph,
            param_attr=ParamAttr(
                initializer=fluid.initializer.UniformInitializer(
                    low=-self.init_scale_, high=self.init_scale_)),
            bias_attr=ParamAttr(initializer=fluid.initializer.Constant(0.0)),
            forget_bias=0.0)
        return rnn_out, last_hidden, last_cell
