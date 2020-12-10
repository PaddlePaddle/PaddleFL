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
mpc network
"""

import os
import math
import logging

import paddle
from paddle import fluid
import paddle_fl.mpc as pfl_mpc

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)


def FM(args, inputs, seed=0):
    """
    FM model
    """
    init_value_ = 0.1

    raw_feat_idx = inputs[0]
    feat_idx = raw_feat_idx
    raw_feat_value = inputs[1]
    label = inputs[2]

    feat_value = pfl_mpc.layers.reshape(
        raw_feat_value, [args.share_num, args.batch_size, args.num_field])

    # ------------------------- first order term --------------------------
    feat_idx_re = pfl_mpc.layers.reshape(feat_idx, [
        args.share_num, args.batch_size * args.num_field,
        args.sparse_feature_number + 1
    ])

    first_weights_re = pfl_mpc.input.embedding(
        input=feat_idx_re,
        is_sparse=False,
        is_distributed=False,
        dtype='int64',
        size=[args.sparse_feature_number + 1, 1],
        padding_idx=0, )

    first_weights = pfl_mpc.layers.reshape(
        first_weights_re,
        shape=[args.share_num, args.batch_size, args.num_field])

    y_first_order = pfl_mpc.layers.reduce_sum(
        (first_weights * feat_value), 2, keep_dim=True)

    b_linear = pfl_mpc.layers.create_mpc_parameter(
        shape=[1],
        dtype='int64',
        default_initializer=fluid.initializer.ConstantInitializer(value=0))

    # ------------------------- second order term --------------------------
    feat_embeddings_re = pfl_mpc.input.embedding(
        input=feat_idx_re,
        is_sparse=False,
        is_distributed=False,
        dtype='int64',
        size=[args.sparse_feature_number + 1, args.sparse_feature_dim],
        padding_idx=0)

    feat_embeddings = pfl_mpc.layers.reshape(
        feat_embeddings_re,
        shape=[
            args.share_num, args.batch_size, args.num_field,
            args.sparse_feature_dim
        ])

    feat_embeddings = pfl_mpc.layers.elementwise_mul(
        feat_embeddings, feat_value, axis=0)

    # sum_square part
    summed_features_emb = pfl_mpc.layers.reduce_sum(feat_embeddings, 2)

    summed_features_emb_square = pfl_mpc.layers.square(summed_features_emb)

    # square_sum part
    squared_features_emb = pfl_mpc.layers.square(feat_embeddings)
    squared_sum_features_emb = pfl_mpc.layers.reduce_sum(squared_features_emb,
                                                         2)

    y_FM_ = pfl_mpc.layers.reduce_sum(
        summed_features_emb_square - squared_sum_features_emb,
        dim=2,
        keep_dim=True)

    y_FM = pfl_mpc.layers.scale(y_FM_, 0.5)

    # ------------------------- Predict --------------------------
    cost = pfl_mpc.layers.sigmoid_cross_entropy_with_logits(
        y_first_order + y_FM + b_linear, label)
    avg_cost = pfl_mpc.layers.reduce_sum(cost, 1)

    return avg_cost, cost
