# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
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
args
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import sys
import numpy as np


def parse_args():
    """
    args
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch_num', type=int, default=10, help='epoch_num')
    parser.add_argument('--batch_size', type=int, default=5, help='batch_size')
    parser.add_argument('--share_num', type=int, default=2, help='share_num')
    parser.add_argument('--base_lr', type=float, default=0.01, help='base_lr')
    parser.add_argument(
        '--dense_feature_dim', type=int, default=13, help='dense_feature_dim')
    parser.add_argument(
        '--sparse_feature_number',
        type=int,
        default=100,
        help='sparse_feature_number')
    parser.add_argument(
        '--sparse_feature_dim',
        type=int,
        default=26,
        help='sparse_feature_dim')
    parser.add_argument(
        '--embedding_size', type=int, default=9, help='embedding_size')
    parser.add_argument('--num_field', type=int, default=39, help='num_field')
    parser.add_argument('--reg', type=float, default=0.001, help='reg')

    parser.add_argument(
        '--paddle_sample_data_dir',
        type=str,
        default='./data/sample_data/train',
        help='paddle_sample_data_dir')
    parser.add_argument(
        '--paddle_train_data_dir',
        type=str,
        default='./data/train',
        help='paddle_train_data_dir')
    parser.add_argument(
        '--paddle_test_data_dir',
        type=str,
        default='./data/test',
        help='paddle_test_data_dir')

    parser.add_argument('--role', type=int, default=0, help='role')
    parser.add_argument(
        '--server', type=str, default='localhost', help='server ip')
    parser.add_argument('--port', type=int, default=12345, help='server port')

    parser.add_argument(
        '--mpc_data_dir', type=str, default='./mpc_data/', help='mpc_data_dir')
    parser.add_argument(
        '--model_dir', type=str, default='./model_dir/', help='model_dir')
    parser.add_argument(
        '--watch_vec_size', type=int, default=64, help='watch_vec_size')
    parser.add_argument(
        '--search_vec_size', type=int, default=64, help='search_vec_size')
    parser.add_argument(
        '--other_feat_size', type=int, default=32, help='other_feat_size')
    parser.add_argument(
        '--output_size', type=int, default=3952, help='output_size')
    parser.add_argument('--topk', type=int, default=10, help='topk')

    args = parser.parse_args()
    return args
