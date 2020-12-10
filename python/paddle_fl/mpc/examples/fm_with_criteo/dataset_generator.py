#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
dataset generate
"""

import os
import numpy as np
import paddle.fluid.incubate.data_generator as dg

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
#cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [cont_max_[i] - cont_min_[i] for i in range(len(cont_min_))]
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)


def generate_sample(hash_dim_, paddle_train_data_dir):
    """
    generate_sample
    """
    files = [
        str(paddle_train_data_dir) + "/%s" % x
        for x in os.listdir(paddle_train_data_dir)
    ]

    #print("file_list : {}".format(files))

    def reader():
        """
        reader
        """
        for file in files:
            with open(file, 'r') as f:
                for line in f:
                    features = line.rstrip('\n').split('\t')

                    feat_idx = []
                    feat_value = []
                    for idx in continuous_range_:
                        #feat_idx.append(idx)
                        feat_idx.append(
                            hash('dense_feat_id' + str(idx)) % hash_dim_)
                        if features[idx] == '':
                            feat_value.append(0.0)
                        else:
                            feat_value.append(
                                (float(features[idx]) - cont_min_[idx - 1]) /
                                cont_diff_[idx - 1])
                    for idx in categorical_range_:
                        if features[idx] == '':
                            feat_idx.append(
                                hash('sparse_feat_id' + str(idx)) % hash_dim_)
                            feat_value.append(0.0)
                        else:
                            feat_idx.append(
                                hash(str(idx) + features[idx]) % hash_dim_)
                            feat_value.append(1.0)
                    label = [int(features[0])]
                    yield feat_idx[:], feat_value[:], label[:]

    return reader


def train(hash_dim_, paddle_train_data_dir):
    """
    generate train sample
    """
    return generate_sample(hash_dim_, paddle_train_data_dir)
