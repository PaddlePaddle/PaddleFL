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
This module provides utils for model processing.
"""
import os
import numpy as np
import six
import paddle
import paddle.fluid as fluid
from paddle_fl.mpc.data_utils.data_utils import get_datautils


mpc_du = get_datautils('aby3')

#BATCH_SIZE = 10
#TRAIN_EPOCH = 20
#PADDLE_UPDATE_EPOCH = 10
#MPC_UPDATE_EPOCH = TRAIN_EPOCH - PADDLE_UPDATE_EPOCH


def get_mpc_dataloader(feature_file, label_file, feature_shape, label_shape,
                   feature_name, label_name, role, batch_size):
    """
    Read feature and label training data from files.

    """
    x = fluid.default_main_program().global_block().var(feature_name)
    y = fluid.default_main_program().global_block().var(label_name)
    feature_reader = mpc_du.load_shares(feature_file, id=role, shape=feature_shape)
    label_reader = mpc_du.load_shares(label_file, id=role, shape=label_shape)
    batch_feature = mpc_du.batch(feature_reader, batch_size, drop_last=True)
    batch_label = mpc_du.batch(label_reader, batch_size, drop_last=True)
    # async data loader
    loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=batch_size)
    batch_sample = paddle.reader.compose(batch_feature, batch_label)
    place = fluid.CPUPlace()
    loader.set_batch_generator(batch_sample, places=place)
    return loader


def get_mpc_test_dataloader(feature_file, feature_shape, role, batch_size):
    """
    Read feature test data for prediction.

    """
    feature_reader = mpc_du.load_shares(feature_file, id=role, shape=feature_shape)
    batch_feature = mpc_du.batch(feature_reader, batch_size, drop_last=True)
    return batch_feature


def load_decrypt_data(filepath, shape):
    """
    Load the encrypted data and reconstruct.

    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(
            mpc_du.load_shares(
                filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1],
                                              part_readers[2])

    for instance in mpc_share_reader():
        p = mpc_du.reconstruct(np.array(instance))
        print(p)


def generate_encrypted_data(mpc_data_dir):
    """
    Generate encrypted samples
    """
    sample_reader = paddle.dataset.uci_housing.train()

    def encrypted_housing_features():
        """
        feature reader
        """
        for instance in sample_reader():
            yield mpc_du.make_shares(instance[0])

    def encrypted_housing_labels():
        """
        label reader
        """
        for instance in sample_reader():
            yield mpc_du.make_shares(instance[1])
    mpc_du.save_shares(encrypted_housing_features, mpc_data_dir + "house_feature")
    mpc_du.save_shares(encrypted_housing_labels, mpc_data_dir + "house_label")


if __name__ == '__main__':
    mpc_data_dir = "./mpc_data/"
    if not os.path.exists(mpc_data_dir):
        os.mkdir(mpc_data_dir)
    generate_encrypted_data(mpc_data_dir)
