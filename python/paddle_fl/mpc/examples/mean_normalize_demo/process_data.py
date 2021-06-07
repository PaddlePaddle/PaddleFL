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
Process data for mean normalize demo.
"""
import numpy as np
import six
import os
import paddle
from paddle_fl.mpc.data_utils.data_utils import get_datautils


mpc_du = get_datautils('aby3')
data_path = './data/'


def encrypted_data(data):
    """
    feature stat reader
    """

    def func():
        """
        make shares
        """
        yield mpc_du.make_shares(data)

    return func


def generate_encrypted_data(party_id, f_mat):
    """
    generate encrypted data from feature matrix (np.array)
    """

    f_max = np.amax(f_mat, axis=0)
    f_min = np.amin(f_mat, axis=0)
    f_mean = np.mean(f_mat, axis=0)

    suffix = '.' + str(party_id)

    mpc_du.save_shares(
        encrypted_data(f_max), data_path + "feature_max" + suffix)
    mpc_du.save_shares(
        encrypted_data(f_min), data_path + "feature_min" + suffix)
    mpc_du.save_shares(
        encrypted_data(f_mean), data_path + "feature_mean" + suffix)


def decrypt_data(filepath, shape):
    """
    load the encrypted data and reconstruct
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
        return p
