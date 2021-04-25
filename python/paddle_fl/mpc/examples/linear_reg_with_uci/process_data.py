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
Process data for UCI Housing.
"""
import numpy as np
import paddle
import six
import os
from paddle_fl.mpc.data_utils import aby3
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc

sample_reader = paddle.dataset.uci_housing.train()


def generate_encrypted_data():
    """
    generate encrypted samples
    """

    def encrypted_housing_features():
        """
        feature reader
        """
        for instance in sample_reader():
            yield aby3.make_shares(instance[0])

    def encrypted_housing_labels():
        """
        label reader
        """
        for instance in sample_reader():
            yield aby3.make_shares(instance[1])

    aby3.save_aby3_shares(encrypted_housing_features, "/tmp/house_feature")
    aby3.save_aby3_shares(encrypted_housing_labels, "/tmp/house_label")

def generate_encrypted_data_online(role, server, port):
    """
    generate encrypted samples
    """

    def save_aby3_share(share_reader, role, part_name):
        with open(part_name + ".part" + str(role), 'wb') as share_file:
            for share in share_reader():
                share_file.write(share.tostring())

    feature_list = []
    label_list = []

    def encrypted_housing_features():
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            pfl_mpc.init("aby3", int(role), "localhost", server, int(port))
            input = fluid.data(name='input', shape=[13], dtype='float32')
            data_location_party = 0 # party_0 has features
            out = pfl_mpc.layers.share(input, party_id=data_location_party)

            place=fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for instance in sample_reader():
                if role == data_location_party:
                    feed_data = np.array(instance[0], dtype='float32')
                else:
                    feed_data = np.zeros(shape=(13,), dtype ='float32')#dummy_data
                out_share = exe.run(feed={'input': feed_data}, fetch_list=[out])
                yield np.array(out_share)

    def encrypted_housing_labels():
        main_program = fluid.Program()
        startup_program = fluid.Program()
        with fluid.program_guard(main_program, startup_program):
            pfl_mpc.init("aby3", int(role), "localhost", server, int(port))
            input = fluid.data(name='input', shape=[1], dtype='float32')
            data_location_party = 1 # party_1 has labels
            out = pfl_mpc.layers.share(input, party_id=data_location_party)

            place=fluid.CPUPlace()
            exe = fluid.Executor(place)
            exe.run(fluid.default_startup_program())
            for instance in sample_reader():
                if role == data_location_party:
                    feed_data = np.array(instance[1], dtype='float32')
                else:
                    feed_data = np.zeros(shape=(1,), dtype ='float32')#dummy_data
                out_share = exe.run(feed={'input': feed_data}, fetch_list=[out])
                yield np.array(out_share)

    # ** return generator **
    def feature_reader():
        for feature in feature_list:
            yield np.array(feature).reshape((2, 13))

    def label_reader():
        for label in label_list:
            yield np.array(label).reshape((2, 1))

    feature_list = list(encrypted_housing_features())
    label_list = list(encrypted_housing_labels())

    return feature_reader, label_reader
    # ** return generator **

    # ***write encrypted data into file***
    #save_aby3_share(encrypted_housing_features, role,  "/tmp/house_feature")
    #save_aby3_share(encrypted_housing_labels, role, "/tmp/house_label")
    # ***write encrypted data into file***

def load_decrypt_data(filepath, shape, decrypted_file):
    """
    load the encrypted data and reconstruct
    """
    if os.path.exists(decrypted_file):
        os.remove(decrypted_file)
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(
            aby3.load_aby3_shares(
                filepath, id=id, shape=shape))
    aby3_share_reader = paddle.reader.compose(part_readers[0], part_readers[1],
                                              part_readers[2])

    for instance in aby3_share_reader():
        p = aby3.reconstruct(np.array(instance))
        with open(decrypted_file, 'a+') as f:
            for i in p:
                f.write(str(i) + '\n')


def decrypt_online(shares, shape):
    main_program = fluid.Program()
    startup_program = fluid.Program()
    with fluid.program_guard(main_program, startup_program):
        input = pfl_mpc.data(name='input', shape=shape[1:], dtype='int64')
        out = pfl_mpc.layers.reveal(input)

        place=fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())
        out_ = exe.run(feed={'input': np.array(shares).reshape(shape)}, fetch_list=[out])
        return out_

