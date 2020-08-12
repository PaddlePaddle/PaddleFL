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
Process data for MNIST.
"""
import numpy as np
import paddle
import six
import os
from paddle_fl.mpc.data_utils import aby3

sample_reader = paddle.dataset.mnist.train()
test_reader = paddle.dataset.mnist.test()

def generate_encrypted_data():
    """
    generate encrypted samples
    """
    def encrypted_mnist_features():
        """
        feature reader
        """
        for instance in sample_reader():
            yield aby3.make_shares(instance[0])

    def encrypted_mnist_labels():
        """
        label reader
        """
        for instance in sample_reader():
            yield aby3.make_shares(np.array(1) if instance[1] == 0 else np.array(0))
    
    aby3.save_aby3_shares(encrypted_mnist_features, "/tmp/mnist2_feature")
    aby3.save_aby3_shares(encrypted_mnist_labels, "/tmp/mnist2_label")

def generate_encrypted_test_data():
    """
    generate encrypted samples
    """
    def encrypted_mnist_features():
        """
        feature reader
        """
        for instance in test_reader():
            yield aby3.make_shares(instance[0])

    def encrypted_mnist_labels():
        """
        label reader
        """
        for instance in test_reader():
            yield aby3.make_shares(np.array(1) if instance[1] == 0 else np.array(0))
    
    aby3.save_aby3_shares(encrypted_mnist_features, "/tmp/mnist2_test_feature")
    aby3.save_aby3_shares(encrypted_mnist_labels, "/tmp/mnist2_test_label")

def load_decrypt_data(filepath, shape):
    """
    load the encrypted data and reconstruct
    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(aby3.load_aby3_shares(filepath, id=id, shape=shape))
    aby3_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in aby3_share_reader():
        p = aby3.reconstruct(np.array(instance))
        print(p)

def decrypt_data_to_file(filepath, shape, decrypted_file):
    """
    load the encrypted data and reconstruct to a file
    """
    if os.path.exists(decrypted_file):
        os.remove(decrypted_file)
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(aby3.load_aby3_shares(filepath, id=id, shape=shape))
    aby3_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in aby3_share_reader():
        p = aby3.reconstruct(np.array(instance))
        with open(decrypted_file, 'a+') as f:
            for i in p:
                f.write(str(i) + '\n')
