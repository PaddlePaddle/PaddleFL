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
Process data for MNIST: 10 classes.
"""
import os
import time
import logging
import numpy as np
import six
import paddle
from paddle_fl.mpc.data_utils.data_utils import get_datautils


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


mpc_du = get_datautils('aby3')
sample_reader = paddle.dataset.mnist.train()
test_reader = paddle.dataset.mnist.test()


def generate_encrypted_train_data(mpc_data_dir, class_num):
    """
    generate encrypted samples
    """

    def encrypted_mnist_features():
        """
        feature reader
        """
        for instance in sample_reader():
            yield mpc_du.make_shares(instance[0])

    def encrypted_mnist_labels():
        """
        label reader
        """
        for instance in sample_reader():
            if class_num == 2:
                label = np.array(1) if instance[1] == 0 else np.array(0)
            elif class_num == 10:
                label = np.eye(N=1, M=10, k=instance[1], dtype=float).reshape(10)
            else:
                raise ValueError("class_num should be 2 or 10, but received {}.".format(class_num))
            yield mpc_du.make_shares(label)

    mpc_du.save_shares(encrypted_mnist_features, mpc_data_dir + "mnist{}_feature".format(class_num))
    mpc_du.save_shares(encrypted_mnist_labels, mpc_data_dir + "mnist{}_label".format(class_num))


def generate_encrypted_test_data(mpc_data_dir, class_num, label_mnist_filepath):
    """
    generate encrypted samples
    """

    def encrypted_mnist_features():
        """
        feature reader
        """
        for instance in test_reader():
            yield mpc_du.make_shares(instance[0])

    def encrypted_mnist_labels():
        """
        label reader
        """
        for instance in test_reader():
            if class_num == 2:
                label = np.array(1) if instance[1] == 0 else np.array(0)
                with open(label_mnist_filepath, 'a+') as f:
                    f.write(str(1 if instance[1] == 0 else 0) + '\n')
            elif class_num == 10:
                label = np.eye(N=1, M=10, k=instance[1], dtype=float).reshape(10)
                with open(label_mnist_filepath, 'a+') as f:
                    f.write(str(instance[1]) + '\n')
            else:
                raise ValueError("class_num should be 2 or 10, but received {}.".format(class_num))
            yield mpc_du.make_shares(label)

    mpc_du.save_shares(encrypted_mnist_features, mpc_data_dir + "mnist{}_test_feature".format(class_num))
    mpc_du.save_shares(encrypted_mnist_labels, mpc_data_dir + "mnist{}_test_label".format(class_num))


def load_decrypt_data(filepath, shape):
    """
    load the encrypted data and reconstruct
    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = mpc_du.reconstruct(np.array(instance))
        logger.info(p)


def load_decrypt_bs_data(filepath, shape):
    """
    load the encrypted data and reconstruct
    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = np.bitwise_xor(np.array(instance[0]), np.array(instance[1]))
        p = np.bitwise_xor(p, np.array(instance[2]))
        logger.info(p)


def decrypt_data_to_file(filepath, shape, decrypted_filepath):
    """
    load the encrypted data (arithmetic share) and reconstruct to a file
    """
    if os.path.exists(decrypted_filepath):
        os.remove(decrypted_filepath)
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = mpc_du.reconstruct(np.array(instance))
        with open(decrypted_filepath, 'a+') as f:
            for i in p:
                f.write(str(np.argmax(i)) + '\n')


def decrypt_bs_data_to_file(filepath, shape, decrypted_filepath):
    """
    load the encrypted data (boolean share) and reconstruct to a file
    """
    if os.path.exists(decrypted_filepath):
        os.remove(decrypted_filepath)
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(mpc_du.load_shares(filepath, id=id, shape=shape))
    mpc_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in mpc_share_reader():
        p = np.bitwise_xor(np.array(instance[0]), np.array(instance[1]))
        p = np.bitwise_xor(p, np.array(instance[2]))
        with open(decrypted_filepath, 'a+') as f:
            for i in p:
                f.write(str(i) + '\n')


if __name__ == '__main__':
    mpc_data_dir = './mpc_data/'
    label_mnist_filepath = mpc_data_dir + "label_mnist"
    if not os.path.exists(mpc_data_dir):
        os.mkdir(mpc_data_dir)
    if os.path.exists(label_mnist_filepath):
        os.remove(label_mnist_filepath)

    class_num = 2
    generate_encrypted_train_data(mpc_data_dir, class_num)
    generate_encrypted_test_data(mpc_data_dir, class_num, label_mnist_filepath)
