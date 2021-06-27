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
Process data for Criteo.
"""
import os
import time
import logging
import numpy as np
import six
import paddle
import dataset_generator
import args
from paddle_fl.mpc.data_utils.data_utils import get_datautils


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)

mpc_du = get_datautils('aby3')


def generate_encrypted_data(args, mpc_data_dir, reader, label_filepath=None):
    """
    generate encrypted samples
    """

    def encrypt_feature_idx():
        for instance in reader():
            global count
            feature_idx_ = np.array(instance[0])
            feature_idx = np.eye(args.sparse_feature_number + 1)[feature_idx_.reshape(-1)]
            yield mpc_du.make_shares(feature_idx)

    def encrypt_feature_value():
        for instance in reader():
            #print(np.array(instance[1]).shape)
            yield mpc_du.make_shares(np.array(instance[1]))

    def encrypt_label():
        for instance in reader():
            #print(np.array(instance[2]))
            if label_filepath != None:
                with open(label_filepath, 'a+') as f:
                    f.write(str(instance[2][0]) + '\n')
            yield mpc_du.make_shares(np.array(instance[2]))

    mpc_du.save_shares(encrypt_label, mpc_data_dir + "criteo_label")
    mpc_du.save_shares(encrypt_feature_value, mpc_data_dir + "criteo_feature_value")
    mpc_du.save_shares(encrypt_feature_idx, mpc_data_dir + "criteo_feature_idx")


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


def decrypt_data_to_file(filepath, shape, decrypted_filepath):
    """
    load the encrypted data (arithmetic share) and reconstruct to a file
    """
    #while(not (os.path.exists(filepath + '.part0') 
    #           and os.path.exists(filepath + '.part1') 
    #           and os.path.exists(filepath + '.part2'))):
    #    time.sleep(0.1)

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
                f.write(str(i) + '\n')


if __name__ == '__main__':

    args = args.parse_args()

    mpc_data_dir = './mpc_data/'
    mpc_infer_data_dir = './mpc_infer_data/'
    if not os.path.exists(mpc_data_dir):
        os.mkdir(mpc_data_dir)
    if not os.path.exists(mpc_infer_data_dir):
        os.mkdir(mpc_infer_data_dir)

    mpc_train_data_dir = mpc_data_dir + 'train/'
    if not os.path.exists(mpc_train_data_dir):
        os.mkdir(mpc_train_data_dir)
    train_reader = dataset_generator.train(args.sparse_feature_number + 1, args.paddle_sample_data_dir)
    generate_encrypted_data(args, mpc_train_data_dir, train_reader)

    mpc_test_data_dir = mpc_data_dir + 'test/'
    if not os.path.exists(mpc_test_data_dir):
        os.mkdir(mpc_test_data_dir)
    label_test_filepath = mpc_infer_data_dir + "label_criteo"
    if os.path.exists(label_test_filepath):
        os.remove(label_test_filepath)

    test_reader = dataset_generator.train(args.sparse_feature_number + 1, args.paddle_sample_data_dir)
    generate_encrypted_data(args, mpc_test_data_dir, test_reader, label_test_filepath)
