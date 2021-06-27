#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
This module test aby3 in data_utils.

"""
import os
import unittest
import numpy as np
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')

class TestDataUtilsAby3(unittest.TestCase):

    def test_encrypt_decrypt(self):
        number = 123.4

        number_shares = aby3.encrypt(number)
        self.assertEqual(len(number_shares), 3)

        revealed_number = aby3.decrypt(number_shares)
        self.assertAlmostEqual(number, revealed_number, delta=1e-4)

    def test_make_shares(self):
        num_arr = np.arange(0, 4).reshape((2, 2))

        shares = aby3.make_shares(num_arr)
        self.assertEqual((3, 2, 2), shares.shape)

    def test_reconstruct(self):
        num_arr = np.arange(0, 4).reshape((2, 2)).astype(np.float32)
        shares = aby3.make_shares(num_arr)
        all_3shares = np.array([aby3.get_shares(shares, i) for i in range(3)])
        recon = aby3.reconstruct(all_3shares)
        self.assertTrue(np.allclose(num_arr, recon))

    def test_make_shares_3dim(self):
        num_arr = np.arange(0, 8).reshape((2, 2, 2))
        shares = aby3.make_shares(num_arr)
        self.assertEqual((3, 2, 2, 2), shares.shape)

    def test_get_shares(self):
        raw_shares = np.arange(1, 13).reshape((3, 2, 2))
        share_list = []
        for idx in range(3):
            share = aby3.get_shares(raw_shares, idx)
            self.assertEqual(share.shape, (2, 2, 2))
            share_list.append(share)

        expect_shares = [np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
                         np.array([[[5, 6], [7, 8]], [[9, 10], [11, 12]]]),
                         np.array([[[9, 10], [11, 12]], [[1, 2], [3, 4]]])]
        for value, expect in zip(share_list, expect_shares):
            self.assertTrue(np.allclose(value, expect))

    def dummy_share_reader(self):
        """
        Dummy share_reader for share_reader in aby3.save_shares()
        :return:
        """
        test_data = np.arange(1, 10).reshape((3, 3)).astype(np.float32)
        yield aby3.make_shares(test_data)

    def remove_temp_file(self, filename):
        """
        Remove temp file.
        :param filename:
        :return:
        """
        os.remove(filename)

    def test_save_shares(self):
        part_name = './tmp_test_save_shares'
        aby3.save_shares(share_reader=self.dummy_share_reader,
                         part_name=part_name)
        files = os.listdir('./')
        true_filename = 'tmp_test_save_shares'
        for idx in range(3):
            tmp_file = true_filename + '.part' + str(idx)
            self.assertTrue(tmp_file in files)
            self.remove_temp_file(tmp_file)

    def test_load_shares(self):
        share = np.arange(1, 19).reshape((2, 3, 3)).astype(np.int64)
        tmp_part_name = './tmp_test_load_shares.part0'
        with open(tmp_part_name, 'wb') as f:
            f.write(share.tostring())
        part_name = './tmp_test_load_shares'
        default_loader = aby3.load_shares(part_name=part_name,
                                               id=0,
                                               shape=(3, 3))
        default_loading_data = next(default_loader())
        self.assertTrue(np.allclose(default_loading_data, share))

        loader = aby3.load_shares(part_name=part_name,
                                       id=0,
                                       shape=(2, 3, 3),
                                       append_share_dim=False)
        loading_data = next(loader())
        self.assertTrue(np.allclose(loading_data, share))

        self.remove_temp_file(tmp_part_name)

    def dummy_reader(self):
        """
        Dummy reader for the reader in aby3.batch()
        :return:
        """
        data = [np.arange(1, 9).reshape((2, 2, 2)).astype(np.int64)] * 4
        for item in data:
            yield item

    def test_batch(self):
        default_batch_reader = aby3.batch(reader=self.dummy_reader,
                                          batch_size=3)
        default_batch_sample_shapes = [(2, 3, 2, 2), (2, 1, 2, 2)]
        for item, shape in zip(default_batch_reader(), default_batch_sample_shapes):
            self.assertEqual(item.shape, shape)

        batch_reader = aby3.batch(reader=self.dummy_reader,
                                  batch_size=3,
                                  drop_last=True)
        for item in batch_reader():
            self.assertEqual(item.shape, (2, 3, 2, 2))


if __name__ == '__main__':
    unittest.main()
