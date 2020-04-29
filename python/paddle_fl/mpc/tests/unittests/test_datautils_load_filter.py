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
This module test load_data and data_filter_by_id functions in data_utils module.

"""
import sys
sys.path.append('../../../')

import unittest

import paddle_encrypted as paddle_enc


class TestDataUtilsLoadFilter(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(TestDataUtilsLoadFilter, self).__init__(methodName)
        self.test_tmp_file = './load_data_test.tmp'

    def create_tmp_file(self):
        with open(self.test_tmp_file, 'w') as f:
            f.write('111\n')
            f.write('222\n')
            f.write('333')

    def delete_tmp_file(self):
        import os
        os.remove(self.test_tmp_file)

    def setUp(self):
        self.create_tmp_file()

    def tearDown(self):
        self.delete_tmp_file()

    def test_load_data(self):
        expected_values = ['111', '222', '333']
        du = fluid_enc.data_utils.DataUtils()
        for data, value in zip(
                du.load_data(self.test_tmp_file), expected_values):
            self.assertEqual(data, value)

    def test_filter(self):
        to_filter = [
            "0, 0.1, 0.1, 0.1, 1", "1, 0.2, 0.2, 0.2, 0", "2, 0.3, 0.3, 0.3, 1"
        ]
        id_list = [0, 2]
        expected_results = ["0, 0.1, 0.1, 0.1, 1", "2, 0.3, 0.3, 0.3, 1"]
        du = fluid_enc.data_utils.DataUtils()
        filter_results = du.data_filter_by_id(
            input_list=to_filter, id_list=id_list)
        for result, expect in zip(filter_results, expected_results):
            self.assertEqual(result, expect)


if __name__ == '__main__':
    unittest.main()
