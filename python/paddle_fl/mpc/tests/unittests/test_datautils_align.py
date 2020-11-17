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
This module test align in aby3 module.

"""

import unittest
import multiprocessing as mp

import paddle_fl.mpc.data_utils.alignment as alignment



class TestDataUtilsAlign(unittest.TestCase):

    @staticmethod
    def run_align(input_set, party_id, endpoints, is_receiver, ret_list):
        """
        Call align function in data_utils.
        :param input_set:
        :param party_id:
        :param endpoints:
        :param is_receiver:
        :return:
        """
        result = alignment.align(input_set=input_set,
                                 party_id=party_id,
                                 endpoints=endpoints,
                                 is_receiver=is_receiver)
        ret_list.append(result)

    def test_align(self):
        """
        Test normal case for align function.
        :return:
        """
        endpoints = '0:127.0.0.1:11111,1:127.0.0.1:22222,2:127.0.0.1:33333'
        set_0 = {'0', '10', '20', '30'}
        set_1 = {'0', '10', '11', '111'}
        set_2 = {'0', '30', '33', '333'}

        mp.set_start_method('spawn')

        manager = mp.Manager()
        ret_list = manager.list()

        party_0 = mp.Process(target=self.run_align, args=(set_0, 0, endpoints, True, ret_list))
        party_1 = mp.Process(target=self.run_align, args=(set_1, 1, endpoints, False, ret_list))
        party_2 = mp.Process(target=self.run_align, args=(set_2, 2, endpoints, False, ret_list))

        party_1.start()
        party_2.start()
        party_0.start()

        party_0.join()
        party_1.join()
        party_2.join()

        self.assertEqual(3, len(ret_list))
        self.assertEqual(ret_list[0], ret_list[1])
        self.assertEqual(ret_list[0], ret_list[2])
        self.assertEqual({'0'}, ret_list[0])


if __name__ == '__main__':
    unittest.main()
