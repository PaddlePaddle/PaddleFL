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
This module test embedding op.

"""
import unittest
from multiprocessing import Manager
import numpy as np

from op_test import OpTest
import paddle.fluid.core as core
import paddle.fluid as fluid
from paddle.fluid import Program, program_guard
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')

class TestLookupTableOp(OpTest):
    def to_one_hot(self, x, depth):
        out = np.zeros(shape=(np.product(x.shape), depth)).astype('float')
        for i in range(np.product(x.shape)):
            out[i, x[i]] = 1.0
        return out

    def setUp(self):
        OpTest.setUp(self)
        self.op_type = "mpc_lookup_table_v2"
        self.dtype = "int64"
        table = np.random.random((17, 31)).astype("float")
        ids = np.random.randint(0, 17, 4).astype("int64")
        ids_one_hot = self.to_one_hot(ids, table.shape[0])
        mpc_table = self.lazy_share(table)
        mpc_ids_one_hot = self.lazy_share(ids_one_hot)
        self.inputs = {'W': mpc_table, 'Ids': mpc_ids_one_hot}
        self.outputs = {'Out': self.lazy_share(table[ids])}

    def test_check_output(self):
        place = core.CPUPlace()
        self.check_output_with_place(place, atol=1e-3)

    
    def test_check_grad(self):
        place = core.CPUPlace()
        self.check_grad_with_place(place, ['W'], 'Out', no_grad_set=set('Ids'), max_relative_error=5)


if __name__ == "__main__":
    unittest.main()

