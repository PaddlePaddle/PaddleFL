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
This module test embedding op.

"""
import unittest
from multiprocessing import Manager

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
import paddle_fl.mpc.data_utils.aby3 as aby3

import test_op_base


class TestInput(test_op_base.TestOpBase):

    def gen_one_hot(self, input, depth):
        """
        example for generate mpc one hot tensor
        """
        data_var = fluid.data(name='input_data', shape=input.shape, dtype='int64')
        ret1 = fluid.input.one_hot(input=data_var, depth=3)
        exe =fluid.Executor(place=fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        data = exe.run(program=fluid.default_main_program(),feed={'input_data': input}, fetch_list=[ret1])
        return data[0]

    def embedding_op(self, **kwargs):
        role = kwargs['role']
        #data = kwargs['data']
        data_normal = kwargs['data_normal']
        data_share = kwargs['data_share'][role]

        w_data = kwargs['w_data']
        w_data_share = kwargs['w_data_share'][role]
        return_results = kwargs['return_results']
        expected_result = kwargs['expect_results']

        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))

        w_param_attrs = fluid.ParamAttr(name='emb_weight',
                                        learning_rate=0.5,
                                        initializer=pfl_mpc.initializer.NumpyArrayInitializer(w_data_share),
                                        trainable=True)
        w_param_attrs1 = fluid.ParamAttr(name='emb_weight1',
                                        learning_rate=0.5,
                                        initializer=fluid.initializer.NumpyArrayInitializer(w_data),
                                        trainable=True)
        input_shape = np.delete(data_share.shape, 0, 0)
        data1 = pfl_mpc.data(name='input', shape=input_shape, dtype='int64')
        data2 = fluid.data(name='input1', shape=data_normal.shape, dtype='int64')

        math_embedding = fluid.input.embedding(input=data2, size=w_data.shape, param_attr=w_param_attrs1, dtype='float32')

        op_embedding = pfl_mpc.input.embedding(input=data1, size=(input_shape[1],input_shape[0]), param_attr=w_param_attrs, dtype='int64')

        exe = fluid.Executor(place=fluid.CPUPlace())
        exe.run(fluid.default_startup_program())

        results = exe.run(feed={'input': data_share, 'input1': data_normal}, fetch_list=[op_embedding, math_embedding])

        return_results.append(results[0])
        expected_result.append(results[1])

    def test_embedding_op(self):
        data = np.array([[1, 0, 0], [0, 1, 0]])
        data_normal = np.array([0, 1]).astype('int64')
        w_data = np.array([[1, 2], [2, 3], [3, 4]])

        # data = self.gen_one_hot(data_normal, w_data.shape[0]).astype('int64')

        data_share = aby3.make_shares(np.array(data))
        data_all3shares = np.array([aby3.get_aby3_shares(data_share, i) for i in range(3)])
        w_data_share = aby3.make_shares(w_data)
        w_data_all3shares = np.array([aby3.get_aby3_shares(w_data_share, i) for i in range(3)])

        return_results = Manager().list()
        expect_results = Manager().list()
        ret = self.multi_party_run(target=self.embedding_op,
                                   data=data,
                                   data_normal=data_normal,
                                   w_data=w_data,
                                   data_share=data_all3shares,
                                   w_data_share=w_data_all3shares,
                                   return_results=return_results,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        # print("reveal: ", revealed)
        self.assertTrue(np.allclose(revealed, expect_results[0], atol=1e-4))

    def test_mpc_one_hot(self):
      data = np.array([0, 1]).astype('int64')
      ret = self.gen_one_hot(data, 3)
      mpc_one_hot = aby3.make_shares(ret)

if __name__ == '__main__':
    unittest.main()
