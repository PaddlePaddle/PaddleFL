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
This module test dyanmic_gru op.

"""
import unittest
from multiprocessing import Manager
import test_op_base
import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')


class TestInput(test_op_base.TestOpBase):

    def dyanmic_gru_op(self, **kwargs):
        role = kwargs['role']
        data = kwargs['data']
        data_share = kwargs['data_share'][role]
        weight = kwargs['weight']
        weight_share = kwargs['weight_share'][role]
        return_results = kwargs['return_results']
        return_results_cheb = kwargs['return_results_cheb']
        expected_result = kwargs['expect_results']
        pfl_mpc.init("aby3", role, "localhost", self.server, int(self.port))

        hidden_dim = 1

        data_paddle = fluid.data(name='input_paddle', shape=[3, 3], dtype='float32', lod_level=1)
        ldata_paddle = fluid.create_lod_tensor(data, [[3]], fluid.CPUPlace())
        w_param_attrs = fluid.ParamAttr(name='gru_weight',
                                        learning_rate=0.5,
                                        initializer=fluid.initializer.NumpyArrayInitializer(weight),
                                        trainable=True)
        hidden_paddle = fluid.layers.dynamic_gru(input=data_paddle, size=hidden_dim, param_attr=w_param_attrs,
                                                 gate_activation='sigmoid', candidate_activation='relu')

        data_mpc = fluid.data(name='input_mpc', shape=[3, 2, 3], dtype='int64', lod_level=1)
        # trans batch information to shape[0]
        data_share_trans = np.transpose(data_share, [1, 0, 2])
        ldata_mpc = fluid.create_lod_tensor(data_share_trans, [[3]], fluid.CPUPlace())
        w_param_attrs1 = fluid.ParamAttr(name='mpc_gru_weight',
                                        learning_rate=0.5,
                                        initializer=pfl_mpc.initializer.NumpyArrayInitializer(weight_share),
                                        trainable=True)
        w_param_attrs2 = fluid.ParamAttr(name='mpc_gru_weight_cheb',
                                        learning_rate=0.5,
                                        initializer=pfl_mpc.initializer.NumpyArrayInitializer(weight_share),
                                        trainable=True)
        hidden_mpc = pfl_mpc.layers.dynamic_gru(input=data_mpc, size=hidden_dim,
                                                param_attr=w_param_attrs1)
        hidden_mpc_cheb = pfl_mpc.layers.dynamic_gru(input=data_mpc, size=hidden_dim,
                                                param_attr=w_param_attrs2, gate_activation='sigmoid_chebyshev')

        exe = fluid.Executor(place=fluid.CPUPlace())
        exe.run(fluid.default_startup_program())
        results = exe.run(feed={'input_paddle': ldata_paddle, 'input_mpc': ldata_mpc},
                        fetch_list=[hidden_paddle, hidden_mpc, hidden_mpc_cheb], return_numpy=False)
        return_results.append(np.array(results[1]))
        return_results_cheb.append(np.array(results[2]))
        expected_result.append(np.array(results[0]))

    def test_dyanmic_gru_op(self):
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [-1.0, -2.0, -3.0]]).astype('float32')
        data_share = aby3.make_shares(data)
        data_all3shares = np.array([aby3.get_shares(data_share, i) for i in range(3)])

        weight = np.array([[0.0, 0.0, 0.0]]).astype('float32')
        weight_share = aby3.make_shares(weight)
        weight_all3shares = np.array([aby3.get_shares(weight_share, i) for i in range(3)])

        return_results = Manager().list()
        return_results_cheb = Manager().list()
        expect_results = Manager().list()
        ret = self.multi_party_run(target=self.dyanmic_gru_op,
                                   data=data,
                                   data_share = data_all3shares,
                                   weight=weight,
                                   weight_share=weight_all3shares,
                                   return_results=return_results,
                                   return_results_cheb=return_results_cheb,
                                   expect_results=expect_results)
        self.assertEqual(ret[0], True)
        revealed = aby3.reconstruct(np.array(return_results))
        revealed_cheb = aby3.reconstruct(np.array(return_results_cheb))
        print("expected:", expect_results[0])
        print("reveal: ", revealed)
        print("reveal_cheb: ", revealed_cheb)
        self.assertTrue(np.allclose(revealed, expect_results[0], atol=1e-1*5))
        self.assertTrue(np.allclose(revealed_cheb, expect_results[0], atol=1e-1*5))


if __name__ == '__main__':
    unittest.main()
