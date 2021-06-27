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
This module test model encryption/decryption in aby3 module.
"""

import os
import shutil
import unittest

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle_fl.mpc.data_utils.data_utils import get_datautils


aby3 = get_datautils('aby3')

class TestDataUtilsEnDecryptModel(unittest.TestCase):
    def __init__(self, methodName='runTest'):
        super(TestDataUtilsEnDecryptModel, self).__init__(methodName)
        self.raw_model_dir = './test_model'
        self.enc_model_dir = './mpc_model'
        self.dec_model_dir = './decrypted_model'

    def create_test_model(self):
        """
        Create a model for test.
        :return:
        """
        x = fluid.data(name='x', shape=[None, 13], dtype='float32')
        y = fluid.data(name='y', shape=[None, 1], dtype='float32')
        param_attr = ParamAttr(name="fc_0.w_0")
        bias_attr = ParamAttr(name="fc_0.b_0")
        y_predict = fluid.layers.fc(input=x, size=1, param_attr=param_attr, bias_attr=bias_attr)

        main_prog = fluid.default_main_program()
        startup_program = fluid.default_startup_program()
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(startup_program)

        if not os.path.exists(self.raw_model_dir):
            os.makedirs(self.raw_model_dir)
        fluid.io.save_inference_model(self.raw_model_dir, ['x'], [y_predict], exe)

        vars = ['fc_0.w_0', 'fc_0.b_0']
        vars_tensor = [
            [[-1.0788183212280273], [2.1307122707366943], [-2.646815538406372], [1.6547845602035522],
             [-2.13144588470459], [3.6621456146240234], [-1.553664207458496], [0.18727444112300873],
             [-2.3649044036865234], [-3.407580852508545], [-4.058014392852783], [1.4958711862564087],
             [-3.9899468421936035]],
            [22.361257553100586]]

        global_block = main_prog.global_block()
        g_scope = fluid.global_scope()
        for var, tensor in zip(vars, vars_tensor):
            param = g_scope.find_var(var)
            param.get_tensor().set(tensor, place)
            variable = global_block.var(var)
            fluid.io.save_vars(executor=exe,
                               dirname=self.raw_model_dir,
                               vars=[variable],
                               filename=var)

    def infer_with_decrypted_model(self, model_path):
        """
        Make inference using decrypted model to test the validity of model decryption.
        :param model_path:
        :return:
        """
        place = fluid.CPUPlace()
        exe = fluid.Executor(place=place)
        [inference_program, feed_target_names, fetch_targets] = \
            fluid.io.load_inference_model(model_path, exe)

        feat = [0.42616306, -0.11363636, 0.25525005, -0.06916996, 0.28457807,
                -0.14440207, 0.17327599, -0.19893267, 0.62828665, 0.49191383,
                0.18558153, -0.0686218, 0.40637243]
        infer_feat = np.array(feat).reshape((1, 13)).astype("float32")

        assert feed_target_names[0] == 'x'
        results = exe.run(inference_program,
                          feed={feed_target_names[0]: infer_feat},
                          fetch_list=fetch_targets)
        return results[0][0]

    def setUp(self):
        """
        Create a test model when setup.
        :return:
        """
        self.create_test_model()
        aby3.encrypt_model(plain_model=self.raw_model_dir,
                           mpc_model_dir=self.enc_model_dir)

    def tearDown(self):
        """
        Remove test files or directories.
        """
        shutil.rmtree(self.raw_model_dir)
        shutil.rmtree(self.enc_model_dir)

    def test_model_encrypt(self):
        """
        Test normal case for model encryption.
        """
        share_dirs = [os.path.join(self.enc_model_dir, sub_dir) for sub_dir in
                      os.listdir(self.enc_model_dir) if not sub_dir.startswith(".")]
        self.assertEqual(3, len(share_dirs))

    def test_model_decrypt(self):
        """
        Test normal case for model decryption.
        """
        aby3.decrypt_model(mpc_model_dir=self.enc_model_dir,
                           plain_model_path=self.dec_model_dir)
        infer_result = self.infer_with_decrypted_model(model_path=self.dec_model_dir)
        # accurate result is 13.79
        self.assertAlmostEqual(infer_result[0], 13.79, delta=1e-1)
        shutil.rmtree(self.dec_model_dir)

if __name__ == '__main__':
    unittest.main()  # run case according to their name
