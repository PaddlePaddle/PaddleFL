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
MPC training.
"""
import numpy as np
import os
import sys
import time

import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils

sys.path.append('..')
import network
import process_data


mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

def encrypt_model_and_train(role, ip, server, port, model_save_dir, model_filename):
    """
    Load uci network and train MPC model.

    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Step 1. Initialize MPC environment and load paddle model network and initialize parameter.
    pfl_mpc.init(mpc_protocol_name, role, ip, server, port)
    [_, _, _, loss] = network.uci_network()
    exe.run(fluid.default_startup_program())

    # Step 2. TRANSPILE: encrypt default_main_program into MPC program
    mpc_du.transpile()

    # Step 3. MPC-TRAINING: model training based on MPC program.
    mpc_data_dir = "../mpc_data/"
    feature_file = mpc_data_dir + "house_feature"
    feature_shape = (13,)
    label_file = mpc_data_dir + "house_label"
    label_shape = (1,)
    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')
    loss_file = "./tmp/uci_mpc_loss.part{}".format(role)
    if os.path.exists(loss_file):
        os.remove(loss_file)
    batch_size = network.UCI_BATCH_SIZE
    epoch_num = network.TRAIN_EPOCH
    feature_name = 'x'
    label_name = 'y'
    loader = process_data.get_mpc_dataloader(feature_file, label_file, feature_shape, label_shape,
                               feature_name, label_name, role, batch_size)
    start_time = time.time()
    for epoch_id in range(epoch_num):
        step = 0
        for sample in loader():
            mpc_loss = exe.run(feed=sample, fetch_list=[loss.name])
            if step % 50 == 0:
                print('Epoch={}, Step={}, Loss={}'.format(epoch_id, step, mpc_loss))
                with open(loss_file, 'ab') as f:
                    f.write(np.array(mpc_loss).tostring())
                step += 1
    end_time = time.time()
    print('Mpc Training of Epoch={} Batch_size={}, cost time in seconds:{}'
          .format(epoch_num, batch_size, (end_time - start_time)))

    # Step 4. SAVE trained MPC model as a trainable model.
    mpc_du.save_trainable_model(exe=exe,
                              model_dir=model_save_dir,
                              model_filename=model_filename)
    print('Successfully save mpc trained model into:{}'.format(model_save_dir))


role, server, port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
model_save_dir = './tmp/mpc_models_trained/trained_model_share_{}'.format(role)
trained_model_name = 'mpc_trained_model'
encrypt_model_and_train(role=role,
                        ip='localhost',
                        server=server,
                        port=port,
                        model_save_dir=model_save_dir,
                        model_filename=trained_model_name)
