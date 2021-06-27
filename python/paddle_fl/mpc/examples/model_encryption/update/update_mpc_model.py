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
MPC updating.
"""
import os
import sys
import time

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils

sys.path.append('..')
import network
import process_data


mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

def load_uci_update(role, ip, server, port, mpc_model_dir, mpc_model_filename, updated_model_dir):
    """
    Load, update and save uci MPC model.

    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Step 1. initialize MPC environment and load MPC model into default_main_program to update.
    pfl_mpc.init(mpc_protocol_name, role, ip, server, port)
    mpc_du.load_mpc_model(exe=exe,
                          mpc_model_dir=mpc_model_dir,
                          mpc_model_filename=mpc_model_filename)

    # Step 2. MPC update
    epoch_num = network.MPC_UPDATE_EPOCH
    batch_size = network.BATCH_SIZE
    mpc_data_dir = "../mpc_data/"
    feature_file = mpc_data_dir + "house_feature"
    feature_shape = (13,)
    label_file = mpc_data_dir + "house_label"
    label_shape = (1,)
    loss_file = "./tmp/uci_mpc_loss.part{}".format(role)
    if os.path.exists(loss_file):
        os.remove(loss_file)
    updated_model_name = 'mpc_updated_model'
    feature_name = 'x'
    label_name = 'y'
    # fetch loss if needed
    loss = fluid.default_main_program().global_block().var('mean_0.tmp_0')
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
    print('Mpc Updating of Epoch={} Batch_size={}, cost time in seconds:{}'
          .format(epoch_num, batch_size, (end_time - start_time)))

    # Step 3. save updated MPC model as a trainable model.
    mpc_du.save_trainable_model(exe=exe,
                                model_dir=updated_model_dir,
                                model_filename=updated_model_name)
    print('Successfully save mpc updated model into:{}'.format(updated_model_dir))


if __name__ == '__main__':
    role, server, port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    mpc_model_dir = './tmp/mpc_models_to_update/model_share_{}'.format(role)
    mpc_model_filename = 'model_to_update'
    updated_model_dir = './tmp/mpc_models_updated/updated_model_share_{}'.format(role)
    load_uci_update(role=role,
                    ip='localhost',
                    server=server,
                    port=port,
                    mpc_model_dir=mpc_model_dir,
                    mpc_model_filename=mpc_model_filename,
                    updated_model_dir=updated_model_dir)
