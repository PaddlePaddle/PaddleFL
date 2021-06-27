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
MPC prediction.
"""
import sys
import time

import numpy as np
import paddle.fluid as fluid
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils

sys.path.append('..')
import process_data
import network


mpc_protocol_name = 'aby3'
mpc_du = get_datautils(mpc_protocol_name)

def load_mpc_model_and_predict(role, ip, server, port, mpc_model_dir, mpc_model_filename):
    """
    Predict based on MPC inference model, save prediction results into files.

    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    # Step 1. initialize MPC environment and load MPC model to predict
    pfl_mpc.init(mpc_protocol_name, role, ip, server, port)
    infer_prog, feed_names, fetch_targets = mpc_du.load_mpc_model(exe=exe,
                                                                mpc_model_dir=mpc_model_dir,
                                                                mpc_model_filename=mpc_model_filename,
                                                                inference=True)
    # Step 2. MPC predict
    batch_size = network.BATCH_SIZE
    feature_file = "/tmp/house_feature"
    feature_shape = (13,)
    pred_file = "./tmp/uci_prediction.part{}".format(role)
    loader = process_data.get_mpc_test_dataloader(feature_file, feature_shape, role, batch_size)
    start_time = time.time()
    for sample in loader():
        prediction = exe.run(program=infer_prog, feed={feed_names[0]: np.array(sample)}, fetch_list=fetch_targets)
        # Step 3. save prediction results
        with open(pred_file, 'ab') as f:
            f.write(np.array(prediction).tostring())
        break
    end_time = time.time()
    print('Mpc Predict with samples of {}, cost time in seconds:{}'
          .format(batch_size, (end_time - start_time)))


if __name__ == '__main__':
    role, server, port = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    mpc_model_dir = './tmp/mpc_models_to_predict/model_share_{}'.format(role)
    mpc_model_filename = 'model_to_predict'
    load_mpc_model_and_predict(role=role,
                               ip='localhost',
                               server=server,
                               port=port,
                               mpc_model_dir=mpc_model_dir,
                               mpc_model_filename=mpc_model_filename)
