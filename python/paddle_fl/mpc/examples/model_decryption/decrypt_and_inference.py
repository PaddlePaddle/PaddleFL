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
Decrypt MPC inference model into paddle model and make prediction.
"""
import numpy as np
import paddle
import paddle.fluid as fluid

import paddle_fl.mpc.data_utils.data_utils 


mpc_du = get_datautils('aby3')

mpc_model_dir = '../model_encryption/predict/tmp/mpc_models_to_predict'
mpc_model_filename = 'model_to_predict'

decrypted_paddle_model_dir = './tmp/paddle_inference_model'
paddle_model_filename = 'decrypted_model'


def infer():
    """
    Predict with decrypted model.
    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # Step 1. load decrypted model.
    infer_prog, feed_names, fetch_targets = fluid.io.load_inference_model(
        executor=exe,
        dirname=decrypted_paddle_model_dir,
        model_filename=paddle_model_filename)
    # Step 2. make prediction
    batch_size = 10
    infer_reader = fluid.io.batch(
        paddle.dataset.uci_housing.test(), batch_size=batch_size)
    infer_data = next(infer_reader())
    infer_feat = np.array([data[0] for data in infer_data]).astype("float32")
    assert feed_names[0] == 'x'
    results = exe.run(infer_prog,
                      feed={feed_names[0]: np.array(infer_feat)},
                      fetch_list=fetch_targets)
    print("infer results: (House Price)")
    for idx, val in enumerate(results[0]):
        print("%d: %.2f" % (idx, val))


if __name__ == '__main__':
    # decrypt mpc model
    mpc_du.decrypt_model(
        mpc_model_dir=mpc_model_dir,
        plain_model_path=decrypted_paddle_model_dir,
        mpc_model_filename=mpc_model_filename,
        plain_model_filename=paddle_model_filename)
    print(
        'Successfully decrypt inference model. The decrypted model is saved in: {}'
        .format(decrypted_paddle_model_dir))

    # infer with decrypted model
    infer()
