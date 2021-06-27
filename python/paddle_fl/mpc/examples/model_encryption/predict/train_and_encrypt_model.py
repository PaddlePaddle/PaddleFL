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
Prepare work before MPC model inference, which includes create paddle
model to inference, and encrypt paddle model into MPC model.
"""
import paddle
import paddle.fluid as fluid
import sys
import time
from paddle_fl.mpc.data_utils.data_utils import get_datautils

sys.path.append('..')
import network


mpc_du = get_datautils('aby3')

def train_infer_model(model_dir, model_filename):
    """
    Original Training: train and save paddle inference model.

    """
    # Step 1. load paddle net
    [x, y, y_pre, loss] = network.uci_network()

    # Step 2. train
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    train_reader = paddle.batch(
        paddle.dataset.uci_housing.train(), batch_size=network.BATCH_SIZE, drop_last=True)
    start_time = time.time()
    for epoch_id in range(network.TRAIN_EPOCH):
        step = 0
        for data in train_reader():
            avg_loss = exe.run(feed=feeder.feed(data), fetch_list=[loss.name])
            if step % 50 == 0:
                print('Epoch={}, Step={}, Loss={}'.format(epoch_id, step, avg_loss[0]))
            step += 1
    end_time = time.time()
    print('For Prediction: Paddle Training of Epoch={} Batch_size={}, cost time in seconds:{}'
          .format(network.TRAIN_EPOCH, network.BATCH_SIZE, (end_time - start_time)))
    # Step 3. save inference model
    fluid.io.save_inference_model(executor=exe,
                                  main_program=fluid.default_main_program(),
                                  dirname=model_dir,
                                  model_filename=model_filename,
                                  feeded_var_names=[x.name],
                                  target_vars=[y_pre])


def encrypt_paddle_model(paddle_model_dir, mpc_model_dir, model_filename):
    """
    Load, encrypt and save model.

    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # Step 1. Load inference model.
    main_prog, _, _ = fluid.io.load_inference_model(executor=exe,
                                                    dirname=paddle_model_dir,
                                                    model_filename=model_filename)
    # Step 2. Encrypt inference model.
    mpc_du.encrypt_model(program=main_prog,
                       mpc_model_dir=mpc_model_dir,
                       model_filename=model_filename)

if __name__ == '__main__':
    model_to_predict_dir = './tmp/paddle_model_to_predict'
    model_to_predict_name = 'model_to_predict'
    train_infer_model(model_dir=model_to_predict_dir,
                      model_filename=model_to_predict_name)
    print('Successfully train and save paddle model to predict. The model is saved in: {}.'
          .format(model_to_predict_dir))

    mpc_model_to_predict_dir = './tmp/mpc_models_to_predict'
    encrypt_paddle_model(paddle_model_dir=model_to_predict_dir,
                         mpc_model_dir=mpc_model_to_predict_dir,
                         model_filename=model_to_predict_name)
    print('Successfully encrypt paddle model to predict. The encrypted models are saved in: {}.'
          .format(mpc_model_to_predict_dir))
