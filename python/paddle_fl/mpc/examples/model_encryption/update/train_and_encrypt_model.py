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
Prepare work before MPC model updating, which includes create paddle
model to update, and encrypt paddle model into MPC model.
"""
import paddle
import paddle.fluid as fluid
import time
import sys
from paddle_fl.mpc.data_utils.data_utils import get_datautils

sys.path.append('..')
import network


mpc_du = get_datautils('aby3')


def original_train(model_dir, model_filename):
    """
    Original Training: train and save pre-trained paddle model

    """
    # Step 1. load paddle net
    [x, y, _, loss] = network.uci_network()

    # Step 2. train
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
    train_reader = paddle.batch(
        paddle.dataset.uci_housing.train(), batch_size=network.BATCH_SIZE, drop_last=True)
    start_time = time.time()
    for epoch_id in range(network.PADDLE_UPDATE_EPOCH):
        step = 0
        for data in train_reader():
            avg_loss = exe.run(feed=feeder.feed(data), fetch_list=[loss.name])
            if step % 50 == 0:
                print('Epoch={}, Step={}, Loss={}'.format(epoch_id, step, avg_loss[0]))
            step += 1
    end_time = time.time()
    print('Paddle Training of Epoch={} Batch_size={}, cost time in seconds:{}'
          .format(network.PADDLE_UPDATE_EPOCH, network.BATCH_SIZE, (end_time - start_time)))

    # Step 3. save model to update
    mpc_du.save_trainable_model(exe=exe,
                              program=fluid.default_main_program(),
                              model_dir=model_dir,
                              model_filename=model_filename)


def encrypt_paddle_model(paddle_model_dir, mpc_model_dir, model_filename):
    """
    Load, encrypt and save model.

    """
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    # Step 1. Load pre-trained model.
    main_prog, _, _ = fluid.io.load_inference_model(executor=exe,
                                                    dirname=paddle_model_dir,
                                                    model_filename=model_filename)
    # Step 2. Encrypt pre-trained model.
    mpc_du.encrypt_model(program=main_prog,
                       mpc_model_dir=mpc_model_dir,
                       model_filename=model_filename)


if __name__ == '__main__':

    # train paddle model
    model_to_update_dir = './tmp/paddle_model_to_update'
    model_to_update_name = 'model_to_update'
    original_train(model_dir=model_to_update_dir,
                   model_filename=model_to_update_name)
    print('Successfully train and save paddle model for update. The model is saved in: {}.'
          .format(model_to_update_dir))

    # encrypt paddle model
    mpc_model_to_update_dir = './tmp/mpc_models_to_update'
    encrypt_paddle_model(paddle_model_dir=model_to_update_dir,
                         mpc_model_dir=mpc_model_to_update_dir,
                         model_filename=model_to_update_name)
    print('Successfully encrypt paddle model for update. The encrypted models are saved in: {}.'
          .format(mpc_model_to_update_dir))
