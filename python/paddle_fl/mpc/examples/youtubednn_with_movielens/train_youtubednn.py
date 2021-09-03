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
MPC YoutubeDNN Demo
"""

import numpy as np
import pandas as pd
import os
import random
import time 
import logging
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle_fl.mpc as pfl_mpc
from paddle_fl.mpc.data_utils.data_utils import get_datautils

import args
import mpc_network


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)

aby3 = get_datautils("aby3")

def read_share(file, shape):
    """
    prepare share reader
    """
    ext = '.part{}'.format(args.role)
    shape = (2, ) + shape
    share_size = np.prod(shape) * 8  # size of int64 in bytes
    def reader():
        with open(file + ext, 'rb') as part_file:
            share = part_file.read(share_size)
            while share:
                yield np.frombuffer(share, dtype=np.int64).reshape(shape)
                share = part_file.read(share_size)
    return reader


def train(args):
    """
    train
    """
    # ********************
    # prepare network
    pfl_mpc.init('aby3', int(args.role), 'localhost', args.server, int(args.port))
    youtube_model = mpc_network.YoutubeDNN()
    inputs = youtube_model.input_data(args.batch_size, 
                                     args.watch_vec_size, 
                                     args.search_vec_size, 
                                     args.other_feat_size)
    loss, l3 = youtube_model.net(inputs, args.output_size, layers=[128, 64, 32])

    #boundaries = [200, 500, 800, 1000]
    #values = [0.05, 0.02, 0.01, 0.005, 0.001]
    #lr = fluid.layers.piecewise_decay(boundaries, values)
    lr = args.base_lr
    sgd = pfl_mpc.optimizer.SGD(learning_rate=lr)
    sgd.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # ********************
    # prepare data
    logger.info('Prepare data...')
    mpc_data_dir = args.mpc_data_dir
    if not os.path.exists(mpc_data_dir):
        raise ValueError('mpc_data_dir is not found. Please prepare encrypted data.')

    video_vec_filepath = mpc_data_dir + 'video_vec'
    video_vec_part_filepath = video_vec_filepath + '.part{}'.format(args.role)
    user_vec_filepath = mpc_data_dir + 'user_vec.csv'
    user_vec_part_filepath = user_vec_filepath + '.part{}'.format(args.role)

    watch_vecs = []
    search_vecs = []
    other_feats = []
    labels = []

    watch_vec_reader = read_share(file=mpc_data_dir + 'watch_vec', shape=(args.batch_size, args.watch_vec_size))
    for vec in watch_vec_reader():
        watch_vecs.append(vec)

    search_vec_reader = read_share(file=mpc_data_dir + 'search_vec', shape=(args.batch_size, args.search_vec_size))
    for vec in search_vec_reader():
        search_vecs.append(vec)

    other_feat_reader = read_share(file=mpc_data_dir + 'other_feat', shape=(args.batch_size, args.other_feat_size))
    for vec in other_feat_reader():
        other_feats.append(vec)

    label_reader = read_share(file=mpc_data_dir + 'label', shape=(args.batch_size, args.output_size))
    for vec in label_reader():
        labels.append(vec)

    # ********************
    # train
    logger.info('Start training...')
    begin = time.time()
    for epoch in range(args.epochs):
        for i in range(args.batch_num):
            loss_data = exe.run(fluid.default_main_program(),
                                feed={'watch_vec': watch_vecs[i],
                                      'search_vec': search_vecs[i],
                                      'other_feat': other_feats[i],
                                      'label': np.array(labels[i])
                                      },
                                return_numpy=True,
                                fetch_list=[loss.name])
        
            if i % 100 == 0:
                end = time.time()
                logger.info('Paddle training of epoch_id: {}, batch_id: {}, batch_time: {:.5f}s'
                    .format(epoch, i, end-begin))
        # save model
        logger.info('save mpc model...')
        cur_model_dir = os.path.join(args.model_dir, 'mpc_model', 'epoch_' + str(epoch + 1), 
                                     'checkpoint', 'party_{}'.format(args.role))
        feed_var_names = ['watch_vec', 'search_vec', 'other_feat']
        fetch_vars = [l3]
        fluid.io.save_inference_model(cur_model_dir, feed_var_names, fetch_vars, exe)

        # save all video vector
        video_array = np.array(fluid.global_scope().find_var('l4_weight').get_tensor())
        if os.path.exists(video_vec_part_filepath):
            os.system('rm -rf ' + video_vec_part_filepath)
        with open(video_vec_part_filepath, 'wb') as f:
            f.write(np.array(video_array).tostring())

        end = time.time()
        logger.info('MPC training of epoch: {}, cost_time: {:.5f}s'.format(epoch, end - begin))
    logger.info('End training.')

def infer(args):
    """
    infer
    """
    logger.info('Start inferring...')
    begin = time.time()
    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    cur_model_path = os.path.join(args.model_dir, 'mpc_model', 'epoch_' + str(args.test_epoch),
                                  'checkpoint', 'party_{}'.format(args.role))

    with fluid.scope_guard(fluid.Scope()):
        pfl_mpc.init('aby3', args.role, 'localhost', args.server, args.port)
        infer_program, feed_target_names, fetch_vars = aby3.load_mpc_model(exe=exe,
                                                                    mpc_model_dir=cur_model_path,
                                                                    mpc_model_filename='__model__',
                                                                    inference=True)
        mpc_data_dir = args.mpc_data_dir
        user_vec_filepath = mpc_data_dir + 'user_vec'
        user_vec_part_filepath = user_vec_filepath + '.part{}'.format(args.role)

        sample_batch = args.batch_size
        watch_vecs = []
        search_vecs = []
        other_feats = []

        watch_vec_reader = read_share(file=mpc_data_dir + 'watch_vec', shape=(sample_batch, args.watch_vec_size))
        for vec in watch_vec_reader():
            watch_vecs.append(vec)
        search_vec_reader = read_share(file=mpc_data_dir + 'search_vec', shape=(sample_batch, args.search_vec_size))
        for vec in search_vec_reader():
            search_vecs.append(vec)
        other_feat_reader = read_share(file=mpc_data_dir + 'other_feat', shape=(sample_batch, args.other_feat_size))
        for vec in other_feat_reader():
            other_feats.append(vec)

        if os.path.exists(user_vec_part_filepath):
            os.system('rm -rf ' + user_vec_part_filepath)

        for i in range(args.batch_num):
            l3 = exe.run(infer_program,
                         feed={
                               'watch_vec': watch_vecs[i],
                               'search_vec': search_vecs[i],
                               'other_feat': other_feats[i],
                         },
                         return_numpy=True,
                         fetch_list=fetch_vars)

            with open(user_vec_part_filepath, 'ab+') as f:
                f.write(np.array(l3[0]).tostring())


    end = time.time()
    logger.info('MPC inferring, cost_time: {:.5f}s'.format(end - begin))
    logger.info('End inferring.')


if __name__ == '__main__':
    args = args.parse_args()
    logger.info(
        ('use_gpu: {}, batch_size: {}, epochs: {}, watch_vec_size: {}, search_vec_size: {},' +  
        'other_feat_size: {}, output_size: {}, model_dir: {}, test_epoch: {}, base_lr: {}').format(
        args.use_gpu, args.batch_size, args.epochs, args.watch_vec_size, args.search_vec_size, args.other_feat_size,
        args.output_size, args.model_dir, args.test_epoch, args.base_lr))

    train(args)
    infer(args)
