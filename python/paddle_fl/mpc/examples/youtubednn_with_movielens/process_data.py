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
Prepare movielens dataset for YoutubeDNN.
"""
import numpy as np
import paddle
import os
import time
import six
import pandas as pd
import logging
from paddle_fl.mpc.data_utils.data_utils import get_datautils
import args
import get_topk


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('fluid')
logger.setLevel(logging.INFO)


aby3 = get_datautils("aby3") 
args = args.parse_args()

watch_vec_size = args.watch_vec_size
search_vec_size = args.search_vec_size
other_feat_size = args.other_feat_size
dataset_size = args.dataset_size

batch_size = args.batch_size
sample_size = args.batch_num
output_size = args.output_size # max movie id


def prepare_movielens_data(sample_size, batch_size, watch_vec_size, search_vec_size, 
                           other_feat_size, dataset_size, label_actual_filepath):
    """
    prepare movielens data
    """
    watch_vecs = []
    search_vecs = []
    other_feats = []
    labels = []

    # prepare movielens data
    movie_info = paddle.dataset.movielens.movie_info()
    user_info = paddle.dataset.movielens.user_info()

    max_user_id = paddle.dataset.movielens.max_user_id()
    user_watch = np.zeros((max_user_id, watch_vec_size))
    user_search = np.zeros((max_user_id, search_vec_size))
    user_feat = np.zeros((max_user_id, other_feat_size))
    user_labels = np.zeros((max_user_id, 1))

    MOVIE_EMBED_TAB_HEIGHT = paddle.dataset.movielens.max_movie_id()
    MOVIE_EMBED_TAB_WIDTH = watch_vec_size

    JOB_EMBED_TAB_HEIGHT = paddle.dataset.movielens.max_job_id() + 1
    JOB_EMBED_TAB_WIDTH = paddle.dataset.movielens.max_job_id() + 1

    AGE_EMBED_TAB_HEIGHT = len(paddle.dataset.movielens.age_table)
    AGE_EMBED_TAB_WIDTH = len(paddle.dataset.movielens.age_table)

    GENDER_EMBED_TAB_HEIGHT = 2
    GENDER_EMBED_TAB_WIDTH = 4

    np.random.seed(1)

    MOVIE_EMBED_TAB = np.zeros((MOVIE_EMBED_TAB_HEIGHT, MOVIE_EMBED_TAB_WIDTH))
    AGE_EMBED_TAB = np.zeros((AGE_EMBED_TAB_HEIGHT, AGE_EMBED_TAB_WIDTH))
    GENDER_EMBED_TAB = np.zeros((GENDER_EMBED_TAB_HEIGHT, GENDER_EMBED_TAB_WIDTH))
    JOB_EMBED_TAB = np.zeros((JOB_EMBED_TAB_HEIGHT, JOB_EMBED_TAB_WIDTH))


    for i in range(MOVIE_EMBED_TAB_HEIGHT):
        MOVIE_EMBED_TAB[i][hash(i) % MOVIE_EMBED_TAB_WIDTH] = 1
        MOVIE_EMBED_TAB[i][hash(hash(i)) % MOVIE_EMBED_TAB_WIDTH] = 1

    for i in range(AGE_EMBED_TAB_HEIGHT):
        AGE_EMBED_TAB[i][i] = 1

    for i in range(GENDER_EMBED_TAB_HEIGHT):
        GENDER_EMBED_TAB[i][i] = 1

    for i in range(JOB_EMBED_TAB_HEIGHT):
        JOB_EMBED_TAB[i][i] = 1

    train_set_creator = paddle.dataset.movielens.train()

    pre_uid = 0
    movie_count = 0
    user_watched_movies = [[] for i in range(dataset_size)]
    for instance in train_set_creator():
        uid = int(instance[0]) - 1
        gender_id = int(instance[1])
        age_id = int(instance[2])
        job_id = int(instance[3])
        mov_id = int(instance[4]) - 1
        user_watched_movies[uid].append(mov_id)
        user_watch[uid, :] += MOVIE_EMBED_TAB[mov_id, :]
        user_labels[uid, :] = mov_id

        user_feat[uid, :] = np.concatenate((JOB_EMBED_TAB[job_id, :], 
                                            GENDER_EMBED_TAB[gender_id, :], 
                                            AGE_EMBED_TAB[age_id, :]))

        if uid == pre_uid:
            movie_count += 1
        else:
            user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count
            movie_count = 1
            pre_uid = uid
    user_watch[pre_uid, :] = user_watch[pre_uid, :] / movie_count

    user_search = user_watch

    if (os.path.exists(label_actual_filepath)):
        os.system('rm -rf'  + label_actual_filepath)
    user_watched_movies_vec = pd.DataFrame(user_watched_movies)
    user_watched_movies_vec.to_csv(label_actual_filepath, mode='a', index=False, header=0)

    return user_watch, user_search, user_feat, user_labels


def gen_cypher_sample(mpc_data_dir, sample_size, batch_size, output_size):
    """
    prepare movielens data and encrypt
    """
    logger.info('Prepare data...')
    if not os.path.exists(mpc_data_dir):
        os.makedirs(mpc_data_dir)
    else:
        os.system('rm -rf ' + mpc_data_dir + '*')
    label_actual_filepath = mpc_data_dir + 'label_actual'
    user_watch, user_search, user_feat, user_labels = prepare_movielens_data(
        sample_size, batch_size, watch_vec_size, search_vec_size, other_feat_size, dataset_size, label_actual_filepath)

    #watch_vecs = []
    #search_vecs = []
    #other_feat_vecs = []
    #label_vecss = []

    for i in range(sample_size):
        watch_vec = user_watch[i * batch_size : (i + 1) * batch_size, :]
        search_vec = user_search[i * batch_size : (i + 1) * batch_size, :]
        other_feat_vec = user_feat[i * batch_size : (i + 1) * batch_size, :]
        save_cypher(cypher_file=mpc_data_dir + 'watch_vec', vec=watch_vec)
        save_cypher(cypher_file=mpc_data_dir + 'search_vec', vec=search_vec)
        save_cypher(cypher_file=mpc_data_dir + 'other_feat', vec=other_feat_vec)
        #watch_vecs.append(watch_vec)
        #search_vecs.append(search_vec)
        #other_feat_vecs.append(other_feat_vec)
        label = np.zeros((batch_size, output_size))
        for j in range(batch_size):
            label[j, int(user_labels[j][0])] = 1
        save_cypher(cypher_file=mpc_data_dir + 'label', vec=label)
        #label_vecs.append(label)
    #return [watch_vecs, search_vecs, other_feat_vecs, label_vecs]


def save_cypher(cypher_file, vec):
    """
    save cypertext to file
    """
    shares = aby3.make_shares(vec)
    exts = ['.part0', '.part1', '.part2']
    with open(cypher_file + exts[0], 'ab') as file0, \
            open(cypher_file + exts[1], 'ab') as file1, \
            open(cypher_file + exts[2], 'ab') as file2:
        files = [file0, file1, file2]
        for idx in six.moves.range(0, 3):  # 3 parts
            share = aby3.get_shares(shares, idx)
            files[idx].write(share.tostring())


def load_decrypt_data(filepath, shape):
    """
    load the encrypted data and reconstruct
    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(aby3.load_shares(filepath, id=id, shape=shape))
    aby3_share_reader = paddle.reader.compose(part_readers[0], part_readers[1], part_readers[2])

    for instance in aby3_share_reader():
        p = aby3.reconstruct(np.array(instance))
        print(p)


def decrypt_data_to_file(cypher_filepath, plaintext_filepath, shape):
    """
    Load the encrypted data and reconstruct.

    """
    part_readers = []
    for id in six.moves.range(3):
        part_readers.append(
            aby3.load_shares(
                cypher_filepath, id=id, shape=shape))
    aby3_share_reader = paddle.reader.compose(part_readers[0], part_readers[1],
                                              part_readers[2])

    for instance in aby3_share_reader():
        p = aby3.reconstruct(np.array(instance))
        tmp = pd.DataFrame(p)
        tmp.to_csv(plaintext_filepath, mode='a', index=False, header=0)


def evaluate_hit_ratio(file1, file2):
    count = 0
    same_count = 0
    f1 = open(file1, 'r')
    f2 = open(file2, 'r')
    while 1:
        line1 = f1.readline()
        line2 = f2.readline()
        if (not line1) or (not line2):
            break
        count += 1
        set1 = set([int(float(x if x != '' and x != '\n' else 10000)) for x in line1.split(',')])
        set2 = set([int(x) for x in line2.split(',')])
        if len(set1.intersection(set2)) != 0:
            same_count += 1
    logger.info(float(same_count)/count)


if __name__ == '__main__':
    gen_cypher_sample('./mpc_data/', sample_size, batch_size, output_size)
