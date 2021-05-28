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
Compute the similarity of videos and users, get topK video for each user
"""

import numpy as np
import pandas as pd
import copy
import os
import logging
import args
import process_data

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fluid")
logger.setLevel(logging.INFO)


def cos_sim(vector_a, vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim


def get_topK(args, K, video_vec_path, user_vec_path, label_actual_filepath, label_paddle_filepath):
    video_vec = pd.read_csv(video_vec_path, header=None)
    user_vec = pd.read_csv(user_vec_path, header=None)

    if (os.path.exists(label_paddle_filepath)):
        os.system("rm -rf " + label_paddle_filepath)

    for i in range(user_vec.shape[0]):
        user_video_sim_list = []
        for j in range(video_vec.shape[1]):    
            user_video_sim = cos_sim(np.array(user_vec.loc[i]), np.array(video_vec[j]))
            user_video_sim_list.append(user_video_sim)
        tmp_list=copy.deepcopy(user_video_sim_list)
        tmp_list.sort()
        max_sim_index=[[user_video_sim_list.index(one) for one in tmp_list[::-1][:K]]]

        max_sim_index_vec = pd.DataFrame(max_sim_index)
        max_sim_index_vec.to_csv(label_paddle_filepath, mode="a", index=False, header=0)

        # for debug
        process_data.evaluate_hit_ratio(label_actual_filepath, label_paddle_filepath)


if __name__ == '__main__':
    args = args.parse_args()
    data_dir = './paddle_data/'
    get_topK(args, args.topk, data_dir + 'video_vec.csv', data_dir + 'user_vec.csv', data_dir + 'label_paddle')
