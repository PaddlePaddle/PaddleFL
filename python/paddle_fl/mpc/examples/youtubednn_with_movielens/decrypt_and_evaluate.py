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
decrypt video and user feature and evaluate hit ratio
"""
import os
import args
import get_topk
import process_data

if __name__ == '__main__':
    args = args.parse_args()

    # decrypt video and user feature
    mpc_data_dir = args.mpc_data_dir
    user_vec_filepath = mpc_data_dir + 'user_vec'
    plain_user_vec_filepath = user_vec_filepath + '.csv'
    if os.path.exists(plain_user_vec_filepath):
        os.system('rm -rf ' + plain_user_vec_filepath)
    process_data.decrypt_data_to_file(user_vec_filepath, plain_user_vec_filepath, (args.batch_size, 32))

    video_vec_filepath = mpc_data_dir + 'video_vec'
    plain_video_vec_filepath = video_vec_filepath + '.csv'
    if os.path.exists(plain_video_vec_filepath):
        os.system('rm -rf ' + plain_video_vec_filepath)
    process_data.decrypt_data_to_file(video_vec_filepath, plain_video_vec_filepath, (32, args.output_size))

    # compute similarity between users and videos
    # compute top k videos for each user
    mpc_data_dir = args.mpc_data_dir
    label_mpc_filepath = mpc_data_dir +'label_mpc'
    label_actual_filepath = mpc_data_dir +'label_actual'
    if os.path.exists(label_mpc_filepath):
        os.system('rm -rf ' + label_mpc_filepath)
    get_topk.get_topK(args, args.topk, plain_video_vec_filepath, plain_user_vec_filepath, label_actual_filepath, label_mpc_filepath)

    # evaluate hit ratio
    process_data.evaluate_hit_ratio(label_actual_filepath, label_mpc_filepath)
