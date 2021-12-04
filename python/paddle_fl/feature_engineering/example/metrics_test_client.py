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
# limitations under the License
"""
mpc metrics test client-side 
"""

import time
import grpc
import gen_test_file
import metrics_plain

from paddle_fl.feature_engineering.core.federated_feature_engineering_client import FederatedFeatureEngineeringClient

SERVER_ADRESS = 'localhost:50051'

def gen_client_channel(server_adress):
    """
    gen channel with server_adress
    """
    with open('server.crt', 'rb') as f:
        trusted_certs = f.read()
    credentials = grpc.ssl_channel_credentials(root_certificates = trusted_certs)

    channel = grpc.secure_channel(server_adress, credentials,
                    options=(('grpc.ssl_target_name_override', "metrics_service",),
                    ('grpc.max_send_message_length', 100 * 1024 * 1024),
                    ('grpc.max_receive_message_length', 100 * 1024 * 1024)))
    return channel


def postive_ratio_test_client(file_name):
    """
    postive ratio test 
    """
    labels, features = gen_test_file.read_file(file_name)

    fed_fea_eng_client = FederatedFeatureEngineeringClient(1024)
    channel = gen_client_channel(SERVER_ADRESS)
    fed_fea_eng_client.connect(channel)
    result = fed_fea_eng_client.get_positive_ratio(labels)
    print("client mpc positive ratio is \n", result)
    plain_result = metrics_plain.get_plain_pos_ratio(labels, features)
    print("client plain positive ratio is \n", plain_result)


def woe_test_client(file_name):
    """
    woe test 
    """
    labels, features = gen_test_file.read_file(file_name)

    fed_fea_eng_client = FederatedFeatureEngineeringClient(1024)
    channel = gen_client_channel(SERVER_ADRESS)
    fed_fea_eng_client.connect(channel)
    result = fed_fea_eng_client.get_woe(labels)
    print("client mpc woe is \n", result)
    plain_result = metrics_plain.get_plain_woe(labels, features)
    print("client plain woe is \n", plain_result)


def iv_test_client(file_name):
    """
    iv test
    """
    labels, features = gen_test_file.read_file(file_name)

    fed_fea_eng_client = FederatedFeatureEngineeringClient(1024)
    channel = gen_client_channel(SERVER_ADRESS)
    fed_fea_eng_client.connect(channel)
    result = fed_fea_eng_client.get_iv(labels)
    print("client mpc iv is \n", result)
    plain_result = metrics_plain.get_plain_iv(labels, features)
    print("client plain iv is \n", plain_result)

if __name__ == '__main__':
    file_name = "test_data.txt"
    time_cost = []
    time_start = time.time()
    postive_ratio_test_client(file_name)
    time_pos_ratio = time.time()
    time_cost.append(time_pos_ratio - time_start)
    woe_test_client(file_name)
    time_woe = time.time()
    time_cost.append(time_woe - time_pos_ratio)
    iv_test_client(file_name)
    time_iv = time.time()
    time_cost.append(time_iv - time_woe)
    time_cost = gen_test_file.np.array(time_cost)
    gen_test_file.np.savetxt("time.txt", time_cost, fmt='%f', delimiter=',')
    print("time cost ", time_iv - time_start, 's')
    