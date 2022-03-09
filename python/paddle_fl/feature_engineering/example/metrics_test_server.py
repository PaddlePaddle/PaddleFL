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
mpc metrics test server-side 
"""

from concurrent import futures
import time
import grpc
import gen_test_file
import metrics_plain

from paddle_fl.feature_engineering.core.federated_feature_engineering_server import FederatedFeatureEngineeringServer

SERVER_ADRESS = 'localhost:50051'

def gen_server():
    """
    gen serever listen on port
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers = 10), 
                         options= [('grpc.max_send_message_length', 100 * 1024 * 1024),
                                  ('grpc.max_receive_message_length', 100 * 1024 * 1024)])
    with open('server.key', 'rb') as f:
        private_key = f.read()
    with open('server.crt', 'rb') as f:
        certificate_chain = f.read()
    server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain,),))
    server.add_secure_port('[::]:50051', server_credentials)
    return server


def postive_ratio_test_server(file_name):
    """
    postive ratio test 
    server do not get postive ratio
    """
    labels, features = gen_test_file.read_file(file_name)
    server = gen_server()
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    fed_fea_eng_server.get_positive_ratio(features)


def woe_test_server(file_name):
    """
    woe test
    """
    labels, features = gen_test_file.read_file(file_name)
    server = gen_server()
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    woe_list = fed_fea_eng_server.get_woe(features)
    print("server woe is \n", woe_list)


def iv_test_server(file_name):
    """
    iv test
    """
    labels, features = gen_test_file.read_file(file_name)
    server = gen_server()
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    iv_list = fed_fea_eng_server.get_iv(features)
    print("server iv is \n", iv_list)


def woe_iv_test_server(file_name):
    """
    woe, iv test
    """
    labels, features = gen_test_file.read_file(file_name)
    server = gen_server()
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    woe, iv = fed_fea_eng_server.get_woe_iv(features)
    print("server woe is \n", woe)
    print("server iv is \n", iv)


def ks_test_server(file_name):
    """
    ks test
    """
    labels, features = gen_test_file.read_file_float(file_name)
    server = gen_server()
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    ks_list = fed_fea_eng_server.get_ks(features)
    print("server ks is \n", ks_list)


def auc_test_server(file_name):
    """
    auc test
    """
    labels, features = gen_test_file.read_file_float(file_name)
    server = gen_server()
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    auc_list = fed_fea_eng_server.get_auc(features)
    print("server auc is \n", auc_list)


if __name__ == '__main__':
    file_name = "test_data.txt"
    time_start = time.time()
    ks_test_server(file_name)
    auc_test_server(file_name)
    postive_ratio_test_server(file_name)
    woe_test_server(file_name)
    iv_test_server(file_name)
    ks_test_server(file_name)
    auc_test_server(file_name)
    woe_iv_test_server(file_name)
    time_end = time.time()
    print("time cost ", time_end - time_start, 's')
