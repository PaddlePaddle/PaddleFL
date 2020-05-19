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

from __future__ import print_function
from concurrent import futures
import data_server_pb2
import data_server_pb2_grpc
import grpc
import sys
from utils.logger import logging


class DataServerServicer(object):
    def __init__(self):
        self.data_dict = {}
        self.param_dict = {}
        self.request_num = 0

    def GetUserParams(self, request, context):
        uid = unicode(request.uid)
        if uid in self.param_dict:
            return self.param_dict[uid]
        else:
            user_param = data_server_pb2.UserParams()
            user_param.err_code = -1
            return user_param

    def UpdateUserParams(self, request, context):
        self.param_dict[request.uid] = request
        res = data_server_pb2.Res()
        res.err_code = 0
        return res

    def ClearUserData(self, request, context):
        date = request.date
        self.data_dict[date].clear()
        res = data_server_pb2.Res()
        res.err_code = 0
        return res

    def GetUserData(self, request, context):
        uid = unicode(request.uid)
        date = unicode(request.date)
        if date not in self.data_dict:
            user_data = data_server_pb2.UserData()
            user_data.err_code = -1
            return user_data
        if uid in self.data_dict[date]:
            self.data_dict[date][uid].err_code = 0
            return self.data_dict[date][uid]
        else:
            user_data = data_server_pb2.UserData()
            user_data.err_code = -1
            self.data_dict[date][uid] = user_data
            return self.data_dict[date][uid]
        user_data = data_server_pb2.UserData()
        user_data.err_code = -1
        return user_data

    def SendData(self, request, context):
        date = unicode(request.date)
        uid = unicode(request.uid)
        if date in self.data_dict:
            if uid in self.data_dict[request.date]:
                self.data_dict[date][uid].line_str.extend([request.line])
            else:
                user_data = data_server_pb2.UserData()
                self.data_dict[date][uid] = user_data
                self.data_dict[date][uid].line_str.extend([request.line])
        else:
            user_data = data_server_pb2.UserData()
            self.data_dict[date] = {}
            self.data_dict[date][uid] = user_data
            self.data_dict[date][uid].line_str.extend([request.line])

        res = data_server_pb2.Res()
        res.err_code = 0
        return res


class DataServer(object):
    def __init__(self):
        pass

    def start(self, max_workers=1000, concurrency=100, endpoint=""):
        if endpoint == "":
            logging.error("You should specify endpoint in start function")
            return
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        data_server_pb2_grpc.add_DataServerServicer_to_server(
            DataServerServicer(), server)
        server.add_insecure_port('[::]:{}'.format(endpoint))
        server.start()
        server.wait_for_termination()


if __name__ == "__main__":
    data_server = DataServer()
    data_server.start(endpoint=sys.argv[1])
