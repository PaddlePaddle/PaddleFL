# -*- coding: utf-8 -*-
from __future__ import print_function
import grpc
import servers.scheduler_server_pb2 as scheduler_server_pb2
import servers.scheduler_server_pb2_grpc as scheduler_server_pb2_grpc
import servers.data_server_pb2_grpc as data_server_pb2_grpc
import servers.data_server_pb2 as data_server_pb2
import numpy as np
from concurrent import futures
from multiprocessing import Process

import time
import sys
import os


class SchedulerClient(object):
    def __init__(self):
        self.stub = None
        self.stub_list = []
        self.global_param_info = {}

    def update_user_inst_num(self, date, user_info_dict):
        user_info = scheduler_server_pb2.UserInstInfo()
        for key in user_info_dict:
            single_user_info = scheduler_server_pb2.UserInstNum()
            single_user_info.uid = key
            single_user_info.inst_num = user_info_dict[key]
            user_info.inst_nums.extend([single_user_info])
        user_info.shard_num = len(self.stub_list)
        user_info.date = date
        call_future = self.stub.UpdateUserInstNum.future(user_info)
        res = call_future.result()
        return res.err_code

    def set_scheduler_server_endpoints(self, endpoints):
        options = [('grpc.max_message_length', 1024 * 1024 * 1024),
                   ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
        channel = grpc.insecure_channel(endpoints[0], options=options)
        self.stub = scheduler_server_pb2_grpc.SchedulerServerStub(channel)

    def set_data_server_endpoints(self, endpoints):
        self.stub_list = []
        for ep in endpoints:
            options = [('grpc.max_message_length', 1024 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
            channel = grpc.insecure_channel(ep, options=options)
            stub = data_server_pb2_grpc.DataServerStub(channel)
            self.stub_list.append(stub)

    def uniform_sample_user_list(self, date, node_id, sample_num, shard_num,
                                 node_num, min_ins_num):
        user_info_dict = {}
        req = scheduler_server_pb2.Request()
        req.node_idx = node_id
        req.sample_num = sample_num
        req.shard_num = shard_num
        req.node_num = node_num
        req.date = date
        req.min_ins_num = min_ins_num
        call_future = self.stub.SampleUsersToTrain.future(req)
        user_info = call_future.result()
        for user in user_info.inst_nums:
            user_info_dict[user.uid] = user.inst_num
        return user_info_dict

    def hash_sample_user_list(self, date, node_id, sample_num, shard_num,
                              node_num):
        user_info_dict = {}
        req = scheduler_server_pb2.Request()
        req.node_idx = node_id
        req.sample_num = sample_num
        req.shard_num = shard_num
        req.node_num = node_num
        req.date = date
        call_future = self.stub.SampleUsersWithHash.future(req)
        user_info = call_future.result()
        for user in user_info.inst_nums:
            user_info_dict[user.uid] = user.inst_num
        return user_info_dict

    def sample_test_user_list(self, date, node_id, shard_num, node_num):
        user_info_dict = {}
        req = scheduler_server_pb2.Request()
        req.node_idx = node_id
        req.shard_num = shard_num
        req.node_num = node_num
        req.date = date
        call_future = self.stub.SampleUsersToTest.future(req)
        user_info = call_future.result()
        for user in user_info.inst_nums:
            user_info_dict[user.uid] = user.inst_num
        return user_info_dict

    def fixed_sample_user_list(self, date, node_id, sample_num, shard_num,
                               node_num):
        user_info_dict = {}
        req = scheduler_server_pb2.Request()
        req.node_idx = node_id
        req.sample_num = sample_num
        req.shard_num = shard_num
        req.node_num = node_num
        req.date = date
        call_future = self.stub.FixedUsersToTrain.future(req)
        user_info = call_future.result()
        for user in user_info.inst_nums:
            user_info_dict[user.uid] = user.inst_num
        return user_info_dict

    def get_global_params(self):
        req = scheduler_server_pb2.Request()
        req.node_idx = 0
        req.sample_num = 0
        req.shard_num = 0
        req.node_num = 0
        call_future = self.stub.GetGlobalParams.future(req)
        global_p = call_future.result()
        result_dict = {}
        for param in global_p.global_params:
            result_dict[param.name] = np.array(
                list(param.weight), dtype=np.float32)
            result_dict[param.name].shape = param.shape
        return result_dict

    def update_global_params(self, global_params):
        global_p = scheduler_server_pb2.GlobalParams()
        for key in global_params:
            param = scheduler_server_pb2.Param()
            param.name = key
            var, shape = global_params[key], global_params[key].shape
            self.global_param_info[key] = shape
            param.weight.extend(var.ravel())
            param.shape.extend(shape)
            global_p.global_params.extend([param])
        call_future = self.stub.UpdateGlobalParams.future(global_p)
        res = call_future.result()
        return res.err_code

    def fedavg_update(self, global_param_delta_dict):
        global_p = scheduler_server_pb2.GlobalParams()
        for key in global_param_delta_dict:
            param = scheduler_server_pb2.Param()
            param.name = key
            parameter_delta, shape = global_param_delta_dict[
                param.name], global_param_delta_dict[param.name].shape
            param.weight.extend(parameter_delta.ravel())
            param.shape.extend(shape)
            global_p.global_params.extend([param])
        call_future = self.stub.FedAvgUpdate.future(global_p)
        res = call_future.result()
        return res.err_code

    def stop_scheduler_server(self):
        empty_input = scheduler_server_pb2.SchedulerServerEmptyInput()
        call_future = self.stub.Exit.future(empty_input)
        res = call_future.result()


def test_uniform_sample_user_list():
    client = SchedulerClient()
    client.set_scheduler_server_endpoints(["127.0.0.1:60001"])
    # buggy
    #user_list = client.get_user_list()
    global_user_list = []
    for i in range(10000):
        global_user_list.append((str(i), 100))
    client.update_user_inst_num(date=None, user_info_dict=global_user_list)
    user_info = client.uniform_sample_user_list(
        date=0,
        node_id=None,
        sample_num=None,
        shard_num=None,
        node_num=None,
        min_ins_num=None)
    user_list = [("101", 100), ("102", 100), ("103", 10000)]
    global_param = {"w0": [1.0, 1.0, 1.0], "w1": [2.0, 2.0, 2.0]}
    client.update_global_params(global_param)
    fetched_params = client.get_global_params()


def test_get_global_params():
    client = SchedulerClient()
    client.set_scheduler_server_endpoints(["127.0.0.1:60001"])
    global_param = {"w0": [1.0, 1.0, 1.0], "w1": [2.0, 2.0, 2.0]}
    client.update_global_params(global_param)
    fetched_params = client.get_global_params()
    print(fetched_params)


def test_update_global_params():
    client = SchedulerClient()
    client.set_scheduler_server_endpoints(["127.0.0.1:60001"])
    global_param = {"w0": [1.0, 1.0, 1.0], "w1": [3.0, 3.0, 3.0]}
    client.update_global_params(global_param)
    fetched_params = client.get_global_params()
    print(fetched_params)


if __name__ == "__main__":
    test_uniform_sample_user_list()
    test_update_global_params()
    test_get_global_params()
