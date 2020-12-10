# -*- coding: utf-8 -*- 
from __future__ import print_function
import grpc
import servers.data_server_pb2 as data_server_pb2
import servers.data_server_pb2_grpc as data_server_pb2_grpc
from concurrent import futures
from multiprocessing import Process
from utils.hdfs_utils import HDFSClient, multi_download
import time
import sys
import os
import xxhash
import numpy as np
from utils.logger import logging


class DataClient(object):
    def __init__(self):
        self.stub_list = []
        self.load_data_into_patch = None

    def uid_shard(self, uid):
        try:
            uid_hash = xxhash.xxh32(str(uid), seed=101).intdigest()
        except:
            return -1
        shard_idx = uid_hash % len(self.stub_list)
        return shard_idx

    # should set all params to numpy array with shape and dtype
    # buggy here
    def set_param_by_uid(self, uid, param_dict):
        shard_idx = self.uid_shard(uid)
        if shard_idx == -1:
            return -1
        user_param = data_server_pb2.UserParams()
        user_param.uid = uid
        for key in param_dict:
            param = data_server_pb2.Param()
            param.name = key
            np_var = param_dict[param.name]
            param.shape.extend(np_var.shape)
            param.weight.extend(np_var.ravel())
            user_param.user_params.extend([param])

        call_future = self.stub_list[shard_idx].UpdateUserParams.future(
            user_param)
        err_code = call_future.result().err_code
        return err_code

    def get_param_by_uid(self, uid):
        shard_idx = self.uid_shard(uid)
        if shard_idx == -1:
            return -1
        data = data_server_pb2.Data()
        data.uid = uid
        call_future = self.stub_list[shard_idx].GetUserParams.future(data)
        user_params = call_future.result()
        param_dict = {}
        for param in user_params.user_params:
            param_dict[param.name] = np.array(
                list(param.weight), dtype=np.float32)
            param_dict[param.name].shape = list(param.shape)
        return param_dict

    def clear_user_data(self, date):
        def clear():
            for stub in self.stub_list:
                data = data_server_pb2.Data()
                data.date = date
                call_future = stub.ClearUserData.future(data)
                res = call_future.result()

        p = Process(target=clear, args=())
        p.start()
        p.join()

    def get_data_by_uid(self, uid, date):
        shard_idx = self.uid_shard(uid)
        if shard_idx == -1:
            return -1
        data = data_server_pb2.Data()
        data.uid = uid
        data.date = date
        call_future = self.stub_list[shard_idx].GetUserData.future(data)
        user_data_list = []
        for item in call_future.result().line_str:
            user_data_list.append(item)
        return user_data_list

    def set_data_server_endpoints(self, endpoints):
        self.stub_list = []
        for ep in endpoints:
            options = [('grpc.max_message_length', 1024 * 1024 * 1024),
                       ('grpc.max_receive_message_length', 1024 * 1024 * 1024)]
            channel = grpc.insecure_channel(ep, options=options)
            stub = data_server_pb2_grpc.DataServerStub(channel)
            self.stub_list.append(stub)

    def global_shuffle_by_patch(self, data_patch, date, concurrency):
        shuffle_time = len(data_patch) / concurrency + 1
        for i in range(shuffle_time):
            if i * concurrency >= len(data_patch):
                break
            pros = []
            end = min((i + 1) * concurrency, len(data_patch))
            patch_list = data_patch[i * concurrency:end]
            width = len(patch_list)
            for j in range(width):
                p = Process(
                    target=self.send_one_patch, args=(patch_list[j], date))
                pros.append(p)
            for p in pros:
                p.start()
            for p in pros:
                p.join()
            logging.info("shuffle round {} done.".format(i))

    def send_one_patch(self, patch, date):
        for line in patch:
            group = line.strip().split("\t")
            if len(group) != 3:
                continue
            data = data_server_pb2.Data()
            data.uid = group[0]
            data.date = date
            data.line = line.strip()
            stub_idx = self.uid_shard(data.uid)
            if stub_idx == -1:
                logging.info("send_one_patch continue for uid: %s" % data.uid)
                continue
            call_future = self.stub_list[stub_idx].SendData.future(data)
            u_num = call_future.result()

    def global_shuffle_by_file(self, filelist, concurrency):
        pass

    def set_load_data_into_patch_func(self, func):
        self.load_data_into_patch = func

    def get_local_files(self,
                        base_path,
                        date,
                        node_idx,
                        node_num,
                        hdfs_configs=None):
        full_path = "{}/{}".format(base_path, date)
        if os.path.exists(full_path):
            file_list = os.listdir(full_path)
            local_files = ["{}/{}".format(full_path, x) for x in file_list]
        elif hdfs_configs is not None:
            local_files = self.download_from_hdfs(hdfs_configs, base_path,
                                                  date, node_idx, node_num)
        else:
            local_files = []
        return local_files

    def download_from_hdfs(self, hdfs_configs, base_path, date, node_idx,
                           node_num):
        # return local filelist
        hdfs_client = HDFSClient("$HADOOP_HOME", hdfs_configs)
        multi_download(
            hdfs_client,
            "{}/{}".format(base_path, date),
            date,
            node_idx,
            node_num,
            multi_processes=30)
        filelist = os.listdir(date)
        files = ["{}/{}".format(date, fn) for fn in filelist]
        return files


def test_global_shuffle():
    data_client = DataClient()
    server_endpoints = ["127.0.0.1:{}".format(50050 + i) for i in range(10)]
    data_client.set_data_server_endpoints(server_endpoints)
    date = "0330"
    file_name = ["data_with_uid/part-01991"]
    with open(file_name[0]) as fin:
        for line in fin:
            group = line.strip().split("\t")
            uid = group[0]
            user_data_dict = data_client.get_data_by_uid(uid, date)


def test_set_param():
    data_client = DataClient()
    server_endpoints = ["127.0.0.1:{}".format(50050 + i) for i in range(10)]
    data_client.set_data_server_endpoints(server_endpoints)
    uid = ["1001", "10001", "100001", "101"]
    param_dict = {"w0": [1.0, 1.1, 1.2, 1.3], "b0": [1.1, 1.2, 1.3, 1.5]}
    for cur_i in uid:
        data_client.set_param_by_uid(cur_i, param_dict)


def test_get_param():
    data_client = DataClient()
    server_endpoints = ["127.0.0.1:{}".format(50050 + i) for i in range(10)]
    data_client.set_data_server_endpoints(server_endpoints)
    uid = ["1001", "10001", "100001", "101"]
    for cur_i in uid:
        param_dict = data_client.get_param_by_uid(cur_i)
        print(param_dict)


if __name__ == "__main__":
    #load_data_global_shuffle()
    #test_global_shuffle()
    test_set_param()
    test_get_param()
