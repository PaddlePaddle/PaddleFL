# -*- coding: utf-8 -*-
from __future__ import print_function
from concurrent import futures
import scheduler_server_pb2
import scheduler_server_pb2_grpc
import grpc
import numpy as np
import random
import sys, os
import time
import xxhash
from utils.logger import logging


class SchedulerServerServicer(object):
    def __init__(self):
        self.global_param_dict = {}
        self.uid_inst_num_dict = {}
        self.shard_id_dict = {}

    def uid_shard(self, uid, shard_num):
        try:
            # print("uid_shard uid: %s" % uid)
            uid_hash = xxhash.xxh32(str(uid), seed=101).intdigest()
        except:
            return -1
        shard_idx = uid_hash % shard_num
        return shard_idx

    def is_test_uid(self, uid):
        return xxhash.xxh32(str(uid), seed=222).intdigest() % 100 == 3

    # we suppose shard num will not be changed during one training job
    # but can be changed with another job
    # so we send shard num every time we update user inst num
    def UpdateUserInstNum(self, request, context):
        shard_num = request.shard_num
        date = request.date
        if date not in self.uid_inst_num_dict:
            self.uid_inst_num_dict[date] = {}
        if date not in self.shard_id_dict:
            self.shard_id_dict[date] = {}
        for user in request.inst_nums:
            shard_id = self.uid_shard(user.uid, shard_num)
            if shard_id == -1:
                logging.info("UpdateUserInstNum continue")
                continue
            if user.uid in self.uid_inst_num_dict[date]:
                self.uid_inst_num_dict[date][user.uid] += user.inst_num
            else:
                self.uid_inst_num_dict[date][user.uid] = user.inst_num
            if shard_id not in self.shard_id_dict[date]:
                self.shard_id_dict[date][shard_id] = [user.uid]
            else:
                self.shard_id_dict[date][shard_id].append(user.uid)
        res = scheduler_server_pb2.Res()
        res.err_code = 0
        return res

    '''
    SampleUsersToTrain: 
        request.node_idx: from which worker node the request is from
        request.sample_num: how many users do we need to sample
        request.shard_num: total shard number of this task
        request.node_num: total number of training node
    '''

    def SampleUsersToTrain(self, request, context):
        node_idx = request.node_idx
        sample_num = request.sample_num
        shard_num = request.shard_num
        node_num = request.node_num
        date = request.date
        shard_per_node = shard_num / node_num
        begin_idx = node_idx * shard_per_node
        min_ins_num = request.min_ins_num

        uid_list = []
        i = 0
        while i < sample_num:
            shard_idx = begin_idx + random.randint(0, shard_per_node)
            if shard_idx not in self.shard_id_dict[date]:
                continue
            sample_idx = random.randint(
                0, len(self.shard_id_dict[date][shard_idx]) - 1)
            uid = self.shard_id_dict[date][shard_idx][sample_idx]
            if self.uid_inst_num_dict[date][uid] < min_ins_num:
                continue
            uid_list.append(uid)
            i += 1

        info = scheduler_server_pb2.UserInstInfo()
        for uid in uid_list:
            inst_num = scheduler_server_pb2.UserInstNum()
            inst_num.uid = uid
            inst_num.inst_num = self.uid_inst_num_dict[date][uid]
            info.inst_nums.extend([inst_num])
        return info

    def SampleUsersWithHash(self, request, context):
        node_idx = request.node_idx
        sample_num = request.sample_num
        shard_num = request.shard_num
        node_num = request.node_num
        date = request.date
        shard_per_node = shard_num / node_num
        begin_idx = node_idx * shard_per_node

        uid_list = []
        i = 0
        while i < sample_num:
            shard_idx = begin_idx + random.randint(0, shard_per_node)
            if shard_idx not in self.shard_id_dict[date]:
                continue
            sample_idx = random.randint(
                0, len(self.shard_id_dict[date][shard_idx]) - 1)
            uid = self.shard_id_dict[date][shard_idx][sample_idx]
            if not self.is_test_uid(uid):
                continue
            uid_list.append(uid)
            i += 1

        info = scheduler_server_pb2.UserInstInfo()
        for uid in uid_list:
            inst_num = scheduler_server_pb2.UserInstNum()
            inst_num.uid = uid
            inst_num.inst_num = self.uid_inst_num_dict[date][uid]
            info.inst_nums.extend([inst_num])
        return info

    def SampleUsersToTest(self, request, context):
        node_idx = request.node_idx
        shard_num = request.shard_num
        node_num = request.node_num
        date = request.date
        shard_per_node = shard_num / node_num
        shard_begin_idx = node_idx * shard_per_node
        shard_end_idx = (node_idx + 1) * shard_per_node
        uid_list = []

        for shard_idx in range(shard_begin_idx, shard_end_idx):
            for uid in self.shard_id_dict[date][shard_idx]:
                if self.is_test_uid(uid):
                    uid_list.append(uid)
        info = scheduler_server_pb2.UserInstInfo()
        for uid in uid_list:
            inst_num = scheduler_server_pb2.UserInstNum()
            inst_num.uid = uid
            inst_num.inst_num = self.uid_inst_num_dict[date][uid]
            info.inst_nums.extend([inst_num])
        return info

    def FixedUsersToTrain(self, request, context):
        node_idx = request.node_idx
        sample_num = request.sample_num
        shard_num = request.shard_num
        node_num = request.node_num
        date = request.date
        begin_idx = node_idx * shard_num

        shard_per_node = shard_num / node_num

        uid_list = []
        i = 0
        assert (sample_num <= 100)
        with open("data/test_user_100.txt") as f:
            for line in f.readlines():
                uid_list.append(line.strip())
        # uid_list.extend(["somebody", "nobody"])
        info = scheduler_server_pb2.UserInstInfo()
        for uid in uid_list[:sample_num]:
            inst_num = scheduler_server_pb2.UserInstNum()
            inst_num.uid = uid
            inst_num.inst_num = self.uid_inst_num_dict[date][uid]
            info.inst_nums.extend([inst_num])
        return info

    def GetGlobalParams(self, request, context):
        if self.global_param_dict == {}:
            logging.debug("global param has not been initialized")
            return
        # logging.info("node {} is asking for global params".format(request.node_idx))
        global_param_pb = scheduler_server_pb2.GlobalParams()
        for key in self.global_param_dict:
            param = scheduler_server_pb2.Param()
            param.name = key
            var, shape = self.global_param_dict[key]
            param.weight.extend(var)
            # print("GetGlobalParams before param.shape")
            param.shape.extend(shape)
            # print("GetGlobalParams after param.shape")
            global_param_pb.global_params.extend([param])
        # print("finish GetGlobalParams")
        return global_param_pb

    def UpdateGlobalParams(self, request, context):
        for param in request.global_params:
            self.global_param_dict[param.name] = [param.weight, param.shape]
        res = scheduler_server_pb2.Res()
        res.err_code = 0
        return res

    def FedAvgUpdate(self, request, context):
        for param in request.global_params:
            old_param, shape = self.global_param_dict[param.name]
            for idx, item in enumerate(old_param):
                old_param[idx] += param.weight[idx]
        res = scheduler_server_pb2.Res()
        res.err_code = 0
        return res

    def Exit(self, request, context):
        with open("_shutdown_scheduler", "w") as f:
            f.write("_shutdown_scheduler\n")
        res = scheduler_server_pb2.Res()
        res.err_code = 0
        return res


class SchedulerServer(object):
    def __init__(self):
        pass

    def start(self, max_workers=1000, concurrency=100, endpoint=""):
        if endpoint == "":
            logging.info("You should specify endpoint in start function")
            return
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[('grpc.max_send_message_length', 1024 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 1024 * 1024 * 1024)],
            maximum_concurrent_rpcs=concurrency)
        scheduler_server_pb2_grpc.add_SchedulerServerServicer_to_server(
            SchedulerServerServicer(), server)
        # print("SchedulerServer add endpoint: ", '[::]:{}'.format(endpoint))
        server.add_insecure_port('[::]:{}'.format(endpoint))
        server.start()
        logging.info("server started")
        os.system("rm _shutdown_scheduler")
        while (not os.path.isfile("_shutdown_scheduler")):
            time.sleep(10)


if __name__ == "__main__":
    scheduler_server = SchedulerServer()
    scheduler_server.start(endpoint=60001)
