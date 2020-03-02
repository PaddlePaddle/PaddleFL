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

import zmq
import time
import random


def recv_and_parse_kv(socket):
    message = socket.recv()
    group = message.decode().split("\t")
    if group[0] == "alive":
        return group[0], "0"
    else:
        return group[0], group[1]


WORKER_EP = "WORKER_EP"
SERVER_EP = "SERVER_EP"


class FLServerAgent(object):
    def __init__(self, scheduler_ep, current_ep):
        self.scheduler_ep = scheduler_ep
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://{}".format(scheduler_ep))
        self.current_ep = current_ep

    def connect_scheduler(self):
        while True:
            self.socket.send_string("SERVER_EP\t{}".format(self.current_ep))
            message = self.socket.recv()
            group = message.decode().split("\t")
            if group[0] == 'INIT':
                break


class FLWorkerAgent(object):
    def __init__(self, scheduler_ep, current_ep):
        self.scheduler_ep = scheduler_ep
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://{}".format(scheduler_ep))
        self.current_ep = current_ep

    def connect_scheduler(self):
        while True:
            self.socket.send_string("WORKER_EP\t{}".format(self.current_ep))
            message = self.socket.recv()
            group = message.decode().split("\t")
            if group[0] == 'INIT':
                break

    def finish_training(self):
        self.socket.send_string("FINISH\t{}".format(self.current_ep))
        key, value = recv_and_parse_kv(self.socket)
        if key == "WAIT":
            time.sleep(3)
            return True
        return False

    def can_join_training(self):
        self.socket.send_string("JOIN\t{}".format(self.current_ep))
        key, value = recv_and_parse_kv(self.socket)

        if key == "ACCEPT":
            return True
        elif key == "REJECT":
            return False
        return False


class FLScheduler(object):
    def __init__(self, worker_num, server_num, port=9091, socket=None):
        self.context = zmq.Context()
        if socket == None:
            self.socket = self.context.socket(zmq.REP)
            self.socket.bind("tcp://*:{}".format(port))
        else:
            self.socket = socket
        self.worker_num = worker_num
        self.server_num = server_num
        self.sample_worker_num = 0
        self.fl_workers = []
        self.fl_servers = []

    def set_sample_worker_num(self, sample_worker_num=0):
        if sample_worker_num == 0:
            self.sample_worker_num = int(self.worker_num * 0.1)
        else:
            self.sample_worker_num = sample_worker_num

    def init_env(self):
        ready = False
        while not ready:
            key, value = recv_and_parse_kv(self.socket)
            if key == WORKER_EP:
                self.fl_workers.append(value)
                self.socket.send_string("INIT\t{}".format(value))
            elif key == SERVER_EP:
                self.fl_servers.append(value)
                self.socket.send_string("INIT\t{}".format(value))
            else:
                time.sleep(3)
                self.socket.send_string("REJECT\t0")
            if len(self.fl_workers) == self.worker_num and \
               len(self.fl_servers) == self.server_num:
                ready = True

    def start_fl_training(self):
        # loop until training is done
        while True:
            random.shuffle(self.fl_workers)
            worker_dict = {}
            for worker in self.fl_workers[:self.sample_worker_num]:
                worker_dict[worker] = 0

            ready_workers = []
            all_ready_to_train = False
            while not all_ready_to_train:
                key, value = recv_and_parse_kv(self.socket)
                if key == "JOIN":
                    if value in worker_dict:
                        if worker_dict[value] == 0:
                            ready_workers.append(value)
                            worker_dict[value] = 1
                            self.socket.send_string("ACCEPT\t0")
                            continue
                    else:
                        if value not in ready_workers:
                            ready_workers.append(value)
                self.socket.send_string("REJECT\t0")
                if len(ready_workers) == len(self.fl_workers):
                    all_ready_to_train = True

            all_finish_training = False
            finish_training_dict = {}
            while not all_finish_training:
                key, value = recv_and_parse_kv(self.socket)
                if key == "FINISH":
                    finish_training_dict[value] = 1
                    self.socket.send_string("WAIT\t0")
                else:
                    self.socket.send_string("REJECT\t0")
                if len(finish_training_dict) == len(worker_dict):
                    all_finish_training = True
            time.sleep(5)
