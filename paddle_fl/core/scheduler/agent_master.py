import zmq
import time
import random


def recv_and_parse_kv(socket):
    message = socket.recv()
    socket.send("alive")
    group = message.split("\t")
    print(group)
    assert len(group) == 2
    return group[0], group[1]

WORKER_EP = "WORKER_EP"
SERVER_EP = "SERVER_EP"

class FLAgent(object):
    def __init__(self, scheduler_ep, current_ep):
        self.scheduler_ep = scheduler_ep
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.current_ep = current_ep

    def connect_scheduler(self):
        self.socket.send("WORKER_EP\t{}".format(self.current_ep))
        self.socket.recv()

    def can_join_training(self):
        self.socket.send("JOIN\t{}".format(self.current_ep))
        self.socket.recv()


class FLScheduler(object):
    def __init__(self, worker_num, server_num, port=9091):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:{}".format(port))
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
            if key == SERVER_EP:
                self.fl_servers.append(value)
            if len(self.fl_workers) == self.worker_num and \
               len(self.fl_servers) == self.server_num:
                ready = True

        print("FL training environment started")
        print("fl workers endpoints")
        print(self.fl_workers)
        print("fl servers endpoints")
        print(self.fl_servers)

    def start_fl_step(self):
        # random select some fl_workers here
        random.shuffle(self.workers)
        worker_dict = {}
        for worker in self.workers[:self.sample_worker_num]:
            worker_dict[worker] = 0
        ready = False
        ready_workers = []
        while not ready:
            key, value = recv_and_parse_kv(self.socket)
            if key == "JOIN":
                if value in worker_dict:
                    if worker_dict[value] == 0:
                        ready_workers.append(value)
                        worker_dict[value] = 1
            if len(ready_workers) == len(worker_dict):
                ready = True
        start_workers = []
        while len(start_workers) != len(ready_workers):
            key, value = recv_and_parse_kv(self.socket)
            if key == "REQUEST_START":
                if value in ready_workers:
                    start_workers.append(value)
                    socket.send("ACCEPT_START")
                    continue
            else:
                socket.send("alive")

