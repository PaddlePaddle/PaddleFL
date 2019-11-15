import zmq
import time
import random

def recv_and_parse_kv(socket):
    message = socket.recv()
    group = message.split("\t")
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
        self.socket.connect("tcp://127.0.0.1:9091")
        self.current_ep = current_ep

    def connect_scheduler(self):
        self.socket.send("SERVER_EP\t{}".format(self.current_ep))
        self.socket.recv()


class FLWorkerAgent(object):
    def __init__(self, scheduler_ep, current_ep):
        self.scheduler_ep = scheduler_ep
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect("tcp://127.0.0.1:9091")
        self.current_ep = current_ep

    def connect_scheduler(self):
        self.socket.send("WORKER_EP\t{}".format(self.current_ep))
        self.socket.recv()

    def finish_training(self):
        self.socket.send("FINISH\t{}".format(self.current_ep))
        key, value = recv_and_parse_kv(self.socket)
        if key == "WAIT":
            time.sleep(3)

    def can_join_training(self):
        self.socket.send("JOIN\t{}".format(self.current_ep))
        key, value = recv_and_parse_kv(self.socket)

        if key == "ACCEPT":
            return True
        elif key == "REJECT":
            return False
        return False



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
                self.socket.send("INIT\t{}".format(value))
            if key == SERVER_EP:
                self.fl_servers.append(value)
                self.socket.send("INIT\t{}".format(value))
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
                            self.socket.send("ACCEPT\t0")
                            continue
                else:
                    ready_workers.append(value)
                self.socket.send("REJECT\t0")

                if len(ready_workers) == len(self.fl_workers):
                    all_ready_to_train = True

            all_finish_training = False
            finish_training_dict = {}
            while not all_finish_training:
                key, value = recv_and_parse_kv(self.socket)
                if key == "FINISH":
                    finish_training_dict[value] = 1
                    self.socket.send("WAIT\t0")
                else:
                    self.socket.send("REJECT\t0")
                if len(finish_training_dict) == len(worker_dict):
                    all_finish_training = True
            time.sleep(5)

