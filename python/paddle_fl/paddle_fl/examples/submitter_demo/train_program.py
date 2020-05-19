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

import socket
import random
import zmq
import os
import tarfile
import paddle_fl as fl
import paddle.fluid as fluid
from paddle_fl.core.server.fl_server import FLServer
from paddle_fl.core.master.fl_job import FLRunTimeJob
from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
import numpy as np
import sys
import logging
import time

random_port = 64001
scheduler_conf = {}

#connect to scheduler and get the role and id of the endpoint
with open("scheduler.conf") as fin:
    for line in fin:
        line = line.strip()
        group = line.split("\t")
        scheduler_conf[group[0]] = group[1]

current_ip = socket.gethostbyname(socket.gethostname())
endpoint = "{}:{}".format(current_ip, random_port)
scheduler_ip = scheduler_conf["ENDPOINT"].split(":")
download_url = "{}:8080".format(scheduler_ip[0])
print(download_url)
context = zmq.Context()
zmq_socket = context.socket(zmq.REQ)
zmq_socket.connect("tcp://{}".format(scheduler_conf["ENDPOINT"]))
zmq_socket.send("ENDPOINT\t{}".format(endpoint))
message = zmq_socket.recv()
print(message)

message = ""

#download the config file from scheduler
while True:
    zmq_socket.send("GET_FL_JOB\t{}".format(endpoint))
    message = zmq_socket.recv()
    group = message.split("\t")
    if group[0] == "WAIT":
        continue
    else:
        os.system("wget {}/job_config/{}.tar.gz".format(download_url, message))
        print(message)
        break

os.system("ls")
os.system("gzip -d {}.tar.gz".format(message))
print("gzip finish")
os.system("tar -xf {}.tar".format(message))
os.system("ls")
zmq_socket.close()
print("close socket")

#program start
if 'server' in message:
    server = FLServer()
    server_id = 0
    job_path = "job_config"
    job = FLRunTimeJob()
    job.load_server_job(job_path, server_id)
    job._scheduler_ep = scheduler_conf["ENDPOINT"]
    server.set_server_job(job)
    server._current_ep = endpoint
    server.start()
else:

    def reader():
        for i in range(1000):
            data_dict = {}
            for i in range(3):
                data_dict[str(i)] = np.random.rand(1, 5).astype('float32')
        data_dict["label"] = np.random.randint(2, size=(1, 1)).astype('int64')
        yield data_dict

    trainer_id = message.split("trainer")[1]
    job_path = "job_config"
    job = FLRunTimeJob()
    job.load_trainer_job(job_path, int(trainer_id))
    job._scheduler_ep = scheduler_conf["ENDPOINT"]
    trainer = FLTrainerFactory().create_fl_trainer(job)
    trainer._current_ep = endpoint
    trainer.start()
    print(trainer._scheduler_ep, trainer._current_ep)
    output_folder = "fl_model"
    epoch_id = 0
    while not trainer.stop():
        print("epoch %d start train" % (epoch_id))
        step_i = 0
        for data in reader():
            trainer.run(feed=data, fetch=[])
            step_i += 1
            if step_i == trainer._step:
                break
        epoch_id += 1
        if epoch_id % 5 == 0:
            trainer.save_inference_program(output_folder)
