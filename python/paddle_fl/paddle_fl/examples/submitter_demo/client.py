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

import os
import socket
import random
import zmq
import time
import sys
from paddle_fl.core.submitter.client_base import HPCClient
from paddle_fl.core.scheduler.agent_master import FLScheduler
import paddle.fluid as fluid
from paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
from model import Model
import tarfile

#random_port = random.randint(60001, 64001)
random_port = 64001
print(random_port)
current_ip = socket.gethostbyname(socket.gethostname())
endpoints = "{}:{}".format(current_ip, random_port)
#start a web server for remote endpoints to download their config 
os.system("python -m SimpleHTTPServer 8080 &")
#os.system("python -m http.server 8080 &")
if os.path.exists("job_config"):
    os.system("rm -rf job_config")
if os.path.exists("package"):
    os.system("rm -rf package")
os.system("mkdir package")
os.system("cp train_program.py package")
with open("package/scheduler.conf", "w") as fout:
    fout.write("ENDPOINT\t{}\n".format(endpoints))

# submit a job with current endpoint

default_dict = {
    "task_name": "",
    "hdfs_path": "",
    "ugi": "",
    "worker_nodes": 5,
    "server_nodes": 1,
    "hadoop_home": "/path/to/hadoop",
    "hpc_home": "/path/to/hpc",
    "package_path": "./package",
    "priority": "high",
    "queue": "",
    "server": "",
    "mpi_node_mem": 11000,
    "pcpu": 180,
    "python_tar": "./python.tar.gz",
    "wheel": "./paddlepaddle-0.0.0-cp27-cp27mu-linux_x86_64-0.whl"
}


def load_conf(conf_file, local_dict):
    with open(conf_file) as fin:
        for line in fin:
            group = line.strip().split("=")
            if len(group) != 2:
                continue
            local_dict[group[0]] = group[1]
    return local_dict


client = HPCClient()
default_dict = load_conf(sys.argv[1], default_dict)

client.submit(
    task_name=default_dict["task_name"],
    hdfs_path=default_dict["hdfs_path"],
    ugi=default_dict["ugi"],
    hdfs_output=default_dict["hdfs_output"],
    worker_nodes=default_dict["worker_nodes"],
    server_nodes=default_dict["server_nodes"],
    hadoop_home=default_dict["hadoop_home"],
    hpc_home=default_dict["hpc_home"],
    train_cmd=default_dict["train_cmd"],
    monitor_cmd=default_dict["monitor_cmd"],
    package_path=default_dict["package_path"],
    priority=default_dict["priority"],
    queue=default_dict["queue"],
    server=default_dict["server"],
    mpi_node_mem=default_dict["mpi_node_mem"],
    pcpu=default_dict["pcpu"],
    python_tar=default_dict["python_tar"],
    wheel=default_dict["wheel"])

print("submit mpi job done.")

# start scheduler and receive the ip of allocated endpoints
context = zmq.Context()
zmq_socket = context.socket(zmq.REP)
zmq_socket.bind("tcp://{}:{}".format(current_ip, random_port))

print("binding tcp://{}:{}".format(current_ip, random_port))

all_ips_ready = False

ip_list = []

scheduler = FLScheduler(
    int(default_dict["worker_nodes"]),
    int(default_dict["server_nodes"]),
    port=random_port,
    socket=zmq_socket)

scheduler.set_sample_worker_num(int(default_dict["worker_nodes"]))

print("going to wait all ips ready")

while not all_ips_ready:
    message = zmq_socket.recv()
    group = message.split("\t")
    if group[0] == "ENDPOINT":
        ip_list.append(group[1])
        zmq_socket.send("ACCEPT\t{}".format(group[1]))
    else:
        zmq_socket.send("WAIT\t0")
    if len(ip_list) == \
       int(default_dict["worker_nodes"]) + \
       int(default_dict["server_nodes"]):
        all_ips_ready = True

print("all worker ips are collected")
print(ip_list)

#allocate the role of each endpoint and their ids
ip_role = {}
for i in range(len(ip_list)):
    if i < int(default_dict["server_nodes"]):
        ip_role[ip_list[i]] = 'server%d' % i
    else:
        ip_role[ip_list[i]] = 'trainer%d' % (
            i - int(default_dict["server_nodes"]))
print(ip_role)


def job_generate():
    #generate a fl job which is the same as fl_master
    inputs = [fluid.layers.data( \
                name=str(slot_id), shape=[5],
                dtype="float32")
               for slot_id in range(3)]
    label = fluid.layers.data( \
                name="label",
                shape=[1],
                dtype='int64')

    model = Model()
    model.mlp(inputs, label)

    job_generator = JobGenerator()
    optimizer = fluid.optimizer.SGD(learning_rate=0.1)
    job_generator.set_optimizer(optimizer)
    job_generator.set_losses([model.loss])
    job_generator.set_startup_program(model.startup_program)
    job_generator.set_infer_feed_and_target_names([x.name for x in inputs],
                                                  [model.predict.name])

    build_strategy = FLStrategyFactory()
    build_strategy.fed_avg = True
    build_strategy.inner_step = 10
    strategy = build_strategy.create_fl_strategy()

    # endpoints will be collected through the cluster
    # in this example, we suppose endpoints have been collected
    server_ip = ["{}".format(ip_list[0])]

    output = "job_config"
    job_generator.generate_fl_job(
        strategy,
        server_endpoints=server_ip,
        worker_num=int(default_dict["worker_nodes"]),
        output=output)

    file_list = os.listdir(output)
    for file in file_list:
        tar = tarfile.open('{}/{}.tar.gz'.format(output, file), 'w:gz')
        for root, dir, files in os.walk("{}/{}".format(output, file)):
            for f in files:
                fullpath = os.path.join(root, f)
                tar.add(fullpath)
        tar.close()


job_generate()

#send the allocated rolls to the remote endpoints
all_job_sent = False
download_job = []
while not all_job_sent:
    message = zmq_socket.recv()
    group = message.split("\t")
    if group[0] == "GET_FL_JOB":
        download_job.append(group[1])
        zmq_socket.send(ip_role[group[1]])
    else:
        zmq_socket.send("WAIT\t0")
    if len(download_job) == len(ip_list):
        all_job_sent = True

#start training
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
