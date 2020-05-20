# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from utils.logger import logging


class FLSimRoleMaker(object):
    def __init__(self):
        from mpi4py import MPI
        self.MPI = MPI
        self._comm = MPI.COMM_WORLD

    def init_env(self, local_shard_num=1):
        if (local_shard_num > 250):
            logging.error("shard num must be less than or equal to 250")
            exit()
        self.free_ports = self._get_free_endpoints(local_shard_num, 40000)
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        self.free_endpoints = ["{}:{}".format(ip, x) for x in self.free_ports]
        self._comm.barrier()
        self.node_type = 1
        if self._comm.Get_rank() == 0:
            self.node_type = 0
        self.group_comm = self._comm.Split(self.node_type)
        self.all_endpoints = self._comm.allgather(self.free_endpoints)

    def simulator_num(self):
        return self._comm.Get_size() - 1

    def simulator_idx(self):
        return self._comm.Get_rank() - 1

    def get_global_scheduler_endpoint(self):
        return self.all_endpoints[0][0]

    def get_data_server_endpoints(self):
        #return self.all_endpoints[2:][::2]
        all_endpoints = []
        for eps in self.all_endpoints[1:]:
            all_endpoints.extend(eps[:len(eps) / 2])
        return all_endpoints

    def get_local_data_server_endpoint(self):
        if self._comm.Get_rank() < 1:
            return None
        local_endpoints = self.all_endpoints[self._comm.Get_rank()]
        return local_endpoints[:len(local_endpoints) / 2]

    def get_local_param_server_endpoint(self):
        if self._comm.Get_rank() < 1:
            return None
        local_endpoints = self.all_endpoints[self._comm.Get_rank()]
        return local_endpoints[len(local_endpoints) / 2:]

    def is_global_scheduler(self):
        rank = self._comm.Get_rank()
        return rank == 0

    def is_simulator(self):
        return self._comm.Get_rank() > 0

    def barrier_simulator(self):
        if self._comm.Get_rank() > 0:
            self.group_comm.barrier()

    def barrier(self):
        self.group_comm.barrier()

    def _get_free_endpoints(self, local_shard_num, start_port):
        import psutil
        conns = psutil.net_connections()
        x = [conn.laddr.port for conn in conns]
        free_endpoints = []
        step = 500
        start_range = start_port + self._comm.Get_rank() * step
        for i in range(start_range, start_range + step, 1):
            if i in x:
                continue
            if i > 65535:
                continue
            else:
                free_endpoints.append(i)
            if len(free_endpoints) == local_shard_num * 2:
                break
        return free_endpoints
