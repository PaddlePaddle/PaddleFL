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

from utils.role_maker import FLSimRoleMaker
from clients import DataClient
from clients import SchedulerClient
from servers import DataServer
from servers import SchedulerServer
from multiprocessing import Process, Pool, Manager, Pipe, Lock
from utils.logger import logging
import pickle
import time
import numpy as np
import sys
import os


class SimulationFramework(object):
    def __init__(self, role_maker):
        self.data_client = None
        self.scheduler_client = None
        self.role_maker = role_maker
        # we suppose currently we train homogeneous model
        self.trainer = None
        # for sampling users
        self.sampler = None
        # for update global weights
        self.fl_optimizer = None
        # for samping users to test
        self.test_sampler = None
        self.profile_file = open("profile", "w")
        self.do_profile = True
        # for data downloading
        self.hdfs_configs = None

    def set_hdfs_configs(self, configs):
        self.hdfs_configs = configs

    def set_trainer(self, trainer):
        self.trainer = trainer

    def set_sampler(self, sampler):
        self.sampler = sampler

    def set_test_sampler(self, sampler):
        self.test_sampler = sampler

    def set_fl_optimizer(self, optimizer):
        self.fl_optimizer = optimizer

    def is_scheduler(self):
        return self.role_maker.is_global_scheduler()

    def is_simulator(self):
        return self.role_maker.is_simulator()

    def run_scheduler_service(self):
        if self.role_maker.is_global_scheduler():
            self._run_global_scheduler()

    def _barrier_simulators(self):
        self.role_maker.barrier_simulator()

    def _start_data_server(self, endpoint):
        data_server = DataServer()
        port = endpoint.split(":")[1]
        data_server.start(endpoint=port)

    def _run_global_scheduler(self):
        scheduler_server = SchedulerServer()
        endpoint = self.role_maker.get_global_scheduler_endpoint()
        port = endpoint.split(":")[1]
        scheduler_server.start(endpoint=port)

    def _get_data_services(self):
        data_server_endpoints = \
          self.role_maker.get_local_data_server_endpoint()
        data_server_pros = []
        for i, ep in enumerate(data_server_endpoints):
            p = Process(target=self._start_data_server, args=(ep, ))
            data_server_pros.append(p)

        return data_server_pros

    def _profile(self, func, *args, **kwargs):
        if self.do_profile:
            start = time.time()
            res = func(*args, **kwargs)
            end = time.time()
            self.profile_file.write("%s\t\t%f s\n" %
                                    (func.__name__, end - start))
            return res
        else:
            return func(*args, **kwargs)

    def _run_sim(self, date, sim_num_everyday=1):
        sim_idx = self.role_maker.simulator_idx()
        sim_num = self.role_maker.simulator_num()
        sim_all_trainer_run_time = 0
        sim_read_praram_and_optimize = 0
        for sim in range(sim_num_everyday):
            logging.info("sim id: %d" % sim)
            # sampler algorithm
            user_info_dict = self._profile(
                self.sampler.sample_user_list, self.scheduler_client, date,
                sim_idx, len(self.data_client.stub_list), sim_num)

            if self.do_profile:
                print("sim_idx: ", sim_idx)
                print("shard num: ", len(self.data_client.stub_list))
                print("sim_num: ", sim_num)
                print("user_info_dict: ", user_info_dict)

            global_param_dict = self._profile(
                self.scheduler_client.get_global_params)
            processes = []
            os.system("rm -rf _global_param")
            os.system("mkdir _global_param")
            start = time.time()
            for idx, user in enumerate(user_info_dict):
                arg_dict = {
                    "uid": str(user),
                    "date": date,
                    "data_endpoints":
                    self.role_maker.get_data_server_endpoints(),
                    "global_params": global_param_dict,
                    "user_param_names": self.trainer.get_user_param_names(),
                    "global_param_names":
                    self.trainer.get_global_param_names(),
                    "write_global_param_file":
                    "_global_param/process_%d" % idx,
                }
                p = Process(
                    target=self.trainer.train_one_user_func,
                    args=(arg_dict, self.trainer.trainer_config))
                p.start()
                processes.append(p)
            if self.do_profile:
                logging.info("wait processes to close")
            for i, p in enumerate(processes):
                processes[i].join()
            end = time.time()
            sim_all_trainer_run_time += (end - start)

            start = time.time()
            train_result = []
            new_global_param_by_user = {}
            training_sample_by_user = {}
            for i, p in enumerate(processes):
                param_dir = "_global_param/process_%d/" % i
                with open(param_dir + "/_info", "r") as f:
                    user, train_sample_num = pickle.load(f)
                param_dict = {}
                for f_name in os.listdir(os.path.join(param_dir, "params")):
                    f_path = os.path.join(param_dir, "params", f_name)
                    if os.path.isdir(f_path):  # layer
                        for layer_param in os.listdir(f_path):
                            layer_param_path = os.path.join(f_path,
                                                            layer_param)
                            with open(layer_param_path) as f:
                                param_dict["{}/{}".format(
                                    f_name, layer_param)] = np.load(f)
                    else:
                        with open(f_path) as f:
                            param_dict[f_name] = np.load(f)
                new_global_param_by_user[user] = param_dict
                training_sample_by_user[user] = train_sample_num

            self.fl_optimizer.update(training_sample_by_user,
                                     new_global_param_by_user,
                                     global_param_dict, self.scheduler_client)
            end = time.time()
            sim_read_praram_and_optimize += (end - start)
        if self.do_profile:
            self.profile_file.write("sim_all_trainer_run_time\t\t%f s\n" %
                                    sim_all_trainer_run_time)
            self.profile_file.write("sim_read_praram_and_optimize\t\t%f s\n" %
                                    sim_read_praram_and_optimize)

        logging.info("training done for date %s." % date)

    def _test(self, date):
        if self.trainer.infer_one_user_func is None:
            pass
        logging.info("doing test...")
        if self.test_sampler is None:
            logging.error("self.test_sampler should not be None when testing")

        sim_idx = self.role_maker.simulator_idx()
        sim_num = self.role_maker.simulator_num()
        user_info_dict = self.test_sampler.sample_user_list(
            self.scheduler_client,
            date,
            sim_idx,
            len(self.data_client.stub_list),
            sim_num, )
        if self.do_profile:
            print("test user info_dict: ", user_info_dict)
        global_param_dict = self.scheduler_client.get_global_params()

        def divide_chunks(l, n):
            for i in range(0, len(l), n):
                yield l[i:i + n]

        # at most 50 process for testing
        chunk_size = 50
        # at most 100 uid for testing
        max_test_uids = 100
        uid_chunks = divide_chunks(user_info_dict.keys(), chunk_size)
        os.system("rm -rf _test_result")
        os.system("mkdir _test_result")

        tested_uids = 0
        for uids in uid_chunks:
            if tested_uids >= max_test_uids:
                break
            processes = []
            for user in uids:
                arg_dict = {
                    "uid": str(user),
                    "date": date,
                    "data_endpoints":
                    self.role_maker.get_data_server_endpoints(),
                    "global_params": global_param_dict,
                    "user_param_names": self.trainer.get_user_param_names(),
                    "global_param_names":
                    self.trainer.get_global_param_names(),
                    "infer_result_dir": "_test_result/uid-%s" % user,
                }
                p = Process(
                    target=self.trainer.infer_one_user_func,
                    args=(arg_dict, self.trainer.trainer_config))
                p.start()
                processes.append(p)
            if self.do_profile:
                logging.info("wait test processes to close")
            for i, p in enumerate(processes):
                processes[i].join()
            tested_uids += chunk_size

        infer_results = []
        # only support one test metric now
        for uid in os.listdir("_test_result"):
            with open("_test_result/" + uid + "/res", 'r') as f:
                sample_cout, metric = f.readlines()[0].strip('\n').split('\t')
                infer_results.append((int(sample_cout), float(metric)))
        if sum([x[0] for x in infer_results]) == 0:
            logging.info("infer results: 0.0")
        else:
            count = sum([x[0] for x in infer_results])
            metric = sum([x[0] * x[1] for x in infer_results]) / count
            logging.info("infer results: %f" % metric)

    def _save_and_upload(self, date, fs_upload_path):
        if self.trainer.save_and_upload_func is None:
            return
        if fs_upload_path is None:
            return
        dfs_upload_path = fs_upload_path + date + "_" + str(
            self.role_maker.simulator_idx())
        global_param_dict = self.scheduler_client.get_global_params()
        arg_dict = {
            "date": date,
            "global_params": global_param_dict,
            "user_param_names": self.trainer.get_user_param_names(),
            "global_param_names": self.trainer.get_global_param_names(),
        }
        self.trainer.save_and_upload_func(
            arg_dict, self.trainer.trainer_config, dfs_upload_path)

    def run_simulation(self,
                       base_path,
                       dates,
                       fs_upload_path=None,
                       sim_num_everyday=1,
                       do_test=False,
                       test_skip_day=6):
        if not self.role_maker.is_simulator():
            pass
        data_services = self._get_data_services()
        for service in data_services:
            service.start()
        self._barrier_simulators()
        self.data_client = DataClient()
        self.data_client.set_load_data_into_patch_func(
            self.trainer.get_load_data_into_patch_func())
        self.data_client.set_data_server_endpoints(
            self.role_maker.get_data_server_endpoints())
        self.scheduler_client = SchedulerClient()
        self.scheduler_client.set_data_server_endpoints(
            self.role_maker.get_data_server_endpoints())
        self.scheduler_client.set_scheduler_server_endpoints(
            [self.role_maker.get_global_scheduler_endpoint()])
        logging.info("trainer config: ", self.trainer.trainer_config)
        self.trainer.prepare(do_test=do_test)

        if self.role_maker.simulator_idx() == 0:
            self.trainer.init_global_model(self.scheduler_client)
        self._barrier_simulators()

        for date_idx, date in enumerate(dates):
            if date_idx > 0:
                self.do_profile = False
                self.profile_file.close()
            logging.info("reading data for date: %s" % date)
            local_files = self._profile(
                self.data_client.get_local_files,
                base_path,
                date,
                self.role_maker.simulator_idx(),
                self.role_maker.simulator_num(),
                hdfs_configs=self.hdfs_configs)

            logging.info("loading data into patch for date: %s" % date)
            data_patch, local_user_dict = self._profile(
                self.data_client.load_data_into_patch, local_files, 10000)
            logging.info("shuffling data for date: %s" % date)
            self._profile(self.data_client.global_shuffle_by_patch, data_patch,
                          date, 30)

            logging.info("updating user inst num for date: %s" % date)
            self._profile(self.scheduler_client.update_user_inst_num, date,
                          local_user_dict)
            self.role_maker.barrier_simulator()

            if do_test and date_idx != 0 and date_idx % test_skip_day == 0:
                self._barrier_simulators()
                self._profile(self._test, date)
                self._barrier_simulators()
                self._profile(self._save_and_upload, date, fs_upload_path)

            self._run_sim(date, sim_num_everyday=sim_num_everyday)
            self.role_maker.barrier_simulator()
            logging.info("clear user data for date: %s" % date)
            self.data_client.clear_user_data(date)

        self._barrier_simulators()
        logging.info("training done all date.")
        logging.info("stoping scheduler")
        self.scheduler_client.stop_scheduler_server()
        for pro in data_services:
            pro.terminate()
        logging.info("after terminate for all server.")
