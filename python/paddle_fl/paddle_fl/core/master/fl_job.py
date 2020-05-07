#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
import paddle.fluid as fluid


class FLJobBase(object):
    """
    FLJobBase is fl job base class, responsible for save and load
    a federated learning job
    """

    def __init__(self):
        pass

    def _save_str_list(self, items, output):
        with open(output, "w") as fout:
            for item in items:
                fout.write(item + "\n")

    def _load_str_list(self, input_file):
        res = []
        with open(input_file, "r") as fin:
            for line in fin:
                res.append(line.strip())
        return res

    def _save_strategy(self, strategy, output_file):
        import pickle
        pickle.dump(strategy, open(output_file, "wb"))

    def _save_endpoints(self, endpoints, output_file):
        with open(output_file, "w") as fout:
            for ep in endpoints:
                fout.write(str(ep) + "\n")

    def _load_endpoints(self, input_file):
        ep_list = []
        with open(input_file, "r") as fin:
            for line in fin:
                ep_list.append(line.strip())
        return ep_list

    def _save_program(self, program, output_file):
        with open(output_file, "wb") as fout:
            fout.write(program.desc.serialize_to_string())

    def _save_readable_program(self, program, output_file):
        with open(output_file, "w") as fout:
            fout.write(str(program))

    def _load_program(self, input_file):
        with open(input_file, "rb") as fin:
            program_desc_str = fin.read()
            return fluid.Program.parse_from_string(program_desc_str)
        return None


class FLCompileTimeJob(FLJobBase):
    """
    FLCompileTimeJob is a container for compile time job in federated learning.
    trainer startup programs, trainer main programs and other trainer programs
    are in FLCompileTimeJob. Also, server main programs and server startup programs
    are in this class. FLCompileTimeJob has server endpoints for debugging as well
    """

    def __init__(self):
        self._trainer_startup_programs = []
        self._trainer_recv_programs = []
        self._trainer_main_programs = []
        self._trainer_send_programs = []
        self._server_startup_programs = []
        self._server_main_programs = []
        self._server_endpoints = []

    def set_strategy(self, strategy):
        self._strategy = strategy

    def set_server_endpoints(self, ps_endpoints):
        self._server_endpoints = ps_endpoints

    def set_feed_names(self, names):
        self._feed_names = names

    def set_target_names(self, names):
        self._target_names = names

    def save(self, folder=None):
        server_num = len(self._server_startup_programs)
        trainer_num = len(self._trainer_startup_programs)
        send_prog_num = len(self._trainer_send_programs)
        for i in range(server_num):
            server_folder = "%s/server%d" % (folder, i)
            os.system("mkdir -p %s" % server_folder)
            server_startup = self._server_startup_programs[i]
            server_main = self._server_main_programs[i]
            self._save_program(server_startup,
                               "%s/server.startup.program" % server_folder)
            self._save_program(server_main,
                               "%s/server.main.program" % server_folder)
            self._save_readable_program(server_startup,
                                        "%s/server.startup.program.txt" %
                                        server_folder)
            self._save_readable_program(
                server_main, "%s/server.main.program.txt" % server_folder)
            self._save_str_list(self._feed_names,
                                "%s/feed_names" % server_folder)
            self._save_str_list(self._target_names,
                                "%s/target_names" % server_folder)
            self._save_endpoints(self._server_endpoints,
                                 "%s/endpoints" % server_folder)
            self._save_strategy(self._strategy,
                                "%s/strategy.pkl" % server_folder)

        for i in range(trainer_num):
            trainer_folder = "%s/trainer%d" % (folder, i)
            os.system("mkdir -p %s" % trainer_folder)
            trainer_startup = self._trainer_startup_programs[i]
            trainer_main = self._trainer_main_programs[i]
            self._save_program(trainer_startup,
                               "%s/trainer.startup.program" % trainer_folder)
            self._save_program(trainer_main,
                               "%s/trainer.main.program" % trainer_folder)
            self._save_readable_program(trainer_startup,
                                        "%s/trainer.startup.program.txt" %
                                        trainer_folder)
            self._save_readable_program(
                trainer_main, "%s/trainer.main.program.txt" % trainer_folder)
            self._save_str_list(self._feed_names,
                                "%s/feed_names" % trainer_folder)
            self._save_str_list(self._target_names,
                                "%s/target_names" % trainer_folder)
            self._save_endpoints(self._server_endpoints,
                                 "%s/endpoints" % trainer_folder)
            self._save_strategy(self._strategy,
                                "%s/strategy.pkl" % trainer_folder)

        for i in range(send_prog_num):
            trainer_folder = "%s/trainer%d" % (folder, i)
            trainer_send = self._trainer_send_programs[i]
            trainer_recv = self._trainer_recv_programs[i]
            self._save_program(trainer_send,
                               "%s/trainer.send.program" % trainer_folder)
            self._save_program(trainer_recv,
                               "%s/trainer.recv.program" % trainer_folder)
            self._save_readable_program(
                trainer_send, "%s/trainer.send.program.txt" % trainer_folder)
            self._save_readable_program(
                trainer_recv, "%s/trainer.recv.program.txt" % trainer_folder)


class FLRunTimeJob(FLJobBase):
    """
    FLRunTimeJob is a contrainer for run time job in federated leanring.
    A trainer or a server can load FLRunTimeJob. Only necessary programs
     can be loaded in FLRunTimeJob
    """

    def __init__(self):
        self._trainer_startup_program = None
        self._trainer_recv_program = None
        self._trainer_main_program = None
        self._trainer_send_program = None
        self._server_startup_program = None
        self._server_main_program = None
        self._feed_names = None
        self._target_names = None
        self._scheduler_ep = None

    def _load_strategy(self, input_file):
        import pickle
        return pickle.load(open(input_file, "rb"))

    def load_trainer_job(self, folder=None, trainer_id=0):
        """
        Load trainer job given training folder and trainer id
        Currently, a trainer_id is assigned to a trainer node, and
        corresponding FL Job will be sent to the trainer node.

        Args:
            folder(str): FL Job folder name
            trainer_id(int): trainer index for current job

        Return:
            None
        """
        folder_name = "%s/trainer%d" % (folder, trainer_id)
        startup_fn = "%s/trainer.startup.program" % folder_name
        self._trainer_startup_program = self._load_program(startup_fn)

        main_fn = "%s/trainer.main.program" % folder_name
        self._trainer_main_program = self._load_program(main_fn)

        try:
            send_fn = "%s/trainer.send.program" % folder_name
            self._trainer_send_program = self._load_program(send_fn)

            recv_fn = "%s/trainer.recv.program" % folder_name
            self._trainer_recv_program = self._load_program(recv_fn)
        except:
            pass

        endpoints_fn = "%s/endpoints" % folder_name
        self._endpoints = self._load_endpoints(endpoints_fn)

        strategy_fn = "%s/strategy.pkl" % folder_name
        self._strategy = self._load_strategy(strategy_fn)

        feed_names_fn = "%s/feed_names" % folder_name
        self._feed_names = self._load_str_list(feed_names_fn)

        target_names_fn = "%s/target_names" % folder_name
        self._target_names = self._load_str_list(target_names_fn)

    def load_server_job(self, folder=None, server_id=0):
        """
        Load server job given training folder and server_id
        Currently, a server_id is assigned to a server node, and
        corresponding FL Job will be sent to the server node.

        Args:
            folder(str): FL Job folder name
            server_id(int): server index for current job

        Return:
            None
        """
        folder_name = "%s/server%d" % (folder, server_id)
        startup_fn = "%s/server.startup.program" % folder_name
        self._server_startup_program = self._load_program(startup_fn)

        main_fn = "%s/server.main.program" % folder_name
        self._server_main_program = self._load_program(main_fn)

        endpoints_fn = "%s/endpoints" % folder_name
        self._endpoints = self._load_endpoints(endpoints_fn)

        import pickle
        strategy_fn = "%s/strategy.pkl" % folder_name
        self._strategy = self._load_strategy(strategy_fn)
