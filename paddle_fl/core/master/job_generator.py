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
from .fl_job import FLCompileTimeJob


class JobGenerator(object):
    """
    A JobGenerator is responsible for generating distributed federated
    learning configs. Before federated learning job starts, organizations
    need to define a deep learning model together to do horizontal federated
    learning.
    """

    def __init__(self):
        # worker num for federated learning
        self._worker_num = 0
        # startup program
        self._startup_prog = None
        # inner optimizer
        self._optimizer = \
            fluid.optimizer.SGD(learning_rate=0.001)
        self._feed_names = []
        self._target_names = []

    def set_optimizer(self, optimizer):
        """
        Set optimizer of current job
        """
        self._optimizer = optimizer

    def set_losses(self, losses):
        """
        Set losses of current job
        losses can be a list of loss so that we can do
        optimization on multiple losses
        """
        self._losses = losses

    def set_startup_program(self, startup=None):
        """
        set startup program for user defined program
        """
        if startup == None:
            startup = fluid.default_startup_program()
        self._startup_prog = startup

    def set_infer_feed_and_target_names(self, feed_names, target_names):
        if not isinstance(feed_names, list) or not isinstance(target_names,
                                                              list):
            raise ValueError(
                "input should be list in set_infer_feed_and_target_names")
        '''
        print(feed_names)
        print(target_names)
        for item in feed_names:
            if type(item) != str:
                raise ValueError("item in feed_names should be string")
        for item in target_names:
            if type(item) != str:
                raise ValueError("item in target_names should be string")
        '''
        self._feed_names = feed_names
        self._target_names = target_names

    def generate_fl_job(self,
                        fl_strategy,
                        server_endpoints=[],
                        worker_num=1,
                        output=None):
        """
        Generate Federated Learning Job, based on user defined configs

        Args:
            fl_strategy(FLStrategyBase): federated learning strategy defined by current federated users
            server_endpoints(List(str)): endpoints for federated server nodes
            worker_num(int): number of training nodes
            output(str): output directory of generated fl job

        Returns:
            None

        Examples:
            import paddle.fluid as fluid
            import paddle_fl as fl
            from paddle_fl.core.master.job_generator import JobGenerator
            from paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory

            input_x = fluid.layers.data(name="input_x", shape=[10], dtype="float32")
            label = fluid.layers.data(name="label", shape[1], dtype="int64")
            fc0 = fluid.layers.fc(input=input_x, size=2, act='sigmoid')
            cost = fluid.layers.cross_entropy(input=fc0, label=label)
            loss = fluid.layers.reduce_mean(cost)

            job_generator = JobGenerator()
            optimizer = fluid.optimizer.SGD(learning_rate=0.1)
            job_generator.set_optimizer(optimizer)
            job_generator.set_losses([loss])
            server_endpoints = [127.0.0.1:8181]
            worker_num = 10
            build_strategy = FLStrategyFactor()
            build_strategy.fed_avg = True
            strategy = build_strategy.create_fl_strategy()
            job_output_dir = "fl_job_config"
            job_generator.generate_fl_job(strategy,
                                          server_endpoints=server_endpoints,
                                          worker_num=1,
                                          output=output)

        """
        local_job = FLCompileTimeJob()
        assert len(self._losses) > 0
        assert self._startup_prog != None
        assert fl_strategy != None
        assert output != None
        fl_strategy.minimize(self._optimizer, self._losses)

        # strategy can generate startup and main program
        # of a single worker and servers
        for trainer_id in range(worker_num):
            startup_program = self._startup_prog.clone()
            main_program = self._losses[0].block.program.clone()
            fl_strategy._build_trainer_program_for_job(
                trainer_id,
                program=main_program,
                ps_endpoints=server_endpoints,
                trainers=worker_num,
                sync_mode=True,
                startup_program=startup_program,
                job=local_job)

        startup_program = self._startup_prog.clone()
        main_program = self._losses[0].block.program.clone()
        fl_strategy._build_server_programs_for_job(
            program=main_program,
            ps_endpoints=server_endpoints,
            trainers=worker_num,
            sync_mode=True,
            startup_program=startup_program,
            job=local_job)

        local_job.set_feed_names(self._feed_names)
        local_job.set_target_names(self._target_names)
        local_job.set_strategy(fl_strategy)
        local_job.save(output)

    def generate_fl_job_for_k8s(self,
                                fl_strategy,
                                server_pod_endpoints=[],
                                server_service_endpoints=[],
                                worker_num=1,
                                output=None):

        local_job = FLCompileTimeJob()
        assert len(self._losses) > 0
        assert self._startup_prog != None
        assert fl_strategy != None
        assert output != None
        fl_strategy.minimize(self._optimizer, self._losses)

        # strategy can generate startup and main program
        # of a single worker and servers
        for trainer_id in range(worker_num):
            startup_program = self._startup_prog.clone()
            main_program = self._losses[0].block.program.clone()
            fl_strategy._build_trainer_program_for_job(
                trainer_id,
                program=main_program,
                ps_endpoints=server_service_endpoints,
                trainers=worker_num,
                sync_mode=True,
                startup_program=startup_program,
                job=local_job)

        startup_program = self._startup_prog.clone()
        main_program = self._losses[0].block.program.clone()
        fl_strategy._build_server_programs_for_job(
            program=main_program,
            ps_endpoints=server_pod_endpoints,
            trainers=worker_num,
            sync_mode=True,
            startup_program=startup_program,
            job=local_job)

        local_job.set_feed_names(self._feed_names)
        local_job.set_target_names(self._target_names)
        local_job.set_strategy(fl_strategy)
        local_job.save(output)

    def save_program(self, program_path, input_list, hidden_vars, loss):
        if not os.path.exists(program_path):
            os.makedirs(program_path)
        main_program_str = fluid.default_main_program(
        ).desc.serialize_to_string()
        startup_program_str = fluid.default_startup_program(
        ).desc.serialize_to_string()
        params = fluid.default_main_program().global_block().all_parameters()
        para_info = []
        for pa in params:
            para_info.append(pa.name)
        with open(program_path + '/input_names', 'w') as fout:
            for input in input_list:
                fout.write("%s\n" % input.name)
        with open(program_path + '/hidden_vars', 'w') as fout:
            for var in hidden_vars:
                fout.write("%s:%s\n" % (var[0], var[1].name))
        with open(program_path + '/para_info', 'w') as fout:
            for item in para_info:
                fout.write("%s\n" % item)
        with open(program_path + '/startup_program', "wb") as fout:
            fout.write(startup_program_str)
        with open(program_path + '/main_program', "wb") as fout:
            fout.write(main_program_str)
        with open(program_path + '/loss_name', 'w') as fout:
            fout.write(loss.name)

    def generate_fl_job_from_program(self, strategy, endpoints, worker_num,
                                     program_input, output):
        local_job = FLCompileTimeJob()
        with open(program_input + '/startup_program', "rb") as fin:
            program_desc_str = fin.read()
            new_startup = fluid.Program.parse_from_string(program_desc_str)

        with open(program_input + '/main_program', "rb") as fin:
            program_desc_str = fin.read()
            new_main = fluid.Program.parse_from_string(program_desc_str)

        para_list = []
        with open(program_input + '/para_info', 'r') as fin:
            for line in fin:
                current_para = line[:-1]
                para_list.append(current_para)

        input_list = []
        with open(program_input + '/input_names', 'r') as fin:
            for line in fin:
                current_input = line[:-1]
                input_list.append(current_input)

        with open(program_input + '/loss_name', 'r') as fin:
            loss_name = fin.read()

        for item in para_list:
            para = new_main.global_block().var(item)
            para.regularizer = None
            para.optimize_attr = {'learning_rate': 1.0}
            para.trainable = True
        exe = fluid.Executor(fluid.CPUPlace())
        loss = None
        for var in new_main.list_vars():
            if var.name == loss_name:
                loss = var
        with fluid.program_guard(new_main, new_startup):
            optimizer = fluid.optimizer.SGD(learning_rate=0.1,
                                            parameter_list=para_list)
            exe.run(new_startup)
            strategy.minimize(optimizer, loss)

        for trainer_id in range(worker_num):
            startup_program = new_startup.clone()
            main_program = loss.block.program.clone()
            strategy._build_trainer_program_for_job(
                trainer_id,
                program=main_program,
                ps_endpoints=endpoints,
                trainers=worker_num,
                sync_mode=True,
                startup_program=startup_program,
                job=local_job)

        startup_program = new_startup.clone()
        main_program = loss.block.program.clone()
        strategy._build_server_programs_for_job(
            program=main_program,
            ps_endpoints=endpoints,
            trainers=worker_num,
            sync_mode=True,
            startup_program=startup_program,
            job=local_job)

        local_job.set_feed_names(input_list)
        local_job.set_target_names([loss.name])
        local_job.set_strategy(strategy)
        local_job.save(output)
