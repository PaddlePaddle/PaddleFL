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
        if not isinstance(feed_names, list) or not isinstance(target_names, list):
            raise ValueError("input should be list in set_infer_feed_and_target_names")
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
                trainer_id, program=main_program,
                ps_endpoints=server_endpoints, trainers=worker_num,
                sync_mode=True, startup_program=startup_program,
                job=local_job)

        startup_program = self._startup_prog.clone()
        main_program = self._losses[0].block.program.clone()
        fl_strategy._build_server_programs_for_job(
            program=main_program, ps_endpoints=server_endpoints,
            trainers=worker_num, sync_mode=True,
            startup_program=startup_program, job=local_job)

        local_job.set_feed_names(self._feed_names)
        local_job.set_target_names(self._target_names)
        local_job.set_strategy(fl_strategy)
        local_job.save(output)
