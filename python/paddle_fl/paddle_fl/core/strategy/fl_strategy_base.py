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

from .fl_distribute_transpiler import FLDistributeTranspiler
from paddle.fluid.optimizer import SGD
import paddle.fluid as fluid


class FLStrategyFactory(object):
    """
    FLStrategyFactory is a FLStrategy builder
    Users can define strategy config to create different FLStrategy
    """

    def __init__(self):
        self._fed_avg = False
        self._dpsgd = False
        self._sec_agg = False
        self._inner_step = 1

    @property
    def fed_avg(self):
        return self._fed_avg

    @fed_avg.setter
    def fed_avg(self, s):
        self._fed_avg = s

    @property
    def dpsgd(self):
        return self._dpsgd

    @fed_avg.setter
    def dpsgd(self, s):
        self._dpsgd = s

    @property
    def sec_agg(self):
        return self._sec_agg

    @sec_agg.setter
    def sec_agg(self, s):
        self._sec_agg = s

    @property
    def inner_step(self):
        return self._inner_step

    @inner_step.setter
    def inner_step(self, s):
        self._inner_step = s

    def create_fl_strategy(self):
        """
        The factory method
        """
        if self._fed_avg == True:
            strategy = FedAvgStrategy()
            strategy._fed_avg = True
            strategy._dpsgd = False
            strategy._sec_agg = False
        elif self._dpsgd == True:
            strategy = DPSGDStrategy()
            strategy._fed_avg = False
            strategy._dpsgd = True
            strategy._sec_agg = False
        elif self._sec_agg == True:
            strategy = SecAggStrategy()
            strategy._fed_avg = False
            strategy._dpsgd = False
            strategy._sec_agg = True
        strategy._inner_step = self._inner_step
        return strategy


class FLStrategyBase(object):
    """
    FLStrategyBase is federated learning algorithm container
    """

    def __init__(self):
        self._fed_avg = False
        self._dpsgd = False
        self._inner_step = 1
        pass

    def minimize(self, optimizer=None, losses=[]):
        """
        minmize can do minimization as paddle.fluid.Optimizer.minimize does
        this function can be overloaded so that for some FLStrategy, the
        program should be transpiled before minimize

        Args:
            optimizer(paddle.fluid.optimizer): the user defined optimizer
            losses(List(Variable)): list of loss variables in paddle.fluid
        """
        for loss in losses:
            optimizer.minimize(loss)

    def _build_trainer_program_for_job(self,
                                       trainer_id=0,
                                       program=None,
                                       ps_endpoints=[],
                                       trainers=0,
                                       sync_mode=True,
                                       startup_program=None,
                                       job=None):
        pass

    def _build_server_programs_for_job(self,
                                       program=None,
                                       ps_endpoints=[],
                                       trainers=0,
                                       sync_mode=True,
                                       startup_program=None,
                                       job=None):
        pass


class DPSGDStrategy(FLStrategyBase):
    """
    DPSGDStrategy: Deep Learning with Differential Privacy. 2016
    """

    def __init__(self):
        super(DPSGDStrategy, self).__init__()

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, s):
        self._learning_rate = s

    @property
    def clip(self):
        return self._clip

    @clip.setter
    def clip(self, s):
        self._clip = s

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, s):
        self._batch_size = s

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, s):
        self._sigma = s

    def minimize(self, optimizer=None, losses=[]):
        """
        Define Dpsgd optimizer
        """
        optimizer = fluid.optimizer.Dpsgd(
            self._learning_rate,
            clip=self._clip,
            batch_size=self._batch_size,
            sigma=self._sigma)
        optimizer.minimize(losses[0])

    def _build_trainer_program_for_job(self,
                                       trainer_id=0,
                                       program=None,
                                       ps_endpoints=[],
                                       trainers=0,
                                       sync_mode=True,
                                       startup_program=None,
                                       job=None):
        transpiler = fluid.DistributeTranspiler()
        transpiler.transpile(
            trainer_id,
            program=program,
            pservers=",".join(ps_endpoints),
            trainers=trainers,
            sync_mode=sync_mode,
            startup_program=startup_program)
        main = transpiler.get_trainer_program(wait_port=False)
        job._trainer_startup_programs.append(startup_program)
        job._trainer_main_programs.append(main)

    def _build_server_programs_for_job(self,
                                       program=None,
                                       ps_endpoints=[],
                                       trainers=0,
                                       sync_mode=True,
                                       startup_program=None,
                                       job=None):
        transpiler = fluid.DistributeTranspiler()
        trainer_id = 0
        transpiler.transpile(
            trainer_id,
            program=program,
            pservers=",".join(ps_endpoints),
            trainers=trainers,
            sync_mode=sync_mode,
            startup_program=startup_program)
        job.set_server_endpoints(ps_endpoints)
        for endpoint in ps_endpoints:
            main_prog = transpiler.get_pserver_program(endpoint)
            startup_prog = transpiler.get_startup_program(endpoint, main_prog)
            job._server_startup_programs.append(startup_prog)
            job._server_main_programs.append(main_prog)


class FedAvgStrategy(FLStrategyBase):
    """
    FedAvgStrategy: this is model averaging optimization proposed in
    H. Brendan McMahan, Eider Moore, Daniel Ramage, Blaise Aguera y Arcas. Federated Learning of Deep Networks using Model Averaging. 2017
    """

    def __init__(self):
        super(FedAvgStrategy, self).__init__()

    def minimize(self, optimizer=None, losses=[]):
        """
        minimize the first loss as in paddle.fluid
        """
        optimizer.minimize(losses[0])

    def _build_trainer_program_for_job(self,
                                       trainer_id=0,
                                       program=None,
                                       ps_endpoints=[],
                                       trainers=0,
                                       sync_mode=True,
                                       startup_program=None,
                                       job=None):
        transpiler = FLDistributeTranspiler()
        transpiler.transpile(
            trainer_id,
            program=program,
            pservers=",".join(ps_endpoints),
            trainers=trainers,
            sync_mode=sync_mode,
            startup_program=startup_program)
        recv, main, send = transpiler.get_trainer_program()
        job._trainer_startup_programs.append(startup_program)
        job._trainer_main_programs.append(main)
        job._trainer_send_programs.append(send)
        job._trainer_recv_programs.append(recv)

    def _build_server_programs_for_job(self,
                                       program=None,
                                       ps_endpoints=[],
                                       trainers=0,
                                       sync_mode=True,
                                       startup_program=None,
                                       job=None):
        transpiler = FLDistributeTranspiler()
        trainer_id = 0
        transpiler.transpile(
            trainer_id,
            program=program,
            pservers=",".join(ps_endpoints),
            trainers=trainers,
            sync_mode=sync_mode,
            startup_program=startup_program)
        job.set_server_endpoints(ps_endpoints)
        for endpoint in ps_endpoints:
            main_prog = transpiler.get_pserver_program(endpoint)
            startup_prog = transpiler.get_startup_program(endpoint, main_prog)
            job._server_startup_programs.append(startup_prog)
            job._server_main_programs.append(main_prog)


class SecAggStrategy(FedAvgStrategy):
    """
    DPSGDStrategy: this is model averaging optimization proposed in
    Aaron Segal, Antonio Marcedone, Benjamin Kreuter, et al.
    Practical Secure Aggregation  for Privacy-Preserving Machine Learning,
    The 24th ACM Conference on Computer and Communications Security ( CCS2017 ).
    """

    def __init__(self):
        super(SecAggStrategy, self).__init__()
        self._param_name_list = []

    @property
    def param_name_list(self):
        return self._param_name_list

    @param_name_list.setter
    def param_name_list(self, s):
        self._param_name_list = s
