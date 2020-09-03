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
import logging
from paddle.fluid.executor import global_scope
import pickle
from paddle.fluid.io import is_belong_to_optimizer
from paddle_fl.paddle_fl.core.scheduler.agent_master import FLWorkerAgent
import numpy
import hmac
import hashlib
from .diffiehellman.diffiehellman import DiffieHellman


class FLTrainerFactory(object):
    def __init__(self):
        pass

    def create_fl_trainer(self, job):
        strategy = job._strategy
        trainer = None
        if strategy._fed_avg == True:
            trainer = FedAvgTrainer()
            trainer.set_trainer_job(job)
        elif strategy._dpsgd == True:
            trainer = FLTrainer()
            trainer.set_trainer_job(job)
        elif strategy._sec_agg == True:
            trainer = SecAggTrainer()
            trainer.set_trainer_job(job)
        trainer.set_trainer_job(job)
        return trainer


class FLTrainer(object):
    def __init__(self):
        self._logger = logging.getLogger("FLTrainer")
        pass

    def set_trainer_job(self, job):
        self._startup_program = \
            job._trainer_startup_program
        self._main_program = \
            job._trainer_main_program
        self._step = job._strategy._inner_step
        self._feed_names = job._feed_names
        self._target_names = job._target_names
        self._scheduler_ep = job._scheduler_ep
        self._current_ep = None
        self.cur_step = 0

    def start(self, place):
        #current_ep = "to be added"
        self.agent = FLWorkerAgent(self._scheduler_ep, self._current_ep)
        self.agent.connect_scheduler()
        self.exe = fluid.Executor(place)
        self.exe.run(self._startup_program)

    def run(self, feed, fetch):
        self._logger.debug("begin to run")
        self.exe.run(self._main_program, feed=feed, fetch_list=fetch)
        self._logger.debug("end to run current batch")
        self.cur_step += 1

    def save_inference_program(self, output_folder):
        target_vars = []
        infer_program = self._main_program.clone(for_test=True)
        for name in self._target_names:
            tmp_var = self._main_program.block(0)._find_var_recursive(name)
            target_vars.append(tmp_var)
        fluid.io.save_inference_model(
            output_folder,
            self._feed_names,
            target_vars,
            self.exe,
            main_program=infer_program)

    def save(self, parameter_dir, model_path):
        base_name = os.path.basename(model_path)
        assert base_name != "", \
            "The input model_path MUST be format of dirname/filename [dirname\\filename in Windows system], but received model_path is empty string."

        dir_name = os.path.dirname(model_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        def get_tensor(var_name):
            t = global_scope().find_var(var_name).get_tensor()
            return numpy.array(t)

        parameter_list = []
        with open(parameter_dir + '/para_info', 'r') as fin:
            for line in fin:
                current_para = line[:-1]
                parameter_list.append(current_para)
        param_dict = {p: get_tensor(p) for p in parameter_list}
        with open(model_path + ".pdparams", 'wb') as f:
            pickle.dump(param_dict, f, protocol=2)

        optimizer_var_list = list(
            filter(is_belong_to_optimizer, self._main_program.list_vars()))

        opt_dict = {p.name: get_tensor(p.name) for p in optimizer_var_list}
        with open(model_path + ".pdopt", 'wb') as f:
            pickle.dump(opt_dict, f, protocol=2)

        main_program = self._main_program.clone()
        self._main_program.desc.flush()
        main_program.desc._set_version()
        fluid.core.save_op_compatible_info(self._main_program.desc)

        with open(model_path + ".pdmodel", "wb") as f:
            f.write(self._main_program.desc.serialize_to_string())

    def save_serving_model(self, model_path, client_conf_path):
        feed_vars = {}
        target_vars = {}
        for target in self._target_names:
            tmp_target = self._main_program.block(0)._find_var_recursive(
                target)
            target_vars[target] = tmp_target

        for feed in self._feed_names:
            tmp_feed = self._main_program.block(0)._find_var_recursive(feed)
            feed_vars[feed] = tmp_feed

        serving_io.save_model(model_path, client_conf_path, feed_vars,
                              target_vars, self._main_program)

    def stop(self):
        # ask for termination with master endpoint
        # currently not open sourced, will release the code later
        # TODO(guru4elephant): add connection with master
        if self.cur_step != 0:
            while not self.agent.finish_training():
                self._logger.debug("Wait others finish")
                continue
        while not self.agent.can_join_training():
            self._logger.debug("Wait permit")
            continue
        self._logger.debug("Ready to train")
        return False


class FedAvgTrainer(FLTrainer):
    def __init__(self):
        super(FedAvgTrainer, self).__init__()
        pass

    def start(self, place):
        #current_ep = "to be added"
        self.agent = FLWorkerAgent(self._scheduler_ep, self._current_ep)
        self.agent.connect_scheduler()
        self.exe = fluid.Executor(place)
        self.exe.run(self._startup_program)

    def set_trainer_job(self, job):
        super(FedAvgTrainer, self).set_trainer_job(job)
        self._send_program = job._trainer_send_program
        self._recv_program = job._trainer_recv_program

    def reset(self):
        self.cur_step = 0

    def run_with_epoch(self, reader, feeder, fetch, num_epoch):
        self._logger.debug("begin to run recv program")
        self.exe.run(self._recv_program)
        epoch = 0
        for i in range(num_epoch):
            for data in reader():
                self.exe.run(self._main_program,
                             feed=feeder.feed(data),
                             fetch_list=fetch)
            self.cur_step += 1
            epoch += 1
        self._logger.debug("begin to run send program")
        self.exe.run(self._send_program)

    def run(self, feed, fetch):
        self._logger.debug(
            "begin to run FedAvgTrainer, cur_step=%d, inner_step=%d" %
            (self.cur_step, self._step))
        if self.cur_step % self._step == 0:
            self._logger.debug("begin to run recv program")
            self.exe.run(self._recv_program)
        self._logger.debug("begin to run current step")
        loss = self.exe.run(self._main_program, feed=feed, fetch_list=fetch)
        if self.cur_step % self._step == 0:
            self._logger.debug("begin to run send program")
            self.exe.run(self._send_program)
        self.cur_step += 1
        return loss


class SecAggTrainer(FLTrainer):
    def __init__(self):
        super(SecAggTrainer, self).__init__()
        pass

    @property
    def trainer_id(self):
        return self._trainer_id

    @trainer_id.setter
    def trainer_id(self, s):
        self._trainer_id = s

    @property
    def trainer_num(self):
        return self._trainer_num

    @trainer_num.setter
    def trainer_num(self, s):
        self._trainer_num = s

    @property
    def key_dir(self):
        return self._key_dir

    @key_dir.setter
    def key_dir(self, s):
        self._key_dir = s

    @property
    def step_id(self):
        return self._step_id

    @step_id.setter
    def step_id(self, s):
        self._step_id = s

    def start(self, place):
        self.agent = FLWorkerAgent(self._scheduler_ep, self._current_ep)
        self.agent.connect_scheduler()
        self.exe = fluid.Executor(place)
        self.exe.run(self._startup_program)
        self.cur_step = 0

    def set_trainer_job(self, job):
        super(SecAggTrainer, self).set_trainer_job(job)
        self._send_program = job._trainer_send_program
        self._recv_program = job._trainer_recv_program
        self_step = job._strategy._inner_step
        self._param_name_list = job._strategy._param_name_list

    def reset(self):
        self.cur_step = 0

    def run(self, feed, fetch):
        self._logger.debug(
            "begin to run SecAggTrainer, cur_step=%d, inner_step=%d" %
            (self.cur_step, self._step))
        if self.cur_step % self._step == 0:
            self._logger.debug("begin to run recv program")
            self.exe.run(self._recv_program)
        scope = fluid.global_scope()
        self._logger.debug("begin to run current step")
        loss = self.exe.run(self._main_program, feed=feed, fetch_list=fetch)
        if self.cur_step % self._step == 0:
            self._logger.debug("begin to run send program")
            noise = 0.0
            scale = pow(10.0, 5)
            digestmod = hashlib.sha256
            # 1. load priv key and other's pub key
            dh = DiffieHellman(group=15, key_length=256)
            dh.load_private_key(self._key_dir + str(self._trainer_id) +
                                "_priv_key.txt")
            key = str(self._step_id).encode("utf-8")
            for i in range(self._trainer_num):
                if i != self._trainer_id:
                    f = open(self._key_dir + str(i) + "_pub_key.txt", "r")
                    public_key = int(f.read())
                    dh.generate_shared_secret(public_key, echo_return_key=True)
                    msg = dh.shared_key.encode("utf-8")
                    hex_res1 = hmac.new(key=key, msg=msg,
                                        digestmod=digestmod).hexdigest()
                    current_noise = int(hex_res1[0:8], 16) / scale
                    if i > self._trainer_id:
                        noise = noise + current_noise
                    else:
                        noise = noise - current_noise

            scope = fluid.global_scope()
            for param_name in self._param_name_list:
                fluid.global_scope().var(param_name + str(
                    self._trainer_id)).get_tensor().set(
                        numpy.array(
                            scope.find_var(param_name + str(self._trainer_id))
                            .get_tensor()) + noise, fluid.CPUPlace())
            self.exe.run(self._send_program)
        self.cur_step += 1
        return loss
