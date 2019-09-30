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
        trainer.set_trainer_job(job)
        return trainer


class FLTrainer(object):
    def __init__(self):
        pass

    def set_trainer_job(self, job):
        self._startup_program = \
            job._trainer_startup_program
        self._main_program = \
            job._trainer_main_program
        self._step = job._strategy._inner_step
        self._feed_names = job._feed_names
        self._target_names = job._target_names

    def start(self):
        self.exe = fluid.Executor(fluid.CPUPlace())
        self.exe.run(self._startup_program)

    def train_inner_loop(self, reader):
        now_step = 0
        for data in reader():
            now_step += 1
            if now_step > self._step:
                break
            self.exe.run(self._main_program,
                         feed=data,
                         fetch_list=[])

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

    def stop(self):
        # ask for termination with master endpoint
        # currently not open sourced, will release the code later
        # TODO(guru4elephant): add connection with master
        return False

class FedAvgTrainer(FLTrainer):
    def __init__(self):
        super(FedAvgTrainer, self).__init__()
        pass

    def start(self):
        self.exe = fluid.Executor(fluid.CPUPlace())
        self.exe.run(self._startup_program)
        self.step = 0

    def set_trainer_job(self, job):
        super(FedAvgTrainer, self).set_trainer_job(job)
        self._send_program = job._trainer_send_program
        self._recv_program = job._trainer_recv_program

    def reset(self):
        self.cur_step = 0

    def run(self, feed, fetch):
        if self.cur_step % self._step == 0:
            self.exe.run(self._recv_program)
        self.exe.run(self._main_program, 
                     feed=feed,
                     fetch_list=fetch)
        if self.cur_step % self._step == 0:
            self.exe.run(self._send_program)
        self.cur_step += 1

    def train_inner_loop(self, reader):
        self.exe.run(self._recv_program)
        now_step = 0
        for data in reader():
            now_step += 1
            if now_step > self._step:
                break
            self.exe.run(self._main_program,
                         feed=data,
                         fetch_list=[])
        self.exe.run(self._send_program)

    def stop(self):
        return False
        
