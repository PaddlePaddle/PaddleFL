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
'''
For large scale mobile devices simulation, we need to consider:
- model structure and input features
- trainer, i.e. how to design training strategy for each simulation
- user sampler, i.e. users that need to participate training
- optimizer, i.e. how to update global weights

Currently, we couple trainer and model together for simplicity
'''
from utils import FLSimRoleMaker
from framework import SimulationFramework
from trainer import LanguageModelTrainer
from optimizer import FedAvgOptimizer
from sampler import UniformSampler, Test1percentSampler
from datetime import date, timedelta
import sys
import time

role_maker = FLSimRoleMaker()
role_maker.init_env(local_shard_num=30)
simulator = SimulationFramework(role_maker)

language_model_trainer = LanguageModelTrainer()

language_model_trainer.set_trainer_configs({
    "epoch": 1,
    "max_steps_in_epoch": -1,
    "lr": 1.0,
    "batch_size": 5,
    "max_grad_norm": 5,
    "n_hidden": 256,
    "num_layers": 2,
    "init_scale": 0.1,
    "dropout_prob": 0.0,
})

sampler = UniformSampler()
sampler.set_sample_num(10)
sampler.set_min_ins_num(1)
test_sampler = Test1percentSampler()
fed_avg_optimizer = FedAvgOptimizer(learning_rate=1.85)

simulator.set_trainer(language_model_trainer)
simulator.set_sampler(sampler)
simulator.set_test_sampler(test_sampler)
simulator.set_fl_optimizer(fed_avg_optimizer)

if simulator.is_scheduler():
    simulator.run_scheduler_service()
elif simulator.is_simulator():
    base_path = sys.argv[1]
    dates = []
    start_date = date(2020, 1, 1)
    end_date = date(2020, 1, 2)
    delta = timedelta(days=1)
    while start_date <= end_date:
        dates.append(start_date.strftime("%Y%m%d"))
        start_date += delta

    print("dates: {}".format(dates))

    time.sleep(10)
    simulator.run_simulation(base_path,
                             dates,
                             sim_num_everyday=100,
                             do_test=True,
                             test_skip_day=1)
