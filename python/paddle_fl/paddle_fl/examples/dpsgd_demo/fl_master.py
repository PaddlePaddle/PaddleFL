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

import paddle.fluid as fluid
import paddle_fl.paddle_fl as fl
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
import math


class Model(object):
    def __init__(self):
        pass

    def lr_network(self):
        self.inputs = fluid.layers.data(
            name='img', shape=[1, 28, 28], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.predict = fluid.layers.fc(input=self.inputs,
                                       size=10,
                                       act='softmax')
        self.sum_cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()


model = Model()
model.lr_network()

STEP_EPSILON = 0.1
DELTA = 0.00001
SIGMA = math.sqrt(2.0 * math.log(1.25 / DELTA)) / STEP_EPSILON
CLIP = 4.0
batch_size = 64

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.1)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    [model.loss.name, model.accuracy.name])

build_strategy = FLStrategyFactory()
build_strategy.dpsgd = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()
strategy.learning_rate = 0.1
strategy.clip = CLIP
strategy.batch_size = float(batch_size)
strategy.sigma = CLIP * SIGMA

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=4, output=output)
# fl_job_config will  be dispatched to workers
