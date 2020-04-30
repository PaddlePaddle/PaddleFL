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


class Model(object):
    def __init__(self):
        pass

    def linear_regression(self, inputs, label):
        param_attrs = fluid.ParamAttr(
            name="fc_0.b_0",
            initializer=fluid.initializer.ConstantInitializer(0.0))
        param_attrs = fluid.ParamAttr(
            name="fc_0.w_0",
            initializer=fluid.initializer.ConstantInitializer(0.0))
        self.predict = fluid.layers.fc(input=inputs,
                                       size=10,
                                       act='softmax',
                                       param_attr=param_attrs)
        self.sum_cost = fluid.layers.cross_entropy(
            input=self.predict, label=label)
        self.loss = fluid.layers.mean(self.sum_cost)
        self.accuracy = fluid.layers.accuracy(input=self.predict, label=label)
        self.startup_program = fluid.default_startup_program()


inputs = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='y', shape=[1], dtype='int64')

model = Model()
model.linear_regression(inputs, label)

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.01)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names([inputs.name, label.name],
                                              [model.loss.name])

build_strategy = FLStrategyFactory()
#build_strategy.fed_avg = True

build_strategy.sec_agg = True
param_name_list = []
param_name_list.append(
    "fc_0.w_0.opti.trainer_")  # need trainer_id when running
param_name_list.append("fc_0.b_0.opti.trainer_")
build_strategy.param_name_list = param_name_list

build_strategy.inner_step = 10
strategy = build_strategy.create_fl_strategy()

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
# fl_job_config will  be dispatched to workers
