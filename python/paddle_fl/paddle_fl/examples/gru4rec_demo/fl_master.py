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

    def gru4rec_network(self,
                        vocab_size=37483,
                        hid_size=100,
                        init_low_bound=-0.04,
                        init_high_bound=0.04):
        """ network definition """
        emb_lr_x = 10.0
        gru_lr_x = 1.0
        fc_lr_x = 1.0
        # Input data
        self.src_wordseq = fluid.layers.data(
            name="src_wordseq", shape=[1], dtype="int64", lod_level=1)
        self.dst_wordseq = fluid.layers.data(
            name="dst_wordseq", shape=[1], dtype="int64", lod_level=1)

        emb = fluid.layers.embedding(
            input=self.src_wordseq,
            size=[vocab_size, hid_size],
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=init_low_bound, high=init_high_bound),
                learning_rate=emb_lr_x),
            is_sparse=False)
        fc0 = fluid.layers.fc(input=emb,
                              size=hid_size * 3,
                              param_attr=fluid.ParamAttr(
                                  initializer=fluid.initializer.Uniform(
                                      low=init_low_bound,
                                      high=init_high_bound),
                                  learning_rate=gru_lr_x))
        gru_h0 = fluid.layers.dynamic_gru(
            input=fc0,
            size=hid_size,
            param_attr=fluid.ParamAttr(
                initializer=fluid.initializer.Uniform(
                    low=init_low_bound, high=init_high_bound),
                learning_rate=gru_lr_x))

        self.fc = fluid.layers.fc(input=gru_h0,
                                  size=vocab_size,
                                  act='softmax',
                                  param_attr=fluid.ParamAttr(
                                      initializer=fluid.initializer.Uniform(
                                          low=init_low_bound,
                                          high=init_high_bound),
                                      learning_rate=fc_lr_x))
        cost = fluid.layers.cross_entropy(
            input=self.fc, label=self.dst_wordseq)
        self.acc = fluid.layers.accuracy(
            input=self.fc, label=self.dst_wordseq, k=20)
        self.loss = fluid.layers.mean(x=cost)
        self.startup_program = fluid.default_startup_program()


model = Model()
model.gru4rec_network()

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=2.0)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.src_wordseq.name, model.dst_wordseq.name],
    [model.loss.name, model.acc.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 10
strategy = build_strategy.create_fl_strategy()

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=4, output=output)
# fl_job_config will  be dispatched to workers
