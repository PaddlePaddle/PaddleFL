import paddle.fluid as fluid
import paddle_fl as fl
from paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory


class Model(object):
    def __init__(self):
        pass

    def cnn(self):
        self.inputs = fluid.layers.data(name='img', shape=[1, 28, 28], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1],dtype='int64')
        self.conv_pool_1 = fluid.nets.simple_img_conv_pool(input=self.inputs,num_filters=20,filter_size=5,pool_size=2,pool_stride=2,act='relu')
        self.conv_pool_2 = fluid.nets.simple_img_conv_pool(input=self.conv_pool_1,num_filters=50,filter_size=5,pool_size=2,pool_stride=2,act='relu')
        self.predict = self.predict = fluid.layers.fc(input=self.conv_pool_2, size=62, act='softmax')
        self.cost = fluid.layers.cross_entropy(input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.cost)
        self.startup_program = fluid.default_startup_program()


model = Model()
model.cnn()

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.1)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name], [model.loss.name, model.accuracy.name])


build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()

endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=4, output=output)
