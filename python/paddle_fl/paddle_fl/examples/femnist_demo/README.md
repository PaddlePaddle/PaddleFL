# Example in LEAF Dataset with FedAvg

This document introduces how to use PaddleFL to train a model with Fl Strategy: FedAvg.

### Dependencies

- paddlepaddle>=1.8

### How to install PaddleFL

Please use pip which has paddlepaddle installed

```sh
pip install paddle_fl
```

### Model

An CNN model which get features with two convolution layers and one fully connected layer and then compute and ouput probabilities of multiple classifications directly via Softmax function.

### Datasets

Public Dataset FEMNIST in [LEAF](https://github.com/TalwalkarLab/leaf)

### How to work in PaddleFL

PaddleFL has two phases , CompileTime and RunTime. In CompileTime, a federated learning task is defined by fl_master. In RunTime, a federated learning job is executed on fl_server and fl_trainer in distributed clusters.

```sh
sh run.sh
```

#### How to work in CompileTime

In this example, we implement compile time programs in fl_master.py

```sh
python fl_master.py
```

In fl_master.py, we first define FL-Strategy, User-Defined-Program and Distributed-Config. Then FL-Job-Generator generate FL-Job for federated server and worker.

```python
import paddle.fluid as fluid
import paddle_fl.paddle_fl as fl
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory

class Model(object):
    def __init__(self):
        pass

    def cnn(self):
        self.inputs = fluid.layers.data(
            name='img', shape=[1, 28, 28], dtype="float32")
        self.label = fluid.layers.data(name='label', shape=[1], dtype='int64')
        self.conv_pool_1 = fluid.nets.simple_img_conv_pool(
            input=self.inputs,
            num_filters=20,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act='relu')
        self.conv_pool_2 = fluid.nets.simple_img_conv_pool(
            input=self.conv_pool_1,
            num_filters=50,
            filter_size=5,
            pool_size=2,
            pool_stride=2,
            act='relu')

        self.predict = self.predict = fluid.layers.fc(input=self.conv_pool_2,
                                                      size=62,
                                                      act='softmax')
        self.cost = fluid.layers.cross_entropy(
            input=self.predict, label=self.label)
        self.accuracy = fluid.layers.accuracy(
            input=self.predict, label=self.label)
        self.loss = fluid.layers.mean(self.cost)
        self.startup_program = fluid.default_startup_program()

model = Model()
model.cnn()

job_generator = JobGenerator()
optimizer = fluid.optimizer.Adam(learning_rate=0.1)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name],
    [model.loss.name, model.accuracy.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()

endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=4, output=output)
```

#### How to work in RunTime

```sh
python -u fl_scheduler.py >scheduler.log &
python -u fl_server.py >server0.log &
for ((i=0;i<4;i++))
do
    python -u fl_trainer.py $i >trainer$i.log &
done
```
In fl_scheduler.py, we let server and trainers to do registeration. 

```python
from paddle_fl.paddle_fl.core.scheduler.agent_master import FLScheduler

worker_num = 4
server_num = 1
# Define the number of worker/server and the port for scheduler
scheduler = FLScheduler(worker_num, server_num, port=9091)
scheduler.set_sample_worker_num(4)
scheduler.init_env()
print("init env done.")
scheduler.start_fl_training()
```
In fl_server.py, we load and run the FL server job.  

```python
import paddle_fl.paddle_fl as fl
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.server.fl_server import FLServer
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob

server = FLServer()
server_id = 0
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
job._scheduler_ep = "127.0.0.1:9091"  # IP address for scheduler
server.set_server_job(job)
server._current_ep = "127.0.0.1:8181"  # IP address for server
server.start()
```

In fl_trainer.py, we load and run the FL trainer job.  

```python
from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import paddle_fl.paddle_fl.dataset.femnist as femnist
import numpy
import sys
import paddle
import paddle.fluid as fluid
import logging
import math

trainer_id = int(sys.argv[1])  # trainer id for each guest
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform the scheduler IP to trainer
print(job._target_names)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)
print(trainer._step)
test_program = trainer._main_program.clone(for_test=True)

img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[img, label], place=fluid.CPUPlace())

def train_test(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    for test_data in train_test_reader():
        acc_np = trainer.exe.run(program=train_test_program,
                                 feed=train_test_feed.feed(test_data),
                                 fetch_list=["accuracy_0.tmp_0"])
        acc_set.append(float(acc_np[0]))
    acc_val_mean = numpy.array(acc_set).mean()
    return acc_val_mean


epoch_id = 0
step = 0
epoch = 3000
count_by_step = False
if count_by_step:
    output_folder = "model_node%d" % trainer_id
else:
    output_folder = "model_node%d_epoch" % trainer_id

while not trainer.stop():
    count = 0
    epoch_id += 1
    if epoch_id > epoch:
        break
    print("epoch %d start train" % (epoch_id))
    #train_data,test_data= data_generater(trainer_id,inner_step=trainer._step,batch_size=64,count_by_step=count_by_step)
    train_reader = paddle.batch(
        paddle.reader.shuffle(
            femnist.train(
                trainer_id,
                inner_step=trainer._step,
                batch_size=64,
                count_by_step=count_by_step),
            buf_size=500),
        batch_size=64)

    test_reader = paddle.batch(
        femnist.test(
            trainer_id,
            inner_step=trainer._step,
            batch_size=64,
            count_by_step=count_by_step),
        batch_size=64)

    if count_by_step:
        for step_id, data in enumerate(train_reader()):
            acc = trainer.run(feeder.feed(data), fetch=["accuracy_0.tmp_0"])
            step += 1
            count += 1
            print(count)
            if count % trainer._step == 0:
                break
    # print("acc:%.3f" % (acc[0]))
    else:
        trainer.run_with_epoch(
            train_reader, feeder, fetch=["accuracy_0.tmp_0"], num_epoch=1)

    acc_val = train_test(
        train_test_program=test_program,
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test with epoch %d, accuracy: %s" % (epoch_id, acc_val))
    if trainer_id == 0:
        save_dir = (output_folder + "/epoch_%d") % epoch_id
        trainer.save_inference_program(output_folder)
```

