# Example to Load Program from a Pre-defined Model

This document introduces how to load a pre-defined model, and transfer into program that is usable by PaddleFL.

### Dependencies 

- paddlepaddle>=1.8
- paddle_fl>=1.0

Please use pip which has paddlepaddle installed

```sh
pip install paddle_fl
``` 

### Compile Time

#### How to save a program

```sh
python program_saver.py
```

In program_saver.py, you can defind a model. And save the program in to 'load_file'

```python
import os
import json
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator

input = fluid.layers.data(name='input', shape=[1, 28, 28], dtype="float32")
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[input, label], place=fluid.CPUPlace())
predict = fluid.layers.fc(input=input, size=10, act='softmax')
sum_cost = fluid.layers.cross_entropy(input=predict, label=label)
accuracy = fluid.layers.accuracy(input=predict, label=label)
avg_cost = fluid.layers.mean(sum_cost, name="loss")
startup_program = fluid.default_startup_program()
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(startup_program)

job_generator = JobGenerator()
program_path = './load_file'
job_generator.save_program(program_path, [input, label],
                           [['predict', predict], ['accuracy', accuracy]],
                           avg_cost)
```

#### How to load a program

In fl_master.py, you can load the program in 'load_file' and transfer it into an fl program.

```python
import paddle_fl.paddle_fl as fl
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 10
strategy = build_strategy.create_fl_strategy()

endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
program_file = "./load_file"
job_generator = JobGenerator()
job_generator.generate_fl_job_from_program(
    strategy=strategy,
    endpoints=endpoints,
    worker_num=2,
    program_input=program_file,
    output=output)
``` 

#### How to work in RunTime

```sh
python -u fl_scheduler.py >scheduler.log &
python -u fl_server.py >server0.log &
python -u fl_trainer.py 0  >trainer0.log &
python -u fl_trainer.py 1  >trainer1.log &
```
In fl_scheduler.py, we let server and trainers to do registeration.

```python
from paddle_fl.paddle_fl.core.scheduler.agent_master import FLScheduler

worker_num = 2
server_num = 1
#Define number of worker/server and the port for scheduler
scheduler = FLScheduler(worker_num, server_num, port=9091)
scheduler.set_sample_worker_num(2)
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

In fl_trainer.py, we load and run the FL trainer job, then evaluate the accuracy with test data.

```python
from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
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
job._scheduler_ep = "127.0.0.1:9091"  # Inform scheduler IP address to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)
test_program = trainer._main_program.clone(for_test=True)

train_reader = paddle.batch(
    paddle.reader.shuffle(
        paddle.dataset.mnist.train(), buf_size=500),
    batch_size=64)
test_reader = paddle.batch(paddle.dataset.mnist.test(), batch_size=64)

input = fluid.layers.data(name='input', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[input, label], place=fluid.CPUPlace())

def train_test(train_test_program, train_test_feed, train_test_reader):
    acc_set = []
    for test_data in train_test_reader():
        acc_np = trainer.exe.run(program=train_test_program,
                                 feed=train_test_feed.feed(test_data),
                                 fetch_list=["accuracy_0.tmp_0"])
        acc_set.append(float(acc_np[0]))
    acc_val_mean = numpy.array(acc_set).mean()
    return acc_val_mean


output_folder = "model_node%d" % trainer_id
epoch_id = 0
step = 0

while not trainer.stop():
    epoch_id += 1
    if epoch_id > 40:
        break
    print("epoch %d start train" % (epoch_id))
    for step_id, data in enumerate(train_reader()):
        acc = trainer.run(feeder.feed(data), fetch=["accuracy_0.tmp_0"])
        step += 1

    acc_val = train_test(
        train_test_program=test_program,
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test with epoch %d, accuracy: %s" % (epoch_id, acc_val))

    save_dir = (output_folder + "/epoch_%d") % epoch_id
    trainer.save_inference_program(output_folder)
```
