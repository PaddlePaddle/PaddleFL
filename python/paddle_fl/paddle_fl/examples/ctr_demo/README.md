# Example in CTR(Click-Through-Rate) with FedAvg

This document introduces how to use PaddleFL to train a model with Fl Strategy.

### Dependencies

- paddlepaddle>=1.8

### How to install PaddleFL

Please use pip which has paddlepaddle installed

```sh
pip install paddle_fl
```

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

    def mlp(self, inputs, label, hidden_size=128):
        self.concat = fluid.layers.concat(inputs, axis=1)
        self.fc1 = fluid.layers.fc(input=self.concat, size=256, act='relu')
        self.fc2 = fluid.layers.fc(input=self.fc1, size=128, act='relu')
        self.predict = fluid.layers.fc(input=self.fc2, size=2, act='softmax')
        self.sum_cost = fluid.layers.cross_entropy(
            input=self.predict, label=label)
        self.accuracy = fluid.layers.accuracy(input=self.predict, label=label)
        self.loss = fluid.layers.reduce_mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()

inputs = [fluid.layers.data( \
            name=str(slot_id), shape=[5],
            dtype="float32")
          for slot_id in range(3)]
label = fluid.layers.data( \
            name="label",
            shape=[1],
            dtype='int64')

model = Model()
model.mlp(inputs, label)

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.1)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names([x.name for x in inputs],
                                              [model.predict.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 10
strategy = build_strategy.create_fl_strategy()

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
# fl_job_config will  be dispatched to workers
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
# Define the number of worker/server and the port for scheduler
scheduler = FLScheduler(worker_num, server_num, port=9091)
scheduler.set_sample_worker_num(worker_num)
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

In fl_trainer.py, we load and run the FL trainer job, then evaluate the accuracy with test data and compute the privacy budget. The DataSet is ramdomly generated. 

```python
import sys
import time
import logging
import numpy as np
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob

def reader():
    for i in range(1000):
        data_dict = {}
        for i in range(3):
            data_dict[str(i)] = np.random.rand(1, 5).astype('float32')
        data_dict["label"] = np.random.randint(2, size=(1, 1)).astype('int64')
        yield data_dict


trainer_id = int(sys.argv[1])  # trainer id for each guest
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
job._scheduler_ep = "127.0.0.1:9091"  # Inform the scheduler IP to trainer
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer._current_ep = "127.0.0.1:{}".format(9000 + trainer_id)
place = fluid.CPUPlace()
trainer.start(place)
print(trainer._scheduler_ep, trainer._current_ep)
output_folder = "fl_model"
epoch_id = 0

while not trainer.stop():
    print("batch %d start train" % (epoch_id))
    train_step = 0
    for data in reader():
        trainer.run(feed=data, fetch=[])
        train_step += 1
        if train_step == trainer._step:
            break
    epoch_id += 1
    if epoch_id % 5 == 0:
        trainer.save_inference_program(output_folder)
```
