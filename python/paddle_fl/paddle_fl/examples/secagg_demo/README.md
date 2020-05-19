# Example in Recognize Digits with Secure Aggragate

This document introduces how to use PaddleFL to train a model with Fl Strategy: Secure Aggregation. Using Secure Aggregation strategy, the server can aggregate the model parameters without learning the value of the parameters.

### Dependencies

- paddlepaddle>=1.8

### How to install PaddleFL

Please use pip which has paddlepaddle installed

```sh
pip install paddle_fl
```

### Model

The simplest Softmax regression model is to get features with input layer passing through a fully connected layer and then compute and ouput probabilities of multiple classes directly via Softmax function [[PaddlePaddle tutorial: recognize digits](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits#references)].

### Datasets

Public Dataset [MNIST](http://yann.lecun.com/exdb/mnist/)

The dataset will downloaded automatically in the API and will be located under `/home/username/.cache/paddle/dataset/mnist`:

| filename                | note                            |
| ----------------------- | ------------------------------- |
| train-images-idx3-ubyte | train data picture, 60,000 data |
| train-labels-idx1-ubyte | train data label, 60,000 data   |
| t10k-images-idx3-ubyte  | test data picture, 10,000 data  |
| t10k-labels-idx1-ubyte  | test data label, 10,000 data    |

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

def linear_regression(self, inputs, label):
        param_attrs = fluid.ParamAttr(
            name="fc_0.b_0",
            initializer=fluid.initializer.ConstantInitializer(0.0))
        param_attrs = fluid.ParamAttr(
            name="fc_0.w_0",
            initializer=fluid.initializer.ConstantInitializer(0.0))
        self.predict = fluid.layers.fc(input=inputs, size=10, act='softmax', param_attr=param_attrs)
        self.sum_cost = fluid.layers.cross_entropy(input=self.predict, label=label)
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
job_generator.set_infer_feed_and_target_names(
    [inputs.name, label.name], [model.loss.name])

build_strategy = FLStrategyFactory()
build_strategy.sec_agg = True
param_name_list = []
param_name_list.append("fc_0.w_0.opti.trainer_") # need trainer_id when running
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

```

#### How to work in RunTime

```sh
python3 fl_master.py
sleep 2
python3 -u fl_server.py >log/server0.log &
sleep 2
python3 -u fl_trainer.py 0 >log/trainer0.log &
sleep 2
python3 -u fl_trainer.py 1 >log/trainer1.log &
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
server.set_server_job(job)
server.start()
```

In fl_trainer.py, we prepare the MNIST dataset, load and run the FL trainer job, then evaluate the accuracy.  Before training , we first prepare the party's private key and other party's public key. Then, each party generates a random noise using Diffie-Hellman key aggregate protocol with its private key and each other's public key [1]. If the other party's id is larger than this party's id, the model parameters add this random noise. If the other party's id is less than this party's id, the model parameters subtract this random noise. So, and the model parameters is masked before uploading to the server. Finally, the random noises can be removed when aggregating the masked parameters from all the parties.

```python
import numpy
import sys
import logging
import time
import datetime
import math
import hashlib
import hmac
import paddle
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob

logging.basicConfig(filename="log/test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)
logger = logging.getLogger("FLTrainer")

BATCH_SIZE = 64

train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=BATCH_SIZE)
test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

trainer_num = 2
trainer_id = int(sys.argv[1]) # trainer id for each guest

job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.trainer_id = trainer_id
trainer.trainer_num = trainer_num
trainer.key_dir = "./keys/"
trainer.start()

output_folder = "fl_model"
epoch_id = 0
step_i = 0

inputs = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
label = fluid.layers.data(name='y', shape=[1], dtype='int64')
feeder = fluid.DataFeeder(feed_list=[inputs, label], place=fluid.CPUPlace())

# for test
test_program = trainer._main_program.clone(for_test=True)

def train_test(train_test_program,
                   train_test_feed, train_test_reader):
    acc_set = []
    avg_loss_set = []
    for test_data in train_test_reader():
        acc_np, avg_loss_np = trainer.exe.run(
            program=train_test_program,
            feed=train_test_feed.feed(test_data),
            fetch_list=["accuracy_0.tmp_0", "mean_0.tmp_0"])
        acc_set.append(float(acc_np))
        avg_loss_set.append(float(avg_loss_np))
    acc_val_mean = numpy.array(acc_set).mean()
    avg_loss_val_mean = numpy.array(avg_loss_set).mean()
    return avg_loss_val_mean, acc_val_mean
# for test

while not trainer.stop():
    epoch_id += 1
    print("epoch %d start train" % (epoch_id))

    for data in train_reader():
        step_i += 1
        trainer.step_id = step_i
        accuracy, = trainer.run(feed=feeder.feed(data),
            fetch=["accuracy_0.tmp_0"])
        if step_i % 100 == 0:
            print("Epoch: {0}, step: {1}, accuracy: {2}".format(epoch_id, step_i, accuracy[0]))

    avg_loss_val, acc_val = train_test(train_test_program=test_program,
                                       train_test_reader=test_reader,
                                       train_test_feed=feeder)
    print("Test with Epoch %d, avg_cost: %s, acc: %s" %(epoch_id, avg_loss_val, acc_val))

    if epoch_id > 40:
        break
    if step_i % 100 == 0:
        trainer.save_inference_program(output_folder)
```



[1] Aaron Segal, Antonio Marcedone, Benjamin Kreuter, Daniel Ramage, H. Brendan McMahan, Karn Seth, Keith Bonawitz, Sarvar Patel, Vladimir Ivanov. **Practical Secure Aggregation  for Privacy-Preserving Machine Learning**, The 24th ACM Conference on Computer and Communications Security (**CCS**), 2017
