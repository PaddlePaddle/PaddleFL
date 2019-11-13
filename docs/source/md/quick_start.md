## Step 1: Define Federated Learning Compile-Time

We define very simple multiple layer perceptron for demonstration. When multiple organizations
agree to share data knowledge through PaddleFL, a model can be defined with agreement from these organizations. A FLJob can be generated and saved. Programs needed to be run each node will be generated separately in FLJob.

```python
import paddle.fluid as fluid
import paddle_fl as fl
from paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory

class Model(object):
    def __init__(self):
        pass

    def mlp(self, inputs, label, hidden_size=128):
        self.concat = fluid.layers.concat(inputs, axis=1)
        self.fc1 = fluid.layers.fc(input=self.concat, size=256, act='relu')
        self.fc2 = fluid.layers.fc(input=self.fc1, size=128, act='relu')
	self.predict = fluid.layers.fc(input=self.fc2, size=2, act='softmax')
        self.sum_cost = fluid.layers.cross_entropy(input=self.predict, label=label)
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
job_generator.set_infer_feed_and_target_names(
    [x.name for x in inputs], [model.predict.name])

build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()

endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
```

## Step 2: Issue FL Job to Organizations

We can define a secure service to send programs to each node in FLJob. There are two types of nodes in distributed federated learning job. One is FL Server, the other is FL Trainer. A FL Trainer is owned by individual organization and an organization can have multiple FL Trainers given different amount of data knowledge the organization is willing to share. A FL Server is owned by a secure distributed training cluster. By means of security of the cluster, all organizations participated in the Federated Training Job should agree to trust the cluster is secure.

## Step 3: Start Federated Learning Run-Time

On FL Trainer Node, a training script is defined as follows:

``` python
from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy as np
import sys

def reader():
    for i in range(1000):
        data_dict = {}
        for i in range(3):
            data_dict[str(i)] = np.random.rand(1, 5).astype('float32')
        data_dict["label"] = np.random.randint(2, size=(1, 1)).astype('int64')
        yield data_dict

trainer_id = int(sys.argv[1]) # trainer id for each guest
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.start()

output_folder = "fl_model"
step_i = 0
while not trainer.stop():
    step_i += 1
    print("batch %d start train" % (step_i))
    trainer.run(feed=data, fetch=[])
    if step_i % 100 == 0:
        trainer.save_inference_program(output_folder)
```

On FL Server Node, a training script is defined as follows:

```python
import paddle_fl as fl
import paddle.fluid as fluid
from paddle_fl.core.server.fl_server import FLServer
from paddle_fl.core.master.fl_job import FLRunTimeJob
server = FLServer()
server_id = 0
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
server.set_server_job(job)
server.start()
```
