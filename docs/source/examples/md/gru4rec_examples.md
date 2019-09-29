# Train gru4rec model with FedAvg Strategy

This doc introduce how to use PaddleFL to train model with Fl Strategy.

### Dependencies
- paddlepaddle>=1.6

### How to install PaddleFL
please use the python which has installed paddlepaddle.
```sh
python setup.py install
```

### Model
[Gru4rec](https://arxiv.org/abs/1511.06939) is the classical session-based recommendation model. The details implement by paddlepaddle is [here](https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec).


### Datasets
We use [Rsc15](https://2015.recsyschallenge.com) dataset as our data. 

```sh
#download data
cd example/gru4rec_demo
sh download.sh
```

### How to work in PaddleFL
PaddleFL has two period , CompileTime and RunTime. In CompileTime, define a federated learning task by fl_master. In RunTime, train a federated learning job by fl_server and fl_trainer .

### How to work in CompileTime
In this example, we implement it in fl_master.py
```sh
# please run fl_master to generate fl_job
python fl_master.py
```
In fl_master.py,  we first define FL-Strategy, User-Defined-Program and Distributed-Config. Then FL-Job-Generator generate FL-Job for federated server and worker.
```python
# define model
model = Model()
model.gru4rec_network()

# define JobGenerator and set model config
# feed_name and target_name are config for save model.
job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=2.0)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.loss])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [x.name for x in model.inputs], [model.loss.name, model.recall.name])

# define FL-Strategy , we now support two flstrategy, fed_avg and dpsgd. Inner_step means fl_trainer locally train inner_step mini-batch.
build_strategy = FLStrategyFactory()
build_strategy.fed_avg = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()

# define Distributed-Config and generate fl_job 
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)

```

To build a gcn layer, one can use our pre-defined ```pgl.layers.gcn``` or just write a gcn layer with message passing interface.
```python
import paddle.fluid as fluid
def gcn_layer(graph_wrapper, node_feature, hidden_size, act):
    def send_func(src_feat, dst_feat, edge_feat):
        return src_feat["h"]
    
    def recv_func(msg):
        return fluid.layers.sequence_pool(msg, "sum")
    
    message = graph_wrapper.send(send_func, nfeat_list=[("h", node_feature)])
    output = graph_wrapper.recv(recv_func, message)
    output = fluid.layers.fc(output, size=hidden_size, act=act)
    return output
```

### Datasets

The datasets contain three citation networks: CORA, PUBMED, CITESEER. The details for these three datasets can be found in the [paper](https://arxiv.org/abs/1609.02907).

### Dependencies

- paddlepaddle>=1.4 (The speed can be faster in 1.5.)
- pgl

### Performance

We train our models for 200 epochs and report the accuracy on the test dataset.

| Dataset | Accuracy | Speed with paddle 1.4 <br> (epoch time) | Speed with paddle 1.5 <br> (epoch time)|
| --- | --- | --- |---|
| Cora | ~81% | 0.0106s | 0.0104s | 
| Pubmed | ~79% | 0.0210s  | 0.0154s |
| Citeseer | ~71% | 0.0175s | 0.0177s | 


### How to run

For examples, use gpu to train gcn on cora dataset.
```
python train.py --dataset cora --use_cuda
```

#### Hyperparameters

- dataset: The citation dataset "cora", "citeseer", "pubmed".
- use_cuda: Use gpu if assign use_cuda. 


### View the Code

See the code [here](gcn_examples_code.html)
