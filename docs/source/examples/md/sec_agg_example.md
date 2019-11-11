# Example in Recognize Digits with DPSGD

This document introduces how to use PaddleFL to train a model with Fl Strategy: Secure Aggregation. Using Secure Aggregation strategy, the server can aggregate the model parameters without learning the value of the parameters.

### Dependencies

- paddlepaddle>=1.6

### How to install PaddleFL

Please use python which has paddlepaddle installed

```
python setup.py install
```

### Model

The simplest Softmax regression model is to get features with input layer passing through a fully connected layer and then compute and ouput probabilities of multiple classifications directly via Softmax function [[PaddlePaddle tutorial: recognize digits](https://github.com/PaddlePaddle/book/tree/develop/02.recognize_digits#references)].

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

```
sh run.sh
```

#### How to work in CompileTime

In this example, we implement compile time programs in fl_master.py

```
python fl_master.py
```

In fl_master.py, we first define FL-Strategy, User-Defined-Program and Distributed-Config. Then FL-Job-Generator generate FL-Job for federated server and worker.

```python
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
build_strategy.inner_step = 10
strategy = build_strategy.create_fl_strategy()

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)

```

How to work in RunTime

```shell
python3 -u fl_server.py >server0.log &
python3 -u fl_trainer.py 0 data/ >trainer0.log &
python3 -u fl_trainer.py 1 data/ >trainer1.log &
```

In fl_server.py, we load and run the FL server job.  

```
server = FLServer()
server_id = 0
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
server.set_server_job(job)
server.start()
```

In fl_trainer.py, we prepare the MNIST dataset, load and run the FL trainer job, then evaluate the accuracy.  Before training , we first prepare the party's private key and other party's public key. Then, each party generates mask using Diffie-Hellman key aggregate protocal with its parivate key and other's public key [1], and masks the model parameters before uploading to the server. Finally, the server can remove the masks by aggregating  the parameters from all the parties.

```python
train_reader = paddle.batch(
    paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
    batch_size=16)

trainer_num = 2
trainer_id = int(sys.argv[1]) # trainer id for each guest

job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.start()

output_folder = "fl_model"
epoch_id = 0
step_i = 0
while not trainer.stop():
    epoch_id += 1
    print("epoch %d start train" % (epoch_id))
    starttime = datetime.datetime.now()

    # prepare the aggregated parameters
    param_name_list = []
    param_name_list.append("fc_0.b_0.opti.trainer_" + str(trainer_id))
    param_name_list.append("fc_0.b_0.opti.trainer_" + str(trainer_id))

    inputs = fluid.layers.data(name='x', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='y', shape=[1], dtype='int64')
    feeder = fluid.DataFeeder(feed_list=[inputs, label], place=fluid.CPUPlace())

    scale = pow(10.0, 5)
     # 1. load priv key and other's pub key
    # party_name = Party(trainer_id + 1)
    dh = DiffieHellman(group=15, key_length=256)
    dh.load_private_key(str(trainer_id) + "_priv_key.txt")

    digestmod="SHA256"

    for data in train_reader():
        step_i += 1
        noise = 0.0

        # 2. generate noise
        secagg_starttime = datetime.datetime.now()
        key = str(step_i).encode("utf-8")
        for i in range(trainer_num):
            if i != trainer_id:
                f = open(str(i) + "_pub_key.txt", "r")
                public_key = int(f.read())
                dh.generate_shared_secret(public_key, echo_return_key=True)
                msg = dh.shared_key.encode("utf-8")
                hex_res1 = hmac.new(key=key, msg=msg, digestmod=digestmod).hexdigest()
                current_noise = int(hex_res1[0:8], 16) / scale
                if i > trainer_id:
                    noise = noise + current_noise
                else:
                    noise = noise - current_noise
        if step_i % 100 == 0:
            print("Step: {0}".format(step_i))
        # 3. add noise between training and sending.
        accuracy, = trainer.run(feed=feeder.feed(data),
            fetch=["top_k_0.tmp_0"],
            param_name_list=param_name_list,
            mask=noise)

    print("Epoch: {0}, step: {1}, accuracy: {2}".format(epoch_id, step_i, accuracy[0]))
    endtime = datetime.datetime.now()
    print("time cost: {0}".format(endtime - starttime))

    if epoch_id > 40:
        break
    if step_i % 100 == 0:
        trainer.save_inference_program(output_folder)
```



[1] Aaron Segal, Antonio Marcedone, Benjamin Kreuter, Daniel Ramage, H. Brendan McMahan, Karn Seth, Keith Bonawitz, Sarvar Patel, Vladimir Ivanov. **Practical Secure Aggregation  for Privacy-Preserving Machine Learning**, The 24th ACM Conference on Computer and Communications Security (**CCS**), 2017