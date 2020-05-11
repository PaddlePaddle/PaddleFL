## PE - Paddle Encrypted 

Paddle Encrypted is a framework for privacy-preserving deep learning based on PaddlePaddle. It follows the same running mechanism and programming paradigm with PaddlePaddle, while using secure multi-party computation (MPC) to enable secure training and prediction. 

With Paddle Encrypted, it is easy to train models or conduct prediction as on PaddlePaddle over encrypted data, without the need for cryptography expertise. Furthermore, the rich industry-oriented models and algorithms built on PaddlePaddle can be smoothly migrated to secure versions on Paddle Encrypted with little effort.

As a key product of PaddleFL, Paddle Encrypted intrinsically supports federated learning well, including horizontal, vertical and transfer learning scenarios. It provides both provable security (semantic security) and competitive performance.

Below please see the installation, examples, or visit the documentation to learn more about the technical details.

## Design Overview

![img](http://icode.baidu.com/path/to/iamge)

Paddle Encrypted implements secure training and inference tasks based on the underlying MPC protocol of ABY3[], in which participants can be classified into roles of Input Party (IP), Computing Party (CP) and Result Party (RP). 

Input Parties (e.g., the training data/model owners) encrypt and distribute data or models to Computing Parties. Computing Parties (e.g., the VM on the cloud) conduct training or inference tasks based on specific MPC protocols, being restricted to see only the encrypted data or models, and thus guarantee the data privacy. When the computation is completed, one or more Result Parties (e.g., data owners or specified third-party) receive the encrypted results from Computing Parties, and reconstruct the plaintext results. Roles can be overlapped, e.g., a data owner can also act as a computing party.

A full training or inference process in Paddle Encrypted consists of mainly three phases: data preparation, training/inference, and result reconstruction.

#### Data preparation

##### Private data alignment

Paddle Encrypted enables data owners (IPs) to find out records with identical keys (like UUID) without revealing private data to each other. This is especially useful in the vertical learning cases where segmented features with same keys need to be identified and aligned from all owners in a private manner before training. Using the OT-based PSI (Private Set Intersection) algorithm[], PE can perform private alignment at a speed of up to 60k records per second.

##### Encryption and distribution

In Paddle Encrypted, data and models from IPs will be encrypted using Secret-Sharing[], and then be sent to CPs, via directly transmission or distributed storage like HDFS. Each CP can only obtain one share of each piece of data, and thus is unable to recover the original value in the Semi-honest model[].

#### Training/inference

![img](http://icode.baidu.com/path/to/iamge)

As in PaddlePaddle, a training or inference job can be separated into the compile-time phase and the run-time phase:

##### Compile time

* **MPC environment specification**: a user needs to choose a MPC protocol, and configure the network settings. In current version, PE provides only the "ABY3" protocol. More protocol implementation will be provided in future.
* **User-defined job program**: a user can define the machine learning model structure and the training strategies (or inference task) in a PE program, using the secure operators.

##### Run time

A PE program is exactly a PaddlePaddle program, and will be executed as normal PaddlePaddle programs. For example, in run-time a  PE program will be transpiled into ProgramDesc, and then be passed to and run by the Executor. The main concepts in the run-time phase are as follows:

* **Computing nodes**: a computing node is an entity corresponding to a Computing Party. In real deployment, it can be a bare-metal machine, a cloud VM, a docker or even a process. PE requires exactly three computing nodes in each run, which is determined by the underlying ABY3 protocol. A PE program will be deployed and run in parallel on all three computing nodes. 
* **Operators using MPC**: PE provides typical machine learning operators in `paddle.fluid_encrypted` over encrypted data. Such operators are implemented upon PaddlePaddle framework, based on MPC protocols like ABY3. Like other PaddlePaddle operators, in run time, instances of PE operators are created and run in order by Executor (see [] for details).

#### Result reconstruction

Upon completion of the secure training (or inference) job, the models (or prediction results) will be output by CPs in encrypted form. Result Parties can collect the encrypted results, decrypt them using the tools in PE, and deliver the plaintext results to users.

## Compilation and Installation

### Docker Installation 

```sh
#Pull and run the docker
docker pull hub.baidubce.com/paddlefl/paddle_mpc:latest
docker run --name <docker_name> --net=host -it -v $PWD:/root <image id> /bin/bash

#Install paddle_fl
pip install paddle_fl
```

### Compile From Source Code

#### Environment preparation

* CentOS 6 or CentOS 7 (64 bit)
* Python 2.7.15+/3.5.1+/3.6/3.7 ( 64 bit) or above 
* pip or pip3 9.0.1+ (64 bit)
* PaddlePaddle release 1.6.3
* Redis 5.0.8 (64 bit)
* GCC or G++ 4.8.3+
* cmake 3.15+

#### Clone the source code, compile and install

Fetch the source code and checkout stable release
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL

# Checkout stable release
mkdir build && cd build
```

Execute compile commands, where `PYTHON_EXECUTABLE` is path to the python binary where the PaddlePaddle is installed, and `PYTHON_INCLUDE_DIRS` is the corresponding python include directory. You can get the `PYTHON_INCLUDE_DIRS` via the following command:

```sh
${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc;print(get_python_inc())"
```
Then you can put the directory in the following command and make:
```sh
cmake ../ -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIRS=${python_include_dir}
make -j$(nproc)
```

Install the package:

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip or pip3 install dist/***.whl -U
```

Validate the installation by running the `python` or `python3`, then runs `import paddle_encrypted as pe` and `pe.version()`. The installation succeeds if you see `Paddle Encrypted Version: 1.0.0`.

## Example

#### Build your model

In Paddle Encrypted, you can build models as it is in PaddlePaddle, but using the variables and operators over encrypted data. First, prepare a training script as the example below. It is worth to note that the operators and variables are created using the `paddle.fluid_encrypted` package.

```python
# An example to build an LR model, named train.py (USE THE HOUSE PRICE CASE)
import sys
import paddle_fl.mpc as pfl_mpc
import paddle.fluid as fluid
import numpy

# read role from command line
role, addr, port = sys.argv[1], sys.argv[2], sys.argv[3]

# init the MPC environment
pfl_mpc.init("aby3", (int)role, net_server_addr=addr, net_server_port=(int)port)

#data processing
BATCH_SIZE = 10

feature_reader = aby3.load_aby3_shares("/tmp/house_feature", id=role, shape=(13, ))
label_reader = aby3.load_aby3_shares("/tmp/house_label", id=role, shape=(1, ))
batch_feature = aby3.batch(feature_reader, BATCH_SIZE, drop_last=True)
batch_label = aby3.batch(label_reader, BATCH_SIZE, drop_last=True)

# define encrypted variables
x = pfl_mpc.data(name='x', shape=[BATCH_SIZE, 13], dtype='int64')
y = pfl_mpc.data(name='y', shape=[BATCH_SIZE, 1], dtype='int64')

# async data loader
loader = fluid.io.DataLoader.from_generator(feed_list=[x, y], capacity=BATCH_SIZE)
batch_sample = paddle.reader.compose(batch_feature, batch_label)
place = fluid.CPUPlace()
loader.set_batch_generator(batch_sample, places=place)


# define a secure training network
y_pre = pfl_mpc.layers.fc(input=x, size=1)
cost = pfl_mpc.layers.square_error_cost(input=y_pre, label=y)
avg_loss = pfl_mpc.layers.mean(cost)
optimizer = pfl_mpc.optimizer.SGD(learning_rate=0.001)
optimizer.minimize(avg_loss)

# loss file that store encrypted loss
loss_file = "/tmp/uci_loss.part{}".format(role)

# start training
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
epoch_num = 20

start_time = time.time()
for epoch_id in range(epoch_num):
    step = 0

    # feed data via loader
    for sample in loader():
        mpc_loss = exe.run(feed=sample, fetch_list=[avg_loss])

        if step % 50 == 0:
            print('Epoch={}, Step={}, Loss={}'.format(epoch_id, step, mpc_loss))
            with open(loss_file, 'ab') as f:
                f.write(np.array(mpc_loss).tostring())
            step += 1

end_time = time.time()

# training time
print('Mpc Training of Epoch={} Batch_size={}, cost time in seconds:{}'
      .format(epoch_num, BATCH_SIZE, (end_time - start_time)))

# do prediction
prediction_file = "/tmp/uci_prediction.part{}".format(role)
for sample in loader():
    prediction = exe.run(program=infer_program,
                         feed=sample,
                         fetch_list=[y_pre])
    with open(prediction_file, 'ab') as f:
        f.write(np.array(prediction).tostring())
    break

# reveal the loss and prediction 
import prepare_data
print("uci_loss:")
prepare_data.load_decrypt_data("/tmp/uci_loss", (1, ))
print("prediction:")
prepare_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE, ))
```

#### Execution and results

To make the MPC training run, we need to deploy the training processes on multiple machines (i.e., three machines in current version), and use a discovery service to let them find each other. We use Redis as the discovery service here.

1. Start a Redis service, and keep the service address:

```sh
# we provide a stable redis package for you to download 

wget https://paddlefl.bj.bcebos.com/redis-stable.tar --no-check-certificate
tar -xf redis-stable.tar
cd redis-stable && make

# start service
cd src
./redis-server --port ${port}
```

2. Deploy the above `train.py` on three machines, and run with different role settings (from 0 to 2):

```sh
# run python code
# on machine1:
python train.py 0 ${redis_addr} ${port}
# on machine2:
python train.py 1 ${redis_addr} ${port}
# on machine3
python train.py 2 ${redis_addr} ${port}
```

Then the training process will start and the underlying MPC-based operators will be executed to complete the secure training.

## Benchmark Task

#### Convergence of paddle_fl.mpc vs paddle 

##### Training Parameters
- Dataset: Boston house price dataset
- Number of Epoch: 20
- Batch Size: 10

##### Experiment Results

| Epoch/Step | paddle_fl.mpc | Paddle |
| ---------- | ------------- | ------ |
| Epoch=0, Step=0  | 738.39491 | 738.46204 |
| Epoch=1, Step=0  | 630.68834 | 629.9071 |
| Epoch=2, Step=0  | 539.54683 | 538.1757 |
| Epoch=3, Step=0  | 462.41159 | 460.64722 |
| Epoch=4, Step=0  | 397.11516 | 395.11017 |
| Epoch=5, Step=0  | 341.83102 | 339.69815 |
| Epoch=6, Step=0  | 295.01114 | 292.83597 |
| Epoch=7, Step=0  | 255.35141 | 253.19429 |
| Epoch=8, Step=0  | 221.74739 | 219.65132 |
| Epoch=9, Step=0  | 193.26459 | 191.25981 |
| Epoch=10, Step=0  | 169.11423 | 167.2204 |
| Epoch=11, Step=0  | 148.63138 | 146.85835 |
| Epoch=12, Step=0  | 131.25081 | 129.60391 |
| Epoch=13, Step=0  | 116.49708 | 114.97599 |
| Epoch=14, Step=0  | 103.96669 | 102.56854 |
| Epoch=15, Step=0  | 93.31706 | 92.03858 |
| Epoch=16, Step=0  | 84.26219 | 83.09653 |
| Epoch=17, Step=0  | 76.55664 | 75.49785 |
| Epoch=18, Step=0  | 69.99673 | 69.03561 |
| Epoch=19, Step=0  | 64.40562 | 63.53539 |

## On Going and Future Work

- more features

## Reference

[1].
