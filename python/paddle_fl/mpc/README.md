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

# define encrypted variables
image = pfl_mpc.data(name='image', shape=[None, 784], dtype='int64')
label = pfl_mpc.data(name='label', shape=[None, 1], dtype='int64')

# define a secure training network
hidden = pfl_mpc.layers.fc(input=image, size=100, act='relu')
prediction = pfl_mpc.layers.fc(input=hidden, size=10, act='softmax')
cost = pfl_mpc.layers.square_error_cost(input=prediction, label=label)
loss = pfl_mpc.layers.mean(cost)

sgd = pfl_mpc.optimizer.SGD(learning_rate=0.001)
sgd.minimize(loss)

# Place the training on CPU
exe = fluid.Executor(place=fluid.CPUPlace())

# use random numbers to simulate encrypted data, and start training
x = numpy.random.random(size=(128, 2, 784)).astype('int64')
y = numpy.random.random(size=(128, 2, 1)).astype('int64')
loss_data, = exe.run(feed={'image':x, 'lable':y},
                     fetch_list=[loss.name])
```

#### Execution and results

To make the MPC training run, we need to deploy the training processes on multiple machines (i.e., three machines in current version), and use a discovery service to let them find each other. We use Redis as the discovery service here.

1. Start a Redis service, and keep the service address:

```sh
redis-server --port ${port}
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

put result here as a table? | DataSet/Task | training methods | Result | | --- | --- | --- |

## On Going and Future Work

- more features

## Reference

[1].
