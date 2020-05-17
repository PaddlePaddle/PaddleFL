
<img src='https://github.com/PaddlePaddle/PaddleFL/blob/master/docs/source/_static/FL-logo.png' width = "400" height = "160">

[DOC](https://paddlefl.readthedocs.io/en/latest/) | [Quick Start](https://paddlefl.readthedocs.io/en/latest/instruction.html) | [中文](./README_cn.md)

PaddleFL is an open source federated learning framework based on PaddlePaddle. Researchers can easily replicate and compare different federated learning algorithms with PaddleFL. Developers can also benefit from PaddleFL in that it is easy to deploy a federated learning system in large scale distributed clusters. In PaddleFL, serveral federated learning strategies will be provided with application in computer vision, natural language processing, recommendation and so on. Application of traditional machine learning training strategies such as Multi-task learning, Transfer Learning in Federated Learning settings will be provided. Based on PaddlePaddle's large scale distributed training and elastic scheduling of training job on Kubernetes, PaddleFL can be easily deployed based on full-stack open sourced software.

## Federated Learning

Data is becoming more and more expensive nowadays, and sharing of raw data is very hard across organizations. Federated Learning aims to solve the problem of data isolation and secure sharing of data knowledge among organizations. The concept of federated learning is proposed by researchers in Google [1, 2, 3]. 

## Compilation and Installation

### Docker Installation

```sh
#Pull and run the docker
docker pull hub.baidubce.com/paddlefl/paddle_fl:latest
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


## Overview of PaddleFL

### Horizontal Federated Learning

<img src='images/FL-framework.png' width = "1000" height = "320" align="middle"/>

In PaddleFL, horizontal and vertical federated learning strategies will be implemented according to the categorization given in [4]. Application demonstrations in natural language processing, computer vision and recommendation will be provided in PaddleFL. 

#### Federated Learning Strategy

- **Vertical Federated Learning**: Logistic Regression with PrivC, Neural Network with third-party PrivC [5]

- **Horizontal Federated Learning**: Federated Averaging [2], Differential Privacy [6], Secure Aggregation

#### Training Strategy

- **Multi Task Learning** [7]

- **Transfer Learning** [8]

- **Active Learning**

### Paddle Encrypted

Paddle Fluid Encrypted is a framework for privacy-preserving deep learning based on PaddlePaddle. It follows the same running mechanism and programming paradigm with PaddlePaddle, while using secure multi-party computation (MPC) to enable secure training and prediction. 

With Paddle Fluid Encrypted, it is easy to train models or conduct prediction as on PaddlePaddle over encrypted data, without the need for cryptography expertise. Furthermore, the rich industry-oriented models and algorithms built on PaddlePaddle can be smoothly migrated to secure versions on Paddle Fluid Encrypted with little effort.

As a key product of PaddleFL, Paddle Fluid Encrypted intrinsically supports federated learning well, including horizontal, vertical and transfer learning scenarios. It provides both provable security (semantic security) and competitive performance.

## Framework design of PaddleFL

### Horizontal Federated Learning
<img src='images/FL-training.png' width = "1000" height = "400" align="middle"/>

In PaddleFL, components for defining a federated learning task and training a federated learning job are as follows:

#### Compile Time

- **FL-Strategy**: a user can define federated learning strategies with FL-Strategy such as Fed-Avg[2]

- **User-Defined-Program**: PaddlePaddle's program that defines the machine learning model structure and training strategies such as multi-task learning.

- **Distributed-Config**: In federated learning, a system should be deployed in distributed settings. Distributed Training Config defines distributed training node information.

- **FL-Job-Generator**: Given FL-Strategy, User-Defined Program and Distributed Training Config, FL-Job for federated server and worker will be generated through FL Job Generator. FL-Jobs will be sent to organizations and federated parameter server for run-time execution.

#### Run Time

- **FL-Server**: federated parameter server that usually runs in cloud or third-party clusters.

- **FL-Worker**: Each organization participates in federated learning will have one or more federated workers that will communicate with the federated parameter server.

- **FL-scheduler**: Decide which set of trainers can join the training before each updating cycle. 

### Paddle Encrypted

Paddle Fluid Encrypted implements secure training and inference tasks based on the underlying MPC protocol of ABY3[], in which participants can be classified into roles of Input Party (IP), Computing Party (CP) and Result Party (RP). 

Input Parties (e.g., the training data/model owners) encrypt and distribute data or models to Computing Parties. Computing Parties (e.g., the VM on the cloud) conduct training or inference tasks based on specific MPC protocols, being restricted to see only the encrypted data or models, and thus guarantee the data privacy. When the computation is completed, one or more Result Parties (e.g., data owners or specified third-party) receive the encrypted results from Computing Parties, and reconstruct the plaintext results. Roles can be overlapped, e.g., a data owner can also act as a computing party.

A full training or inference process in Paddle Fluid Encrypted consists of mainly three phases: data preparation, training/inference, and result reconstruction.

#### Data preparation

##### Private data alignment

Paddle Fluid Encrypted enables data owners (IPs) to find out records with identical keys (like UUID) without revealing private data to each other. This is especially useful in the vertical learning cases where segmented features with same keys need to be identified and aligned from all owners in a private manner before training. Using the OT-based PSI (Private Set Intersection) algorithm[], PFE can perform private alignment at a speed of up to 60k records per second.

##### Encryption and distribution

In Paddle Fluid Encrypted, data and models from IPs will be encrypted using Secret-Sharing[], and then be sent to CPs, via directly transmission or distributed storage like HDFS. Each CP can only obtain one share of each piece of data, and thus is unable to recover the original value in the Semi-honest model[].

#### Training/inference

![img](http://icode.baidu.com/path/to/iamge)

As in PaddlePaddle, a training or inference job can be separated into the compile-time phase and the run-time phase:

##### Compile time

* **MPC environment specification**: a user needs to choose a MPC protocol, and configure the network settings. In current version, PFE provides only the "ABY3" protocol. More protocol implementation will be provided in future.
* **User-defined job program**: a user can define the machine learning model structure and the training strategies (or inference task) in a PFE program, using the secure operators.

##### Run time

A PFE program is exactly a PaddlePaddle program, and will be executed as normal PaddlePaddle programs. For example, in run-time a  PFE program will be transpiled into ProgramDesc, and then be passed to and run by the Executor. The main concepts in the run-time phase are as follows:

* **Computing nodes**: a computing node is an entity corresponding to a Computing Party. In real deployment, it can be a bare-metal machine, a cloud VM, a docker or even a process. PFE requires exactly three computing nodes in each run, which is determined by the underlying ABY3 protocol. A PFE program will be deployed and run in parallel on all three computing nodes. 
* **Operators using MPC**: PFE provides typical machine learning operators in `paddle.fluid_encrypted` over encrypted data. Such operators are implemented upon PaddlePaddle framework, based on MPC protocols like ABY3. Like other PaddlePaddle operators, in run time, instances of PFE operators are created and run in order by Executor (see [] for details).

#### Result reconstruction

Upon completion of the secure training (or inference) job, the models (or prediction results) will be output by CPs in encrypted form. Result Parties can collect the encrypted results, decrypt them using the tools in PFE, and deliver the plaintext results to users.

## Easy deployment with kubernetes

### Horizontal Federated Learning
```sh

kubectl apply -f ./python/paddle_fl/paddle_fl/examples/k8s_deployment/master.yaml

```
Please refer [K8S deployment example](./python/paddle_fl/paddle_fl/examples/k8s_deployment/README.md) for details

You can also refer [K8S cluster application and kubectl installation](./python/paddle_fl/paddle_fl/examples/k8s_deployment/deploy_instruction.md) to deploy your K8S cluster

### Paddle Encrypted

To be added.

## Benchmark task

### Horzontal Federated Learning

Gru4Rec [9] introduces recurrent neural network model in session-based recommendation. PaddlePaddle's Gru4Rec implementation is in https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec. An example is given in [Gru4Rec in Federated Learning](https://paddlefl.readthedocs.io/en/latest/examples/gru4rec_examples.html)

### Paddle Encrypted 

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

- Vertial Federated Learning will support more algorithms.

- Add K8S deployment scheme for Paddle Encrypted.

## Reference

[1]. Jakub Konečný, H. Brendan McMahan, Daniel Ramage, Peter Richtárik. **Federated Optimization: Distributed Machine Learning for On-Device Intelligence.** 2016

[2]. H. Brendan McMahan, Eider Moore, Daniel Ramage, Blaise Agüera y Arcas. **Federated Learning of Deep Networks using Model Averaging.** 2017

[3]. Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, Dave Bacon. **Federated Learning: Strategies for Improving Communication Efficiency.** 2016

[4]. Qiang Yang, Yang Liu, Tianjian Chen, Yongxin Tong. **Federated Machine Learning: Concept and Applications.** 2019

[5]. Kai He, Liu Yang, Jue Hong, Jinghua Jiang, Jieming Wu, Xu Dong et al. **PrivC  - A framework for efficient Secure Two-Party Computation. In Proceedings of 15th EAI International Conference on Security and Privacy in Communication Networks.** SecureComm 2019

[6]. Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. **Deep Learning with Differential Privacy.** 2016

[7]. Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, Ameet Talwalkar. **Federated Multi-Task Learning** 2016

[8]. Yang Liu, Tianjian Chen, Qiang Yang. **Secure Federated Transfer Learning.** 2018

[9]. Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk. **Session-based Recommendations with Recurrent Neural Networks.** 2016
