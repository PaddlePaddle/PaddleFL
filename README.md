
<img src='https://github.com/PaddlePaddle/PaddleFL/blob/master/docs/source/_static/FL-logo.png' width = "400" height = "160">

[DOC](https://paddlefl.readthedocs.io/en/latest/) | [Quick Start](https://paddlefl.readthedocs.io/en/latest/instruction.html) | [中文](./README_cn.md)

PaddleFL is an open source federated learning framework based on PaddlePaddle. Researchers can easily replicate and compare different federated learning algorithms with PaddleFL. Developers can also benefit from PaddleFL in that it is easy to deploy a federated learning system in large scale distributed clusters. In PaddleFL, serveral federated learning strategies will be provided with application in computer vision, natural language processing, recommendation and so on. Application of traditional machine learning training strategies such as Multi-task learning, Transfer Learning in Federated Learning settings will be provided. Based on PaddlePaddle's large scale distributed training and elastic scheduling of training job on Kubernetes, PaddleFL can be easily deployed based on full-stack open sourced software.

## Federated Learning

Data is becoming more and more expensive nowadays, and sharing of raw data is very hard across organizations. Federated Learning aims to solve the problem of data isolation and secure sharing of data knowledge among organizations. The concept of federated learning is proposed by researchers in Google [1, 2, 3]. 

## Overview of PaddleFL

<img src='images/FL-framework.png' width = "1000" height = "320" align="middle"/>

In PaddleFL, horizontal and vertical federated learning strategies will be implemented according to the categorization given in [4]. Application demonstrations in natural language processing, computer vision and recommendation will be provided in PaddleFL. 

#### A. Federated Learning Strategy

- **Vertical Federated Learning**: Logistic Regression with PrivC[5], Neural Network with MPC [11]

- **Horizontal Federated Learning**: Federated Averaging [2], Differential Privacy [6], Secure Aggregation

#### B. Training Strategy

- **Multi Task Learning** [7]

- **Transfer Learning** [8]

- **Active Learning**

There are mainly two components in PaddleFL: **Data Parallel** and **Federated Learning with MPC (PFM)**.

With Data Parallel, distributed data holders can finish their Federated Learning tasks based on common horizontal federated strategies, such as FedAvg, DPSGD, etc.

Besides, PFM is implemented based on secure multi-party computation (MPC) to enable secure training and prediction. As a key product of PaddleFL, PFM intrinsically supports federated learning well, including horizontal, vertical and transfer learning scenarios. Users with little cryptography expertise can also train models or conduct prediction on encrypted data.

## Installation

We **highly recommend** to run PaddleFL in Docker 

```sh
#Pull and run the docker
docker pull hub.baidubce.com/paddlefl/paddle_fl:latest
docker run --name <docker_name> --net=host -it -v $PWD:/paddle <image id> /bin/bash

#Install paddle_fl
pip install paddle_fl
```

If you want to compile and install from source code, please click [here](./docs/source/md/compile_and_install.md) to get instructions. 

We also prepare a stable redis package for you to download and install, which will be used in tasks with MPC. 

```sh
wget --no-check-certificate https://paddlefl.bj.bcebos.com/redis-stable.tar
tar -xf redis-stable.tar
cd redis-stable &&  make
```

## Easy deployment with kubernetes

### Data Parallel
```sh

kubectl apply -f ./python/paddle_fl/paddle_fl/examples/k8s_deployment/master.yaml

```
Please refer [K8S deployment example](./python/paddle_fl/paddle_fl/examples/k8s_deployment/README.md) for details

You can also refer [K8S cluster application and kubectl installation](./python/paddle_fl/paddle_fl/examples/k8s_deployment/deploy_instruction.md) to deploy your K8S cluster

### Federated Learning with MPC

To be added.

## Framework design of PaddleFL

### Data Parallel

<img src='images/FL-training.png' width = "1000" height = "400" align="middle"/>

In Data Parallel, components for defining a federated learning task and training a federated learning job are as follows:

#### A. Compile Time

- **FL-Strategy**: a user can define federated learning strategies with FL-Strategy such as Fed-Avg[2]

- **User-Defined-Program**: PaddlePaddle's program that defines the machine learning model structure and training strategies such as multi-task learning.

- **Distributed-Config**: In federated learning, a system should be deployed in distributed settings. Distributed Training Config defines distributed training node information.

- **FL-Job-Generator**: Given FL-Strategy, User-Defined Program and Distributed Training Config, FL-Job for federated server and worker will be generated through FL Job Generator. FL-Jobs will be sent to organizations and federated parameter server for run-time execution.

#### B. Run Time

- **FL-Server**: federated parameter server that usually runs in cloud or third-party clusters.

- **FL-Worker**: Each organization participates in federated learning will have one or more federated workers that will communicate with the federated parameter server.

- **FL-scheduler**: Decide which set of trainers can join the training before each updating cycle.

For more instructions, please refer to the [examples](./python/paddle_fl/paddle_fl/examples)

### Federated Learning with MPC

<img src='images/PFM-overview.png' width = "1000" height = "446" align="middle"/>

Paddle FL MPC implements secure training and inference tasks based on the underlying MPC protocol like ABY3[11], which is a high efficient three-party computing model.

In ABY3, participants can be classified into roles of Input Party (IP), Computing Party (CP) and Result Party (RP). Input Parties (e.g., the training data/model owners) encrypt and distribute data or models to Computing Parties. Computing Parties (e.g., the VM on the cloud) conduct training or inference tasks based on specific MPC protocols, being restricted to see only the encrypted data or models, and thus guarantee the data privacy. When the computation is completed, one or more Result Parties (e.g., data owners or specified third-party) receive the encrypted results from Computing Parties, and reconstruct the plaintext results. Roles can be overlapped, e.g., a data owner can also act as a computing party. 

A full training or inference process in PFM consists of mainly three phases: data preparation, training/inference, and result reconstruction.

#### A. Data preparation

- **Private data alignment**: PFM enables data owners (IPs) to find out records with identical keys (like UUID) without revealing private data to each other. This is especially useful in the vertical learning cases where segmented features with same keys need to be identified and aligned from all owners in a private manner before training.

- **Encryption and distribution**: In PFM, data and models from IPs will be encrypted using Secret-Sharing[10], and then be sent to CPs, via directly transmission or distributed storage like HDFS. Each CP can only obtain one share of each piece of data, and thus is unable to recover the original value in the Semi-honest model.

#### B. Training/inference

A PFM program is exactly a PaddlePaddle program, and will be executed as normal PaddlePaddle programs. Before training/inference, user needs to choose a MPC protocol, define a machine learning model and their training strategies. Typical machine learning operators are provided in `paddle_fl.mpc` over encrypted data, of which the instances are created and run in order by Executor during run-time.

For more information of Training/inference phase, please refer to the following [doc](./docs/source/md/mpc_train.md).

#### C. Result reconstruction

Upon completion of the secure training (or inference) job, the models (or prediction results) will be output by CPs in encrypted form. Result Parties can collect the encrypted results, decrypt them using the tools in PFM, and deliver the plaintext results to users.

For more instructions, please refer to [mpc examples](./python/paddle_fl/mpc/examples) 

## Benchmark task

### Data Parallel 

Gru4Rec [9] introduces recurrent neural network model in session-based recommendation. PaddlePaddle's Gru4Rec implementation is in https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec. An example is given in [Gru4Rec in Federated Learning](https://paddlefl.readthedocs.io/en/latest/examples/gru4rec_examples.html)

### Federated Learning with MPC 

We conduct tests on PFM using Boston house price dataset, and the implementation is given in [uci_demo](./python/paddle_fl/mpc/examples/uci_demo)

## On Going and Future Work

- Vertial Federated Learning will support more algorithms.

- Add K8S deployment scheme for PFM.

- FL mobile simulator will be open sourced in following versions.

## Reference

[1]. Jakub Konečný, H. Brendan McMahan, Daniel Ramage, Peter Richtárik. **Federated Optimization: Distributed Machine Learning for On-Device Intelligence.** 2016

[2]. H. Brendan McMahan, Eider Moore, Daniel Ramage, Blaise Agüera y Arcas. **Federated Learning of Deep Networks using Model Averaging.** 2017

[3]. Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, Dave Bacon. **Federated Learning: Strategies for Improving Communication Efficiency.** 2016

[4]. Qiang Yang, Yang Liu, Tianjian Chen, Yongxin Tong. **Federated Machine Learning: Concept and Applications.** 2019

[5]. Kai He, Liu Yang, Jue Hong, Jinghua Jiang, Jieming Wu, Xu Dong et al. **PrivC  - A framework for efficient Secure Two-Party Computation.** In Proc. of SecureComm 2019

[6]. Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. **Deep Learning with Differential Privacy.** 2016

[7]. Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, Ameet Talwalkar. **Federated Multi-Task Learning** 2016

[8]. Yang Liu, Tianjian Chen, Qiang Yang. **Secure Federated Transfer Learning.** 2018

[9]. Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk. **Session-based Recommendations with Recurrent Neural Networks.** 2016

[10]. https://en.wikipedia.org/wiki/Secret_sharing

[11]. Payman Mohassel and Peter Rindal. **ABY3: A Mixed Protocol Framework for Machine Learning.** In Proc. of CCS 2018
