# PaddleFL

PaddleFL是一个基于PaddlePaddle的开源联邦学习框架。研究人员可以很轻松地用PaddleFL复制和比较不同的联邦学习算法。开发人员也可以从paddleFL中获益，因为用PaddleFL在大规模分布式集群中部署联邦学习系统很容易。PaddleFL提供很多联邦学习策略及其在计算机视觉、自然语言处理、推荐算法等领域的应用。此外，PaddleFL还将提供传统机器学习训练策略的应用，例如多任务学习、联邦学习环境下的迁移学习。依靠着PaddlePaddle的大规模分布式训练和Kubernetes对训练任务的弹性调度能力，PaddleFL可以基于全栈开源软件轻松地部署。

## 联邦学习

如今，数据变得越来越昂贵，而且跨组织共享原始数据非常困难。联合学习旨在解决组织间数据隔离和数据知识安全共享的问题。联邦学习的概念是由谷歌的研究人员提出的[1，2，3]。

## PaddleFL概述

<img src='images/FL-framework-zh.png' width = "1300" height = "310" align="middle"/>

在PaddleFL中，横向和纵向联邦学习策略将根据[4]中给出的分类来实现。PaddleFL也将提供在自然语言处理，计算机视觉和推荐算法等领域的应用示例。

#### A. 联邦学习策略

- **纵向联邦学习**: 带privc的逻辑回归，带ABY3[11]的神经网络

- **横向联邦学习**: 联邦平均 [2]，差分隐私 [6]，安全聚合

#### B. 训练策略

- **多任务学习** [7]

- **迁移学习** [8]

- **主动学习**

PaddleFL 中主要提供两种解决方案：**Data Parallel** 以及 **Federated Learning with MPC (PFM)**。

通过Data Parallel，各数据方可以基于经典的横向联邦学习策略（如 FedAvg，DPSGD等）完成模型训练。

此外，PFM是基于多方安全计算（MPC）实现的联邦学习方案。作为PaddleFL的一个重要组成部分，PFM可以很好地支持联邦学习，包括横向、纵向及联邦迁移学习等多个场景。既提供了可靠的安全性，也拥有可观的性能。

## 安装

我们**强烈建议** 您在docker中使用PaddleFL。

```sh
#Pull and run the docker
docker pull paddlepaddle/paddlefl:latest
docker run --name <docker_name> --net=host -it -v $PWD:/paddle <image id> /bin/bash

#Install paddle_fl
pip install paddle_fl
```

若您希望从源码编译安装，请点击[这里](./docs/source/md/compile_and_install_cn.md)。

我们也提供了稳定的redis安装包, 可供下载。

```sh
wget --no-check-certificate https://paddlefl.bj.bcebos.com/redis-stable.tar
tar -xf redis-stable.tar
cd redis-stable &&  make
```

## Kubernetes简单部署

### 横向联邦方案
```sh

kubectl apply -f ./python/paddle_fl/paddle_fl/examples/k8s_deployment/master.yaml

```
请参考[K8S部署实例](./python/paddle_fl/paddle_fl/examples/k8s_deployment/README.md)

也可以参考[K8S集群申请及kubectl安装](./python/paddle_fl/paddle_fl/examples/k8s_deployment/deploy_instruction.md) 配置自己的K8S集群

### Federated Learning with MPC

会在后续版本中发布。

## PaddleFL框架设计

### Data Parallel 

<img src='images/FL-training.png' width = "1300" height = "450" align="middle"/>

在PaddeFL中，用于定义联邦学习任务和联邦学习训练工作的组件如下：

#### A. 编译时

- **FL-Strategy**: 用户可以使用FL-Strategy定义联邦学习策略，例如Fed-Avg[2]。

- **User-Defined-Program**: PaddlePaddle的程序定义了机器学习模型结构和训练策略，如多任务学习。

- **Distributed-Config**: 在联邦学习中，系统会部署在分布式环境中。分布式训练配置定义分布式训练节点信息。

- **FL-Job-Generator**: 给定FL-Strategy, User-Defined Program 和 Distributed Training Config，联邦参数的Server端和Worker端的FL-Job将通过FL Job Generator生成。FL-Jobs 被发送到组织和联邦参数服务器以进行联合训练。

#### B. 运行时

- **FL-Server**: 在云或第三方集群中运行的联邦参数服务器。

- **FL-Worker**: 参与联合学习的每个组织都将有一个或多个与联合参数服务器通信的Worker。

- **FL-Scheduler**: 训练过程中起到调度Worker的作用，在每个更新周期前，决定哪些Worker可以参与训练。

请参考更多的[例子](./python/paddle_fl/paddle_fl/examples), 获取更多的信息。

### Federated Learning with MPC

<img src='images/PFM-overview.png' width = "1000" height = "446" align="middle"/>

Paddle FL MPC 中的安全训练和推理任务是基于高效的多方计算协议实现的，如ABY3[11]

在ABY3[11]中，参与方可分为：输入方、计算方和结果方。输入方为训练数据及模型的持有方，负责加密数据和模型，并将其发送到计算方。计算方为训练的执行方，基于特定的多方安全计算协议完成训练任务。计算方只能得到加密后的数据
及模型，以保证数据隐私。计算结束后，结果方会拿到计算结果并恢复出明文数据。每个参与方可充当多个角色，如一个数据拥有方也可以作为计算方参与训练。

PFM的整个训练及推理过程主要由三个部分组成：数据准备，训练/推理，结果解析。

#### A. 数据准备

- **私有数据对齐**： PFM允许数据拥有方（数据方）在不泄露自己数据的情况下，找出多方共有的样本集合。此功能在纵向联邦学习中非常必要，因为其要求多个数据方在训练前进行数据对齐，并且保护用户的数据隐私。
- **数据加密及分发**：在PFM中，数据方将数据和模型用秘密共享[10]的方法加密，然后用直接传输或者数据库存储的方式传到计算方。每个计算方只会拿到数据的一部分，因此计算方无法还原真实数据。

#### B. 训练/推理

PFM 拥有与PaddlePaddle相同的运行模式。在训练前，用户需要定义MPC协议，训练模型以及训练策略。`paddle_fl.mpc`中提供了可以操作加密数据的算子，在运行时算子的实例会被创建并被执行器依次运行。

请参考以下[文档](./docs/source/md/mpc_train_cn.md), 以获得更多关于训练阶段的信息。
#### C. 结果重构

安全训练和推理工作完成后，模型（或预测结果）将由计算方以加密形式输出。结果方可以收集加密的结果，使用PFM中的工具对其进行解密，并将明文结果传递给用户。

请参考[MPC的例子](./python/paddle_fl/mpc/examples)，以获取更多的信息。

## 性能测试

### 横向联邦方案
Gru4Rec [9] 在基于会话的推荐中引入了递归神经网络模型。PaddlePaddle的GRU4RC实现代码在 https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec. 一个基于联邦学习训练Gru4Rec模型的示例请参考[Gru4Rec in Federated Learning](https://paddlefl.readthedocs.io/en/latest/examples/gru4rec_examples.html)

### Federated Learning with MPC 
我们基于波士顿房价数据集对PFM进行了测试，具体的事例及实现请参考 [uci_demo](./python/paddle_fl/mpc/examples/uci_demo)

## 正在进行的工作

- PFM支持更多的模型。
- 发布PFM的K8S部署方案。
- 手机端的联邦学习模拟器将在下一版本开源。

## 参考文献

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
