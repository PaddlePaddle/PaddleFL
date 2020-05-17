# PaddleFL

PaddleFL是一个基于PaddlePaddle的开源联邦学习框架。研究人员可以很轻松地用PaddleFL复制和比较不同的联邦学习算法。开发人员也可以从padderFL中获益，因为用PaddleFL在大规模分布式集群中部署联邦学习系统很容易。PaddleFL提供很多联邦学习策略及其在计算机视觉、自然语言处理、推荐算法等领域的应用。此外，PaddleFL还将提供传统机器学习训练策略的应用，例如多任务学习、联邦学习环境下的迁移学习。依靠着PaddlePaddle的大规模分布式训练和Kubernetes对训练任务的弹性调度能力，PaddleFL可以基于全栈开源软件轻松地部署。

## 联邦学习

如今，数据变得越来越昂贵，而且跨组织共享原始数据非常困难。联合学习旨在解决组织间数据隔离和数据知识安全共享的问题。联邦学习的概念是由谷歌的研究人员提出的[1，2，3]。

## 编译与安装

## PaddleFL概述

### 横向联邦方案

<img src='images/FL-framework-zh.png' width = "1300" height = "310" align="middle"/>

在PaddleFL中，横向和纵向联邦学习策略将根据[4]中给出的分类来实现。PaddleFL也将提供在自然语言处理，计算机视觉和推荐算法等领域的应用示例。

#### 联邦学习策略

- **纵向联邦学习**: 带privc的逻辑回归，带第三方privc的神经网络[5]

- **横向联邦学习**: 联邦平均 [2]，差分隐私 [6]，安全聚合

#### 训练策略

- **多任务学习** [7]

- **迁移学习** [8]

- **主动学习**

### Paddle Encrypted

Paddle Encrypted 是一个基于PaddlePaddle的隐私保护深度学习框架。Paddle Encrypted基于多方计算（MPC）实现安全训练及预测，拥有与PaddlePaddle相同的运行机制及编程范式。

Paddle Encrypted 设计与PaddlePaddle相似，没有密码学相关背景的用户亦可简单的对加密的数据进行训练和预测。同时，PaddlePaddle中丰富的模型和算法可以轻易地迁移到Paddle Encrypted中。

作为PaddleFL的一个重要组成部分，Paddle Encrypted可以很好滴支持联邦学习，包括横向、纵向及联邦迁移学习等多个场景。既提供了可靠的安全性，也拥有可观的性能。

## PaddleFL框架设计

### 横向联邦方案

<img src='images/FL-training.png' width = "1300" height = "450" align="middle"/>

在PaddeFL中，用于定义联邦学习任务和联邦学习训练工作的组件如下：

#### 编译时

- **FL-Strategy**: 用户可以使用FL-Strategy定义联邦学习策略，例如Fed-Avg[2]。

- **User-Defined-Program**: PaddlePaddle的程序定义了机器学习模型结构和训练策略，如多任务学习。

- **Distributed-Config**: 在联邦学习中，系统会部署在分布式环境中。分布式训练配置定义分布式训练节点信息。

- **FL-Job-Generator**: 给定FL-Strategy, User-Defined Program 和 Distributed Training Config，联邦参数的Server端和Worker端的FL-Job将通过FL Job Generator生成。FL-Jobs 被发送到组织和联邦参数服务器以进行联合训练。

#### 运行时

- **FL-Server**: 在云或第三方集群中运行的联邦参数服务器。

- **FL-Worker**: 参与联合学习的每个组织都将有一个或多个与联合参数服务器通信的Worker。

- **FL-Scheduler**: 训练过程中起到调度Worker的作用，在每个更新周期前，决定哪些Worker可以参与训练。

请参考更多的[例子](./python/paddle_fl/paddle_fl/examples), 获取更多的信息。
### Paddle Encrypted

Paddle Encrypted 中的安全训练和推理任务是基于底层的ABY3多方计算协议实现的。在ABY3中，参与方可分为：输入方、计算方和结果方。

输入方为训练数据及模型的持有方，负责加密数据和模型，并将其发送到计算方。计算方为训练的执行方，基于特定的多方安全计算协议完成训练任务。计算方只能得到加密后的数据及模型，以保证数据隐>私。计算结束后，结果方会拿到计算结果并恢复出明文数据。每个参与方可充当多个角色，如一个数据拥有方也可以作为计算方参与训练。

Paddle Encrypted的整个训练及推理过程主要由三个部分组成：数据准备，训练/推理，结果解析。

#### 数据准备

##### 私有数据对齐

Paddle Encrypted允许数据拥有方（数据方）在不泄露自己数据的情况下，找出多方共有的样本集合。此功能在纵向联邦学习中非常必要，因为其要求多个数据方在训练前进行数据对齐，并且保护用户的数>据隐私。凭借PSI算法，Paddle Encrypted可以在一秒内完成6万条数据的对齐。

##### 数据加密及分发

在Paddle Encrypted中，数据方将数据和模型用秘密共享的方法加密，然后用直接传输或者数据库存储的方式传到计算方。每个计算方只会拿到数据的一部分，因此计算方无法还原真实数据。

#### 训练及推理

像PaddlePaddle一样，训练和推理任务可以分为编译阶段和运行阶段。

##### 编译时

* **确定MPC环境**：用户需要指定用到的MPC协议，并配置网络环境。现有版本的Paddle Encrypted只支持"ABY3"协议。更多的协议将在后续版本中支持。
* **用户定义训练任务**：用户可以根据Paddle Encrypted提供的安全接口，定义集齐学习网络以及训练策略。
##### 运行时

一个Paddle Encrypted程序实际上就是一个PaddlePaddle程序。在运行时，Paddle Encrypted的程序将会转变为PaddlePaddle中的ProgramDesc，并交给Executor运行。以下是运行阶段的主要概念：
* **运算节点**：计算节点是与计算方相对应的实体。在实际部署中，它可以是裸机、云虚拟机、docker甚至进程。Paddle Encrypted在每次运行中只需要三个计算节点，这由底层ABY3协议决定。Paddle Encrypted程序将在所有三个计算节点上并行部署和运行。
* **基于MPC的算子**：Paddle Encrypted 为操作加密数据提供了特殊的算子，这些算子在PaddlePaddle框架中实现，基于像ABY3一样的MPC协议。像PaddlePaddle中一样，在运行时Paddle Encrypted的算子将被创建并按照顺序执行。
#### 结果重构

安全训练和推理工作完成后，模型（或预测结果）将由计算方以加密形式输出。结果方可以收集加密的结果，使用Paddle Encrypted中的工具对其进行解密，并将明文结果传递给用户。

请参考[MPC的例子](./python/paddle_fl/mpc/examples)，以获取更多的信息。
## Kubernetes简单部署

### 横向联邦方案
```sh
 
kubectl apply -f ./python/paddle_fl/paddle_fl/examples/k8s_deployment/master.yaml

```
请参考[K8S部署实例](./python/paddle_fl/paddle_fl/examples/k8s_deployment/README.md)

也可以参考[K8S集群申请及kubectl安装](./python/paddle_fl/paddle_fl/examples/k8s_deployment/deploy_instruction.md) 配置自己的K8S集群

### Paddle Encrypted

会在后续版本中发布。
## 性能测试

### 横向联邦方案
Gru4Rec [9] 在基于会话的推荐中引入了递归神经网络模型。PaddlePaddle的GRU4RC实现代码在 https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec. 一个基于联邦学习训练Gru4Rec模型的示例请参考[Gru4Rec in Federated Learning](https://paddlefl.readthedocs.io/en/latest/examples/gru4rec_examples.html)

### Paddle Encrypted

#### 精度测试

##### 训练参数

- 数据集：波士顿房价预测。
- 训练轮数： 20
- Batch Size：10

#####实验结果

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

## 正在进行的工作

- 纵向联合学习支持更多的模型。
- 发布纵向联邦学习K8S部署方案。

## 参考文献

[1]. Jakub Konečný, H. Brendan McMahan, Daniel Ramage, Peter Richtárik. **Federated Optimization: Distributed Machine Learning for On-Device Intelligence.** 2016

[2]. H. Brendan McMahan, Eider Moore, Daniel Ramage, Blaise Agüera y Arcas. **Federated Learning of Deep Networks using Model Averaging.** 2017

[3]. Jakub Konečný, H. Brendan McMahan, Felix X. Yu, Peter Richtárik, Ananda Theertha Suresh, Dave Bacon. **Federated Learning: Strategies for Improving Communication Efficiency.** 2016

[4]. Qiang Yang, Yang Liu, Tianjian Chen, Yongxin Tong. **Federated Machine Learning: Concept and Applications.** 2019

[5]. Kai He, Liu Yang, Jue Hong, Jinghua Jiang, Jieming Wu, Xu Dong et al. **PrivC  - A framework for efficient Secure Two-Party Computation. In Proceedings of 15th EAI International Conference on Security and Privacy in Communication Networks.** SecureComm 2019

[6]. Martín Abadi, Andy Chu, Ian Goodfellow, H. Brendan McMahan, Ilya Mironov, Kunal Talwar, Li Zhang. **Deep Learning with Differential Privacy.** 2016

[7]. Virginia Smith, Chao-Kai Chiang, Maziar Sanjabi, Ameet Talwalkar. **Federated Multi-Task Learning** 2016

[8]. Yang Liu, Tianjian Chen, Qiang Yang. **Secure Federated Transfer Learning.** 2018

[9]. Balázs Hidasi, Alexandros Karatzoglou, Linas Baltrunas, Domonkos Tikk. **Session-based Recommendations with Recurrent Neural Networks.** 2016
