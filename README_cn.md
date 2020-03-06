# PaddleFL

PaddleFL是一个基于PaddlePaddle的开源联邦学习框架。研究人员可以很轻松地用PaddleFL复制和比较不同的联邦学习算法。开发人员也可以从padderFL中获益，因为用PaddleFL在大规模分布式集群中部署联邦学习系统很容易。PaddleFL提供很多联邦学习策略及其在计算机视觉、自然语言处理、推荐算法等领域的应用。此外，PaddleFL还将提供传统机器学习训练策略的应用，例如多任务学习、联邦学习环境下的迁移学习。依靠着PaddlePaddle的大规模分布式训练和Kubernetes对训练任务的弹性调度能力，PaddleFL可以基于全栈开源软件轻松地部署。

## 联邦学习

如今，数据变得越来越昂贵，而且跨组织共享原始数据非常困难。联合学习旨在解决组织间数据隔离和数据知识安全共享的问题。联邦学习的概念是由谷歌的研究人员提出的[1，2，3]。

## PaddleFL概述

<img src='images/FL-framework-zh.png' width = "1300" height = "310" align="middle"/>

在PaddleFL中，横向和纵向联邦学习策略将根据[4]中给出的分类来实现。PaddleFL也将提供在自然语言处理，计算机视觉和推荐算法等领域的应用示例。

#### 联邦学习策略

- **纵向联邦学习**: 带privc的逻辑回归，带第三方privc的神经网络[5]

- **横向联邦学习**: 联邦平均 [2]，差分隐私 [6]

#### 训练策略

- **多任务学习** [7]

- **迁移学习** [8]

- **主动学习**


## PaddleFL框架设计

<img src='images/FL-training.png' width = "1300" height = "450" align="middle"/>

在PaddeFL中，用于定义联邦学习任务和联邦学习训练工作的组件如下：

#### 编译时

- **FL-Strategy**: 用户可以使用FL-Strategy定义联邦学习策略，例如Fed-Avg[1]。

- **User-Defined-Program**: PaddlePaddle的程序定义了机器学习模型结构和训练策略，如多任务学习。

- **Distributed-Config**: 在联邦学习中，系统会部署在分布式环境中。分布式训练配置定义分布式训练节点信息。

- **FL-Job-Generator**: 给定FL-Strategy, User-Defined Program 和 Distributed Training Config，联邦参数的Server端和Worker端的FL-Job将通过FL Job Generator生成。FL-Jobs 被发送到组织和联邦参数服务器以进行联合训练。

#### 运行时

- **FL-Server**: 在云或第三方集群中运行的联邦参数服务器。

- **FL-Worker**: 参与联合学习的每个组织都将有一个或多个与联合参数服务器通信的Worker。

- **FL-Scheduler**: 训练过程中起到调度Worker的作用，在每个更新周期前，决定哪些Worker可以参与训练。

## 安装指南和快速入门

请参考[快速开始](https://paddlefl.readthedocs.io/en/latest/instruction.html)。

## Kubernetes简单部署

```sh
 
kubectl apply -f ./paddle_fl/examples/k8s_deployment/master.yaml

```
请参考[K8S 部署实例](./paddle_fl/examples/k8s_deployment/README.md)
## 性能测试

Gru4Rec [9] 在基于会话的推荐中引入了递归神经网络模型。PaddlePaddle的GRU4RC实现代码在 https://github.com/PaddlePaddle/models/tree/develop/PaddleRec/gru4rec. 一个基于联邦学习训练Gru4Rec模型的示例请参考[Gru4Rec in Federated Learning](https://paddlefl.readthedocs.io/en/latest/examples/gru4rec_examples.html)


## 正在进行的工作

- 联邦学习在公共数据集上的实验基准。

- kubernetes中联邦学习系统的部署方法。

- 垂直联合学习策略和更多的水平联合学习策略将是开源的。

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
