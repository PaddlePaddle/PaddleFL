
## 联邦算法模拟器 (fl-mobile simulator)

FL-mobile是一个集移动端算法模拟调研、训练和部署为一体的框架。算法模拟器 (simulator) 是FL-mobile的一部分。

该模拟器的设计目的，是为了模拟实际线上多个移动端设备配合训练的场景。框架的设计思想在服务器上模拟数个端上设备，快速验证算法效果。模拟器的优势为：

- 支持单机和分布式训练
- 支持常见开源数据集的训练
- 支持模型中的私有参数和共享参数，私有参数不参与全局更新

## 准备工作

- 安装mpirun

- python安装grpc
    ```shell
    pip install grpcio==1.28.1
    ```
    
- 安装Paddle

    ```shell
    pip install paddlepaddle==1.8.0
    ```

    

## 快速开始

我们以Leaf数据集中的[reddit数据](https://github.com/TalwalkarLab/leaf/tree/master/data/reddit)为例，参考[这篇论文](https://arxiv.org/pdf/1812.01097.pdf)，用LSTM建模，在simulator中给出一个单机训练的例子。通过这个例子，您能了解simulator的基础用法。

### 准备数据

```
wget https://paddle-serving.bj.bcebos.com/temporary_files_for_docker/reddit_subsampled.zip --no-check-certificate
unzip reddit_subsampled.zip
```
在模拟器中，我们假设用户的数据都是天级别的，因此我们将下载的数据重新归置如下

```
tree lm_data
lm_data
|-- 20200101
|   `-- train_data.json
|-- 20200102
|   `-- test_data.json
`-- vocab.json
```
可以看到，我们将训练数据作为20200101的数据，测试数据作为20200102的数据。

### 生成server代码

```
cd protos
python run_codegen.py
cd ..
```

### 开始训练

在训练中，我们每轮用均匀采样（Uniform Sample）方式选取`10`个Client进行训练，每个Client在本地用该Client对应的全部数据（未经shuffle）训练`1`个epoch，总共训练`100`轮。在本实验中使用的Client学习率为`1.0`，FedAvg学习率为`1.85	`。

```shell
export PYTHONPATH=$PWD:$PYTHONPATH
mpirun -np 2 python application.py lm_data
```

### 训练结果

在测试集上，测试Top1为 `11.6% `。

```shell
framework.py : INFO  infer results: 0.116334
```

相同参数的非联邦训练测试Top1为`11.1%`。

## 添加自己的数据集和Trainer

如果您想要训练自己的联邦模型，您需要做四件事：

1. 创建reader，参考`reader/leaf_reddit_reader.py`
2. 创建trainer，参考`trainer/language_model_trainer.py`
3. 创建model，即组网，参考`model/language_model.py`
4. 创建application，参考`application.py`

## 模拟器(simulator) 介绍

框架主要由scheduler和simulator构成；其中scheduler负责统筹规划数据和全局参数；simulator负责做实际的训练和私有参数更新。

- scheduler
在一次训练流程中，只会有一个global scheduler， 而每个机器上都会有一个scheduler client，负责和global scheduler做参数、数据的通信。

- simulator
每个机器上都会有1个Simulator，每个Simulator又会有多个shard，shard是用于本机并行训练的。

作为一个分布式框架，FL-mobile simulator 也是包含模型初始化、模型分发、模型训练、模型更新四部分，下面通过一次实际的训练流程，来了解一下各个模型的工作吧：

- Step1 模型初始化

    1. 全局参数初始化：由编号为0的simulator来做模型初始化工作，初始化之后，它会通过UpdateGlobalParams()接口将参数传递给Scheduler；
    
    2. 个性化参数初始化

- Step2 模型分发

    1. 全局参数分发：每个simulator开始训练之前，都需要先找SchedulerServer拿全局参数，即通过scheduler_client.get_global_params()获得全局参数；

    2. 个性化参数分发：个性化参数由每个local trainer训练前，向data server获取，获取接口为：get_param_by_uid；

- Step3 模型训练

    模型训练是整个流程的核心，多个trainer并行训练，训练中参数不做同步；每个trainer的训练步数由这个用户的数据量决定。

- Step4 模型更新

    1. 全局参数更新：在所有trainer训练结束后，会做一次同步，并且通过FedAvg算法计算参数梯度；之后上传梯度至scheduler，再拉取新的全局参数，回到第二步；
    
    2. 个性化参数更新：个性化参数更新简单一些，每个trainer调用set_param_by_uid就可完成自己的个性化参数更新；
