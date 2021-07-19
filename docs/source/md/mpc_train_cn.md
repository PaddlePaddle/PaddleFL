## MPC 训练及推理

<img src='../../../images/PFM-design.png' width = "1000" height = "622" align="middle"/>

像PaddlePaddle一样，训练和推理任务可以分为编译阶段和运行阶段。

##### 1. 编译时

* **确定MPC环境**：用户需要指定用到的MPC协议，并配置网络环境。现有版本支持基于PrivC的两方联邦学习以及基于ABY3的三方联邦学习。
* **用户定义训练任务**：用户可以根据PFM提供的安全接口，定义联邦学习网络模型以及训练策略。

##### 2. 运行时

一个Paddle Encrypted程序实际上就是一个PaddlePaddle程序。在运行时，PFM的程序将会转变为PaddlePaddle中的ProgramDesc，并交给Executor运行。以下是运行阶段的主要概念：
* **运算节点**：计算节点是与计算方相对应的实体。在实际部署中，它可以是裸机、云虚拟机、docker甚至进程。Paddle Encrypted程序将在所有计算节点上并行部署和运行。
* **基于MPC的算子**：PFM 为操作加密数据提供了特殊的算子，这些算子在PaddlePaddle框架中基于像ABY3和PrivC一样的MPC协议进行实现。像PaddlePaddle中一样，在运行时PFM的算子将被创建并按照顺序执行。


