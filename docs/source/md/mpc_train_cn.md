## B. 训练及推理

<img src='../../../images/PFM-design.png' width = "1000" height = "622" align="middle"/>

像PaddlePaddle一样，训练和推理任务可以分为编译阶段和运行阶段。

##### 1. 编译时

* **确定MPC环境**：用户需要指定用到的MPC协议，并配置网络环境。现有版本的Paddle Encrypted只支持"ABY3"协议。更多的协议将在后续版本中支持。
* **用户定义训练任务**：用户可以根据PFM提供的安全接口，定义集齐学习网络以及训练策略。
##### 2. 运行时

一个Paddle Encrypted程序实际上就是一个PaddlePaddle程序。在运行时，PFM的程序将会转变为PaddlePaddle中的ProgramDesc，并交给Executor运行。以下是运行阶段的主要概念：
* **运算节点**：计算节点是与计算方相对应的实体。在实际部署中，它可以是裸机、云虚拟机、docker甚至进程。PFM在每次运行中只需要三个计算节点，这由底层ABY3协议决定。Paddle Encrypted程序将在所有三个计算节点上并行部署和>运行。
* **基于MPC的算子**：PFM 为操作加密数据提供了特殊的算子，这些算子在PaddlePaddle框架中实现，基于像ABY3一样的MPC协议。像PaddlePaddle中一样，在运行时PFM的算子将被创建并按照顺序执行。
