# 两方方差Demo

## 应用场景

两个数据方想要在不暴露自己的原始数据的前提下联合计算方差。

## 算法原理及整体流程

1. 双方本地对自己的原始数据求平均数和方差。

2. 将双方本地计算的平均数和方差以及一些用于除法的参数按照aby3秘密共享的形式拆分，每方数据分为三份。

3. 将三份数据交给三个不同的计算节点（本demo中的run.sh文件提供了单机模拟三方的示例，实际使用中拆开运行即可）。

4. 双方使用Paddlefl提供的基础运算算子，按照下面的计算公式进行两方联合计算。
   $$
   avg = (num0*avg0+num1*avg1)/(num0+num1)
   $$

   $$
   prod = ((num0-1)*variance0+(num1-1)*variance1+num0*(avg0-avg)*(avg0-avg)+num1*(avg1-avg)*(avg1-avg))/(num0+num1-1)
   $$

5. 将秘密共享形式的计算结果交给数据合并方（由于本demo是单机模拟三方，这一步在实际使用中需要自行实现）。

6. 重构计算结果。

## 代码执行过程

1. `python3 data.py` 生成双方原始数据，得到Input-P0.list和Input-P1.list。
2. `python3 variance2_share.py` 双方本地计算平均数和方差并拆分，得到data_C0_P0_avg、data_C0_P0_variance、data_C0_P1_avg.npy、data_C0_P1_variance、data_C1_P0_avg、data_C1_P0_variance、data_C1_P1_avg.npy、data_C1_P1_variance、data_C2_P0_avg、data_C2_P0_variance、data_C2_P1_avg.npy、data_C2_P1_variance，以及对一些用于除法的参数（见variance2_share.py中第47行）进行拆分，得到data_C0_tmp.npy、data_C1_tmp.npy、data_C2_tmp.npy，同时会在命令行打印应当得到的中位数计算结果。
3. 在真正的多方运算时需要分发第2步中的秘密数据（本demo不需要这一步）。
4. `python3 flush.py` 清除redis中上一次残留的注册信息，以免下面运行的三方进程连接到错误的地方。
5. `./run.sh` 运行单机模拟三方的脚本执行多方联合计算，logs文件夹中会生成运行日志，同时还会生成结果文件result_C0.npy、result_C1.npy、result_C2.npy。
6. 在真正的多方运算时需要将第5步秘密共享形式的计算结果交给数据合并方（本demo不需要这一步）。
7. `python3 reconstruct.py` 重构计算结果，并在命令行打印实际计算得到的中位数结果。

## 其他说明

1. 文件名中C代表计算方的编号，P代表数据方的编号，例如data_C0_P1.npy代表这份秘密来自数据方1，需要交给计算方0。
2. 数据方编号与计算方编号均由0开始，
3. 由于paddlefl未提供除法算子，这里我们采用乘以倒数近似，例如除以(num0+num1-1)，需要表示为乘以1/(num0+num1-1)，data_Ci_tmp.npy文件中即存储秘密共享形式的1/(num0+num1-1)等参数。
4. 和paddlefl提供的原始test_op_base.py相比，本demo中注释掉了第70行代码，这一行代码会引起redis注册混乱，从而影响到三方模拟运算；同时将multi_party_run改造成了三个不同的函数，以便三方分别调用单独生成进程。
5. 安全性：使用本demo需要提前知晓双方的数据量，用于实现近似的除法，需要在variance2_share.py中45、46行手动设置。