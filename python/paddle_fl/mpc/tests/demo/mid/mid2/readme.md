# 两方中位数Demo

## 应用场景

两个数据方想要在不暴露自己的原始数据的前提下联合计算中位数。

## 算法原理及整体流程

1. 双方本地对自己的原始数据升序排列。
2. 将双方排序好的数据按照aby3秘密共享的形式拆分，每方数据分为三份。
3. 将三份数据交给三个不同的计算节点（本demo中的run.sh文件提供了单机模拟三方的示例，实际使用中拆开运行即可）。
4. 三个计算方使用Paddlefl提供的基础运算算子，按照[leetcode上的寻找两个有序数组的中位数算法（题解方法一：二分查找）](https://leetcode.cn/problems/median-of-two-sorted-arrays/solution/xun-zhao-liang-ge-you-xu-shu-zu-de-zhong-wei-s-114/)进行多方联合计算。
5. 将秘密共享形式的计算结果交给数据合并方（由于本demo是单机模拟三方，这一步在实际使用中需要自行实现）。
6. 重构计算结果。

## 代码执行过程

1. `python3 data.py` 生成双方原始数据，得到Input-P1.list和Input-P2.list。
2. `python3 mid2_share.py` 将双方原始数据排序并拆分，得到data_C0_P1.npy、data_C1_P1.npy、data_C2_P1.npy、data_C0_P2.npy、data_C1_P2.npy、data_C2_P2.npy，如果双方数据量之和为偶数，还将得到data_C0_tmp.npy、data_C1_tmp.npy、data_C2_tmp.npy，同时会在命令行打印应当得到的中位数计算结果。
3. 在真正的多方运算时需要分发第2步中的秘密数据（本demo不需要这一步）。
4. `python3 flush.py` 清除redis中上一次残留的注册信息，以免下面运行的三方进程连接到错误的地方。
5. `./run.sh` 运行单机模拟三方的脚本执行多方联合计算，logs文件夹中会生成运行日志，同时还会生成结果文件result_C0.npy、result_C1.npy、result_C2.npy。
6. 在真正的多方运算时需要将第5步秘密共享形式的计算结果交给数据合并方（本demo不需要这一步）。
7. `python3 reconstruct.py` 重构计算结果，并在命令行打印实际计算得到的中位数结果。

## 其他说明

1. 文件名中C代表计算方的编号，P代表数据方的编号，例如data_C0_P1.npy代表这份秘密来自数据方1，需要交给计算方0。
2. 数据方编号由1开始，这是为了尽量保证寻找两个有序数组的中位数算法的核心代码与leetcode提供的算法一致，以增加代码可读性；计算方编号由0开始，这是为了尽量保证除了中位数算法核心代码以外的部分和paddlefl提供的/python/paddle_fl/mpc/tests/unittests中的其他示例对于参与方的描述一致。同时，这两部分代码的变量命名规范不同，中位数算法的核心代码使用驼峰命名法，其余部分和paddlefl一致使用下划线命名法。
3. 由于paddlefl未提供除法算子，这里我们采用乘以倒数近似，即当双方数据总量为偶数n时，求第n/2和n/2+1个数据的平均值，需要除以2，data_Ci_tmp.npy文件中即存储秘密共享形式的0.5。
4. 和paddlefl提供的原始test_op_base.py相比，本demo中注释掉了第70行代码，这一行代码会引起redis注册混乱，从而影响到三方模拟运算；同时将multi_party_run改造成了三个不同的函数，以便三方分别调用单独生成进程。