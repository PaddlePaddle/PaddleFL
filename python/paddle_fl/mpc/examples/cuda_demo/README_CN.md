## PaddleFL CUDA Demo运行说明

(简体中文|[English](./README.md))

本示例介绍基于如何运行PaddleFL CUDA Demo.

用以下命令启动各个计算参与方:
```
NCCL_SOCKET_IFNAME=$IP_INTERFACE $PYTHON_EXECUTABLE cuda_demo.py $PARTY_ID $PARTY0_IP_ADDR
```
这里IP_INTERFACE指定通信使用的IP接口,
    PYTHON_EXECUTABLE指定安装了PaddleFL-GPU的python,
    PARTY_ID指定计算参与方的ID值为0, 1, 或 2,
    PARTY0_IP_ADDR为参与方0的IP地址.

可以通过IP地址获取IP接口名称:
```sh
ifconfig | grep -B1 "<ip-address>" | awk 'NR==1{print $1}'
```

注意, 可以通过环境变量传递其他NCCL参数.

Demo脚本`cuda_demo.py`默认使用0号CUDA 设备, 如希望使用其他CUDA设备或在单机多卡
机器运行本Demo, 请修改155行:
```python
place = fluid.CUDAPlace(0)
```

为了性能测试本Demo使用了dummy reader, 如希望处理真实数据, 请参照MNIST Demo准备
加密数据并用真实数据reader替换145, 146行.
