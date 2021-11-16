## Instructions for PaddleFL CUDA Demo

([简体中文](./README_CN.md)|English)

This document introduces how to run CUDA demo based on PaddleFL.

Launch demo on each computation party with the following command:
```
NCCL_SOCKET_IFNAME=$IP_INTERFACE $PYTHON_EXECUTABLE cuda_demo.py $PARTY_ID $PARTY0_IP_ADDR
```
where IP_INTERFACE specifies which IP interface to use for communication,
      PYTHON_EXECUTABLE is the python which installs PaddleFL-GPU,
      PARTY_ID is the ID of computation party, which is 0, 1, or 2,
      PARTY0_IP_ADDR represents the IP address of party 0.

You can get interface name by ip address:
```sh
ifconfig | grep -B1 "<ip-address>" | awk 'NR==1{print $1}'
```

Note that you can also pass other arguments to NCCL using enviroment variables.

In the demo script `cuda_demo.py`, CUDA devive 0 is selected.
If you want to use another CUDA device, or run this demo on a single machine
with mutiple CUDA devices, modify line 155:
```python
place = fluid.CUDAPlace(0)
```

In this demo we use dummy reader for benchmarking.
If you wish to process real data, please refer to MNIST demo and prepare encrypted data.
Then substitude line 145, 146 with real data reader.
