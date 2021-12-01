## 联邦特征工程

支持计算正样本占比、woe、iv

## 单独编译

### 环境准备
* CentOS 7 (64 bit) or Ubuntu 16.04
* Python 3.5/3.6/3.7 ( 64 bit) or above
* pip3 9.0.1+ (64 bit)
* GCC or G++ 8.2.0+
* cmake 3.15+
* grpcio
* grpcio-tools


### 克隆源码并安装

1.获取源代码
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL
mkdir build && cd build
```

2.执行部分编译指令（参照 docs/source/md/compile_and_install_cn.md ）

```
cmake .. -DCMAKE_C_COMPILER=${gcc_path} -DCMAKE_CXX_COMPILER=${g++_path} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS} -DBUILD_PADDLE_FROM_SOURCE=ON -DWITH_GRPC=ON -DWITH_GPU=OFF
```

```
cd core/he
make -j48
make install
```

3.生成grpc_pb
```
cd /path/to/PaddleFL/python
python3 paddle_fl/feature_engineering/proto/run_protogen.py
```

4.pip打包并安装
```
cd /path/to/PaddleFL/python/paddle_fl
mkdir build && cd build
python3 ../feature_engineering/setup.py sdist bdist_wheel
pip3 install dist/paddle_fl_feature_engineering-1.2.0-py3-none-any.whl -U
```
## 跟随paddlefl编译
不久后将支持

## 测试

1.准备数据
```
cd /path/to/PaddleFL/python/paddle_fl/feature_engineering/example
python3 gen_test_file.py
```
简单测试: gen_simple_file  性能测试: gen_bench_file

2.生成证书
生成grpc证书 grpc secure channel 需要

```
openssl req -newkey rsa:2048 -nodes -keyout server.key -x509 -days 3650 -out server.crt
```
示例中定义Common Name 为 metrics_service 其余为空

在example目录下会生成 server.key server.crt

3.进行测试

服务器端：python3 metrics_test_server.py 

客户端： python3 metrics_test_client.py

## 构建自己的程序

我们提供了pip打包支持，用户只需在自己的程序中 import paddle_fl.feature_engineering.core 即可，grpc通信模块可由用户自定义

示例如下：

channel: grpc client channel 自定义

server: grpc server 自定义

```
#client
    from paddle_fl.feature_engineering.core.federated_feature_engineering_client import FederatedFeatureEngineeringClient
    fed_fea_eng_client = FederatedFeatureEngineeringClient(1024)
    fed_fea_eng_client.connect(channel)
    result = fed_fea_eng_client.get_woe(labels)

#server
    from paddle_fl.feature_engineering.core.federated_feature_engineering_server import FederatedFeatureEngineeringServer
    fed_fea_eng_server = FederatedFeatureEngineeringServer()
    fed_fea_eng_server.serve(server)
    woe_list = fed_fea_eng_server.get_woe(features)
```