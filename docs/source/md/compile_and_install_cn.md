## 从源码编译

#### A. 环境准备

* CentOS 7 (64 bit)
* Python 3.5/3.6/3.7 ( 64 bit) or above
* pip3 9.0.1+ (64 bit)
* PaddlePaddle release 1.8.5 (如果选择不从源代码构建paddle)
* Redis 5.0.8 (64 bit)
* GCC or G++ 8.3.1
* cmake 3.15+

#### B. 克隆源代码并编译安装

1.获取源代码
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL

# Checkout stable release
mkdir build && cd build
```

2.执行编译指令, `CMAKE_C_COMPILER` 为指定的gcc路径, `CMAKE_CXX_COMPILER` 为指定的g++路径,`PYTHON_EXECUTABLE` 为安装了PaddlePaddle的可执行python路径,`DPYTHON_INCLUDE_DIRS`为指定Python.h文件所在路径。

之后就可以执行编译和安装的指令
```sh
cmake .. -DCMAKE_C_COMPILER=${gcc_path} -DCMAKE_CXX_COMPILER=${g++_path} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS} -DBUILD_PADDLE_FROM_SOURCE=ON -DWITH_GRPC=ON
make -j$(nproc)
```
例如：
```
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DPYTHON_EXECUTABLE=/usr/local/python/bin/python3.8 -DPYTHON_INCLUDE_DIRS=/usr/local/python/include/python3.8/ -DBUILD_PADDLE_FROM_SOURCE=ON -DWITH_GRPC=ON
```


3.安装paddle (如果选择从源代码构建paddle):
```sh
pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle-1.8.5-cp38-cp38-linux_x86_64.whl -U
```
安装对应的安装包

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip3 install dist/***.whl -U
```
