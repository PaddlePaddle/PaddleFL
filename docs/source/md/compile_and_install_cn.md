## 从源码编译

#### A. 环境准备

* CentOS 7 (64 bit) or Ubuntu 16.04
* Python 3.5/3.6/3.7 ( 64 bit) or above
* pip3 9.0.1+ (64 bit)
* PaddlePaddle 1.8.5 (or PaddlePaddle-GPU 1.8.5 如希望编译GPU版本)
* Redis 5.0.8 (64 bit)
* GCC or G++ 8.2.0+
* cmake 3.15+

#### B. 克隆源代码并编译安装

1.获取源代码
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL
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
可以通过`-DWITH_GPU`决定编译GPU版本还是CPU版本. 注意, GPU版本只支持在CUDAPlace运行.

如果您事先安装好了PaddlePaddle，希望只编译PaddleFL，那么将上面命令中 "-DBUILD_PADDLE_FROM_SOURCE=ON" 改为 "-DBUILD_PADDLE_FROM_SOURCE=OFF" 即可。


3.安装paddle (如果选择从源代码构建paddle):
CPU版:
```sh
pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle-1.8.5-cp38-cp38-linux_x86_64.whl -U
```

GPU版:
```sh
pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle_gpu-1.8.5-cp38-cp38-linux_x86_64.whl -U
```

4.安装PaddleFL对应的安装包

CPU版:
```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip3 install dist/***.whl -U
```

GPU版本:
```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel --gpu
pip3 install dist/***.whl -U
```
