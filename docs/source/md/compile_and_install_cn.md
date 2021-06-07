## 从源码编译

#### A. 环境准备

* CentOS 6 or CentOS 7 (64 bit)
* Python 2.7.15+/3.5.1+/3.6/3.7 ( 64 bit) or above
* pip or pip3 9.0.1+ (64 bit)
* PaddlePaddle release 1.8 (如果选择不从源代码构建paddle)
* Redis 5.0.8 (64 bit)
* GCC or G++ 4.8.3+
* cmake 3.15+

#### B. 克隆源代码并编译安装

获取源代码
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL

# Checkout stable release
mkdir build && cd build
```

执行编译指令, `PYTHON_EXECUTABLE` 为安装了PaddlePaddle的可执行python路径, `CMAKE_C_COMPILER` 为指定的gcc路径, `CMAKE_CXX_COMPILER` 为指定的g++路径。

之后就可以执行编译和安装的指令
```sh
cmake ../ -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DCMAKE_C_COMPILER=${gcc_path} -DCMAKE_CXX_COMPILER=${g++_path}
make -j$(nproc)
```

注意, paddle-fl 1.1.2 版(或更高)必须从源码构建paddle, 即打开cmake选项`BUILD_PADDLE_FROM_SOURCE`.

安装paddle (如果选择从源代码构建paddle):
```sh
pip or pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle-1.8.0-cp36-cp36m-linux_x86_64.whl -U
```
安装对应的安装包

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip or pip3 install dist/***.whl -U
```
