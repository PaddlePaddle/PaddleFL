## 从源码编译

#### A. 环境准备

* CentOS 6 or CentOS 7 (64 bit)
* Python 2.7.15+/3.5.1+/3.6/3.7 ( 64 bit) or above
* pip or pip3 9.0.1+ (64 bit)
* PaddlePaddle release 1.8
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

执行编译指令, `PYTHON_EXECUTABLE` 为安装了PaddlePaddle的可执行python路径, `CMAKE_CXX_COMPILER` 为指定的g++路径。 `PYTHON_INCLUDE_DIRS` 是相应的include路径，可以用如下指令获得：

```sh
${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc;print(get_python_inc())"
```
之后就可以执行编译和安装的指令
```sh
cmake ../ -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIRS=${python_include_dir} -DCMAKE_CXX_COMPILER=${g++_path}
make -j$(nproc)
```
安装对应的安装包

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip or pip3 install dist/***.whl -U
```
