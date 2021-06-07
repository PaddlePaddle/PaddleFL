# Compile and Install

## Installation

We **highly recommend** to run PaddleFL in Docker

```sh
#Pull and run the docker
docker pull hub.baidubce.com/paddlefl/paddle_fl:latest
docker run --name <docker_name> --net=host -it -v $PWD:/root <image id> /bin/bash

#Install paddle_fl
pip install paddle_fl
```

We also prepare a stable redis package for you to download and install, which will be used in tasks with MPC.

```sh
wget --no-check-certificate https://paddlefl.bj.bcebos.com/redis-stable.tar
tar -xf redis-stable.tar
cd redis-stable &&  make
```

## Compile From Source Code

#### A. Environment preparation

* CentOS 6 or CentOS 7 (64 bit)
* Python 2.7.15+/3.5.1+/3.6/3.7 ( 64 bit) or above
* pip or pip3 9.0.1+ (64 bit)
* PaddlePaddle release 1.8 (if not build paddle from source)
* Redis 5.0.8 (64 bit)
* GCC or G++ 6+
* cmake 3.15+

#### B. Clone the source code, compile and install

Fetch the source code and checkout stable release
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL

# Checkout stable release
mkdir build && cd build
```

Execute compile commands, where `PYTHON_EXECUTABLE` is path to the python binary where the PaddlePaddle is installed, `CMAKE_C_COMPILER` is the path of gcc and `CMAKE_CXX_COMPILER` is the path of g++.

Then you can put the directory in the following command and make:
```sh
cmake ../ -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DCMAKE_C_COMPILER=${gcc_path} -DCMAKE_CXX_COMPILER=${g++_path}
make -j$(nproc)
```

Note that paddle must be built from source after paddle-fl version 1.1.2 by turning on cmake option `BUILD_PADDLE_FROM_SOURCE` (paddlepaddle1.8.5).

Install paddle if BUILD_PADDLE_FROM_SOURCE=on:
```sh
pip or pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle-1.8.5-cp38-cp38-linux_x86_64.whl -U
```

Install the package:

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip or pip3 install dist/***.whl -U
```
