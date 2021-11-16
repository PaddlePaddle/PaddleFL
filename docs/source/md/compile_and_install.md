## Compile From Source Code

#### A. Environment preparation

* CentOS 7 (64 bit) or Ubuntu 16.04
* Python 3.5/3.6/3.7 ( 64 bit) or above
* pip3 9.0.1+ (64 bit)
* PaddlePaddle 1.8.5 (or PaddlePaddle-GPU 1.8.5 to compile GPU version)
* Redis 5.0.8 (64 bit)
* GCC or G++ 8.2.0+
* cmake 3.15+

#### B. Clone the source code, compile and install

Fetch the source code and prepare for compilation

```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL
mkdir build && cd build
```

Execute compile commands, where `CMAKE_C_COMPILER` is the path of gcc and `CMAKE_CXX_COMPILER` is the path of g++,  `PYTHON_EXECUTABLE` is path to the python binary where the PaddlePaddle is installed, `PYTHON_INCLUDE_DIRS` is path to the file `Python.h`.

Then you can put the directory in the following command and make:
```sh
cmake .. -DCMAKE_C_COMPILER=${gcc_path} -DCMAKE_CXX_COMPILER=${g++_path} -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIRS=${PYTHON_INCLUDE_DIRS} -DBUILD_PADDLE_FROM_SOURCE=ON -DWITH_GRPC=ON
make -j$(nproc)
```
For example,
```
cmake .. -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DPYTHON_EXECUTABLE=/usr/local/python/bin/python3.8 -DPYTHON_INCLUDE_DIRS=/usr/local/python/include/python3.8/ -DBUILD_PADDLE_FROM_SOURCE=ON -DWITH_GRPC=ON
```

Choose compile GPU version or CPU version by set `-DWITH_GPU`.
Note that GPU verion only run within CUDAPlace.

If you have installed PaddlePaddle in advance and only want to compile PaddleFL, then change "-DBUILD_PADDLE_FROM_SOURCE=ON" in the above command to "-DBUILD_PADDLE_FROM_SOURCE=OFF".

Install paddle if BUILD_PADDLE_FROM_SOURCE=on:
```sh
pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle-1.8.5-cp38-cp38-linux_x86_64.whl -U
```

GPU version:
```sh
pip3 install ./third_party/paddle/src/extern_paddle-build/python/dist/paddlepaddle_gpu-1.8.5-cp38-cp38-linux_x86_64.whl -U
```

Install the package:

CPU version:

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip3 install dist/***.whl -U
```

GPU version:
```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel --gpu
pip3 install dist/***.whl -U
```
