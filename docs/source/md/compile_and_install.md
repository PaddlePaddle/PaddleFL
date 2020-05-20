## Compile From Source Code

#### A. Environment preparation

* CentOS 6 or CentOS 7 (64 bit)
* Python 2.7.15+/3.5.1+/3.6/3.7 ( 64 bit) or above
* pip or pip3 9.0.1+ (64 bit)
* PaddlePaddle release 1.8
* Redis 5.0.8 (64 bit)
* GCC or G++ 4.8.3+
* cmake 3.15+

#### B. Clone the source code, compile and install

Fetch the source code and checkout stable release
```sh
git clone https://github.com/PaddlePaddle/PaddleFL
cd /path/to/PaddleFL

# Checkout stable release
mkdir build && cd build
```

Execute compile commands, where `PYTHON_EXECUTABLE` is path to the python binary where the PaddlePaddle is installed, `CMAKE_CXX_COMPILER` is the path of G++ and `PYTHON_INCLUDE_DIRS` is the corresponding python include directory. You can get the `PYTHON_INCLUDE_DIRS` via the following command:

```sh
${PYTHON_EXECUTABLE} -c "from distutils.sysconfig import get_python_inc;print(get_python_inc())"
```
Then you can put the directory in the following command and make:
```sh
cmake ../ -DPYTHON_EXECUTABLE=${PYTHON_EXECUTABLE} -DPYTHON_INCLUDE_DIRS=${python_include_dir} -DCMAKE_CXX_COMPILER=${g++_path}
make -j$(nproc)
```
Install the package:

```sh
make install
cd /path/to/PaddleFL/python
${PYTHON_EXECUTABLE} setup.py sdist bdist_wheel
pip or pip3 install dist/***.whl -U
```
