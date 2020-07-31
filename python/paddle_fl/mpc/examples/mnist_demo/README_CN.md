## PaddleFL-MPC MNIST Demo运行说明

(简体中文|[English](./README.md))

本示例介绍基于PaddleFL-MPC进行MNIST数据集模型训练和预测的使用说明，分为单机运行和多机运行两种方式。

### 一. 单机运行

#### 1. 准备数据

使用`process_data.py`脚本中的`generate_encrypted_data()`和`generate_encrypted_test_data()`产生加密训练数据和测试数据，比如将如下内容写到一个`prepare.py`脚本中，然后`python prepare.py`

```python
import process_data
process_data.generate_encrypted_data()
process_data.generate_encrypted_test_data()
```

将在/tmp目录下生成对应于3个计算party的feature和label的加密数据文件，以后缀名区分属于不同party的数据。比如，`mnist2_feature.part0`表示属于party0的feature数据。

#### 2. 使用shell脚本启动demo

使用`run_standalone.sh`脚本，启动并运行demo，命令如下：

```bash 
bash run_standalone.sh mnist_demo.py
```

运行之后将在屏幕上打印训练过程中所处的epoch和step，并在完成训练后打印出训练花费的时间。

此外，在完成训练之后，demo会继续进行预测，并将预测密文结果保存到/tmp目录下的文件中，文件命名格式类似于步骤1中所述。

#### 3. 解密数据

使用`process_data.py`脚本中的`decrypt_data_to_file()`，将保存的密文预测结果进行解密，并且将解密得到的明文预测结果保存到指定文件中。例如，将下面的内容写到一个`decrypt_save.py`脚本中，然后`python decrypt_save.py decrypt_file`，将把明文预测结果保存在`decrypt_file`文件中。

```python
import sys

decrypt_file=sys.argv[1]
import process_data
process_data.decrypt_data_to_file("/tmp/mnist_output_prediction", (BATCH_SIZE,), decrypt_file)
```

**注意**：再次启动运行demo之前，请先将上次在`/tmp`保存的预测密文结果文件删除，以免影响本次密文数据的恢复结果。为了简化用户操作，我们在`run_standalone.sh`脚本中加入了如下的内容，可以在执行脚本时删除上次的数据。

```bash
# remove temp data generated in last time
PRED_FILE="/tmp/mnist_output_prediction.*"
if [ "$PRED_FILE" ]; then
        rm -rf $PRED_FILE
fi
```



### 二. 多机运行

#### 1. 准备数据

数据方对数据进行加密处理。具体操作和单机运行中的准备数据步骤一致。

#### 2. 分发数据

按照后缀名，将步骤1中准备好的数据分别发送到对应的计算party的/tmp目录下。比如，使用scp命令，将

`mnist2_feature.part0`和`mnist2_label.part0`发送到party0的/tmp目录下。

#### 3. 计算party修改mnist_demo.py脚本

各计算party根据自己的机器环境，将脚本如下内容中的`localhost`修改为自己的IP地址：

```python
pfl_mpc.init("aby3", int(role), "localhost", server, int(port))
```

#### 4. 各计算party启动demo

**注意**：运行需要用到redis服务。为了确保redis中已保存的数据不会影响demo的运行，请在各计算party启动demo之前，使用如下命令清空redis。其中，REDIS_BIN表示redis-cli可执行程序，SERVER和PORT分别表示redis server的IP地址和端口号。

```
$REDIS_BIN -h $SERVER -p $PORT flushall
```

在各计算party分别执行以下命令，启动demo：

```
$PYTHON_EXECUTABLE mnist_demo.py $PARTY_ID $SERVER $PORT
```

其中，PYTHON_EXECUTABLE表示自己安装了PaddleFL的python，PARTY_ID表示计算party的编号，值为0、1或2，SERVER和PORT分别表示redis server的IP地址和端口号。

同样地，密文prediction数据将会保存到`/tmp`目录下的文件中。比如，在party0中将保存为文件`mnist_output_prediction.part0`.

**注意**：再次启动运行demo之前，请先将上次在`/tmp`保存的prediction文件删除，以免影响本次密文数据的恢复结果。

#### 5. 解密预测数据

各计算party将`/tmp`目录下的`mnist_output_prediction.part`文件发送到数据方的/tmp目录下。数据方使用`process_data.py`脚本中的`decrypt_data_to_file()`，将密文预测结果进行解密，并且将解密得到的明文预测结果保存到指定文件中。例如，将下面的内容写到一个`decrypt_save.py`脚本中，然后`python decrypt_save.py decrypt_file`，将把明文预测结果保存在`decrypt_file`文件中。

```python
import sys

decrypt_file=sys.argv[1]
import process_data
process_data.decrypt_data_to_file("/tmp/mnist_output_prediction", (BATCH_SIZE,), decrypt_file)
```

