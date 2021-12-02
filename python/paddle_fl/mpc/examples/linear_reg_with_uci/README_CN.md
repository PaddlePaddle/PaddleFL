## PaddleFL-MPC UCI Housing Demo运行说明

(简体中文|[English](./README.md))

本示例介绍基于PaddleFL-MPC进行UCI房价预测模型训练和预测的使用说明，分为单机运行和多机运行两种方式。

### 一. 单机运行

#### 1. 准备数据

使用`process_data.py`脚本中的`generate_encrypted_data()`产生加密数据，比如将如下内容写到一个`prepare.py`脚本中，然后`python prepare.py`

```python
import process_data
process_data.generate_encrypted_data()
```

将在/tmp目录下生成对应于3个计算party的feature和label的加密数据文件，以后缀名区分属于不同party的数据。比如，`house_feature.part0`表示属于party0的feature数据。

#### 2. 使用shell脚本启动demo

运行demo之前，需设置以下环境变量：

```
export PYTHON=/yor/python
export PATH_TO_REDIS_BIN=/path/to/redis_bin
export LOCALHOST=/your/localhost
export REDIS_PORT=/your/redis/port
```

然后使用`run_standalone.sh`脚本，启动并运行demo，命令如下：

```bash 
bash run_standalone.sh uci_demo.py
```

运行之后将在屏幕上打印训练过程中的密文loss数据，同时，对应的密文loss数据将会保存到/tmp目录下的文件中，文件命名格式类似于步骤1中所述。

此外，在完成训练之后，demo会继续进行预测，并将预测密文结果也保存到/tmp目录下的文件中。

#### 3. 解密数据

最后，demo会使用`process_data.py`脚本中的`load_decrypt_data()`，恢复并打印出明文的loss数据和prediction结果，用以和明文Paddle模型结果进行对比。
例如，将下面的内容写到一个decrypt_save.py脚本中，然后python decrypt_save.py decrypt_loss_file decrypt_prediction_file，将把明文losss数据和预测结果分别保存在文件中。

```python
import sys

import process_data


decrypt_loss_file=sys.argv[1]
decrypt_prediction_file=sys.argv[2]
BATCH_SIZE=10
process_data.load_decrypt_data("/tmp/uci_loss", (1, ), decrypt_loss_file)
process_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE, ), decrypt_prediction_file)
```

**注意**：再次启动运行demo之前，请先将上次在`/tmp`保存的loss和prediction文件删除，以免影响本次密文数据的恢复结果。为了简化用户操作，我们在`run_standalone.sh`脚本中加入了如下的内容，可以在执行脚本时删除上次数据。

```bash
# remove temp data generated in last time
LOSS_FILE="/tmp/uci_loss.*"
PRED_FILE="/tmp/uci_prediction.*"
if [ "$LOSS_FILE" ]; then
        rm -rf $LOSS_FILE
fi

if [ "$PRED_FILE" ]; then
        rm -rf $PRED_FILE
fi
```



### 二. 多机运行

#### 1. 准备数据

数据方对数据进行加密处理。具体操作和单机运行中的准备数据步骤一致。

#### 2. 分发数据

按照后缀名，将步骤1中准备好的数据分别发送到对应的计算party的/tmp目录下。比如，使用scp命令，将

`house_feature.part0`和`house_label.part0`发送到party0的/tmp目录下。


#### 3. 各计算party启动demo

**注意**：运行需要用到redis服务。为了确保redis中已保存的数据不会影响demo的运行，请在各计算party启动demo之前，使用如下命令清空redis。其中，REDIS_BIN表示redis-cli可执行程序，SERVER和PORT分别表示redis server的IP地址和端口号。

```
$REDIS_BIN -h $SERVER -p $PORT flushall
```

在各计算party分别执行以下命令，启动demo：

```
$PYTHON_EXECUTABLE uci_demo.py $PARTY_ID $SERVER $PORT $SELF_ADDR
```

其中，PYTHON_EXECUTABLE表示自己安装了PaddleFL的python，PARTY_ID表示计算party的编号，值为0、1或2，SERVER和PORT分别表示redis server的IP地址和端口号, SELF_ADDR为自己的IP地址。

同样地，运行之后将在各计算party的屏幕上打印训练过程中的密文loss数据。同时，对应的密文loss和prediction数据将会保存到`/tmp`目录下的文件中，文件命名格式类似于步骤1中所述。

**注意**：再次启动运行demo之前，请先将上次在`/tmp`保存的loss和prediction文件删除，以免影响本次密文数据的恢复结果。

#### 4. 数据方解密loss和prediction

各计算party将`/tmp`目录下的`uci_loss.part`和`uci_prediction.part`文件发送到数据方的/tmp目录下。数据方使用process_data.py脚本中的load_decrypt_data()解密恢复出loss数据和prediction数据。

例如，将下面的内容写到一个decrypt_save.py脚本中，然后python decrypt_save.py decrypt_loss_file decrypt_prediction_file，将把明文losss数据和预测结果分别保存在文件中。

```python
import sys

import process_data


decrypt_loss_file=sys.argv[1]
decrypt_prediction_file=sys.argv[2]
BATCH_SIZE=10
process_data.load_decrypt_data("/tmp/uci_loss", (1, ), decrypt_loss_file)
process_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE, ), decrypt_prediction_file)
```

### 三. 单机精度测试

#### 1. 训练参数

- 数据集：波士顿房价预测。
- 训练轮数： 20
- Batch Size：10

#### 2. 实验结果

| Epoch/Step | paddle_fl.mpc | Paddle |
| ---------- | ------------- | ------ |
| Epoch=0, Step=0  | 738.39491 | 738.46204 |
| Epoch=1, Step=0  | 630.68834 | 629.9071 |
| Epoch=2, Step=0  | 539.54683 | 538.1757 |
| Epoch=3, Step=0  | 462.41159 | 460.64722 |
| Epoch=4, Step=0  | 397.11516 | 395.11017 |
| Epoch=5, Step=0  | 341.83102 | 339.69815 |
| Epoch=6, Step=0  | 295.01114 | 292.83597 |
| Epoch=7, Step=0  | 255.35141 | 253.19429 |
| Epoch=8, Step=0  | 221.74739 | 219.65132 |
| Epoch=9, Step=0  | 193.26459 | 191.25981 |
| Epoch=10, Step=0  | 169.11423 | 167.2204 |
| Epoch=11, Step=0  | 148.63138 | 146.85835 |
| Epoch=12, Step=0  | 131.25081 | 129.60391 |
| Epoch=13, Step=0  | 116.49708 | 114.97599 |
| Epoch=14, Step=0  | 103.96669 | 102.56854 |
| Epoch=15, Step=0  | 93.31706 | 92.03858 |
| Epoch=16, Step=0  | 84.26219 | 83.09653 |
| Epoch=17, Step=0  | 76.55664 | 75.49785 |
| Epoch=18, Step=0  | 69.99673 | 69.03561 |
| Epoch=19, Step=0  | 64.40562 | 63.53539 |

