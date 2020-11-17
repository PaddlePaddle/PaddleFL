## PaddleFL-MPC FM Demo运行说明

(简体中文|[English](./README.md))

本示例介绍基于PaddleFL-MPC进行Criteo数据集FM模型训练和预测的使用说明，分为单机运行和多机运行两种方式。

### 一. 单机运行

#### 1. 准备数据

使用脚本`data/download.sh`下载Criteo数据集，并拆分成训练数据集和预测数据集（少量数据集`data/sample_data/train/sample_train.txt`可用于验证模型训练和预测）。使用`process_data.py`脚本中的`generate_encrypted_data()`产生加密训练数据和测试数据，用户可以直接运行脚本`python process_data.py`在指定的目录下（比如`./mpc_data/`）产生加密训练数据和测试数据（特征和标签）。在指定目录下生成对应于3个计算party的feature和label的加密数据文件，以后缀名区分属于不同party的数据。比如，`criteo_feature_idx.part0`表示属于party0的feature id，`criteo_feature_value.part0`表示属于party0的feature value，`criteo_label.part0`表示属于party0的label。

#### 2. 使用shell脚本启动训练demo

运行demo之前，需设置以下环境变量：

```
export PYTHON=/yor/python
export PATH_TO_REDIS_BIN=/path/to/redis_bin
export LOCALHOST=/your/localhost
export REDIS_PORT=/your/redis/port
```

然后使用`run_standalone.sh`脚本，启动并运行训练demo，命令如下：

```bash 
bash run_standalone.sh train_fm.py
```

运行之后将在屏幕上打印训练过程中所处的epoch和step，并在完成训练后打印训练花费的时间，并在每个epoch训练结束后，保存可用于执行加密预测的模型。

#### 3. 使用shell脚本启动预测demo

在完成训练之后，使用`run_standalone.sh`脚本，启动并运行预测demo，命令如下：

```bash
bash run_standalone.sh load_model_and_infer.py
```

加载训练时保存的模型，对测试数据进行预测，并将预测密文结果保存到./mpc_infer_data/目录下的文件中，文件命名格式类似于步骤1中所述。

#### 4. 解密数据

预测结束后，可以使用`process_data.py`脚本中的`decrypt_data_to_file()`，将保存的密文预测结果进行解密，并且将解密得到的明文预测结果保存到指定文件中。然后再调用脚本`evaluate_metrics.py`中的`evaluate_accuracy`和`evaluate_auc`接口统计预测的准确率和AUC（Area Under Curve）。（现已包含再infer过程中，用户也可以单独写在一个脚本执行）。


### 二. 多机运行

#### 1. 准备数据

数据方对数据进行加密处理。具体操作和单机运行中的准备数据步骤一致。

#### 2. 分发数据

按照后缀名，将步骤1中准备好的数据分别发送到对应的计算party的./mpc_data/目录下。比如，使用scp命令，将后缀为`part0`的加密数据发送到party0的./mpc_data/目录下。

#### 3. 修改各计算party的训练脚本traim_fm.py

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
$PYTHON_EXECUTABLE train_fm.py $PARTY_ID $SERVER $PORT
```

其中，PYTHON_EXECUTABLE表示自己安装了PaddleFL的python，PARTY_ID表示计算party的编号，值为0、1或2，SERVER和PORT分别表示redis server的IP地址和端口号。

同样地，密文prediction数据将会保存到`./mpc_infer_data/`目录下的文件中。比如，在party0中将保存为文件`prediction.part0`.

#### 5. 解密预测数据

各计算party将`./mpc_infer_data/`目录下的`prediction.part*`文件发送到数据方的`./mpc_infer_data/`目录下。数据方使用`process_data.py`脚本中的`decrypt_data_to_file()`，将密文预测结果进行解密，并且将解密得到的明文预测结果保存到指定文件中。然后可以使用`process_data.py`脚本中的`decrypt_data_to_file()`，将保存的密文预测结果进行解密。

