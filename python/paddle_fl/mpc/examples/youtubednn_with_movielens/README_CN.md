## PaddleFL-MPC YoutubeDNN Demo运行说明

(简体中文|[English](./README.md))

本示例介绍基于PaddleFL-MPC进行YoutubeDNN模型训练和预测的使用说明，分为单机运行和多机运行两种方式。

### 一. 单机运行

#### 1. 准备数据

使用`process_data.py`脚本中的`gen_cypher_sample()`产生加密训练数据和测试数据，用户可以直接运行脚本`python process_data.py`在指定的目录下（比如`./mpc_data/`）产生加密特征和标签。在指定目录下生成对应于3个计算party的feature和label的加密数据文件，以后缀名区分属于不同party的数据。

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
bash run_standalone.sh train_youtubednn.py
```

运行之后将在屏幕上打印训练进度：当前epoch和step，以及当前训练耗时，并在完成训练后保存参数`l4_weight`作为电影特征。

此外，在完成训练之后，demo会继续进行预测，并将预测密文结果(第三个fc的输出`l3`)保存到./mpc_data/目录下的文件中，作为对应用户的特征。

#### 3. 解密特征数据并计算命中率

首先使用`process_data.py`脚本中的`decrypt_data_to_file()`，将保存的密文电影和用户特征进行解密，并且将解密得到的明文预测结果保存到指定文件中。

然后使用`get_topk.py`脚本中的`get_topK()`计算解密的用户特征和视频特征的相似度，排序选出推荐给用户的k个电影。然后在使用`process_data.py`脚本中的`evaluate_hit_ratio()`计算命中率。

解密数据和计算命中率，可参考脚本`decrypt_and_evaluate.py`。


### 二. 多机运行

#### 1. 准备数据

数据方对数据进行加密处理。具体操作和单机运行中的准备数据步骤一致。

#### 2. 分发数据

按照后缀名，将步骤1中准备好的数据分别发送到对应的计算party的./mpc_data/目录下。比如，使用scp命令，将

`*.part0`发送到party0的./mpc_data/目录下。

#### 3. 修改各计算party的train_youtubednn.py脚本

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
$PYTHON_EXECUTABLE train_youtubednn.py --role $PARTY_ID --server $SERVER --port $PORT
```

其中，PYTHON_EXECUTABLE表示自己安装了PaddleFL的python，PARTY_ID表示计算party的编号，值为0、1或2，SERVER和PORT分别表示redis server的IP地址和端口号。

同样地，密文电影特征和用户特征数据将会保存到`./mpc_data/`目录下的文件中。比如，在party0中将保存为文件`video_vec.part0`和`user_vec.part0`.

#### 5. 解密特征数据

各计算party将`./mpc_data/`目录下的`video.part*`和`user_vec.part*`文件发送到数据方的`./mpc_data/`目录下。然后按照`单机运行`中步骤3`解密特征数据并计算命中率`。
