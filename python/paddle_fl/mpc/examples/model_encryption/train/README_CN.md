## UCI房价预测模型加密训练

(简体中文|[English](./README.md))

本示例简单介绍了基于PaddleFL-MPC对明文空白模型加密后再训练的使用说明。

### 1. 准备加密数据

执行脚本`../process_data.py`完成训练数据的加密处理。

### 2. 加密空白明文模型并训练保存

使用如下命令完成模型的加密、训练与保存:

```bash
bash run_standalone.sh encrypt_and_train_model.py
```

### 3. 解密loss数据验证密文模型训练过程正确性

使用如下命令对训练过程中保存的loss数据进行解密查看，验证训练的正确性：

```bash
python decrypt_mpc_loss.py
```

