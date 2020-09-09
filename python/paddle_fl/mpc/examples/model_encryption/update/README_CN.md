## UCI房价预测模型加密更新

(简体中文|[English](./README.md))

本示例介绍基于PaddleFL-MPC对预训练明文模型加密后再训练更新的使用说明。

### 1. 训练明文模型并加密保存

使用如下命令训练明文模型并加密保存：

```bash
python train_and_encrypt_model.py
```

### 2. 准备用于更新模型的加密数据

执行脚本`../process_data.py`生成更新模型所需的加密数据。

### 3. 更新密文模型

使用如下命令训练更新密文模型并保存：

```bash
bash run_standalone.sh update_mpc_model.py
```

### 4. 解密loss数据验证密文模型更新过程的正确性

使用如下命令对更新过程中保存的loss数据进行解密查看，验证更新过程的正确性：

```bash
python decrypt_mpc_loss.py 
```

