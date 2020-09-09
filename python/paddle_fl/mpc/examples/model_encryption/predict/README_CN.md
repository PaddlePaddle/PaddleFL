## UCI房价预测模型加密预测

(简体中文|[English](./README.md))

### 1. 训练明文模型并加密保存

使用如下命令完成明文模型的训练、加密和保存：

```bash
python train_and_encrypt_model.py
```

### 2. 准备用于预测的加密数据

执行脚本`../process_data.py`加密待预测的数据。

### 3. 加密预测

使用如下命令完成密文模型预测：

```bash
bash run_standalone.sh predict_with_mpc_model.py
```

### 4. 解密loss数据验证密文模型预测过程

使用如下命令对保存的预测结果进行解密查看：

```bash
python decrypt_mpc_prediction.py 
```

