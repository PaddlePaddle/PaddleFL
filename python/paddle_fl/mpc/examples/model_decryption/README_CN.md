## 模型解密使用手册

(简体中文|[English](./README.md))

### 1. 介绍

基于paddle-mpc提供的功能，用户可以实现对MPC密文模型的解密，得到明文模型。具体地，模型解密可以满足用户对于明文模型的需求：在从各方获取密文模型之后，通过解密得到最终的明文模型，该明文模型和paddle模型的功能完全一致。

### 2. 使用场景

基于多方训练、更新得到的密文模型，解密恢复出完整的明文模型。该明文模型可用于继续训练和预测。

### 3. 使用方法

由于针对训练、更新和预测模型的解密步骤基本是一致的，所以这里以预测模型的解密为例，介绍模型解密的主要使用步骤。

1. **解密模型**：模型解密需求方从各方获取保存的密文预测模型（即模型分片），使用paddle-mpc提供的模型解密接口`aby3.decrypt_model`解密恢复出明文预测模型。

   假设获取到的三个密文模型分片存放在`mpc_model_dir`目录下，使用`aby3.decrypt_model`进行解密：

   ```python
   aby3.decrypt_model(mpc_model_dir=mpc_model_dir,
                      plain_model_path=decrypted_paddle_model_dir,
                      mpc_model_filename=mpc_model_filename,
                      plain_model_filename=paddle_model_filename)
   ```

2. **预测**：使用解密后的预测模型对输入的待预测数据进行预测，输出预测的结果。

  该步骤同paddle预测模型的使用方法一致，首先使用`fluid.io.load_inference_model`加载预测模型：
  
  ```python
  infer_prog, feed_names, fetch_targets = fluid.io.load_inference_model(executor=exe,
                                                                        dirname=decrypted_paddle_model_dir,
                                                                        model_filename=paddle_model_filename)
  ```
  
  然后进行预测，得到预测结果：
  
  ```python
  results = exe.run(infer_prog,
                    feed={feed_names[0]: np.array(infer_feature)},
                    fetch_list=fetch_targets)
  ```

### 4. 使用示例

提供了对UCI Housing房价预测模型进行解密并使用的示例，可直接运行`decrypt_inference_model.py`脚本得到预测结果。**需要注意的是**，`decrypt_inference_model.py`脚本中待解密的模型设置为了`model_encryption/predict/predict.py`脚本内指定的模型，因此，执行脚本前请确保对应路径下已经保存了密文预测模型。

