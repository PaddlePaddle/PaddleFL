## 模型解密使用手册

(简体中文|[English](./README.md))

### 1. 介绍

基于paddle-mpc提供的功能，用户可以实现对MPC密文模型的解密，得到明文模型，然后可以使用明文模型进行再训练/预测。具体地，用户从各方获取密文模型（基于多方训练/更新得到的密文模型）之后，通过调用解密接口可以得到明文模型，该明文模型和paddle模型的功能完全一致。

### 2. 使用方法

由于针对训练、更新和预测模型的解密步骤基本是一致的，所以这里以预测模型的解密为例，介绍模型解密使用的主要步骤。

1. **解密模型**：模型解密需求方从各方获取保存的密文预测模型（即模型分片），使用paddle-mpc提供的模型解密接口`mpc_du.decrypt_model`解密恢复出明文预测模型。

   假设获取到的三个密文模型分片存放于`mpc_model_dir`目录，使用`mpc_du.decrypt_model`进行解密，分别指定密文模型的路径和名字，明文模型的存放路径和名字：

   ```python
   mpc_du.decrypt_model(mpc_model_dir=mpc_model_dir,
                      plain_model_path=decrypted_paddle_model_dir,
                      mpc_model_filename=mpc_model_filename,
                      plain_model_filename=paddle_model_filename)
   ```

2. **预测**：使用解密后的预测模型对待预测的数据进行预测，输出预测结果。

  该步骤同paddle预测模型的使用方法一致，首先使用`fluid.io.load_inference_model`加载明文预测模型：
  
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

### 3. 使用示例

脚本`decrypt_and_inference.py`提供了对UCI Housing房价预测模型进行解密并使用的示例，可直接运行`decrypt_inference_model.py`脚本得到预测结果。**需要注意的是**，`decrypt_inference_model.py`脚本中待解密的模型设置为`../model_encryption/predict/train_and_encrypt_model.py`脚本内指定的模型，因此，执行脚本前请确保对应路径下已经存在密文预测模型。

