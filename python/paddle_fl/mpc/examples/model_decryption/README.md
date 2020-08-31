## Instructions for PaddleFL-MPC Model Decryption Demo

([简体中文](./README_CN.md)|English)

### 1. Introduction

Users can decrypt encrypted model with Paddle-MPC. The decrypted model can be used for training and prediction.

### 2. Usages

We will show how to decrypt prediction model.

1. **Decrypt Model**：Users decrypt encryped model with api `aby3.decrypt_model`.

   ```python
   aby3.decrypt_model(mpc_model_dir=mpc_model_dir,
                      plain_model_path=decrypted_paddle_model_dir,
                      mpc_model_filename=mpc_model_filename,
                      plain_model_filename=paddle_model_filename)
   ```

2. **Predict**：Users can predict plaintext data with decrypted model.

  1) User loads decrypted model with api `fluid.io.load_inference_model`.
  
  ```python
  infer_prog, feed_names, fetch_targets = fluid.io.load_inference_model(executor=exe,
                                                                        dirname=decrypted_paddle_model_dir,
                                                                        model_filename=paddle_model_filename)
  ```
  
  2) User predict plaintext data with decrypted model.
  
  ```python
  results = exe.run(infer_prog,
                    feed={feed_names[0]: np.array(infer_feature)},
                    fetch_list=fetch_targets)
  ```

