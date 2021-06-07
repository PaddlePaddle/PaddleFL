## Instructions for PaddleFL-MPC Model Decryption Demo

([简体中文](./README_CN.md)|English)

### 1. Introduction

User can decrypt encrypted model (three model shares) with Paddle-MPC. The decrypted model can be used for training and prediction.

### 2. Usages

How to decrypt and use prediction model:

1. **Decrypt Model**：user decrypts encryped model with api `mpc_du.decrypt_model`.

   ```python
   mpc_du.decrypt_model(mpc_model_dir=mpc_model_dir,
                      plain_model_path=decrypted_paddle_model_dir,
                      mpc_model_filename=mpc_model_filename,
                      plain_model_filename=paddle_model_filename)
   ```

2. **Predict**：user predicts plaintext data with decrypted model.

  1) Load decrypted model with api `fluid.io.load_inference_model`.
  
  ```python
  infer_prog, feed_names, fetch_targets = fluid.io.load_inference_model(executor=exe,
                                                                        dirname=decrypted_paddle_model_dir,
                                                                        model_filename=paddle_model_filename)
  ```
  
  2) Predict plaintext data with decrypted model.
  
  ```python
  results = exe.run(infer_prog,
                    feed={feed_names[0]: np.array(infer_feature)},
                    fetch_list=fetch_targets)
  ```

### 3. Demo

Script `decrypt_inference_model.py` shows model decryption and prediction. Note that, encryption model should be saved in specified directory before running the script. Script `../model_encryption/predict/train_and_encrypt_model.py` can be used to generate encryption model.
