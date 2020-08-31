## Instructions for Pre-trained Model Encryption and Prediction with Paddle-MPC

([简体中文](./README_CN.md)|English)

This document introduces how to encrypt pre-trained plaintext model and predict encrypted data with the encrypted model based on Paddle-MPC.

### 1. Train PaddlePaddle Model, Encrypt, and Save

Train plaintext PaddlePaddle model, encrypt, and save it with the following script.

```bash
python train_and_encrypt_model.py
```

### 2. Prepare Data

Run script `../process_data.py` to generate encrypted prediction input data.

### 3. Predict with MPC Model

Predict encrypted data using encrypted model with the following script.

```bash
bash run_standalone.sh predict_with_mpc_model.py
```

### 4. Decrypt Prediction Output Data

Decrypt predition output data with the following script.

```bash
python decrypt_mpc_prediction.py
```

