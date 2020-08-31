## Instructions for PaddleFL-MPC Model Encryption and Training

([简体中文](./README_CN.md)|English)

This document introduces how to encrypt plaintext model and train the encrypted model based on Paddle-MPC.

### 1. Prepare Data

Run script `../process_data.py` to generate encrypted training and testing data.

### 2. Encrypt Model, Train, and Save

Encrypt plaintext PaddlePaddle model, train the encrypted model, and save the trained encrypted model with the following script.

```bash
bash run_standalone.sh encrypt_and_train_model.py
```

### 3. Decrypt Loss Data

Decrypt the loss data to test the correctness of mpc training with the following script.

```bash
python decrypt_mpc_loss.py
```

