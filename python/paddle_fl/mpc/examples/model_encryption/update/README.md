## Instructions for Pre-trained Model Encryption and Update with Paddle-MPC

([简体中文](./README_CN.md)|English)

This document introduces how to encrypt pre-trained plaintext model and update it based on Paddle-MPC.

### 1. Train PaddlePaddle Model, Encrypt, and Save

Train plaintext PaddlePaddle model, encrypt, and save with the following script.

```bash
python train_and_encrypt_model.py
```

### 2. Prepare Data

Run script `../process_data.py` to generate encrypted training and testing data for updating encrypted model.

### 3. Update MPC Model

Update mpc model with the following script.

```bash
bash run_standalone.sh update_mpc_model.py
```

### 4. Decrypt Loss Data

Decrypt the loss data to test the correctness of encrypted model updating by running the following script.

```bash
python decrypt_mpc_loss.py
```

