## Instructions for PaddleFL-MPC model encryption and training

([简体中文](./README_CN.md)|English)

This document introduces how to encrypt plaintext model and training based on Paddle-MPC.


###1. Prepare Data

Run script `../process_data.py` to generate encrypted training and testing data.

###2. Encrypt Model, Train, and Save

Encrypt plaintext PaddlePaddle model, train the model, and save it with the following script.

```bash
bash run_standalone.sh encrypt_and_train_model.py
```

###3. Decrypt Loss Data

Decrypt the loss data to test the correctness of mpc training by running the following script.

```bash
python decrypt_mpc_loss.py
```

