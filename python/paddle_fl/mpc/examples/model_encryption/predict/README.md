## Instructions for pre-trained model encryption and prediction with Paddle-MPC

([简体中文](./README_CN.md)|English)

This document introduces how to encrypt pre-trained plaintext model and predict based it on Paddle-MPC.


###1. Train PaddlePaddle Model, Encrypt, and Save

Train plaintext PaddlePaddle model, encrypt, and save with the following script.

```bash
python train_and_encrypt_model.py
```

###2. Prepare Data

Run script `../process_data.py` to generate encrypted training and testing data.

###3. Predict with MPC Model

Predict mpc model with the following script.

```bash
bash run_standalone.sh predict_with_mpc_model.py
```

###4. Decrypt Prediction Data

Decrypt predition data.

```bash
python decrypt_mpc_prediction.py
```

