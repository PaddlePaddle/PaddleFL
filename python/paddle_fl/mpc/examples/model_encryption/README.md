## Instructions for PaddleFL-MPC Model Encryption Demo

([简体中文](./README_CN.md)|English)

### 1. Introduction

This document introduces how to run encrypt PaddlePaddle  model, then train or update encrypted model, or predict encrypted data with encrypted model. Model encryption is suitable for protecting training/prediction data and model.

### 2. Scenarios

Model encryption demo contains three scenarios:

*  **Encrypt Model and Train**

Each party loads PaddlePadlde model and encrypts it. Each party feeds the encrypted data to train the encrypted model. Each party can get one share for the encrypted model. PaddlePaddle model can be reconstructed with three encrypted model shares.

*  **Encrypt Pre-trained Model and Update**

Pre-trained model is encryption and distributed to multipel parties. Parties update the encrypted model with encrypted data. PaddlePaddle model can be reconstructed with three encrypted model shares.

*  **Encrypt Pre-trained Model and Predict**

Pre-trained model is encryption and distributed to multipel parties. Parties predict encrypted data with the encrypted model. Prediction output can be reconstructed with three encrypted prediction output shares.

### 3. Usage

#### 3.1 Train Model

<img src='images/model_training.png' width = "500" height = "550" align="middle"/>

This figure shows model encryption and training with Paddle-MPC.

1). **Load PaddlePaddle Model**: Users init mpc context with mpc_init OP, then load or define PaddlePaddle network.

   ```python
   pfl_mpc.init("aby3", role, ip, server, port)
   [_, _, _, loss] = network.model_network()
   exe.run(fluid.default_startup_program())
   ```

2). **Transpile(Encrypt) Model**: Users use api `aby3.transpile` to encrypt curent default PaddlePaddle model.

   ```python
   aby3.transpile()
   ```

3). **Train Model**: Users train encrypted model with encrypted data.

   ```python
   for epoch_id in range(epoch_num):
       for mpc_sample in loader():
           mpc_loss = exe.run(feed=mpc_sample, fetch_list=[loss.name])
   ```

4). **Save Model**：Users save encrypted model using `aby3.save_trainable_model`.

   ```python
   aby3.save_trainable_model(exe=exe,
                             model_dir=model_save_dir,
                             model_filename=model_filename)
   ```

5). **Decrypt Model**：PaddlePaddle model can be reconstructed with three model shares (encrypted model). 

#### 3.2 Update Model

<img src='images/model_updating.png' width = "500" height = "380" align="middle"/>

This figure shows how to update pre-trained model with Paddle-MPC.

1). **Pre-train Model**: PaddlePaddle model is trained with plaintext data.

2). **Encrypt Model**: User encrypts pre-trained model with api `aby3.encrypt_model` and distributes three model shares to three parties.

   ```python
   # Step 1. Load pre-trained model.
   main_prog, _, _ = fluid.io.load_inference_model(executor=exe,
                                                   dirname=paddle_model_dir,
                                                   model_filename=model_filename)
   # Step 2. Encrypt pre-trained model.
   aby3.encrypt_model(program=main_prog,
                      mpc_model_dir=mpc_model_dir,
                      model_filename=model_filename)
   ```

3). **Update Model**：Users init mpc context with mpc_init OP, then load encrypted model with `aby3.load_mpc_model`. Users update the encrypted model with encrypted data.

   ```python
   # Step 1. initialize MPC environment and load MPC model into
   # default_main_program to update.
   pfl_mpc.init("aby3", role, ip, server, port)
   aby3.load_mpc_model(exe=exe,
                       mpc_model_dir=mpc_model_dir,
                       mpc_model_filename=mpc_model_filename)
   
   # Step 2. MPC update
   for epoch_id in range(epoch_num):
       for mpc_sample in loader():
           mpc_loss = exe.run(feed=mpc_sample, fetch_list=[loss.name])
   ```

4). **Decrypt Model**：User can decrypt model with three model shares. 

#### 3.3 Predict Model

<img src='images/model_infer.png' width = "500" height = "380" align="middle"/>

This figure shows how to predict encryted data with encrypted model.

1). **Train Model**：User trains PaddlePaddle model with plaintext data.

2). **Encrypt Model**: User encrypts model with api `aby3.encrypt_model` and distributes model shares to three users. The api is same with `Update Model`.

3). **Predict/Infer**: Users initialize mpc context with `mpc_init OP`, then load encrypted model with api `aby3.load_mpc_model`. Users predict encryped data with encryted model.

   ```python
   # Step 1. initialize MPC environment and load MPC model to predict
   pfl_mpc.init("aby3", role, ip, server, port)
   infer_prog, feed_names, fetch_targets = 
   						aby3.load_mpc_model(exe=exe,
                                   mpc_model_dir=mpc_model_dir,                                                    															  mpc_model_filename=mpc_model_filename, inference=True)
   
   # Step 2. MPC predict
   prediction = exe.run(program=infer_prog, feed={feed_names[0]: np.array(mpc_sample)}, fetch_list=fetch_targets)
   
   # Step 3. save prediction results
   with open(pred_file, 'ab') as f:
       f.write(np.array(prediction).tostring())
   ```

4. **Decrypt Model**：User can decrypt model with the model shares. 

### 4. Usage Demo

**Train Model**: Instructions for model encryption and training with PaddleFL-MPC using UCI Housing dataset: [Here](./train).
**Update Model**: Instructions for pre-trained model encryption and update with Paddle-MPC using UCI Housing dataset: [Here](./update).
**Predict Model**: Instructions for pre-trained model encryption and prediction with Paddle-MPC using UCI Housing dataset: [Here](./predict).

