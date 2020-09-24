## Instructions for PaddleFL-MPC Factorization Machine(FM) Demo

([简体中文](./README_CN.md)|English)

This document introduces how to run FM demo based on Paddle-MPC, which has two ways of running, i.e., single machine and multi machines.

### 1. Running on Single Machine

#### (1). Prepare Data

Download Criteo dataset using script `./data/download.sh`, then split the dataset into training data and testing data (sample data: `./data/sample_data/train/sample_train.txt`). Generate encrypted training and testing data utilizing `generate_encrypted_data()` in `process_data.py` script. Users can run the script with command `python process_data.py` to generate encrypted feature and label in given directory, e.g., `./mpc_data/`. Different suffix names are used for these files to indicate the ownership of different computation parties. For instance, a file named `criteo_feature_idx.part0` means it is a feature file of party 0.

#### (2). Launch Train Demo with A Shell Script

You should set the env params as follow:

```
export PYTHON=/your/python
export PATH_TO_REDIS_BIN=/path/to/redis_bin
export LOCALHOST=/your/localhost
export REDIS_PORT=/your/redis/port
```

Launch train demo with the `run_standalone.sh` script. The concrete command is:

```bash
bash run_standalone.sh train_fm.py
```

The information of current epoch and step will be displayed on screen while training, as well as the total cost time when traning finished. Encrypted infer model will be stored after each epoch's training.

#### (3). Launch Train Demo with A Shell Script

Launch infer demo with the `run_standalone.sh` script. The concrete command is:

```bash
bash run_standalone.sh load_model_and_infer.py
```

Load encrypted infer model and infer encrypted test data. The predictions with cypher text format would be save in `./mpc_infer_data/` directory (users can modify it in the python script `load_model_and_infer.py`), and the format of file name is similar to what is described in Step 1.

#### (4). Decrypt Data

Decrypt the saved prediction data and save the decrypted prediction results into a specified file using `decrypt_data_to_file()` in `process_data.py` script. The decrypted prediction results would be saved. `Accuracy` and `AUC` can be evaluated using the interface `evaluate_accuracy` and `evaluate_auc` in the script `evaluate_metrics.py`. (This phase is include in the infer phase, user also can run that in single script.)


### 2. Running on Multi Machines

#### (1). Prepare Data

Data owner encrypts data. Concrete operations are consistent with “Prepare Data” in “Running on Single Machine”.

#### (2). Distribute Encrypted Data

According to the suffix of file name, distribute encrypted data files to `./mpc_data/ ` directories of all 3 computation parties. For example, send encrypted data with suffix `.part0` to `./mpc_data/` of party 0 with `scp` command.

#### (3). Modify training and infering script `train_fm.py` and `load_model_and_infer.py`

Each computation party modifies `localhost` in the following code as the IP address of it's machine.

```python
pfl_mpc.init("aby3", int(role), "localhost", server, int(port))
```

#### (4). Launch Demo on Each Party

**Note** that Redis service is necessary for demo running. Remember to clear the cache of Redis server before launching demo on each computation party, in order to avoid any negative influences caused by the cached records in Redis. The following command can be used for clear Redis, where REDIS_BIN is the executable binary of redis-cli, SERVER and PORT represent the IP and port of Redis server respectively.

```
$REDIS_BIN -h $SERVER -p $PORT flushall
```

Launch train demo on each computation party with the following command,

```
$PYTHON_EXECUTABLE train_fm.py $PARTY_ID $SERVER $PORT
```

Launch infer demo on each computation party with the following command,
```
$PYTHON_EXECUTABLE load_model_and_infer.py $PARTY_ID $SERVER $PORT
```

where PYTHON_EXECUTABLE is the python which installs PaddleFL, PARTY_ID is the ID of computation party, which is 0, 1, or 2, SERVER and PORT represent the IP and port of Redis server respectively.

Similarly, predictions with cypher text format would be saved in `./mpc_infer_data/` directory, for example, a file named `prediction.part0` for party 0.

#### (5). Decrypt Prediction Data

Each computation party sends  `prediction.part?` file in `./mpc_infer_data/` directory to the `./mpc_infer_data/` directory of data owner. Data owner decrypts the prediction data and saves the decrypted prediction results into a specified file using `decrypt_data_to_file()` in `process_data.py` script. For example, users can write the following code into a python script named `decrypt_save.py`, and then run the script with command `python decrypt_save.py decrypted_file`. The decrypted prediction results would be saved into file `decrypted_file`. Then, users can evaluate the accuracy and AUC metrics.

```python
import sys

decrypt_file=sys.argv[1]
import process_data
process_data.decrypt_data_to_file("./mpc_infer_data/prediction", (BATCH_SIZE,), decrypted_file)
```

