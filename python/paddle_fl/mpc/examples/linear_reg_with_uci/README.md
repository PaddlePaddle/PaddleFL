## Instructions for PaddleFL-MPC UCI Housing Demo

([简体中文](./README_CN.md)|English)

This document introduces how to run UCI Housing demo based on Paddle-MPC, which has two ways of running, i.e., single machine and multi machines.

### 1. Running on Single Machine

#### (1). Prepare Data

Generate encrypted data utilizing `generate_encrypted_data()` in `process_data.py` script. For example, users can write the following code into a python script named `prepare.py`, and then run the script with command `python prepare.py`.

```python
import process_data
process_data.generate_encrypted_data()
```

Encrypted data files of feature and label would be generated and saved in `/tmp` directory. Different suffix names are used for these files to indicate the ownership of different computation parties. For instance, a file named `house_feature.part0` means it is a feature file of party 0.

#### (2). Launch Demo with A Shell Script

You should set the env params as follow:

```
export PYTHON=/yor/python
export PATH_TO_REDIS_BIN=/path/to/redis_bin
export LOCALHOST=/your/localhost
export REDIS_PORT=/your/redis/port
```

Launch demo with the `run_standalone.sh` script. The concrete command is:

```bash
bash run_standalone.sh uci_demo.py
```

The loss with cypher text format will be displayed on screen while training. At the same time, the loss data would be also save in `/tmp` directory, and the format of file name is similar to what is described in Step 1.

Besides, predictions would be made in this demo once training is finished. The predictions with cypher text format would also be save in `/tmp` directory.

#### (3). Decrypt Data

Finally, using `load_decrypt_data()` in `process_data.py` script, this demo would decrypt and print the loss and predictions, which can be compared with related results of Paddle plain text model.

For example, users can write the following code into a python script named `decrypt_save.py`, and then run the script with command `python decrypt_save.py decrypt_loss_file decrypt_prediction_file`. The decrypted loss and prediction results would be saved into two files correspondingly.

```python
import sys

import process_data


decrypt_loss_file=sys.argv[1]
decrypt_prediction_file=sys.argv[2]
BATCH_SIZE=10
process_data.load_decrypt_data("/tmp/uci_loss", (1, ), decrypt_loss_file)
process_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE, ), decrypt_prediction_file)
```

**Note** that remember to delete the loss and prediction files in `/tmp` directory generated in last running, in case of any influence on the decrypted results of current running. For simplifying users operations, we provide the following commands in `run_standalone.sh`, which can delete the files mentioned above when running this script.

```bash
# remove temp data generated in last time
LOSS_FILE="/tmp/uci_loss.*"
PRED_FILE="/tmp/uci_prediction.*"
if [ "$LOSS_FILE" ]; then
        rm -rf $LOSS_FILE
fi

if [ "$PRED_FILE" ]; then
        rm -rf $PRED_FILE
fi
```



### 2. Running on Multi Machines

#### (1). Prepare Data

Data owner encrypts data. Concrete operations are consistent with “Prepare Data” in “Running on Single Machine”.

#### (2). Distribute Encrypted Data

According to the suffix of file name, distribute encrypted data files to `/tmp ` directories of all 3 computation parties. For example, send `house_feature.part0` and `house_label.part0` to `/tmp` of party 0 with `scp` command.

#### (3). Launch Demo on Each Party

**Note** that Redis service is necessary for demo running. Remember to clear the cache of Redis server before launching demo on each computation party, in order to avoid any negative influences caused by the cached records in Redis. The following command can be used for clear Redis, where REDIS_BIN is the executable binary of redis-cli, SERVER and PORT represent the IP and port of Redis server respectively.

```
$REDIS_BIN -h $SERVER -p $PORT flushall
```

Launch demo on each computation party with the following command,

```
$PYTHON_EXECUTABLE uci_demo.py $PARTY_ID $SERVER $PORT $SELF_ADDR
```

where PYTHON_EXECUTABLE is the python which installs PaddleFL, PARTY_ID is the ID of computation party, which is 0, 1, or 2, SERVER and PORT represent the IP and port of Redis server respectively, SELF_ADDR represents the IP address of the machine.

Similarly, training loss with cypher text format would be printed on the screen of each computation party. And at the same time, the loss and predictions would be saved in `/tmp` directory.

**Note** that remember to delete the loss and prediction files in `/tmp` directory generated in last running, in case of any influence on the decrypted results of current running.

#### (4). Decrypt Loss and Prediction Data

Each computation party sends `uci_loss.part` and `uci_prediction.part` files in `/tmp` directory to the `/tmp` directory of data owner. Data owner decrypts and gets the plain text of loss and predictions with ` load_decrypt_data()` in `process_data.py`.

For example, the following code can be written into a python script to decrypt and print training loss and predictions.

```python
import sys

import process_data


decrypt_loss_file=sys.argv[1]
decrypt_prediction_file=sys.argv[2]
BATCH_SIZE=10
process_data.load_decrypt_data("/tmp/uci_loss", (1, ), decrypt_loss_file)
process_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE, ), decrypt_prediction_file)
```

### 3. Convergence of paddle_fl.mpc vs paddle

Below, is the result of our experiment to test the convergence of paddle_fl.mpc on single machine.


#### (1). Training Parameters

- Dataset: Boston house price dataset
- Number of Epoch: 20
- Batch Size: 10

#### (2). Experiment Results

| Epoch/Step | paddle_fl.mpc | Paddle |
| ---------- | ------------- | ------ |
| Epoch=0, Step=0  | 738.39491 | 738.46204 |
| Epoch=1, Step=0  | 630.68834 | 629.9071 |
| Epoch=2, Step=0  | 539.54683 | 538.1757 |
| Epoch=3, Step=0  | 462.41159 | 460.64722 |
| Epoch=4, Step=0  | 397.11516 | 395.11017 |
| Epoch=5, Step=0  | 341.83102 | 339.69815 |
| Epoch=6, Step=0  | 295.01114 | 292.83597 |
| Epoch=7, Step=0  | 255.35141 | 253.19429 |
| Epoch=8, Step=0  | 221.74739 | 219.65132 |
| Epoch=9, Step=0  | 193.26459 | 191.25981 |
| Epoch=10, Step=0  | 169.11423 | 167.2204 |
| Epoch=11, Step=0  | 148.63138 | 146.85835 |
| Epoch=12, Step=0  | 131.25081 | 129.60391 |
| Epoch=13, Step=0  | 116.49708 | 114.97599 |
| Epoch=14, Step=0  | 103.96669 | 102.56854 |
| Epoch=15, Step=0  | 93.31706 | 92.03858 |
| Epoch=16, Step=0  | 84.26219 | 83.09653 |
| Epoch=17, Step=0  | 76.55664 | 75.49785 |
| Epoch=18, Step=0  | 69.99673 | 69.03561 |
| Epoch=19, Step=0  | 64.40562 | 63.53539 |

