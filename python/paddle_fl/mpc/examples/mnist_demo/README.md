## Instructions for PaddleFL-MPC MNIST Demo

([简体中文](./README_CN.md)|English)

This document introduces how to run MNIST demo based on Paddle-MPC, which has two ways of running, i.e., single machine and multi machines.

### 1. Running on Single Machine

#### (1). Prepare Data

Generate encrypted training and testing data utilizing `generate_encrypted_data()` and `generate_encrypted_test_data()` in `process_data.py` script. For example, users can write the following code into a python script named `prepare.py`, and then run the script with command `python prepare.py`.

```python
import process_data
process_data.generate_encrypted_data()
process_data.generate_encrypted_test_data()
```

Encrypted data files of feature and label would be generated and saved in `/tmp` directory. Different suffix names are used for these files to indicate the ownership of different computation parties. For instance, a file named `mnist2_feature.part0` means it is a feature file of party 0.

#### (2). Launch Demo with A Shell Script

Launch demo with the `run_standalone.sh` script. The concrete command is:

```bash
bash run_standalone.sh mnist_demo.py
```

The information of current epoch and step will be displayed on screen while training, as well as the total cost time when traning finished.

Besides, predictions would be made in this demo once training is finished. The predictions with cypher text format would be save in `/tmp` directory, and the format of file name is similar to what is described in Step 1.

#### (3). Decrypt Data

Decrypt the saved prediction data and save the decrypted prediction results into a specified file using `decrypt_data_to_file()` in `process_data.py` script. For example, users can write the following code into a python script named `decrypt_save.py`, and then run the script with command `python decrypt_save.py`. The decrypted prediction results would be saved into `mpc_label`.

```python
import process_data
process_data.decrypt_data_to_file("/tmp/mnist_output_prediction", (BATCH_SIZE,), "mpc_label")
```

**Note** that remember to delete the prediction files in `/tmp` directory generated in last running, in case of any influence on the decrypted results of current running. For simplifying users operations, we provide the following commands in `run_standalone.sh`, which can delete the files mentioned above when running this script.

```bash
# remove temp data generated in last time
PRED_FILE="/tmp/mnist_output_prediction.*"
if [ "$PRED_FILE" ]; then
        rm -rf $PRED_FILE
fi
```



### 2. Running on Multi Machines

#### (1). Prepare Data

Data owner encrypts data. Concrete operations are consistent with “Prepare Data” in “Running on Single Machine”.

#### (2). Distribute Encrypted Data

According to the suffix of file name, distribute encrypted data files to `/tmp ` directories of all 3 computation parties. For example, send `mnist2_feature.part0` and `mnist2_label.part0` to `/tmp` of party 0 with `scp` command.

#### (3). Modify mnist_demo.py

Each computation party modifies `localhost` in the following code as the IP address of it's machine.

```python
pfl_mpc.init("aby3", int(role), "localhost", server, int(port))
```

#### (4). Launch Demo on Each Party

**Note** that Redis service is necessary for demo running. Remember to clear the cache of Redis server before launching demo on each computation party, in order to avoid any negative influences caused by the cached records in Redis. The following command can be used for clear Redis, where REDIS_BIN is the executable binary of redis-cli, SERVER and PORT represent the IP and port of Redis server respectively.

```
$REDIS_BIN -h $SERVER -p $PORT flushall
```

Launch demo on each computation party with the following command,

```
$PYTHON_EXECUTABLE mnist_demo.py $PARTY_ID $SERVER $PORT
```

where PYTHON_EXECUTABLE is the python which installs PaddleFL, PARTY_ID is the ID of computation party, which is 0, 1, or 2, SERVER and PORT represent the IP and port of Redis server respectively.

Similarly, predictions with cypher text format would be saved in `/tmp` directory, for example, a file named `mnist_output_prediction.part0` for party 0.

**Note** that remember to delete the precidtion files in `/tmp` directory generated in last running, in case of any influence on the decrypted results of current running.

#### (5). Decrypt Prediction Data

Each computation party sends  `mnist_output_prediction.part` file in `/tmp` directory to the `/tmp` directory of data owner. Data owner decrypts the prediction data and saves the decrypted prediction results into a specified file using `decrypt_data_to_file()` in `process_data.py` script. For example, users can write the following code into a python script named `decrypt_save.py`, and then run the script with command `python decrypt_save.py`. The decrypted prediction results would be saved into file `mpc_label`.

```python
import process_data
process_data.decrypt_data_to_file("/tmp/mnist_output_prediction", (BATCH_SIZE,), "mpc_label")
```

