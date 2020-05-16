##Instructions for PaddleFL-MPC UCI Housing Demo

([简体中文](./README_CN.md)|English)

This document introduces how to run UCI Housing demo based on Paddle-MPC, which has two ways of running, i.e., single machine and multi machines.

###1. Running on Single Machine

####(1). Prepare Data

Generate encrypted data utilizing `generate_encrypted_data()` in `process_data.py` script. For example, users can write the following code into a python script named `prepare.py`, and then run the script with command `python prepare.py`.

```python
import process_data
process_data.generate_encrypted_data()
```

Encrypted data files of feature and label would be generated and saved in `/tmp` directory. Different suffix names are used for these files to indicate the ownership of different computation parties. For instance, a file named `house_feature.part0` means it is a feature file of party 0.

####(2). Launch Demo with A Shell Script

Launch demo with the `run_standalone.sh` script. The concrete command is:

```bash
bash run_standalone.sh uci_housing_demo.py
```

The loss with cypher text format will be displayed on screen while training. At the same time, the loss data would be also save in `/tmp` directory, and the format of file name is similar to what is described in Step 1.

Besides, predictions would be made in this demo once training is finished. The predictions with cypher text format would also be save in `/tmp` directory.

Finally, using `load_decrypt_data()` in `process_data.py` script, this demo would decrypt and print the loss and predictions, which can be compared with related results of Paddle plain text model.

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



###2. Running on Multi Machines

####(1). Prepare Data

Data owner encrypts data. Concrete operations are consistent with “Prepare Data” in “Running on Single Machine”.

####(2). Distribute Encrypted Data

According to the suffix of file name, distribute encrypted data files to `/tmp ` directories of all 3 computation parties. For example, send `house_feature.part0` and `house_label.part0` to `/tmp` of party 0 with `scp` command.

####(3). Modify uci_housing_demo.py

Each computation party makes the following modifications on `uci_housing_demo.py` according to the environment of machine.

* Modify IP Information

  Modify `localhost` in the following code as the IP address of the machine.

  ```python
  pfl_mpc.init("aby3", int(role), "localhost", server, int(port))
  ```

* Comment Out Codes for Single Machine Running

  Comment out the following codes which are used when running on single machine.

  ```python
  import process_data
  print("uci_loss:")
  process_data.load_decrypt_data("/tmp/uci_loss", (1,))
  print("prediction:")
  process_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE,))
  ```

####(4). Launch Demo on Each Party

**Note** that Redis service is necessary for demo running. Remember to clear the cache of Redis server before launching demo on each computation party, in order to avoid any negative influences caused by the cached records in Redis. The following command can be used for clear Redis, where REDIS_BIN is the executable binary of redis-cli, SERVER and PORT represent the IP and port of Redis server respectively.

```
$REDIS_BIN -h $SERVER -p $PORT flushall
```

Launch demo on each computation party with the following command,

```
$PYTHON_EXECUTABLE uci_housing_demo.py $PARTY_ID $SERVER $PORT
```

where PYTHON_EXECUTABLE is the python which installs PaddleFL, PARTY_ID is the ID of computation party, which is 0, 1, or 2, SERVER and PORT represent the IP and port of Redis server respectively.

Similarly, training loss with cypher text format would be printed on the screen of each computation party. And at the same time, the loss and predictions would be saved in `/tmp` directory.

**Note** that remember to delete the loss and precidtion files in `/tmp` directory generated in last running, in case of any influence on the decrypted results of current running.

####(5). Decrypt Loss and Prediction Data

Each computation party sends `uci_loss.part` and `uci_prediction.part` files in `/tmp` directory to the `/tmp` directory of data owner. Data owner decrypts and gets the plain text of loss and predictions with ` load_decrypt_data()` in `process_data.py`.

For example, the following code can be written into a python script to decrypt and print training loss.

```python
import process_data
print("uci_loss:")
process_data.load_decrypt_data("/tmp/uci_loss", (1,))
```

And the following code can be written into a python script to decrypt and print predictions.

```python
import process_data
print("prediction:")
process_data.load_decrypt_data("/tmp/uci_prediction", (BATCH_SIZE,))
```

