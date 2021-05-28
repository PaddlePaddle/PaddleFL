## Instructions for PaddleFL-MPC YoutubeDNN Demo

([简体中文](./README_CN.md)|English)

This document introduces how to run YoutubeDNN demo based on Paddle-MPC, which has two ways of running, i.e., single machine and multi machines.

### 1. Running on Single Machine

#### (1). Prepare Data

Generate encrypted training and testing data utilizing `gen_cypher_sample()` in `process_data.py` script. Users can run the script with command `python process_data.py` to generate encrypted feature and label in given directory, e.g., `./mpc_data/`. Different suffix names are used for these files to indicate the ownership of different computation parties.

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
bash run_standalone.sh train_youtubednn.py
```

The information of current epoch and step will be displayed on screen while training, as well as the total cost time when traning finished.

Besides, predictions would be made in this demo once training is finished. The predictions (l3: third fc's output) with cypher text format would be save in `./mpc_data/`  and the format of file name is similar to what is described in Step 1.

#### (3). Decrypt Data and Evaluate Hit Ratio

Decrypt the saved prediction data (video and user feature) and save the decrypted prediction results into a specified file using `decrypt_data_to_file()` in `process_data.py` script.  

The similarity of all videos and users will be evaluate with the api `get_topK()` in script `get_topk.py`, then top-K videos will be chosen for each user with api `evaluate_hit_ratio()` in script `process_data.py`.

User can run the shell script `decrypt_and_evaluate.py` to decrypt data and evaluate hit ratio..

### 2. Running on Multi Machines

#### (1). Prepare Data

Data owner encrypts data. Concrete operations are consistent with “Prepare Data” in “Running on Single Machine”.

#### (2). Distribute Encrypted Data

According to the suffix of file name, distribute encrypted data files to `./mpc_data/ ` directories of all 3 computation parties. For example, send `*.part0` to `./mpc_data/` of party 0 with `scp` command.

#### (3). Modify train_youtubednn.py

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
$PYTHON_EXECUTABLE train_youtubednn.py --role $PARTY_ID --server $SERVER --port $PORT
```

where PYTHON_EXECUTABLE is the python which installs PaddleFL, PARTY_ID is the ID of computation party, which is 0, 1, or 2, SERVER and PORT represent the IP and port of Redis server respectively.

Similarly, video and user feature in cypher text format would be saved in `./mpc_data/` directory, for example, `video_vec.part0` and `user_vec.part0` will be saved for party 0.

#### (5). Decrypt Feature Data

Each computation party sends  `video_vec.part*` and `user_vec.part*` file in `./mpc_data/` directory to the `./mpc_infer_data/` directory of data owner. Then, `decrypt and evaluate hit ratios` are are consistent with `Decrypt Data and Evaluate Hit Ratio` in `Running on Single Machine`.

