## Instructions for PaddleFL-MPC Mean Normalize Demo

This document introduces how to run Mean Normalize demo based on Paddle-MPC,
which is single machine demo.

### Running on Single Machine

#### (1). Prepare Data

Create a empty dir for data, and modify `data_path` in `process_data.py`,
default dir path is `./data`.

Then run the script with command `python prepare.py` to generate random data
for demo, which is dumped by numpy and named `feature_data.{i}.npy` located
in `data_path`. Otherwise generate your own data, move them to `data_path`,
name as the same way, and  modify corresponding meta info in `prepare.py`.

Encrypted data files of feature statstics would be generated and saved in
`data_path` directory. Different suffix names are used for these files to
indicate the ownership of different data source and computation parties.
For instance, a file named `feature_max.1.part2` means it contains the max
feature values from data owner 1 and needs to be feed to computing party 2.

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
bash ../run_standalone.sh mean_normalize_demo.py
```

The ciphertext result of global feature range and feature mean will be save in
`data_path` directory, named `result.part{i}`.

#### (3). Decrypt Data

Finally, using `decrypt_data()` in `process_data.py` script, this demo would
decrypt and returns the result, which can be used to rescale local feature data
by all data owners respectively.

```python
import prepare
import process_data

# 0 for f_range, 1 for f_mean
# use decrypted global f_range and f_mean to rescaling local feature data
res = process_data.decrypt_data(prepare.data_path + 'result', (2, prepare.feat_width, ))
```

Or use `decrypt_and_rescale.py` to decrypt, rescale the feature data which has
been saved in `feature_data.{i}.npy`, and dump the normalized data to
`normalized_data.{i}.npy` which is located in `data_path`.

Also, `verify.py` could be used to calculate error of `f_range` and `f_mean`
between direct plaintext numpy calculation and mpc mean normalize.
