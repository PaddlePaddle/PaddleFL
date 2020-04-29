#!/bin/bash
# This script is used for run unit tests.

# modify the following vars according to your environment
LD_LIB_PATH="path_to_needed_libs"
MPC_DATA_UTILS_MODULE_PATH="path_to_mpc_data_utils_so_file"

export LD_LIBRARY_PATH=$LD_LIB_PATH:$LD_LIBRARY_PATH
export PYTHONPATH=$MPC_DATA_UTILS_MODULE_PATH:$PYTHON_PATH

PYTHON_TEST="python -m unittest"

# add your test modules here
TEST_MODULES=("test_datautils_load_filter")

for MODULE in ${TEST_MODULES[@]} 
do
    $PYTHON_TEST $MODULE
done
