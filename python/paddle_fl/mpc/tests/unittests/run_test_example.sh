#!/bin/bash

# set redis server ip and port for test
export TEST_REDIS_IP=${LOCALHOST}
export TEST_REDIS_PORT=${REDIS_PORT}

# unittest command
PYTHON_TEST="python -m unittest"

# add the modules to test
TEST_MODULES=("test_datautils_aby3"
"test_datautils_align"
"test_op_add"
"test_op_sub"
"test_op_mul"
"test_op_square"
"test_op_sum"
"test_op_mean"
"test_op_square_error_cost"
"test_op_fc"
"test_op_relu"
"test_op_compare"
"test_op_embedding"
"test_op_softmax_with_cross_entropy"
"test_op_batch_norm"
"test_op_conv"
"test_op_pool"
"test_op_metric"
"test_data_preprocessing"
"test_op_reshape"
"test_op_reduce_sum"
"test_op_elementwise_mul"
"test_gru_op"
)

# run unittest
for MODULE in ${TEST_MODULES[@]}
do
    echo Test Module $MODULE
    $PYTHON_TEST $MODULE
done
