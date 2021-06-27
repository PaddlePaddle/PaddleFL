#!/bin/bash

# set redis server ip and port for test
export TEST_REDIS_IP=${LOCALHOST}
export TEST_REDIS_PORT=${REDIS_PORT}

# unittest command
PYTHON_TEST="python -m unittest"

# add the modules to test
TEST_MODULES=(
"test_datautils_privc"
"test_op_add"
"test_op_sub"
"test_op_elementwise_mul"
"test_op_sum"
"test_op_square"
"test_op_square_error_cost"
"test_op_mul"
"test_op_mean"
"test_op_fc"
"test_elementwise_add_op"
"test_mul_op"
#"test_relu_op"
"test_sigmoid_op"
"test_softmax_op"
)

for MODULE in ${TEST_MODULES[@]}
do
    echo Test Module $MODULE
    $PYTHON_TEST $MODULE
done