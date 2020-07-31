# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
#
# A tools to faciliate the parallel running of fluid_encrypted test scrips.
# A test script is EXPECTED to accepted arguments in the following format:
#
# SCRIPT_NAME $ROLE $SERVER $PORT
#   ROLE:    the role of the running party
#   SERVER:  the address of the party discovering service
#   PORT:    the port of the party discovering service
#
# This tool will try to fill the above three argument to the test script,
# so that totally three processes running the script will be started, to
# simulate run of three party in a standalone machine.
#
# Usage of this script:
#
# bash run_standalone.sh TEST_SCRIPT_NAME
#

# please set the following environment vars according in your environment
PYTHON=${PYTHON}
REDIS_HOME=${PATH_TO_REDIS_BIN}
SERVER=${LOCALHOST}
PORT=${REDIS_PORT}

echo "redis home in ${REDIS_HOME}, server is ${SERVER}, port is ${PORT}"
function usage() {
    echo 'run_standalone.sh SCRIPT_NAME [ARG...]'
    exit 0
}

if [ $# -lt 1 ]; then
    usage
fi

SCRIPT=$1
if [ ! -f $SCRIPT ]; then
    echo 'Could not find script of '$SCRIPT
    exit 1
fi

REDIS_BIN=$REDIS_HOME/redis-cli
if [ ! -f $REDIS_BIN ]; then
    echo 'Could not find redis cli in '$REDIS_HOME
    exit 1
fi

# clear the redis cache
$REDIS_BIN -h $SERVER -p $PORT flushall

# remove temp data generated in last time
PRED_FILE="/tmp/mnist_output_prediction.*"
if [ "$PRED_FILE" ]; then
        rm -rf $PRED_FILE
fi

PRED_FILE="/tmp/mnist2_feature.part*"
if [ ! "$PRED_FILE" ]; then
    echo "There is no data in /tmp, please prepare data with "python prepare.py" firstly"
    exit 1
else
    echo "There are data for mnist:"
    echo "`ls ${PRED_FILE}`"
fi


# kick off script with roles of 1 and 2, and redirect output to /dev/null
for role in {1..2}; do
    $PYTHON $SCRIPT $role $SERVER $PORT 2>&1 >/dev/null &
done

# for party of role 0, run in a foreground mode and show the output
$PYTHON $SCRIPT 0 $SERVER $PORT

