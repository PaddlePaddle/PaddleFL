#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

import paddle_fl.paddle_fl as fl
import os
import paddle.fluid as fluid
from paddle_fl.paddle_fl.core.server.fl_server import FLServer
from paddle_fl.paddle_fl.core.master.fl_job import FLRunTimeJob
import time
server = FLServer()
server_id = 0
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_server_job(job_path, server_id)
job._scheduler_ep = os.environ['FL_SCHEDULER_SERVICE_HOST'] + ":" + os.environ[
    'FL_SCHEDULER_SERVICE_PORT_FL_SCHEDULER']  # IP address for scheduler
#job._endpoints = os.environ['POD_IP'] + ":" + os.environ['FL_SERVER_SERVICE_PORT_FL_SERVER']  # IP address for server
server.set_server_job(job)
server._current_ep = os.environ['FL_SERVER_SERVICE_HOST'] + ":" + os.environ[
    'FL_SERVER_SERVICE_PORT_FL_SERVER']  # IP address for server
print(job._scheduler_ep, server._current_ep)
server.start()
print("connect")
