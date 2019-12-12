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
import paddle.fluid as fluid
from paddle_fl.core.scheduler.agent_master import FLServerAgent

class FLServer(object):

    def __init__(self):
        self._startup_program = None
        self._main_program = None
        self._scheduler_ep = None
        self._current_ep = None

    def set_server_job(self, job):
        # need to parse startup and main program in job
        # need to parse current endpoint
        # need to parse master endpoint
        self._startup_program = job._server_startup_program
        self._main_program = job._server_main_program
        self._scheduler_ep = job._scheduler_ep
        self._current_ep = None

    def start(self):
        self.agent = FLServerAgent(self._scheduler_ep, self._current_ep)
        self.agent.connect_scheduler()
        exe = fluid.Executor(fluid.CPUPlace())
        exe.run(self._startup_program)
        exe.run(self._main_program)
