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

from role_maker import FLSimRoleMaker

role_maker = FLSimRoleMaker()
role_maker.init_env(local_shard_num=30)
print("simulator num: {}".format(role_maker.simulator_num()))
print("simulator idx: {}".format(role_maker.simulator_idx()))
print("global scheduler endpoint: {}".format(
    role_maker.get_global_scheduler_endpoint()))
print("data server endpoints")
print(role_maker.get_data_server_endpoints())
print("local data server")
print(role_maker.get_local_data_server_endpoint())
print("local param server")
print(role_maker.get_local_param_server_endpoint())
role_maker.barrier_simulator()
