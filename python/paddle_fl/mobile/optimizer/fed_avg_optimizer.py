# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .optimizer_base import OptimizerBase


class FedAvgOptimizer(OptimizerBase):
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        pass

    def update(self, user_info, new_global_param_by_user, old_global_param,
               scheduler_client):
        total_weight = 0.0
        for uid in user_info:
            total_weight += float(user_info[uid])
        update_dict = {}
        for key in new_global_param_by_user:
            uid = key
            uid_global_w = new_global_param_by_user[key]
            weight = float(user_info[key]) / total_weight
            for param_name in uid_global_w:
                if param_name in update_dict:
                    update_dict[param_name] += \
                        self.learning_rate * weight * (uid_global_w[param_name] - old_global_param[param_name])
                else:
                    update_dict[param_name] = \
                        self.learning_rate * weight * (uid_global_w[param_name] - old_global_param[param_name])

        scheduler_client.fedavg_update(update_dict)


class SumOptimizer(OptimizerBase):
    def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
        pass

    def update(self, user_info, new_global_param_by_user, old_global_param,
               scheduler_client):
        update_dict = {}
        for key in new_global_param_by_user:
            uid = key
            uid_global_w = new_global_param_by_user[key]
            weight = 1.0
            for param_name in uid_global_w:
                if param_name in update_dict:
                    update_dict[param_name] += \
                        self.learning_rate * weight * (uid_global_w[param_name] - old_global_param[param_name])
                else:
                    update_dict[param_name] = \
                        self.learning_rate * weight * (uid_global_w[param_name] - old_global_param[param_name])

        scheduler_client.fedavg_update(update_dict)
