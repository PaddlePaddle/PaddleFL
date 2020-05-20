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


class TrainerBase(object):
    def __init__(self):
        self.trainer_config = None
        self.train_one_user_func = None
        self.infer_one_user_func = None
        self.save_and_upload_func = None

    def get_user_param_names(self):
        pass

    def get_global_param_names(self):
        pass

    def set_trainer_configs(self, trainer_configs):
        """
        config training parameter, only support the basic types of python
        """
        self.trainer_config = trainer_configs

    def update_trainer_configs(self, key, val):
        self.trainer_config[key] = val

    def prepare(self):
        """
        generate network description string;
        """
        pass

    def init_global_model(self, scheduler_client):
        """
        initialize the network parameters, which will be broadcasted to all simulator
        """
        pass
