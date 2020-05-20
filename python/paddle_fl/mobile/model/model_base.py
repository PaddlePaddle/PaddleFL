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

import paddle.fluid as fluid
import numpy as np


def set_user_param_dict(param_names, param_dict, scope):
    place = fluid.CPUPlace()
    for var_name in param_names:
        param = scope.find_var(var_name)
        if param is None:
            print("var name: {} does not exist in memory".format(var_name))
            continue
        param.get_tensor().set(param_dict[var_name], place)
    return


def set_global_param_dict(param_names, param_dict, scope):
    place = fluid.CPUPlace()
    for var_name in param_names:
        param = scope.find_var(var_name)
        if param is None:
            print("var name: {} does not exist in memory".format(var_name))
            continue
        if var_name not in param_dict:
            print("var name: {} does not exist in global param dict".format(
                var_name))
            exit()
        var_numpy = param_dict[var_name]
        param.get_tensor().set(var_numpy, place)
    return


class ModelBase(object):
    def __init__(self):
        pass

    def init_model(self):
        pass

    def build_model(self, model_configs):
        pass

    def get_model_inputs(self):
        pass

    def get_model_loss(self):
        pass

    def get_model_metrics(self):
        pass

    def get_startup_program(self):
        pass

    def get_main_program(self):
        pass

    def get_user_param_dict(self):
        param_dict = {}
        scope = fluid.global_scope()
        for var_pair in self.get_user_param_names():
            param = scope.find_var(var_pair[0])
            if param is None:
                print("var name: {} does not exist in memory".format(var_pair[
                    0]))
                continue
            var = param.get_tensor().__array__()
            param_dict[var_pair[0]] = [var, var_pair[1].shape]
        return param_dict

    def get_global_param_dict(self):
        param_dict = {}
        scope = fluid.global_scope()
        for var_pair in self.get_global_param_names():
            param = scope.find_var(var_pair[0])
            if param is None:
                print("var name: {} does not exist in memory".format(var_pair[
                    0]))
                continue
            var = param.get_tensor().__array__()
            param_dict[var_pair[0]] = var
        return param_dict

    def get_user_param_names(self):
        user_params = []
        for var_name, var in self.startup_program_.global_block().vars.items():
            if var.persistable and "@USER" in var_name and \
               "learning_rate" not in var_name:
                user_params.append((var_name, var))
        return user_params

    def get_global_param_names(self):
        global_params = []
        for var_name, var in self.startup_program_.global_block().vars.items():
            if var.persistable and "@USER" not in var_name and \
               "learning_rate" not in var_name and "generated_var" not in var_name:
                global_params.append((var_name, var))
        return global_params
