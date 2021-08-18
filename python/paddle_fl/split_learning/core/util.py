# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
import os
import subprocess
from typing import Dict, Union
import time
import json
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.framework import Parameter, Program
from .proto import common_pb2_grpc, common_pb2
from .static import reformer


def parse_proto_to_tensor(
        proto: common_pb2.Features, 
        to_tensor: bool = True, 
        place=paddle.fluid.CPUPlace()) \
                -> Dict[str, Union[np.ndarray, paddle.Tensor, paddle.fluid.core_avx.LoDTensor]]:
    vars_map = {}
    for pb_var in proto.tensors:
        dtype = pb_var.dtype
        name = pb_var.name
        data = getattr(pb_var, "data_{}".format(dtype))
        shape = pb_var.shape
        np_data = np.array(data).astype(dtype).reshape(shape)
        if to_tensor:
            tensor = paddle.to_tensor(np_data, dtype, place)
            vars_map[name] = tensor
        else:
            vars_map[name] = np_data
    for pb_var in proto.vars:
        dtype = pb_var.dtype
        name = pb_var.name
        data = getattr(pb_var, "data_{}".format(dtype))
        shape = pb_var.shape
        recursive_seq_lens = []
        for sequence_length in pb_var.recursive_sequence_lengths.sequence_lengths:
            recursive_seq_lens.append(sequence_length)
        np_data = np.array(data).astype(dtype).reshape(shape)
        tensor = fluid.create_lod_tensor(np_data, recursive_seq_lens, place)
        vars_map[name] = tensor
    return vars_map

def pack_tensor_to_proto(vars_map: Dict[str, Union[np.ndarray, paddle.fluid.core_avx.LoDTensor]]):
    proto = common_pb2.Features()
    for name, tensor in vars_map.items():
        if isinstance(tensor, paddle.fluid.core_avx.LoDTensor):
            np_data = np.array(tensor)
            params = {
                "name": name,
                "shape": list(np_data.shape),
                "dtype": np_data.dtype.name,
                "data_{}".format(np_data.dtype): np_data.reshape(-1).tolist()
            }
            pb_var = common_pb2.Variable(**params)
            recursive_seq_lens = tensor.recursive_sequence_lengths()
            for seq_lens in recursive_seq_lens:
                pb_seq_lens = common_pb2.Variable.RecursiveSequenceLength.SequenceLength()
                pb_seq_lens.lengths.extend(seq_lens)
                pb_var.recursive_sequence_lengths.append(pb_seq_lens)
            proto.vars.append(pb_var)
        else:
            if isinstance(tensor, np.ndarray):
                np_data = tensor
            else:
                np_data = tensor.numpy()
            params = {
                "name": name,
                "shape": list(np_data.shape),
                "dtype": np_data.dtype.name,
                "data_{}".format(np_data.dtype): np_data.reshape(-1).tolist()
            }
            pb_var = common_pb2.Tensor(**params)
            proto.tensors.append(pb_var)
    proto.state.succ = True
    return proto

def save_whole_program(main_prog, startup_prog, program_path):
    if not os.path.exists(program_path):
        os.makedirs(program_path)
    main_program_str = main_prog.desc.serialize_to_string()
    startup_program_str = startup_prog.desc.serialize_to_string()
    params = main_prog.global_block().all_parameters()

    with open(program_path + '/para_info', 'w') as fout:
        for item in params:
            fout.write("%s:%s\n" % (item.name, item.trainable))

    with open(program_path + '/startup_program', "wb") as fout:
        fout.write(startup_program_str)

    with open(program_path + '/main_program', "wb") as fout:
        fout.write(main_program_str)

    stop_vars = []
    for check_stop in main_prog.list_vars():
        if check_stop.stop_gradient == True:
            stop_vars.append(check_stop.name)
    with open(program_path + '/stop_gradient', 'w') as fout:
        for stop_item in stop_vars:
            fout.write("%s\n" % stop_item)

def load_whole_program(program_input):
    with open(program_input + '/startup_program', "rb") as fin:
        new_startup = Program().parse_from_string(fin.read())

    with open(program_input + '/main_program', "rb") as fin:
        new_main = Program().parse_from_string(fin.read())

    para_list = []
    with open(program_input + '/para_info', 'r') as fin:
        for line in fin:
            current_para = {}
            para = line[:-1].split(":")
            current_para["name"] = para[0]
            if para[1] == 'True':
                current_para['trainable'] = True
            else:
                current_para['trainable'] = False
            para_list.append(current_para)
    with open(program_input + '/stop_gradient', 'r') as fin:
        for line in fin:
            stop_name = line[:-1]
            stop_var = new_main.global_block().var(stop_name)
            stop_var.stop_gradient = True

    for item in para_list:
        main_para = new_main.global_block().var(item['name'])
        main_para.__class__ = Parameter
        main_para.regularizer = None
        main_para.optimize_attr = {'learning_rate': 1.0}
        main_para.trainable = item['trainable']
        main_para.is_distributed = False

        startup_para = new_startup.global_block().var(item['name'])
        startup_para.__class__ = Parameter
        startup_para.regularizer = None
        startup_para.optimize_attr = {'learning_rate': 1.0}
        startup_para.trainable = item['trainable']
        startup_para.is_distributed = False

    return new_startup, new_main


def split_program_by_name_and_save(
        startup_program,
        main_program,
        save_path,
        feeded_var_names,
        target_var_names):
    split_program_by_key_prefix_and_save(
        startup_program,
        main_program,
        "Host|",
        save_path,
        feeded_var_names,
        target_var_names)


def split_program_by_key_prefix_and_save(
        startup_program,
        main_program,
        key_prefix,
        save_path,
        feeded_var_names,
        target_var_names):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    token = str(time.time())
    # split program by key_prefix
    splited_programs = reformer.Reformer.split_program_by_key_prefix(main_program, key_prefix)

    model_infos = []
    # common var name
    for i in range(len(splited_programs)):
        model_infos.append({"common": {"in": [], "out": []}})
    common_vars = intersection_vars(splited_programs[0], splited_programs[1])
    for name in common_vars:
        model_infos[0]["common"]["out"].append(name)
        model_infos[1]["common"]["in"].append(name)
    common_vars = intersection_vars(splited_programs[1], splited_programs[2])
    for name in common_vars:
        model_infos[1]["common"]["out"].append(name)
        model_infos[2]["common"]["in"].append(name)

    # save splited_program
    for i, program in enumerate(splited_programs):
        program_save_path = os.path.join(save_path, "part{}".format(i))
        if not os.path.exists(program_save_path):
            os.makedirs(program_save_path)
        # save startup_program
        with open(os.path.join(program_save_path, 'startup_program'), "wb") as fout:
            fout.write(startup_program.desc.serialize_to_string())
        # save main_pargram part
        with open(os.path.join(program_save_path, "main_program"), "wb") as fout:
            fout.write(program.desc.serialize_to_string())
        model_info = {
            "token": token,
            "params": [],
            "stop_gradient_vars": [],
            "target_var_names": [],
            "feeded_var_names": [],
            "persistable_vars": [],
        }
        # param name with trainable
        for param in program.global_block().all_parameters():
            model_info["params"].append({"name": param.name, "trainable": param.trainable})
        # stop_gradient var name
        for check_stop in program.list_vars():
            if check_stop.stop_gradient == True:
                model_info["stop_gradient_vars"].append(check_stop.name)
        # target_var_names
        for name in target_var_names:
            if find_var(program, name) is not None:
                model_info["target_var_names"].append(name)
        # feeded_var_names
        for name in feeded_var_names:
            if find_var(program, name) is not None:
                model_info["feeded_var_names"].append(name)
        # persistable var names
        for var in program.list_vars():
            if var.persistable == True:
                model_info["persistable_vars"].append(var.name)
        model_infos[i].update(model_info)

        with open(os.path.join(program_save_path, "model_info"), "w") as fout:
            fout.write(json.dumps(model_infos[i]))


def load_splited_program(save_path):
    startup_program, main_program = None, None
    with open(os.path.join(save_path, "startup_program"), "rb") as fin:
        startup_program = Program().parse_from_string(fin.read())
    with open(os.path.join(save_path, 'main_program'), "rb") as fin:
        main_program = Program().parse_from_string(fin.read())
    with open(os.path.join(save_path, "model_info"), "r") as fin:
        model_info = json.loads(fin.read())

    # params
    for item in model_info["params"]:
        main_para = main_program.global_block().var(item['name'])
        main_para.__class__ = Parameter
        main_para.regularizer = None
        main_para.optimize_attr = {'learning_rate': 1.0}
        main_para.trainable = item['trainable']
        main_para.is_distributed = False

        startup_para = startup_program.global_block().var(item['name'])
        startup_para.__class__ = Parameter
        startup_para.regularizer = None
        startup_para.optimize_attr = {'learning_rate': 1.0}
        startup_para.trainable = item['trainable']
        startup_para.is_distributed = False
    # stop_gradient
    for stop_name in model_info["stop_gradient_vars"]:
        stop_var = main_program.global_block().var(stop_name)
        stop_var.stop_gradient = True
    return startup_program, main_program, model_info


def intersection_vars(p1_program, p2_program):
    p1_whole_vars = [var.name for var in p1_program.list_vars()]
    p2_whole_vars = [var.name for var in p2_program.list_vars()]
    return set(p1_whole_vars) & set(p2_whole_vars)


def make_vars_persistable(program, var_names):
    for name in var_names:
        var = find_var(program, name)
        var.persistable = True


def find_var(program, var_name):
    whole_vars = program.list_vars()
    for var in whole_vars:
        if var.name == var_name:
            return var
    return None


def parse_bns_by_name(bns_name='', default_ip_port=''):
    """
    return proxy ip list
    """
    final_ip_port = default_ip_port
    (s, o) = subprocess.getstatusoutput(
        'get_instance_by_service -ip %s' % bns_name)

    if int(s) == 0:
        lns = o.split('\n')
        final_ip_port = list()
        for line in lns:
            ll = line.split(' ')
            ip_port = ""
            if len(ll) == 3:
                ip_port = (ll[1], ll[2])
            elif len(ll) == 2:
                ip_port = (ll[0], ll[1])
            final_ip_port.append(ip_port)
    return final_ip_port
