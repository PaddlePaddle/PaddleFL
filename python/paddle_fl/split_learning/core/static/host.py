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
from concurrent import futures
import contextlib
import socket
import grpc
import json
import logging
import paddle.fluid as fluid
from .. import util
from ..proto import common_pb2_grpc, common_pb2
from . import reformer

_LOGGER = logging.getLogger(__name__)


class FLExecutorServicer(common_pb2_grpc.FLExecutorServicer):
    def __init__(self, program_loader, lookup_table, reader):
        super(FLExecutorServicer, self).__init__()
        self.run_type = program_loader.run_type
        self.exe = program_loader.exe
        self.p1_program = program_loader.p1_program
        self.p1_common_vars = program_loader.p1_common_vars
        self.p3_program = program_loader.p3_program
        self.p3_common_vars = program_loader.p3_common_vars
        self.vars_need_saved = program_loader.vars_need_saved
        self.token = program_loader.token
        self.table = lookup_table
        self.reader = reader
        self.feed_data = None

        if self.run_type == "INFER":
            #self.target_vars = [
            #        util.find_var(
            #            self.p1_program, "{}.tmp_0".format(x))
            #        for x in self.p1_common_vars["out"]]
            self.target_vars = [
                    util.find_var(
                        self.p1_program, "save_infer_model/scale_{}.tmp_0".format(idx))
                    for idx in range(len(self.p1_common_vars["out"]))]

    def execute_forward_host_part(self, request, context):
        if request.token != self.token:
            err_msg = "Failed: token({}) is not valid.".format(request.token)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_err_features("[Host] {}".format(err_msg))

        uid = request.uid

        try:
            value = self.table.lookup(uid)
            inputs = self.reader.parse(value)
        except Exception as e:
            err_msg = "Failed to lookup for input: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_err_features("[Host] {}".format(err_msg))

        # can not modify
        self.feed_data = {name: tensor for name, tensor in inputs.items()}
        fetch_vars = None

        try:
            if self.run_type == "TRAIN":
                fetch_names = self.p1_common_vars["out"]
                fetch_vars = self._execute_p1_program(self.feed_data, fetch_names)
            elif self.run_type == "INFER":
                target_vars = self.target_vars
                fetch_vars = self._execute_p1_program(self.feed_data, target_vars)
        except Exception as e:
            err_msg = "Failed to run forward program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_err_features("[Host] {}".format(err_msg))

        try:
            resp = self._pack_vars_to_client(fetch_vars, self.p1_common_vars["out"])
        except Exception as e:
            err_msg = "Failed to pack vars to client: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_err_features("[Host] {}".format(err_msg))
        return resp

    def execute_backward_host_part(self, request, context):
        if request.token != self.token:
            err_msg = "Failed: token({}) is not valid.".format(req_token)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_nil_response("[Host] {}".format(err_msg))
        try:
            common_map = self._parse_vars_from_client(request, self.p3_common_vars["in"])
        except Exception as e:
            err_msg = "Failed to parse vars from client: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_nil_response("[Host] {}".format(err_msg))

        # generate feed
        self.feed_data.update(common_map)
        fetch_names = self.p3_common_vars["out"]  # []
        try:
            fetch_vars = self._execute_p3_program(self.feed_data, fetch_names)
        except Exception as e:
            err_msg = "Failed to run backward program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_nil_response("[Host] {}".format(err_msg))
        return self.__generate_nil_response()

    def save_persistables(self, request, context):
        if request.token != self.token:
            err_msg = "Failed: token({}) is not valid.".format(req_token)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_nil_response("[Host] {}".format(err_msg))

        try:
            HostProgramSaver.save_persistables(
                request.path, self.exe,
                self.p1_program, self.p3_program,
                self.vars_need_saved, request.save_token)
        except Exception as e:
            err_msg = "Failed to save vars: {}".format(e)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_nil_response("[Host] {}".format(err_msg))
        return self.__generate_nil_response()

    def save_inference_model(self, request, context):
        if request.token != self.token:
            err_msg = "Failed: token({}) is not valid.".format(req_token)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_nil_response("[Host] {}".format(err_msg))

        try:
            HostProgramSaver.save_inference_model(
                request.path, self.exe, self.p1_program,
                self.p1_common_vars, list(request.feeded_var_names),
                request.save_token)
        except Exception as e:
            err_msg = "Failed to save inference model: {}".format(e)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_nil_response("[Host] {}".format(err_msg))
        return self.__generate_nil_response()

    def cancel_current_step(self, request, context):
        if request.token != self.token:
            err_msg = "Failed: token({}) is not valid.".format(req_token)
            _LOGGER.error(err_msg, exc_info=True)
            return self.__generate_nil_response("[Host] {}".format(err_msg))
        self._inner_cancel_current_step(request.state.error_message)
        return self.__generate_nil_response()

    def _parse_vars_from_client(self, request, required_common_vars):
        vars_map = util.parse_proto_to_tensor(request)
        # check common in 
        for name in required_common_vars:
            if name not in vars_map:
                raise KeyError(
                    "Failed to parse vars from client: {} not found in response.".format(name))
        return vars_map

    def _pack_vars_to_client(self, fetch_vars, required_common_vars):
        vars_map = {name: fetch_vars[idx] for idx, name in enumerate(required_common_vars)}
        req = util.pack_tensor_to_proto(vars_map)
        req.token = self.token
        return req

    def _execute_p1_program(self, feed_data, fetch_list):
        fetch_vars = self.exe.run(
            program=self.p1_program,
            feed=feed_data,
            fetch_list=fetch_list,
            return_numpy=False)  # same lod_tensor can not transfer to numpy
        return fetch_vars

    def _execute_p3_program(self, feed_data, fetch_list):
        fetch_vars = self.exe.run(
            program=self.p3_program,
            feed=feed_data,
            fetch_list=fetch_list,
            return_numpy=False)
        return fetch_vars

    def _inner_cancel_current_step(self, err_msg):
        _LOGGER.error(err_msg, exc_info=True)
        self.feed_data = None

    def __generate_nil_response(self, error_message=None):
        if error_message:
            return common_pb2.NilResponse(
                state=common_pb2.State(
                    succ=False,
                    error_message=error_message))
        else:
            return common_pb2.NilResponse(
                state=common_pb2.State(succ=True))

    def __generate_err_features(self, error_message):
        return common_pb2.Features(
            token=self.token,
            state=common_pb2.State(
                succ=False,
                error_message=error_message))


class HostExecutor(object):
    def __init__(self, place, table, reader, max_workers=1):
        self.program_loader = HostProgramLoader(place)
        self.lookup_table = table
        self.max_workers = max_workers
        self.reader = reader

    def load_program_from_full_network(
            self, startup_program, main_program):
        self.program_loader.load_program_from_full_network(
            startup_program, main_program)

    def load_program_from_splited_file(
            self, p1_program_path, p3_program_path):
        self.program_loader.load_program_from_splited_file(
            p1_program_path, p3_program_path)

    def load_inference_model(self, local_path):
        self.program_loader.load_inference_model(local_path)

    def load_persistables(self, path):
        self.program_loader.load_persistables(path)

    def _is_port_available(self, port):
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(2)
            result = sock.connect_ex(('0.0.0.0', port))
        return result != 0

    def start(self, port):
        if not self._is_port_available(port):
            raise ValueError("Failed to start: port {} not available".format(port))
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[('grpc.max_send_message_length', 256 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 256 * 1024 * 1024)])
        common_pb2_grpc.add_FLExecutorServicer_to_server(
            FLExecutorServicer(
                self.program_loader, self.lookup_table, self.reader), server)
        server.add_insecure_port('[::]:{}'.format(port))
        server.start()
        _LOGGER.info("Run service in port: {}".format(port))
        server.wait_for_termination()


class HostProgramLoader(object):
    def __init__(self, place):
        self.exe = fluid.Executor(place)
        self.run_type = None  # TRAIN or INFER
        self.p1_program = None
        self.p1_common_vars = None
        self.p3_program = None
        self.p3_common_vars = None
        self.vars_need_saved = None
        self.token = None

    def load_program_from_full_network(
            self, startup_program, main_program):
        self.run_type = "TRAIN"
        splited_programs = reformer.Reformer.split_program_by_name(main_program)
        self.p1_program = splited_programs[0]
        self.p3_program = splited_programs[2]
        self.p1_common_vars = {
            "in": [],
            "out": list(
                util.intersection_vars(
                    splited_programs[0], splited_programs[1]))}
        self.p3_common_vars = {
            "in": list(
                util.intersection_vars(
                    splited_programs[1], splited_programs[2])),
            "out": []}
        self.vars_need_saved = set()
        for program in [self.p1_program, self.p3_program]:
            for var in program.list_vars():
                if var.persistable == True:
                    self.vars_need_saved.add(var.name)
        self.token = "init_from_full_network"
        self.exe.run(startup_program)
        self._make_temp_vars_persistable()

    def load_program_from_splited_file(
            self, p1_program_path, p3_program_path):
        self.run_type = "TRAIN"
        startup_program, p1_program, p1_model_info \
            = util.load_splited_program(p1_program_path)
        self.p1_program = p1_program
        self.p1_common_vars = p1_model_info["common"]
        _, p3_program, p3_model_info \
            = util.load_splited_program(p3_program_path)
        self.p3_program = p3_program
        self.p3_common_vars = p3_model_info["common"]
        self.token = p1_model_info["token"]
        self.vars_need_saved = set(p1_model_info["persistable_vars"]) & \
                               set(p3_model_info["persistable_vars"])
        self.exe.run(startup_program)
        self._make_temp_vars_persistable()

    def load_inference_model(self, local_path):
        self.run_type = "INFER"
        inference_program, feed_target_names, fetch_targets = \
            fluid.io.load_inference_model(
                dirname=local_path,
                executor=self.exe)
        self.p1_program = inference_program
        # load common var info
        with open(os.path.join(local_path, "model_info")) as f:
            p1_model_info = json.load(f)

        self.p1_common_vars = p1_model_info["common"]
        self.token = p1_model_info["token"]
        self.p3_program = None
        self.p3_common_vars = None
        self.vars_need_saved = None

    def load_persistables(self, path):
        p1_vars = []
        p3_vars = []
        for name in self.vars_need_saved:
            var = util.find_var(self.p1_program, name)
            if var is not None:
                p1_vars.append(var)
            else:
                p3_vars.append(
                    util.find_var(
                        self.p3_program, name))

        fluid.io.load_vars(
            executor=self.exe,
            dirname=path,
            main_program=self.p1_program,
            vars=p1_vars)

        fluid.io.load_vars(
            executor=self.exe,
            dirname=path,
            main_program=self.p3_program,
            vars=p3_vars)
        # load token info
        with open(os.path.join(path, "model_info")) as f:
            model_info = json.load(f)
        self.token = model_info["token"]

    def _make_temp_vars_persistable(self):
        intersection_vars = util.intersection_vars(
            self.p1_program, self.p3_program)
        util.make_vars_persistable(
            self.p1_program, intersection_vars)
        util.make_vars_persistable(
            self.p3_program, intersection_vars)


class HostProgramSaver(object):
    def __init__(self):
        pass

    @staticmethod
    def save_persistables(
            path, exe,
            p1_program, p3_program,
            vars_need_saved, save_token):
        p1_vars = []
        p3_vars = []
        for name in vars_need_saved:
            var = util.find_var(p1_program, name)
            if var is not None:
                p1_vars.append(var)
            else:
                p3_vars.append(
                    util.find_var(p3_program, name))

        fluid.io.save_vars(
            executor=exe,
            dirname=path,
            main_program=p1_program,
            vars=p1_vars)

        fluid.io.save_vars(
            executor=exe,
            dirname=path,
            main_program=p3_program,
            vars=p3_vars)

        model_info = {
            "token": save_token,
        }

        with open(os.path.join(path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))

    @staticmethod
    def save_inference_model(
            path, exe, p1_program, p1_common_vars,
            feeded_var_names, save_token):

        # check
        for name in feeded_var_names:
            if not util.find_var(p1_program, name):
                raise RuntimeError(
                    "feeded_var_names({}) not in host side.".format(name))

        target_vars = [util.find_var(p1_program, name)
                       for name in p1_common_vars["out"]]

        fluid.io.save_inference_model(
            dirname=path,
            feeded_var_names=feeded_var_names,
            target_vars=target_vars,
            executor=exe,
            main_program=p1_program)

        model_info = {
            "common": p1_common_vars,
            "token": save_token,
        }

        # save common var info
        with open(os.path.join(path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
