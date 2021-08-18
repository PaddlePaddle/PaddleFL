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
import paddle
import numpy as np
import os
import json
from concurrent import futures
import contextlib
import socket
import grpc
import logging
from typing import List, Dict, Tuple, Union
from .proto import common_pb2_grpc, common_pb2
from .layer_handler import HostLayerHandler
from .layer_handler.layer_base import LayerBase
from . import util
from .table.table_base import TableBase
from .reader.reader_base import ReaderBase

_LOGGER = logging.getLogger(__name__)


class HostProgramLoader(object):

    def __init__(self):
        self.run_type = None  # TRAIN or INFER
        self.tensor_names_to_customer = None
        self.layer_handler = None
        self.input_specs = None
        self.exe = None
        self.main_program = None
        self.token = None

    def init(self, 
            layer: LayerBase, 
            optimizer: paddle.optimizer.Optimizer, 
            tensor_names_to_customer: List[str],
            input_specs: List[paddle.static.InputSpec]):
        self.run_type = "TRAIN"
        self.layer_handler = HostLayerHandler(layer, optimizer)
        self.tensor_names_to_customer = tensor_names_to_customer
        self.input_specs = input_specs
        self.token = "init_from_full_network"

    def load_inference_model(self, 
            local_path_prefix: str) -> None:
        self.run_type = "INFER"
        self.exe = paddle.static.Executor(paddle.CPUPlace())
        inference_program, feed_target_names, fetch_targets = \
                paddle.static.load_inference_model(
                        path_prefix=local_path_prefix,
                        executor=self.exe)
        self.main_program = inference_program
        # load common var info
        with open("{}.{}".format(local_path_prefix, "model_info")) as f:
            model_info = json.load(f)
        self.tensor_names_to_customer = model_info["tensor_names_to_customer"]
        self.token = model_info["token"]
    
    def load_persistables(self, path: str) -> None:
        layer_state_dict = paddle.load(
                os.path.join(path, "layer.pdparams"))
        opt_state_dict = paddle.load(
                os.path.join(path, "optimizer.pdopt"))
        self.layer_handler.layer.set_state_dict(layer_state_dict)
        self.layer_handler.optimizer.set_state_dict(opt_state_dict)

        # load token info
        with open(os.path.join(path, "model_info")) as f:
            model_info = json.load(f)
        self.token = model_info["token"]


class FLExecutorServicer(common_pb2_grpc.FLExecutorServicer):

    def __init__(self, 
            program_loader: HostProgramLoader, 
            lookup_table: TableBase, 
            reader: ReaderBase):
        super(FLExecutorServicer, self).__init__()
        self.run_type = program_loader.run_type
        self.tensor_names_to_customer = program_loader.tensor_names_to_customer
        self.token = program_loader.token
        self.layer_handler = program_loader.layer_handler
        self.input_specs = program_loader.input_specs
        self.exe = program_loader.exe
        self.main_program = program_loader.main_program
        self.table = lookup_table
        self.reader = reader

        if self.run_type == "INFER":
            self.infer_target_vars = [
                    util.find_var(
                        self.main_program, "save_infer_model/scale_{}.tmp_0".format(idx))
                    for idx in range(len(self.tensor_names_to_customer))]

    def execute_forward_host_part(self, request, context):
        if request.token != self.token:
            err_msg = "Failed: token({}) is not valid.".format(request.token)
            _LOGGER.error(err_msg)
            return self.__generate_err_features("[Host] {}".format(err_msg))

        uid = request.uid

        try:
            value = self.table.lookup(uid)
            inputs = self.reader.parse(value)
        except Exception as e:
            err_msg = "Failed to lookup for input: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_err_features("[Host] {}".format(err_msg))

        feed_data = {name: tensor for name, tensor in inputs.items()}
        fetch_vars = None

        try:
            if self.run_type == "TRAIN":
                # forward only
                fetch_vars = self.layer_handler.call_for_forward(**feed_data)
            elif self.run_type == "INFER":
                fetch_vars = self.exe.run(
                        program=self.main_program,
                        feed=feed_data,
                        fetch_list=self.infer_target_vars,
                        return_numpy=False)
            else:
                raise ValueError("Failed to execute program: "
                        "unknown run type({})".format(self.run_type))
        except Exception as e:
            err_msg = "Failed to run forward program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_err_features("[Host] {}".format(err_msg))

        try:
            resp = self._pack_vars_to_client(
                    fetch_vars, self.tensor_names_to_customer)
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
            grad_map = self._parse_vars_from_client(
                    request, self.tensor_names_to_customer)
        except Exception as e:
            err_msg = "Failed to parse vars from client: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_nil_response("[Host] {}".format(err_msg))

        for name, tensor in grad_map.items():
            _LOGGER.debug("Get grad {}: {}".format(name, tensor))

        try:
            # backward and minimize
            fetch_vars = self.layer_handler.call_for_backward(
                    grad_map, self.tensor_names_to_customer)
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
                    request.path, self.layer_handler, request.save_token)
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
                    request.path, self.layer_handler, 
                    self.input_specs, self.tensor_names_to_customer,
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

    def _parse_vars_from_client(self, 
            request: common_pb2.Features, 
            tensor_names_to_customer: List[str]) \
                    -> Dict[str, Union[paddle.Tensor, np.ndarray]]:
        vars_map = util.parse_proto_to_tensor(request, to_tensor=(self.run_type=="TRAIN"))
        vars_grad_map = {}
        for name in tensor_names_to_customer:
            grad_name = "{}@GRAD".format(name)
            if grad_name not in vars_map:
                raise KeyError(
                        "Failed to parse vars from client: {} not found in response.".format(name))
            vars_grad_map[grad_name] = vars_map[grad_name]
        return vars_grad_map

    def _pack_vars_to_client(self, 
            fetch_vars: List[paddle.Tensor], 
            tensor_names_to_customer: List[str]) -> common_pb2.Features:
        vars_map = {name: fetch_vars[idx] for idx, name in enumerate(tensor_names_to_customer)}
        req = util.pack_tensor_to_proto(vars_map)
        req.token = self.token
        return req

    def _inner_cancel_current_step(self, err_msg):
        _LOGGER.error(err_msg, exc_info=True)
        if self.run_type == "TRAIN":
            self.layer_handler.cancel()

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

    def __init__(self, table, reader, max_workers=1):
        self.program_loader = HostProgramLoader()
        self.table = table
        self.reader = reader
        self.max_workers = max_workers

    def init(self, 
            layer: LayerBase, 
            optimizer: paddle.optimizer.Optimizer, 
            tensor_names_to_customer: List[str], 
            input_specs: List[paddle.static.InputSpec]) -> None:
        self.program_loader.init(
                layer, optimizer, tensor_names_to_customer, input_specs)

    def load_inference_model(self, local_path: str) -> None:
        self.program_loader.load_inference_model(local_path)

    def load_persistables(self, path: str) -> None:
        self.program_loader.load_persistables(path)

    def _is_port_available(self, port: int) -> bool:
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            sock.settimeout(2)
            result = sock.connect_ex(('0.0.0.0', port))
        return result != 0

    def start(self, port: int) -> None:
        if not self._is_port_available(port):
            raise ValueError("Failed to start: port {} not available".format(port))
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=self.max_workers),
            options=[('grpc.max_send_message_length', 256 * 1024 * 1024),
                     ('grpc.max_receive_message_length', 256 * 1024 * 1024)])
        common_pb2_grpc.add_FLExecutorServicer_to_server(
                FLExecutorServicer(self.program_loader, self.table, self.reader), server)
        server.add_insecure_port('[::]:{}'.format(port))
        _LOGGER.info("Run service in port: {}".format(port))
        server.start()
        server.wait_for_termination()


class HostProgramSaver(object):

    def __init__(self):
        pass

    @staticmethod
    def save_persistables(
            dirpath: str, 
            layer_handler: HostLayerHandler, 
            save_token: str) -> None:
        layer = layer_handler.layer
        optimizer = layer_handler.optimizer
        paddle.save(layer.state_dict(), os.path.join(dirpath, "layer.pdparams"))
        paddle.save(optimizer.state_dict(), os.path.join(dirpath, "optimizer.pdopt"))

        model_info = {
            "token": save_token,
        }
        with open(os.path.join(dirpath, "model_info"), "w") as f:
            f.write(json.dumps(model_info))

    @staticmethod
    def save_inference_model(
            path: str, 
            layer_handler: HostLayerHandler, 
            input_specs: List[paddle.static.InputSpec], 
            tensor_names_to_customer: List[str],
            save_token: str) -> None:

        paddle.jit.save(layer_handler.layer, path, input_specs)

        model_info = {
            "tensor_names_to_customer": tensor_names_to_customer,
            "token": save_token,
        }

        with open("{}.{}".format(path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
