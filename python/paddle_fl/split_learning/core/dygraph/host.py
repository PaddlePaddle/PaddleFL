import paddle
import numpy as np
import os
import json
from concurrent import futures
import contextlib
import socket
import grpc
import logging

from core.proto import common_pb2_grpc, common_pb2
from .layer_handler import HostLayerHandler
from core import util
    
_LOGGER = logging.getLogger(__name__)


class FLExecutorServicer(common_pb2_grpc.FLExecutorServicer):
    
    def __init__(self, program_loader, lookup_table, reader):
        super(FLExecutorServicer, self).__init__()
        self.run_type = program_loader.run_type
        self.common_vars = program_loader.common_vars
        self.token = program_loader.token
        self.layer_handler = program_loader.layer_handler
        self.input_spec = program_loader.input_spec
        self.exe = program_loader.exe
        self.main_program = program_loader.main_program
        self.table = lookup_table
        self.reader = reader

        if self.run_type == "INFER":
            self.target_vars = [
                    util.find_var(
                        self.main_program, "save_infer_model/scale_{}.tmp_0".format(idx))
                    for idx in range(len(self.common_vars["out"]))]

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
                        fetch_list=self.target_vars,
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
                    fetch_vars, self.common_vars["out"])
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
            common_map = self._parse_vars_from_client(
                    request, self.common_vars["in"])
        except Exception as e:
            err_msg = "Failed to parse vars from client: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_nil_response("[Host] {}".format(err_msg))

        for name, tensor in common_map.items():
            _LOGGER.debug("Get grad {}: {}".format(name, tensor))

        try:
            # backward and minimize
            fetch_vars = self.layer_handler.call_for_backward(
                    common_map, self.common_vars["out"])
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
                    self.input_spec, self.common_vars,
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
        vars_map = util.parse_proto_to_tensor(
                request, is_train=(self.run_type=="TRAIN"))
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

    def load_layer_handler(self, layer, optimizer, common_vars, input_spec):
        self.program_loader.load_layer_handler(
                layer, optimizer, common_vars, input_spec)

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
                FLExecutorServicer(self.program_loader, self.table, self.reader), server)
        server.add_insecure_port('[::]:{}'.format(port))
        _LOGGER.info("Run service in port: {}".format(port))
        server.start()
        server.wait_for_termination()


class HostProgramLoader(object):

    def __init__(self):
        self.run_type = None  # TRAIN or INFER
        self.common_vars = None
        self.layer_handler = None
        self.input_spec = None
        self.exe = None
        self.main_program = None
        self.token = None

    def load_layer_handler(self, layer, optimizer, common_vars, input_spec):
        self.run_type = "TRAIN"
        self.layer_handler = HostLayerHandler(layer, optimizer)
        self.common_vars = common_vars
        self.input_spec = input_spec
        self.token = "init_from_full_network"

    def load_inference_model(self, local_path_prefix):
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
        self.common_vars = model_info["common"]
        self.token = model_info["token"]
    
    def load_persistables(self, path):
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


class HostProgramSaver(object):

    def __init__(self):
        pass

    @staticmethod
    def save_persistables(
            dirpath, layer_handler, save_token):
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
            path, layer_handler, input_spec, 
            common_vars, save_token):

        paddle.jit.save(layer_handler.layer, path, input_spec)

        model_info = {
            "common": common_vars,
            "token": save_token,
        }

        with open("{}.{}".format(path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
