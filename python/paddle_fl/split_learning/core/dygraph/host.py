import paddle.fluid as fluid
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
        self.table = lookup_table
        self.reader = reader

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
                fetch_vars = self.layer_handler.call_for_forward(feed_data)
            elif self.run_type == "INFER":
                # TODO
                pass
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

        try:
            # backward and minimize
            fetch_vars = self.layer_handler.call_for_backward(common_map)
        except Exception as e:
            err_msg = "Failed to run backward program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return self.__generate_nil_response("[Host] {}".format(err_msg))

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
        vars_map = {name: fetch_vars[name] for name in required_common_vars}
        req = util.pack_tensor_to_proto(vars_map)
        req.token = self.token
        return req

    def _inner_cancel_current_step(self, err_msg):
        _LOGGER.error(err_msg, exc_info=True)
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

    def load_layer_handler(self, layer, optimizer, common_vars):
        self.program_loader.load_layer_handler(layer, optimizer, common_vars)

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
        self.token = None

    def load_layer_handler(self, layer, optimizer, common_vars):
        self.run_type = "TRAIN"
        self.layer_handler = HostLayerHandler(layer, optimizer)
        self.common_vars = common_vars
        self.token = "init_from_full_network"
