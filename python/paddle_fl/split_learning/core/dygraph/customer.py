import paddle
import json
import time
import os
import numpy as np
import grpc
import yaml
import logging

from core.proto import common_pb2_grpc, common_pb2
from .layer_handler import CustomerLayerHandler
from core import util

_LOGGER = logging.getLogger(__name__)


class CustomerExecutor(object):

    def __init__(self, endpoints):
        self._connect(endpoints)
        self.run_type = None  # TRAIN or INFER

    def load_layer_handler(
            self, layer, optimizer, common_vars, input_spec):
        self.run_type = "TRAIN" 
        self.layer_handler = CustomerLayerHandler(layer, optimizer)
        self.common_vars = common_vars
        self.input_spec = input_spec
        self.token = "init_from_full_network"

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

    def load_inference_model(self, local_path_prefix):
        self.run_type = "INFER"
        self.exe = paddle.static.Executor(paddle.CPUPlace())
        inference_program, feed_target_names, fetch_targets = \
            paddle.static.load_inference_model(
                    path_prefix=local_path_prefix,
                    executor=self.exe)
        # load common var info
        with open("{}.{}".format(local_path_prefix, "model_info")) as f:
            model_info = json.load(f)
        self.common_vars = model_info["common"]
        self.token = model_info["token"]
        self.main_program = inference_program
        return feed_target_names, fetch_targets

    def _connect(self, endpoints):
        options = [('grpc.max_receive_message_length', 512 * 1024 * 1024),
                   ('grpc.max_send_message_length', 512 * 1024 * 1024)]
        g_endpoint = 'ipv4:{}'.format(','.join(endpoints))
        self.channel = grpc.insecure_channel(g_endpoint, options=options)
        self.stub = common_pb2_grpc.FLExecutorStub(self.channel)

    def _parse_vars_from_host(self, resp, required_common_vars):
        vars_map = util.parse_proto_to_tensor(
                resp, is_train=(self.run_type=="TRAIN"))
        # check common in
        for name in required_common_vars:
            if name not in vars_map:
                raise KeyError("Failed to calc: {} not found in query response.".format(name))
        return vars_map

    def _pack_vars_to_host(self, grad_vars, required_common_vars):
        vars_map = {name: grad_vars[name] for name in required_common_vars}
        req = util.pack_tensor_to_proto(vars_map)
        return req

    def _inner_cancel_current_step(self, err_msg):
        _LOGGER.error(err_msg, exc_info=True)
        if self.run_type == "TRAIN":
            self.layer_handler.cancel()

    def cancel_host_current_step(self, err_msg):
        self.stub.cancel_current_step(
                common_pb2.NilRequest(
                    token=self.token,
                    state=common_pb2.State(
                        succ=False,
                        error_message=err_msg)))

    def run(self, usr_key, feed, label=None, fetch_targets=[]):
        if self.run_type == "TRAIN":
            return self._run_for_train(usr_key, feed, label)
        elif self.run_type == "INFER":
            return self._run_for_infer(
                    usr_key, feed, 
                    fetch_list=fetch_targets)
        else:
            raise ValueError("Failed to execute program: "
                    "unknown run type({})".format(self.run_type))

    def _execute_forward_host_part(self, usr_key):
        # query for user feature
        user_info = common_pb2.UserInfo(
                uid=usr_key, token=self.token)
        resp = self.stub.execute_forward_host_part(user_info)
        if not resp.state.succ:
            raise RuntimeError(resp.state.error_message)
        return resp

    def _generate_feed_for_customer_part(self, feed, vars_from_host):
        for in_name in self.common_vars["in"]:
            feed[in_name] = vars_from_host[in_name]
        return feed

    def _execute_backward_host_part(self, req):
        resp = self.stub.execute_backward_host_part(req)
        if not resp.state.succ:
            raise RuntimeError(resp.state.error_message)

    def _execute_middle_customer_part(
            self, feed, label=None, fetch_list=[]):
        if self.run_type == "TRAIN":
            fetch_vars = self.layer_handler.call_for_forward(
                    label, **feed)
            return fetch_vars
        elif self.run_type == "INFER":
            fetch_vars = self.exe.run(
                    program=self.main_program,
                    feed=feed,
                    fetch_list=fetch_list,
                    return_numpy=False)
            return fetch_vars
        else:
            raise ValueError("Failed to execute program: "
                    "unknown run type({})".format(self.run_type))
    
    def _run_for_train(self, usr_key, feed, label):
        try:
            resp = self._execute_forward_host_part(usr_key)
        except Exception as e:
            err_msg = "Failed to execute forward host part: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return None
        
        try:
            vars_from_host = self._parse_vars_from_host(
                    resp, self.common_vars["in"])
            for tensor in vars_from_host.values():
                tensor.stop_gradient = False # allow gradient
        except Exception as e:
            err_msg = "Failed to parse vars from host: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        for name in self.common_vars["in"]:
            _LOGGER.debug("Get params {}: {}".format(
                name, vars_from_host[name]))

        try:
            feed = self._generate_feed_for_customer_part(feed, vars_from_host)
        except Exception as e:
            err_msg = "Failed to generate feed data: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None
        
        try:
            # forward and calc grad
            fetch_vars = self._execute_middle_customer_part(
                    feed=feed, label=label)
        except Exception as e:
            err_msg = "Failed to run middle program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None
        
        try:
            grad_vars = {
                    "{}@GRAD".format(name): tensor.grad.numpy()
                    for name, tensor in vars_from_host.items()}
            
            for name, tensor in vars_from_host.items():
                _LOGGER.debug("Send grad {}: {}".format(name, tensor.grad))

            req = self._pack_vars_to_host(
                    grad_vars, self.common_vars["out"])
            req.token = self.token
        except Exception as e:
            err_msg = "Failed to pack vars to host: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        try:
            # update params
            self.layer_handler.call_for_backward()
        except Exception as e:
            err_msg = "Failed to update params: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        try:
            self._execute_backward_host_part(req)
        except Exception as e:
            err_msg = "Failed to execute backward host part: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None
        return fetch_vars

    def _run_for_infer(self, usr_key, feed, fetch_list=[]):
        try:
            resp = self._execute_forward_host_part(usr_key)
        except Exception as e:
            err_msg = "Failed to execute forward host part: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return None

        try:
            vars_from_host = self._parse_vars_from_host(
                    resp, self.common_vars["in"])
        except Exception as e:
            err_msg = "Failed to parse vars from host: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        try:
            feed = self._generate_feed_for_customer_part(feed, vars_from_host)
        except Exception as e:
            err_msg = "Failed to generate feed data: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        try:
            fetch_vars = self._execute_middle_customer_part(
                    feed=feed, fetch_list=fetch_list)
        except Exception as e:
            err_msg = "Failed to run middle program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None
        return fetch_vars
    
    def save_persistables(self, local_path, remote_path):
        token = CustomerProgramSaver.save_persistables(
                local_path, self.layer_handler)

        resp = self.stub.save_persistables(
                common_pb2.SaveInfo(
                    path=remote_path,
                    token=self.token,
                    save_token=token))
        
        if not resp.state.succ:
            err_msg = "Failed to save vars in host side: {}".format(
                    resp.state.error_message)
            raise RuntimeError(err_msg)
        return True

    def save_inference_model(
            self,
            local_path,
            remote_path):
        token = CustomerProgramSaver.save_inference_model(
                local_path, remote_path, self.layer_handler,
                self.common_vars, self.input_spec)

        resp = self.stub.save_inference_model(
                common_pb2.SaveInfo(
                    token=self.token,
                    save_token=token,
                    path=remote_path))

        if not resp.state.succ:
            err_msg = "Failed to save inference model in host side: {}".format(resp.state.error_message)
            raise RuntimeError(err_msg)
        return True


class CustomerProgramSaver(object):

    def __init__(self):
        pass

    @staticmethod
    def save_persistables(
            dirpath, layer_handler):
        layer = layer_handler.layer
        optimizer = layer_handler.optimizer
        paddle.save(layer.state_dict(), os.path.join(dirpath, "layer.pdparams"))
        paddle.save(optimizer.state_dict(), os.path.join(dirpath, "optimizer.pdopt"))

        # token
        token = str(time.time())
        model_info = {
            "token": token,
        }
        with open(os.path.join(dirpath, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
        return token

    @staticmethod
    def save_inference_model(
            local_path, remote_path, layer_handler,
            common_vars, input_spec):
        paddle.jit.save(layer_handler.layer,
                local_path, input_spec)

        token = str(time.time())
        model_info = {
            "common": common_vars,
            "token": token,
        }

        with open("{}.{}".format(local_path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
        return token
