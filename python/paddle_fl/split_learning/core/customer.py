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
import json
import time
import os
import numpy as np
import grpc
import yaml
import logging
from typing import List, Dict, Any, Union, Tuple
from .proto import common_pb2_grpc, common_pb2
from .layer_handler import CustomerLayerHandler
from .layer_handler.layer_base import LayerBase
from . import util

_LOGGER = logging.getLogger(__name__)


class CustomerExecutor(object):

    def __init__(self, endpoints: List[str]):
        self._connect(endpoints)
        self.run_type = None  # TRAIN or INFER

    def init(self, 
            layer: LayerBase, 
            optimizer: paddle.optimizer.Optimizer, 
            tensor_names_from_host: List[str], 
            input_specs: List[paddle.static.InputSpec]) -> None:
        self.run_type = "TRAIN" 
        self.layer_handler = CustomerLayerHandler(layer, optimizer)
        self.tensor_names_from_host = tensor_names_from_host
        self.input_specs = input_specs
        self.customer_feed_names = [name for name in 
                [spec.name for spec in self.input_specs]
                if name not in self.tensor_names_from_host]
        self.token = "init_from_full_network"

    def run(self, 
            usr_key: str, 
            feed: Dict[str, Union[np.ndarray, paddle.Tensor]], 
            label: Union[np.ndarray, paddle.Tensor, None] = None, 
            fetch_targets: List[paddle.fluid.framework.Variable] = []) \
                    -> List[paddle.Tensor]:
        if self.run_type == "TRAIN":
            return self._run_for_train(usr_key, feed, label)
        elif self.run_type == "INFER":
            return self._run_for_infer(
                    usr_key, feed, fetch_list=fetch_targets)
        else:
            raise ValueError("Failed to execute program: "
                    "unknown run type({})".format(self.run_type))

    def _run_for_train(self, 
            usr_key: str, 
            feed: Dict[str, Union[np.ndarray, paddle.Tensor]], 
            label: Union[np.ndarray, paddle.Tensor]) \
                    -> List[paddle.Tensor]:
        # check feed map
        if len(feed) != len(self.customer_feed_names):
            raise KeyError(
                    "Failed: required feed map not match")
        for name in self.customer_feed_names:
            if name not in feed:
                raise KeyError(
                        "Failed: feed '{}' is required but not found".format(name))

        # execute forward host part
        resp = self._execute_forward_host_part(usr_key)
        
        vars_from_host = self._parse_vars_from_host(resp, self.tensor_names_from_host)
        for tensor in vars_from_host.values():
            tensor.stop_gradient = False # allow gradient
        feed = self._generate_feed_for_customer_part(feed, vars_from_host)
        
        # execute forward custmer part (and calcute grad)
        fetch_vars, loss = self._execute_middle_customer_part(feed=feed, label=label)
        
        for name, tensor in vars_from_host.items():
            _LOGGER.debug("Send grad {}: {}".format(name, tensor.grad))

        # Paddle2.0.0
        grad_vars = {"{}@GRAD".format(name): tensor.grad
                for name, tensor in vars_from_host.items()}
        # Paddle2.1.0
        #grad_vars = {"{}@GRAD".format(name): tensor.grad.numpy()
        #        for name, tensor in vars_from_host.items()}
        req = self._pack_vars_to_host(
                grad_vars, self.tensor_names_from_host, token=self.token)

        # execute backward custmer part (update params)
        self.layer_handler.call_for_backward()

        # execute backward host part
        self._execute_backward_host_part(req)
        return fetch_vars, loss

    def _connect(self, endpoints: List[str]) -> None:
        options = [('grpc.max_receive_message_length', 512 * 1024 * 1024),
                   ('grpc.max_send_message_length', 512 * 1024 * 1024)]
        g_endpoint = 'ipv4:{}'.format(','.join(endpoints))
        self.channel = grpc.insecure_channel(g_endpoint, options=options)
        self.stub = common_pb2_grpc.FLExecutorStub(self.channel)

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

    def load_inference_model(self, local_path_prefix: str) \
            -> Tuple[List[str], List[paddle.fluid.framework.Variable]]:
        self.run_type = "INFER"
        self.exe = paddle.static.Executor(paddle.CPUPlace())
        inference_program, feed_target_names, fetch_targets = \
            paddle.static.load_inference_model(
                    path_prefix=local_path_prefix,
                    executor=self.exe)
        # load common var info
        with open("{}.{}".format(local_path_prefix, "model_info")) as f:
            model_info = json.load(f)
        self.tensor_names_from_host = model_info["tensor_names_from_host"]
        self.token = model_info["token"]
        self.main_program = inference_program
        return feed_target_names, fetch_targets

    def _parse_vars_from_host(self, 
            resp: common_pb2.Features, 
            required_var_names: List[str]) -> Dict[str, Union[np.ndarray, paddle.Tensor]]:
        vars_map = util.parse_proto_to_tensor(resp, to_tensor=(self.run_type=="TRAIN"))
        # check common in
        for name in required_var_names:
            if name not in vars_map:
                raise KeyError(
                        "Failed to parse vars from host: {} not found in response."
                        .format(name))
        return vars_map

    def _pack_vars_to_host(self, 
            grad_vars: Dict[str, paddle.Tensor], 
            required_common_vars: List[str],
            token: str) -> common_pb2.Features:
        vars_map = {}
        for name in required_common_vars:
            grad_name = "{}@GRAD".format(name)
            vars_map[grad_name] = grad_vars[grad_name]
        req = util.pack_tensor_to_proto(vars_map)
        req.token = token
        return req

    def cancel_current_step(self, err_msg: str):
        if self.run_type == "TRAIN":
            self.layer_handler.cancel()
        self.stub.cancel_current_step(
                common_pb2.NilRequest(
                    token=self.token,
                    state=common_pb2.State(
                        succ=False,
                        error_message=err_msg)))

    def _execute_forward_host_part(self, usr_key: str) -> common_pb2.Features:
        # query for user feature
        user_info = common_pb2.UserInfo(
                uid=usr_key, token=self.token)
        resp = self.stub.execute_forward_host_part(user_info)
        if not resp.state.succ:
            raise RuntimeError(
                    "Failed to execute forward host part: {}".format(
                        resp.state.error_message))
        return resp

    def _generate_feed_for_customer_part(self, 
            feed: Dict[str, Union[np.ndarray, paddle.Tensor]], 
            vars_from_host: Dict[str, Union[np.ndarray, paddle.Tensor]]) \
                    -> Dict[str, Union[np.ndarray, paddle.Tensor]]:
        for in_name in self.tensor_names_from_host:
            feed[in_name] = vars_from_host[in_name]
        return feed

    def _execute_backward_host_part(self, req: common_pb2.Features) -> None:
        resp = self.stub.execute_backward_host_part(req)
        if not resp.state.succ:
            raise RuntimeError(resp.state.error_message)

    def _execute_middle_customer_part(self, 
            feed: Dict[str, Union[np.ndarray, paddle.Tensor]], 
            label: Union[np.ndarray, paddle.Tensor, None] = None, 
            fetch_list: List[paddle.fluid.framework.Variable] = []) \
                    -> List[paddle.Tensor]:
        if self.run_type == "TRAIN":
            fetch_vars, loss = self.layer_handler.call_for_forward(label, **feed)
            return fetch_vars, loss
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
    
    def _run_for_infer(self, 
            usr_key: str, 
            feed: Dict[str, Union[np.ndarray, paddle.Tensor]], 
            fetch_list: List[paddle.fluid.framework.Variable] = []) \
                    -> List[paddle.Tensor]:
        # execute forward host part
        resp = self._execute_forward_host_part(usr_key)
        vars_from_host = self._parse_vars_from_host(resp, self.tensor_names_from_host)

        feed = self._generate_feed_for_customer_part(feed, vars_from_host)

        # execute forward custmer part (and calcute grad)
        fetch_vars = self._execute_middle_customer_part(feed=feed, fetch_list=fetch_list)
        return fetch_vars
    
    def save_persistables(self, 
            local_path: str, 
            remote_path: str) -> bool:
        token = CustomerProgramSaver.save_persistables(
                local_path, self.layer_handler)

        resp = self.stub.save_persistables(
                common_pb2.SaveInfo(
                    path=remote_path,
                    token=self.token,
                    save_token=token))
        
        if not resp.state.succ:
            _LOGGER.error(
                    "Failed to save vars in host side: {}".format(
                        resp.state.error_message))
            return False
        return True

    def save_inference_model(
            self,
            local_path: str,
            remote_path: str) -> bool:
        token = CustomerProgramSaver.save_inference_model(
                local_path, remote_path, self.layer_handler,
                self.tensor_names_from_host, self.input_specs)

        resp = self.stub.save_inference_model(
                common_pb2.SaveInfo(
                    token=self.token,
                    save_token=token,
                    path=remote_path))

        if not resp.state.succ:
            _LOGGER.error(
                    "Failed to save inference model in host side: {}".format(
                        resp.state.error_message))
            return False
        return True


class CustomerProgramSaver(object):

    def __init__(self):
        pass

    @staticmethod
    def save_persistables(
            dirpath: str, 
            layer_handler: CustomerLayerHandler) -> str:
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
            local_path: str, 
            remote_path: str, 
            layer_handler: CustomerLayerHandler,
            tensor_names_from_host: List[str], 
            input_specs: List[paddle.static.InputSpec]) -> str:
        paddle.jit.save(layer_handler.layer, local_path, input_specs)

        token = str(time.time())
        model_info = {
            "tensor_names_from_host": tensor_names_from_host,
            "token": token,
        }

        with open("{}.{}".format(local_path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
        return token
