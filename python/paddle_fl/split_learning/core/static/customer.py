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
import time
import json
import grpc
import logging
import numpy as np
import paddle.fluid as fluid
from .. import util
from . import reformer
from ..proto import common_pb2_grpc, common_pb2

_LOGGER = logging.getLogger(__name__)


class CustomerExecutor(object):
    def __init__(self, endpoints, place):
        self.exe = fluid.Executor(place)
        self._connect(endpoints)
        self.run_type = None  # TRAIN or INFER

    def load_program_from_full_network(
            self, startup_program, main_program):
        """
        load program from full network for train
        """
        self.run_type = "TRAIN"
        splited_programs = reformer.Reformer.split_program_by_name(main_program)
        self.main_program = splited_programs[1]
        self.common_vars = {
            "in": list(
                util.intersection_vars(
                    splited_programs[0], splited_programs[1])),
            "out": list(
                util.intersection_vars(
                    splited_programs[2], splited_programs[1]))}
        self.vars_need_saved = []
        for var in self.main_program.list_vars():
            if var.persistable == True:
                self.vars_need_saved.append(var.name)
        self.token = "init_from_full_network"
        self.exe.run(startup_program)

    def load_program_from_splited_file(self, load_path):
        """
        load program from splited file for train
        """
        self.run_type = "TRAIN"
        startup_program, main_program, model_infos \
            = util.load_splited_program(load_path)
        self.main_program = main_program
        self.common_vars = model_infos["common"]
        self.vars_need_saved = model_infos["persistable_vars"]
        self.token = model_infos["token"]
        self.exe.run(startup_program)

    def load_inference_model(self, local_path):
        self.run_type = "INFER"
        inference_program, feed_target_names, fetch_targets = \
            fluid.io.load_inference_model(
                dirname=local_path,
                executor=self.exe)
        # load common var info
        with open(os.path.join(local_path, "model_info")) as f:
            model_info = json.load(f)
        self.common_vars = model_info["common"]
        self.token = model_info["token"]
        self.main_program = inference_program
        return feed_target_names, fetch_targets

    def load_persistables(self, path):
        vars = [util.find_var(self.main_program, name)
                for name in self.vars_need_saved]
        fluid.io.load_vars(
            executor=self.exe,
            dirname=path,
            main_program=self.main_program,
            vars=vars)
        # load token info
        with open(os.path.join(path, "model_info")) as f:
            model_info = json.load(f)
        self.token = model_info["token"]

    def _connect(self, endpoints):
        options = [('grpc.max_receive_message_length', 512 * 1024 * 1024),
                   ('grpc.max_send_message_length', 512 * 1024 * 1024)]
        g_endpoint = 'ipv4:{}'.format(','.join(endpoints))
        self.channel = grpc.insecure_channel(g_endpoint, options=options)
        self.stub = common_pb2_grpc.FLExecutorStub(self.channel)

    def _parse_vars_from_host(self, resp, required_common_vars):
        vars_map = util.parse_proto_to_tensor(resp)
        # check common in 
        for name in required_common_vars:
            if name not in vars_map:
                raise KeyError("Failed to calc: {} not found in query response.".format(name))
        return vars_map

    def _pack_vars_to_host(self, fetch_vars, required_common_vars):
        vars_map = {name: fetch_vars[idx] for idx, name in enumerate(required_common_vars)}
        req = util.pack_tensor_to_proto(vars_map)
        return req

    def _inner_cancel_current_step(self, err_msg):
        _LOGGER.error(err_msg, exc_info=True)

    def cancel_host_current_step(self, err_msg):
        self.stub.cancel_current_step(
            common_pb2.NilRequest(
                token=self.token,
                state=common_pb2.State(
                    succ=False,
                    error_message=err_msg)))

    def run(self, usr_key, feed, fetch_list):
        if self.run_type == "TRAIN":
            return self._run_for_train(
                usr_key, feed, fetch_list)
        elif self.run_type == "INFER":
            return self._run_for_infer(
                usr_key, feed, fetch_list)
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

    def _execute_middle_customer_part(self, feed, fetch_list):
        fetch_vars = self.exe.run(
            program=self.main_program,
            feed=feed,
            fetch_list=fetch_list,
            return_numpy=False)
        return fetch_vars

    def _run_for_train(self, usr_key, feed, fetch_list):
        try:
            resp = self._execute_forward_host_part(usr_key)
        except Exception as e:
            err_msg = "Failed to execute forward host part: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return None
        try:
            vars_from_host = self._parse_vars_from_host(resp, self.common_vars["in"])
        except Exception as e:
            err_msg = "Failed to parse vars from host: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        for name in self.common_vars["in"]:
            _LOGGER.debug("Get params {}: {}".format(
                name, np.array(vars_from_host[name])))

        try:
            feed = self._generate_feed_for_customer_part(feed, vars_from_host)
        except Exception as e:
            err_msg = "Failed to generate feed data: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        fetch_names = self.common_vars["out"] + fetch_list
        try:
            fetch_vars = self._execute_middle_customer_part(feed, fetch_names)
        except Exception as e:
            err_msg = "Failed to run middle program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        common_vars = fetch_vars[:len(self.common_vars["out"])]
        ret_vars = fetch_vars[len(self.common_vars["out"]):]

        try:
            req = self._pack_vars_to_host(common_vars, self.common_vars["out"])
            req.token = self.token
        except Exception as e:
            err_msg = "Failed to pack vars to host: {}".format(e)
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
        return ret_vars

    def _run_for_infer(self, usr_key, feed, fetch_list):
        try:
            resp = self._execute_forward_host_part(usr_key)
        except Exception as e:
            err_msg = "Failed to execute forward host part: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            return None

        try:
            vars_from_host = self._parse_vars_from_host(resp, self.common_vars["in"])
        except Exception as e:
            err_msg = "Failed to parse vars from host: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        for name in self.common_vars["in"]:
            _LOGGER.debug("Get params {}: {}".format(
                name, np.array(vars_from_host[name])))

        try:
            feed = self._generate_feed_for_customer_part(feed, vars_from_host)
        except Exception as e:
            err_msg = "Failed to generate feed data: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None

        try:
            fetch_vars = self._execute_middle_customer_part(feed, fetch_list)
        except Exception as e:
            err_msg = "Failed to run middle program: {}".format(e)
            self._inner_cancel_current_step(err_msg)
            self.cancel_host_current_step("[Customer] {}".format(err_msg))
            return None
        return fetch_vars

    def save_persistables(self, local_path, remote_path):
        token = CustomerProgramSaver.save_persistables(
            local_path, self.exe, self.main_program, self.vars_need_saved)

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
            remote_path,
            feeded_var_names,
            target_var_names):

        host_feeded_var_names, token = CustomerProgramSaver.save_inference_model(
            local_path, self.exe, self.main_program, self.common_vars,
            feeded_var_names, target_var_names)

        resp = self.stub.save_inference_model(
            common_pb2.SaveInfo(
                token=self.token,
                save_token=token,
                path=remote_path,
                feeded_var_names=host_feeded_var_names))

        if not resp.state.succ:
            err_msg = "Failed to save inference model in host side: {}".format(
                resp.state.error_message)
            raise RuntimeError(err_msg)
        return True


class CustomerProgramSaver(object):

    def __init__(self):
        pass

    @staticmethod
    def save_persistables(
            local_path, exe, main_program, vars_need_saved):
        vars = [util.find_var(main_program, name)
                for name in vars_need_saved]
        fluid.io.save_vars(
            executor=exe,
            dirname=local_path,
            main_program=main_program,
            vars=vars)

        # token
        token = str(time.time())
        model_info = {
            "token": token,
        }
        with open(os.path.join(local_path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
        return token

    @staticmethod
    def save_inference_model(
            local_path, exe, main_program, common_vars,
            feeded_var_names, target_var_names):

        customer_feeded_var_names = []
        host_feeded_var_names = []
        for name in feeded_var_names:
            if util.find_var(main_program, name):
                customer_feeded_var_names.append(name)
            else:
                host_feeded_var_names.append(name)
        customer_feeded_var_names += common_vars["in"]
        customer_target_vars = []
        for name in target_var_names:
            var = util.find_var(main_program, name)
            if var is None:
                raise ValueError("Failed to save inference model: "
                                 "target_var_names({}) not in customer side.".format(name))
            customer_target_vars.append(var)

        fluid.io.save_inference_model(
            dirname=local_path,
            feeded_var_names=customer_feeded_var_names,
            target_vars=customer_target_vars,
            executor=exe,
            main_program=main_program)

        # save inference info
        token = str(time.time())
        model_info = {
            "common": common_vars,
            "token": token,
        }
        with open(os.path.join(local_path, "model_info"), "w") as f:
            f.write(json.dumps(model_info))
        return host_feeded_var_names, token
