# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
"""
This module provides io modules, which contains io operators related to model
in paddle_mpc.
"""

import paddle.fluid as fluid


def _assign_default_model_name_for_none(model_name, protocol):
    """
    Assign default model file name according to protocol.
    :param model_name:
    :param protocol:
    :return:
    """
    default_model_name = 'mpc_model'
    if model_name is None:
        # the name of model file is "mpc_model.aby3"
        model_name = default_model_name + '.' + protocol
    return model_name


def save_mpc_inference_model(protocol,
                             dirname,
                             feed_var_name,
                             target_var,
                             executor,
                             model_filename=None):
    """
    Save the trained model in to a file.
    :param protocol: the name of mpc protocol used for this model.
    :param dirname: the path of file that save the model.
    :param model_filename: the file name of inference model. Default
    value is None, which means the file is saved as "mpc_model.protocol".
    :param feed_var_name: the name of variables that are used when make inference
    with the trained model.
    :param target_var: the output var of the trained model.
    :param executor: the executor.
    :return:
    """
    model_name = _assign_default_model_name_for_none(model_filename, protocol)
    fluid.io.save_inference_model(
        dirname=dirname,
        feeded_var_names=feed_var_name,
        target_vars=target_var,
        executor=executor,
        model_filename=model_name)


def load_mpc_inference_model(protocol, dirname, executor, model_filename=None):
    """
    Load a trained model from model_path by using
    fluid.io.load_inference_model in Paddle.
    :param protocol: the mpc protocol used for this model
    :param dirname:
    :param executor: the executor
    :param model_filename:
    :return:
    """
    model_name = _assign_default_model_name_for_none(model_filename, protocol)
    [mpc_infer_program, mpc_feed_var_names,
     mpc_fetch_var] = fluid.io.load_inference_model(
         dirname=dirname, executor=executor, model_filename=model_name)
    return mpc_infer_program, mpc_feed_var_names, mpc_fetch_var


def save(program, model_path):
    """
    Refer to fluid.io.save in Paddle 1.6.
    This function save parameters, optimizer information
    and network description to model_path. And this can be
    used in scenarios like model updating.
    :param program: the program to saved.
    :param model_path: the file prefix to save the program.
    :return:
    """
    # Use paddle method regardless what the mpc protocol is.
    # If load the saved model with unmatched protocol,
    # throw TypeError.
    fluid.io.save(program=program, model_path=model_path)


def load(program, model_path):
    """
    Refer to fluid.io.load in Paddle 1.6.
    This function This function get parameters and optimizer
    information from program, and then get corresponding value
    from file. And this can be used in scenarios like model updating.
    :param program: the program to be loaded.
    :param model_path: the file prefix store the program
    :return:
    """
    fluid.io.load(program=program, model_path=model_path)
