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
from .trainer_base import TrainerBase
from model import LanguageModel
from clients import DataClient
import paddle.fluid as fluid
from utils.hdfs_utils import multi_upload, HDFSClient
import reader.leaf_reddit_reader as reader
from utils.logger import logging
from itertools import groupby
import numpy as np
import random
import paddle
import pickle
import os
from model.model_base import set_user_param_dict
from model.model_base import set_global_param_dict


def train_one_user(arg_dict, trainer_config):
    show_metric = trainer_config["show_metric"]
    shuffle = trainer_config["shuffle"]
    max_training_steps = trainer_config["max_training_steps"]
    batch_size = trainer_config["batch_size"]
    # logging.info("training one user...")
    main_program = fluid.Program.parse_from_string(
        trainer_config["main_program_desc"])
    startup_program = fluid.Program.parse_from_string(
        trainer_config["startup_program_desc"])
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.global_scope()
    if (startup_program is None):
        logging.error("startup_program is None")
        exit()
    exe.run(startup_program)

    feeder = fluid.DataFeeder(feed_list=trainer_config["input_names"],
                              place=place,
                              program=main_program)
    data_server_endpoints = arg_dict["data_endpoints"]
    # create data clients
    data_client = DataClient()
    data_client.set_data_server_endpoints(data_server_endpoints)
    uid = arg_dict["uid"]
    date = arg_dict["date"]
    global_param_dict = arg_dict["global_params"]
    user_data = data_client.get_data_by_uid(uid, date)
    train_reader = reader.train_reader(user_data)
    if shuffle == True:
        train_reader = paddle.reader.shuffle(train_reader, buf_size=10000)
    train_reader = paddle.batch(train_reader, batch_size=batch_size)

    # get user param
    # logging.debug("do not need to get user params")

    set_global_param_dict(arg_dict["global_param_names"],
                          arg_dict["global_params"], scope)

    if (main_program is None):
        logging.error("main_program is None")
        exit()

    epoch = trainer_config["epoch"]
    max_steps_in_epoch = trainer_config.get("max_steps_in_epoch", -1)
    metrics = trainer_config["metrics"]
    fetch_list = []
    for var in trainer_config["target_names"]:
        fetch_list.append(var)

    for ei in range(epoch):
        fetch_res_list = []
        trained_sample_num = 0
        step = 0
        num_layers = trainer_config["num_layers"]
        hidden_size = trainer_config["n_hidden"]
        tot_loss, tot_correct = 0, 0
        tot_samples = 0
        init_hidden, init_cell = generate_init_data(batch_size, num_layers,
                                                    hidden_size)
        for data in train_reader():
            feed_data, input_lengths = prepare_input(batch_size, data,
                                                     init_hidden, init_cell)
            fetch_res = exe.run(main_program,
                                feed=feeder.feed(feed_data),
                                fetch_list=fetch_list)
            loss, last_hidden, last_cell, correct = fetch_res

            init_hidden = np.array(last_hidden)
            init_cell = np.array(last_cell)
            tot_loss += np.array(loss)
            tot_correct += np.array(correct)
            tot_samples += np.sum(input_lengths)
            step += 1
            trained_sample_num += len(data)
            fetch_res_list.append([np.array(loss), np.array(correct)])
            if max_steps_in_epoch != -1 and step >= max_steps_in_epoch:
                break

        if show_metric and trained_sample_num > 0:
            loss = tot_loss / step
            acc = float(tot_correct) / tot_samples
            print("loss: {}, acc: {}".format(loss, acc))

    local_updated_param_dict = {}
    # update user param
    # logging.debug("do not need to update user params")

    data_client.set_param_by_uid(uid, local_updated_param_dict)
    # global_updated_param_dict = {}
    write_global_param_file = arg_dict["write_global_param_file"]
    #os.makedirs("%s/params" % write_global_param_file)
    for var_name in arg_dict["global_param_names"]:
        var = scope.var(var_name).get_tensor().__array__().astype(np.float32)
        filename = os.path.join(write_global_param_file, "params", var_name)
        #logging.info("filename: {}".format(filename))
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filename, "w") as f:
            np.save(f, var)
    with open("%s/_info" % write_global_param_file, "w") as f:
        pickle.dump([uid, trained_sample_num], file=f)


def infer_one_user(arg_dict, trainer_config):
    """
    infer a model with global_param and user params
    input:
        global_param
        user_params
        infer_program
        user_data
    output:
        [sample_cout, top1] 
    """
    # run startup program, set params
    uid = arg_dict["uid"]
    batch_size = trainer_config["batch_size"]
    startup_program = fluid.Program.parse_from_string(
        trainer_config["startup_program_desc"])
    infer_program = fluid.Program.parse_from_string(
        trainer_config["infer_program_desc"])
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    scope = fluid.global_scope()

    if (startup_program is None):
        logging.error("startup_program is None")
        exit()
    if (infer_program is None):
        logging.error("infer_program is None")
        exit()

    exe.run(startup_program)

    data_client = DataClient()
    data_client.set_data_server_endpoints(arg_dict["data_endpoints"])

    # get user param
    # logging.debug("do not need to get user params")

    set_global_param_dict(arg_dict["global_param_names"],
                          arg_dict["global_params"], scope)

    # reader
    date = arg_dict["date"]
    global_param_dict = arg_dict["global_params"]
    user_data = data_client.get_data_by_uid(uid, date)
    infer_reader = reader.infer_reader(user_data)
    infer_reader = paddle.batch(infer_reader, batch_size=batch_size)

    # run infer program
    os.mkdir(arg_dict["infer_result_dir"])
    #pred_file = open(arg_dict["infer_result_dir"] + '/' + "pred_file", "w")
    feeder = fluid.DataFeeder(feed_list=trainer_config["input_names"],
                              place=place,
                              program=infer_program)

    fetch_list = trainer_config["target_names"]
    #logging.info("fetch_list: {}".format(fetch_list))
    fetch_res = []
    sample_count = 0

    num_layers = trainer_config["num_layers"]
    hidden_size = trainer_config["n_hidden"]
    tot_correct, tot_loss = 0, 0
    tot_samples, tot_batches = 0, 0
    init_hidden, init_cell = generate_init_data(batch_size, num_layers,
                                                hidden_size)
    for data in infer_reader():
        feed_data, input_lengths = prepare_input(batch_size, data, init_hidden,
                                                 init_cell)
        fetch_res = exe.run(infer_program,
                            feed=feeder.feed(feed_data),
                            fetch_list=fetch_list)
        loss, last_hidden, last_cell, correct = fetch_res

        cost_eval = np.array(loss)
        init_hidden = np.array(last_hidden)
        init_cell = np.array(last_cell)
        correct_val = np.array(correct)
        tot_loss += cost_eval
        tot_correct += correct_val
        tot_samples += np.sum(input_lengths)
        tot_batches += 1

    loss = tot_loss / tot_batches
    acc = float(tot_correct) / tot_samples
    logging.info("infer acc: {}".format(acc))
    with open(arg_dict["infer_result_dir"] + "/res", "w") as f:
        f.write("%d\t%f\n" % (1, acc))


def prepare_input(batch_size, data, init_hidden, init_cell):
    init_hidden = np.split(init_hidden, batch_size)
    init_cell = np.split(init_cell, batch_size)
    data = [[features] + [labels] + [seq_len_ph] + [seq_mask_ph] + [init_hidden[i]] + [init_cell[i]  ] \
              for i, (features, labels, seq_len_ph, seq_mask_ph) in enumerate(data)]
    input_lengths = [x[2] for x in data]
    return data, input_lengths


def generate_init_data(batch_size, num_layers, hidden_size):
    init_hidden = np.zeros((batch_size, num_layers, hidden_size),
                           dtype='float32')
    init_cell = np.zeros((batch_size, num_layers, hidden_size),
                         dtype='float32')
    return init_hidden, init_cell


def save_and_upload(arg_dict, trainer_config, dfs_upload_path):
    logging.info("do not save and upload...")
    return


def evaluate_a_group(group):
    group_list = []
    for label, pred, _ in group:
        group_list.append((int(label), float(pred)))
    random.shuffle(group_list)
    labels = [x[0] for x in group_list]
    preds = [x[1] for x in group_list]
    true_res = labels.index(1) if 1 in labels else -1
    pred_res = preds.index(max(preds))
    if pred_res == true_res:
        return 1
    else:
        return 0


class LanguageModelTrainer(TrainerBase):
    """
    LanguageModelTrainer only support training with PaddlePaddle
    """
    def __init__(self):
        super(LanguageModelTrainer, self).__init__()
        self.main_program_ = fluid.Program()
        self.startup_program_ = fluid.Program()
        self.infer_program_ = fluid.Program()
        self.main_program_desc_ = ""
        self.startup_program_desc_ = ""
        self.infer_program_desc_ = ""
        self.train_one_user_func = train_one_user
        self.infer_one_user_func = infer_one_user
        self.save_and_upload_func = save_and_upload
        self.input_model_ = None

    def get_load_data_into_patch_func(self):
        return reader.load_data_into_patch

    def prepare(self, do_test=False):
        self.generate_program_desc(do_test)
        pass

    def get_user_param_names(self):
        # return [x[0] for x in self.input_model_.get_user_param_names()]
        pass

    def get_global_param_names(self):
        return [x[0] for x in self.input_model_.get_global_param_names()]

    def generate_program_desc(self, do_test=False):
        """
        generate the paddle program desc
        """
        with fluid.program_guard(self.main_program_, self.startup_program_):
            self.input_model_ = LanguageModel()
            model_configs = self.trainer_config
            self.input_model_.build_model(model_configs)

            optimizer = fluid.optimizer.SGD(
                learning_rate=self.trainer_config["lr"],
                grad_clip=fluid.clip.GradientClipByGlobalNorm(
                    clip_norm=self.trainer_config["max_grad_norm"]))
            optimizer.minimize(self.input_model_.get_model_loss())

        self.main_program_desc_ = self.main_program_.desc.serialize_to_string()
        self.startup_program_desc_ = self.startup_program_.desc.serialize_to_string(
        )
        self.update_trainer_configs("loss_name",
                                    self.input_model_.get_model_loss_name())
        self.update_trainer_configs(
            "input_names",
            self.input_model_.get_model_input_names(),
        )
        self.update_trainer_configs(
            "target_names",
            self.input_model_.get_target_names(),
        )
        self.update_trainer_configs(
            "metrics",
            self.input_model_.get_model_metrics(),
        )
        self.update_trainer_configs("show_metric", True)
        self.update_trainer_configs("max_training_steps", "inf")
        self.update_trainer_configs("shuffle", False)
        self.update_trainer_configs("main_program_desc",
                                    self.main_program_desc_)
        self.update_trainer_configs("startup_program_desc",
                                    self.startup_program_desc_)

        if do_test:
            input_names = self.input_model_.get_model_input_names()
            target_var_names = self.input_model_.get_target_names()
            self.infer_program_ = self.main_program_._prune_with_input(
                feeded_var_names=input_names, targets=target_var_names)
            self.infer_program_ = self.infer_program_._inference_optimize(
                prune_read_op=True)
            fluid.io.prepend_feed_ops(self.infer_program_, input_names)
            fluid.io.append_fetch_ops(self.infer_program_, target_var_names)
            self.infer_program_.desc._set_version()
            fluid.core.save_op_compatible_info(self.infer_program_.desc)
            self.infer_program_desc_ = self.infer_program_.desc.serialize_to_string(
            )
            self.update_trainer_configs("infer_program_desc",
                                        self.infer_program_desc_)

    def init_global_model(self, scheduler_client):
        logging.info("initializing global model")
        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(self.startup_program_)
        logging.info("finish initializing global model")

        global_param_dict = self.input_model_.get_global_param_dict()
        scheduler_client.update_global_params(global_param_dict)
