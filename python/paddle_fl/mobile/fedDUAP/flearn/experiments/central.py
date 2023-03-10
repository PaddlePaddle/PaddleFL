"""
central
"""
import paddle
import numpy as np
import random
from flearn.utils.update import LocalUpdate
from flearn.utils.model_utils import average_weights, test_inference, is_conv, ratio_combine, ratio_minus
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.models import cnn, vgg, resnet, lenet
from data.util import get_client_server_noniid_degree, get_global_distribution, get_target_users_distribution, \
    get_noniid_degree
import copy
import os
import time
import math


class CentralTraining(object):
    """
    对于聚合后的模型，进行中心化的训练，share_percent 是共享数据集的大小
    """

    def __init__(self, args, share_percent=0, iid=True, unequal=False,
                 prune_interval=-1, prune_rate=0.6, auto_rate=False, result_dir="central",
                 auto_mu=False, server_mu=0.0, client_mu=0.0):

        self.device = "cuda" if "gpu" in paddle.device.get_device() else "cpu"
        self.args = args

        # 设置随机种子
        self.reset_seed()

        # 数据集划分信息
        self.share_percent = share_percent  # 共享数据占总客户端数据的百分比
        self.num_data = 40000  # 数据集中前 40000 用于分配给客户端进行训练
        self.num_share = int(self.num_data * share_percent / 100)
        self.l = 2  # noniid的程度， l越小 noniid 程度越大， 当 l = 1 时， 就是将数据按照顺序分成 clients 份，
        # 每个设备得到一份， 基本上只包含一个数字

        # 定义FedAvg的一些参数
        self.m = 10  # 每次取m个客户端进行平均
        self.iid = iid
        self.unequal = unequal
        self.decay = self.args.decay
        self.server_mu = server_mu
        self.client_mu = client_mu
        self.auto_mu = auto_mu
        self.server_min = 0
        self.server_max = 1e5

        # 剪枝的参数
        self.channels_list = []
        self.conv_layer_idx = []
        self.current_prune_idx = 0
        self.prune_interval = prune_interval
        self.prune_round = prune_interval
        self.pre_prune_round = 0
        self.recovery_round = None
        self.recovery_interval = 30 if prune_interval > 5 else 2
        self.valid_loss_list = []
        self.valid_acc_list = []
        self.prune_rate = prune_rate
        self.auto_rate = auto_rate
        self.init_compress_rate = [prune_rate] * 3
        self.compress_rate = [prune_rate] * 3
        self.pre_acc = 1
        self.num_weights = []
        self.log_weights = []

        self.result_dir = result_dir

        self.lr_decay = 0.99
        self.init_lr = self.args.lr

        self.v = {}
        self.momentum = 0

    def reset_seed(self):
        # 设置随机种子
        paddle.seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

    def init_data(self):
        if self.args.dataset == "cifar10":
            from data.cifar10.cifar10_data import get_dataset
            self.num_data = 40000
            self.num_share = int(self.num_data * self.share_percent / 100)
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            num_share=self.num_share,
                            l=self.l, unequal=self.unequal, share_l=self.args.share_l)
        elif self.args.dataset == "cifar100":
            from data.cifar100.cifar100_data import get_dataset
            self.num_data = 40000
            self.num_share = int(self.num_data * self.share_percent / 100)
            self.l = self.l * 10
            self.args.share_l = self.args.share_l * 10
            train_dataset, test_dataset, user_groups = \
                get_dataset(num_data=self.num_data, num_users=self.args.num_users, iid=self.iid,
                            num_share=self.num_share,
                            l=self.l, unequal=self.unequal, share_l=self.args.share_l)
        else:
            exit('Error: unrecognized dataset')

        self.num_share = len(user_groups[self.args.num_users])
        self.share_percent = math.ceil(self.num_share / self.num_data * 100)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.user_groups = user_groups
        self.global_distribution = get_global_distribution(train_dataset, user_groups)
        self.server_distribution = get_target_users_distribution(train_dataset, user_groups, [self.args.num_users])
        self.server_noniid_degree = get_noniid_degree(self.server_distribution, self.global_distribution)
        return train_dataset, test_dataset, user_groups

    def load_model(self):
        # BUILD MODEL
        # Convolutional neural network
        if self.args.dataset == "cifar10":
            in_channel = 3
            num_classes = 10
        elif self.args.dataset == "mnist" or self.args.dataset == "fashionmnist":
            self.init_lr = 0.01
            self.args.lr = 0.01
            in_channel = 1
            num_classes = 10
        elif self.args.dataset == "cifar100":
            in_channel = 3
            num_classes = 100
        else:
            exit('Error: unrecognized dataset')
        if self.args.model == "vgg":
            global_model = vgg.VGG11(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "resnet":
            global_model = resnet.resnet18(in_channel=in_channel, num_classes=num_classes)
        elif self.args.model == "lenet":
            global_model = lenet.LENET(in_channel=in_channel, num_classes=num_classes)
        else:
            global_model = cnn.CNN(in_channel=in_channel, num_classes=num_classes)

        self.load_model_info(global_model)
        self.num_weights.append(global_model.calc_num_all_active_params())
        self.channels_list.append(global_model.get_channels())
        self.log_weights.append([0, self.num_weights[-1], self.channels_list[-1]])

        # Set the model to train and send it to device.
        global_model.train()
        return global_model

    def load_model_info(self, model):
        """
        加载模型信息，确定可剪枝层的索引，并按照参数量进行降序排序
        :param model:
        :return:
        """
        num_trainables = []
        for i, layer in enumerate(model.prunable_layers):
            if is_conv(layer):
                self.conv_layer_idx.append(i)
                num_trainables.append(layer.num_weight)
        if self.auto_rate:
            self.init_compress_rate = [0] * len(self.conv_layer_idx)
            self.compress_rate = [0] * len(self.conv_layer_idx)
        else:
            self.init_compress_rate = [self.prune_rate] * len(self.conv_layer_idx)
            self.compress_rate = [self.prune_rate] * len(self.conv_layer_idx)
        self.sorted_conv_layers = self.conv_layer_idx
        # self.sorted_conv_layers = np.argsort(num_trainables)[::-1]
        self.num_trainables = num_trainables
        # print(f"prunable layer idx: {self.conv_layer_idx}")
        # print(f"sorted_layers: {self.sorted_conv_layers} according: {num_trainables}")

    def record_base_message(self, log_path):
        """
        record base message
        """
        record_log(self.args, log_path, "=== " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " === \n")
        # record_log(self.args, log_path, f"=== model: {self.args.model} ===\n")
        # client_noniid_degree, share_noniid_degree = get_client_server_noniid_degree(self.train_dataset,
        #                                                                             self.user_groups)
        # record_log(self.args, log_path,
        #            f"=== noniid_degree client/share/share_l: {client_noniid_degree} / {share_noniid_degree} / "
        #            f"{self.args.share_l} ===\n")
        # record_log(self.args, log_path,
        #            f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===\n")
        # record_log(self.args, log_path,
        #            f"=== share_percent: {self.share_percent} " + ("iid" if self.iid else "noniid") + " ===\n")
        # record_log(self.args, log_path,
        #            f"=== central_training: " + ("yes" if self.args.central_train == 1 else "no") + " ===\n")
        # record_log(self.args, log_path, f"=== server_mu/client_mu: {self.server_mu}/{self.args.client_mu} ===\n")
        # record_log(self.args, log_path,
        #            f"=== server_max/server_min/decay: {self.server_max}/{self.server_min}/{self.decay} ===\n")
        # record_log(self.args, log_path,
        #            f"=== prune_interval/recovery_interval: {self.prune_interval}/{self.recovery_interval} ===\n")
        # record_log(self.args, log_path,
        #            f"=== prune_ratio: {self.prune_rate if not self.auto_rate else 'auto rate'} ===\n")
        # record_log(self.args, log_path, f"=== prunable_layer_idx: {self.conv_layer_idx} ===\n")
        # record_log(self.args, log_path,
        #            f"=== sorted_layers: {self.sorted_conv_layers} according: {self.num_trainables} ===\n")

    def print_info(self, user_groups=None):
        """
        print info
        """
        if user_groups is None:
            user_groups = [[]]
        # print(f"data name: {self.args.dataset}")
        # print(f"=== model: {self.args.model} ===\n")
        # print(f"shared data nums: {self.num_share}")
        # print(f"user nums: {self.args.num_users}")
        # print(f"{'iid' if self.iid else 'noniid'} user sample nums: {len(user_groups[0])}")
        # print(f"=== local_bs/local_ep/epochs: {self.args.local_bs}/{self.args.local_ep}/{self.args.epochs} ===")
        # print(f"=== server_mu/client_mu: {self.server_mu}/{self.client_mu} ===")
        # print(f"=== server_max/server_min/decay: {self.server_max}/{self.server_min}/{self.decay} ===")
        # print(f"=== prune_interval/recovery_interval: {self.prune_interval}/{self.recovery_interval} ===")
        # print(f"=== prune_ratio: {self.prune_rate if not self.auto_rate else 'auto rate'} ===\n")
        # print(f"=== using device {self.device} optim {self.args.optim} ===")

    def get_loss(self, all_trian_data, train_dataset, global_model):
        """
        获取所有训练数据的 loss
        :param user_groups:
        :param train_dataset:
        :param global_model:
        :return:
        """
        # losses = []
        # for idx in range(len(user_groups)):
        #     if user_groups[idx].shape[0] == 0:
        #         continue
        #     local_model = LocalUpdate(args=self.args, dataset=train_dataset,
        #                               idxs=user_groups[idx], device=self.device)
        #     acc, loss = local_model.inference(global_model)
        #     losses.append(loss)
        # loss = sum(losses) / len(losses)
        local_model = LocalUpdate(args=self.args, local_bs=128, dataset=train_dataset,
                                  idxs=all_trian_data, device=self.device)
        acc, loss = local_model.inference(global_model)
        return round(loss, 4)

    def get_mu(self, avg_iter, mu=0.0):
        """
        获取 mu 值
        :return:
        """
        alpha = pow(self.m, 1 / 2) / (pow(avg_iter, 1 / 2) * pow(self.args.epochs, 1 / 6))
        if self.auto_mu:
            mu = alpha / self.args.lr
        return mu

    def client_train(self, idxs_users, global_model, user_groups, epoch, train_dataset, train_losses, local_weights,
                     local_losses, local_P=[], local_v=[]):
        """
        进行客户端训练
        :param idxs_users:
        :param global_model:
        :param user_groups:
        :param epoch:
        :param train_dataset:
        :param train_losses:
        :param local_weights:
        :param local_losses:
        :return:
        """
        num_current = 0
        for idx in idxs_users:
            num_current += len(user_groups[idx])
        start = time.time()
        self.args.lr = self.init_lr * pow(self.lr_decay, epoch)
        # 计算 mu 值
        avg_iter = (num_current * self.args.local_ep) / (self.m * self.args.local_bs)
        client_mu = self.get_mu(avg_iter, self.client_mu)
        # print(f"client mu {client_mu}")
        for idx in idxs_users:
            local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                      idxs=user_groups[idx], device=self.device)
            if self.args.optim == "fsgdm":
                P, v, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, mu=self.args.client_mu,
                    v=copy.deepcopy(self.v))
                if loss < train_losses[0] * 3:
                    local_P.append([len(user_groups[idx]), copy.deepcopy(P)])
                    local_v.append([len(user_groups[idx]), copy.deepcopy(v)])
                    local_losses.append(copy.deepcopy(loss))
            else:
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, mu=self.args.client_mu)
                if loss < train_losses[0] * 3:
                    local_weights.append([len(user_groups[idx]), copy.deepcopy(w)])
                    local_losses.append(copy.deepcopy(loss))
        #     print("{}:{:.4f}".format(idx, loss), end=" ")
        # print("本轮设备总用时：{:.4f}".format(time.time() - start))
        print()

        return num_current, avg_iter

    def server_train(self, epoch, num_current, train_dataset, user_groups, global_model, avg_iter, idxs_users):
        """
        进行服务端训练，首先计算迭代次数，然后 mu
        :param epoch:
        :param num_current:
        :param train_dataset:
        :param user_groups:
        :param global_model:
        :param avg_iter:
        :return:
        """
        # 计算迭代次数
        share_percent = self.num_share / (num_current + self.num_share)
        # print(f"round{epoch} training_data_num/data_percent: {num_current} / {share_percent}")
        local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                  idxs=user_groups[self.args.num_users], device=self.device)
        acc, pre_loss = local_model.inference(model=global_model)
        clients_distribution = get_target_users_distribution(train_dataset, user_groups, idxs_users)
        clients_noniid_degree = get_noniid_degree(clients_distribution, self.global_distribution)
        # print(f"round{epoch} noniid degree client/server: {clients_noniid_degree} / {self.server_noniid_degree}")
        alpha = (1 - acc) * (self.num_share * clients_noniid_degree) / \
                (self.num_share * clients_noniid_degree + num_current * self.server_noniid_degree)
        alpha = alpha * pow(self.decay, epoch)
        server_iter = max(self.server_min, int(alpha * avg_iter))
        # alpha = min(1, alpha)
        alpha = alpha * 1

        # 获取 mu
        if alpha > 0.001:
            server_mu = self.get_mu((avg_iter + server_iter) / 2, self.server_mu)
            # print(f"server_mu {server_mu}")

            if self.args.optim == "fsgdm":
                P, v, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, mu=self.args.client_mu,
                    v=copy.deepcopy(self.v))
                actual_iter = math.ceil(self.num_share / self.args.local_bs) * self.args.local_ep
                server_iter = min(actual_iter, server_iter)
                w = ratio_minus(global_model.state_dict(), P, self.args.lr * alpha)
                # print(f"round{epoch} server fixing with iters/actual_iters/alpha "
                #       f"{server_iter} / {actual_iter} / {alpha}")
            else:
                w, loss = local_model.update_weights(
                    model=copy.deepcopy(global_model), global_round=epoch, mu=server_mu)
                actual_iter = math.ceil(self.num_share / self.args.local_bs) * self.args.local_ep
                server_iter = min(actual_iter, server_iter)
                w = ratio_combine(global_model.state_dict(), w, alpha)
                # print(f"round{epoch} server fixing with iters/actual_iters/alpha "
                #       f"{server_iter} / {actual_iter} / {alpha}")

            origin_w = copy.deepcopy(global_model.state_dict())
            if self.args.global_momentum == 1:
                delta = ratio_minus(w, origin_w, 1)
                self.momentum = ratio_minus(ratio_minus(self.momentum, self.momentum, 0.1),
                                            ratio_minus(delta, delta, 0.9),
                                            -1)
                # 更新权重 w + 1 * m
                w = ratio_minus(origin_w, self.momentum, -1)

            global_model.load_dict(w)
            acc, loss = local_model.inference(model=global_model)
            print("server final loss/acc: {:.4f} / {:.4f}".format(loss, acc))
            if loss < pre_loss * 3:
                # 因为服务器是部分更新的，我们也取它一部分的动量作为累加， (v' - v) * alpha + v
                if self.args.optim == "fsgdm":
                    self.v = ratio_minus(self.v,
                                         ratio_minus(v, self.v, 1),
                                         -alpha)
                pass
            else:
                global_model.load_dict(origin_w)

    def check_pruning_round(self, cur_round):
        """
        check pruning
        """
        return self.pre_prune_round + self.prune_interval == cur_round and \
               self.current_prune_idx < len(self.conv_layer_idx)

    def check_recovery_round(self, cur_round):
        """
        check recovery round
        """
        return self.recovery_round is not None and \
               cur_round == self.recovery_round and self.current_prune_idx < len(self.conv_layer_idx)

    def train(self):
        """
        train
        """
        # 记录日志和结果
        log_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log",
                                str(self.share_percent) + ".txt")
        result_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid")

        # 加载模型
        global_model = self.load_model()
        print(global_model)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        self.print_info(user_groups)
        self.record_base_message(log_path)

        # copy weights
        global_weights = global_model.state_dict()

        # Training
        train_losses = []
        test_accs = []

        all_train_data = np.array([])
        for k, v in user_groups.items():
            all_train_data = np.concatenate(
                (all_train_data, v), axis=0)

        self.reset_seed()

        # 第一次评估
        # loss = self.get_loss(all_train_data, train_dataset, global_model)
        # Test inference after completion of training
        test_acc, test_loss = test_inference(global_model, test_dataset)
        test_accs.append(test_acc)
        train_losses.append(test_loss)
        print("-train loss:{:.4f} -test acc:{:.4f}".format(test_loss, test_acc))

        for epoch in range(self.args.epochs):
            start = time.time()
            local_weights, local_losses = [], []
            local_P, local_v = [], []
            # print(f'\n | Global Training Round : {epoch} |\n')


            # 进行验证操作
            if epoch == self.recovery_round:
                print("recovery...")
                global_model.recovery_model()
                self.num_weights.append(global_model.calc_num_all_active_params())
                self.channels_list.append(global_model.get_channels())
                self.log_weights.append([epoch, self.num_weights[-1], self.channels_list[-1]])
                self.prune_round = epoch + self.prune_interval
                print("finish recovery...")

            # 选择设备，并进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)
            num_current, avg_iter = self.client_train(idxs_users, global_model, user_groups, epoch,
                                                      train_dataset, train_losses, local_weights, local_losses,
                                                      local_P, local_v)

            # 无效轮
            if len(local_weights) == 0 and len(local_P) == 0:
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                continue

            # update global weights
            if self.args.optim == "fsgdm":
                self.v = average_weights(local_v)
                P = average_weights(local_P)
                global_weights = ratio_minus(global_model.state_dict(), P, self.args.lr)

            else:
                global_weights = average_weights(local_weights)

            if self.args.global_momentum == 1:
                # 获取差值 w' - w
                delta = ratio_minus(global_weights, global_model.state_dict(), 1)
                # 更新动量， m = 0.9 * m + 0.1 * delta
                if self.momentum == 0:
                    self.momentum = delta
                else:
                    self.momentum = ratio_minus(ratio_minus(self.momentum, self.momentum, 0.1),
                                                ratio_minus(delta, delta, 0.9),
                                                -1)
                # 更新权重 w + 1 * m
                global_weights = ratio_minus(global_model.state_dict(), self.momentum, -1)

            # update global weights
            pre_model = copy.deepcopy(global_model.state_dict())
            global_model.load_dict(global_weights)

            # 进行中心训练
            if self.share_percent > 0 and self.args.central_train == 1:
                self.server_train(epoch, num_current, train_dataset, user_groups, global_model, avg_iter, idxs_users)

            # loss = self.get_loss(all_train_data, train_dataset, global_model)
            # Test inference after completion of training
            test_acc, test_loss = test_inference(global_model, test_dataset)
            if test_loss < train_losses[0] * 100:
                test_accs.append(test_acc)
                train_losses.append(test_loss)
            else:
                print("recover model test_loss/test_acc : {}/{}".format(test_loss, test_acc))
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                global_model.load_dict(pre_model)

            if (epoch + 11) % 10 == 0 or epoch == self.args.epochs - 1:
                save_result(self.args, os.path.join(result_path, str(self.share_percent) + "_train_loss.txt"),
                            str(train_losses)[1:-1])
                save_result(self.args, os.path.join(result_path, str(self.share_percent) + "_test_accuracy.txt"),
                            str(test_accs)[1:-1])

            print("epoch{:4d} - loss: {:.4f} - accuracy: {:.4f} - lr: {:.4f} - time: {:.2f}".
                  format(epoch, test_loss, test_acc, self.args.lr, time.time() - start))
            print()

        print("prune result:" + str(self.num_weights) + " " + str(self.channels_list))
        # record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        # weight_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log",
        #                            f"weights_{self.share_percent}.txt")
        # record_log(self.args, weight_path, str(self.log_weights)[1:-1])


if __name__ == "__main__":
    args = args_parser()
    t = CentralTraining(args, share_percent=1, iid=False, unequal=False, prune_interval=-1, prune_rate=0.6,
                        auto_rate=True, auto_mu=False, server_mu=0, client_mu=0)
    t.train()
