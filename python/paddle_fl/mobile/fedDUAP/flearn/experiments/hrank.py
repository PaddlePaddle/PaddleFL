import numpy as np
from flearn.utils.model_utils import average_weights, test_inference, is_conv
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.utils.prune import hrank_prune, get_ratio, get_rate_for_each_layers, lt_prune
from data.util import get_client_server_noniid_degree, get_global_distribution, get_target_users_distribution, get_noniid_degree
import copy
import os
import time
from flearn.experiments.central import CentralTraining


class HRank(CentralTraining):
    """
    按照特征图的秩，一次减去一定百分比的卷积层
    """
    def __init__(self, args, share_percent=0, iid=True, unequal=False,
                 prune_interval=-1, prune_rate=0.6, auto_rate=False, result_dir="hrank",
                 auto_mu=False, server_mu=0.0, client_mu=0.0):

        super(HRank, self).__init__(args, share_percent, iid, unequal, prune_interval, prune_rate, auto_rate,
                                    result_dir, auto_mu, server_mu, client_mu)

    def train(self):
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

        degrees = []
        for i in range(self.args.num_users + 1):
            distritubion = get_target_users_distribution(train_dataset, user_groups, [i])
            degree = get_noniid_degree(distritubion, self.global_distribution)
            degrees.append(degree)
        print(degrees)
        degrees = [1 / (i + 0.0001) for i in degrees]
        for i in range(self.args.num_users + 1):
            degrees[i] = degrees[i] * len(self.user_groups[i])
        rates = [i/sum(degrees) for i in degrees]
        print(rates)
        print(sum(rates))

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
            print(f'\n | Global Training Round : {epoch} |\n')

            # 进行剪枝操作
            if epoch == self.prune_interval:
                if self.args.dataset == "cifar10":
                    from data.cifar10.cifar10_data_pt import get_pt_dataset
                    train_dataset_pt = get_pt_dataset()
                elif self.args.dataset == "cifar100":
                    from data.cifar100.cifar100_data_pt import get_pt_dataset
                    train_dataset_pt = get_pt_dataset()
                pt_model = global_model.get_pt_model()
                global_prune_rate = 0
                if self.auto_rate:
                    ratios = []
                    start = time.time()
                    for i in range(self.args.num_users + 1):
                        ret = get_ratio(pt_model, train_dataset_pt, user_groups[i])
                        ratios.append(ret)
                    print("consumed time:")
                    print(time.time() - start)
                    # ratios.sort()
                    # global_prune_rate = ratios[int(self.args.num_users/2)]
                    for i in range(self.args.num_users + 1):
                        global_prune_rate += rates[i] * ratios[i]
                    print(ratios)

                    self.compress_rate = get_rate_for_each_layers(global_model, global_prune_rate)
                    print(f"=== 根据全局剪枝率 {global_prune_rate} 获得的每层卷积层的剪枝率为 {self.compress_rate} ===")
                    record_log(self.args, log_path,
                               f"=== 根据全局剪枝率 {global_prune_rate} 获得的每层卷积层的剪枝率为 {self.compress_rate} ===\n")

                for idx in self.sorted_conv_layers:
                    prune_rate = self.compress_rate[idx]
                    hrank_prune(global_model, train_dataset, user_groups[self.args.num_users],
                                prune_rate, layer_idx=idx, device=self.device)
                    record_log(self.args, log_path, f"=== Pruning === Current pruned layer {idx} using prune rate {prune_rate} ===\n")
                # lt_prune(global_model, global_prune_rate, device=self.device)
                self.num_weights.append(global_model.calc_num_all_active_params())
                self.channels_list.append(global_model.get_channels())
                self.log_weights.append([epoch, self.num_weights[-1], self.channels_list[-1]])
                print("\tTotal: {}/{} = {}".format(self.num_weights[-1], self.num_weights[0],
                                                         self.num_weights[-1] / self.num_weights[0]))
                record_log(self.args, log_path,
                           "=== Total: {}/{} = {} ===\n".format(self.num_weights[-1], self.num_weights[0],
                                                                self.num_weights[-1] / self.num_weights[0]))
                record_log(self.args, log_path,
                           f"=== epoch{epoch} pruning - current_layer: {self.current_prune_idx} ===\n" +
                           f"weights/channel: {self.num_weights}/{self.channels_list} \n\n")

            # 选择设备，并进行训练
            global_model.train()
            idxs_users = np.random.choice(range(self.args.num_users), self.m, replace=False)
            num_current, avg_iter = self.client_train(idxs_users, global_model, user_groups, epoch,
                                                      train_dataset, train_losses, local_weights, local_losses)

            # 无效轮
            if len(local_weights) == 0:
                train_losses.append(train_losses[-1])
                test_accs.append(test_accs[-1])
                continue

            # update global weights
            pre_model = copy.deepcopy(global_model.state_dict())
            global_weights = average_weights(local_weights)

            # update global weights
            global_model.load_dict(global_weights)

            # 进行中心训练
            if self.share_percent > 0 and self.args.central_train == 1:
                self.server_train(epoch, num_current, train_dataset, user_groups, global_model, avg_iter, idxs_users)

            # loss = self.get_loss(all_train_data, train_dataset, global_model)
            # Test inference after completion of training
            test_acc, test_loss = test_inference(global_model, test_dataset)
            if test_loss < train_losses[0] * 3:
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
        record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        weight_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log",
                                f"weights_{self.share_percent}.txt")
        record_log(self.args, weight_path, str(self.log_weights)[1:-1])


if __name__ == "__main__":
    args = args_parser()
    t = HRank(args, share_percent=1, iid=False, unequal=False, prune_interval=5, prune_rate=0.6, auto_rate=True,
              auto_mu=False, server_mu=0, client_mu=0)
    t.train()