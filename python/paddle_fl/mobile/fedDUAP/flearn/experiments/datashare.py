"""
Data share
"""
import copy
import numpy as np
from flearn.utils.model_utils import average_weights, test_inference, is_conv
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.experiments.central import CentralTraining

import os
import time

class DataShare(CentralTraining):
    """
    对照试验1
    每个设备有自己随机的一部分数据，同时还将拥有全局的数据
    """
    def __init__(self, args, share_percent=0, iid=True, unequal=False, result_dir="datashare"):
        self.eta = 1

        super(DataShare, self).__init__(args, share_percent, iid, unequal, result_dir=result_dir)

    # 将共享数据分给每个设备
    def distribute(self, user_groups, eta=0.1):
        """
        distribute
        """
        origin_user_groups = dict()
        shared_data = user_groups[self.args.num_users]
        for k, v in user_groups.items():
            if k == self.args.num_users:
                continue
            selected_shared_data = np.random.choice(shared_data, min(len(shared_data), int(self.num_data * 0.05)),
                                                    replace=False)
            origin_user_groups[k] = v.copy()
            user_groups[k] = np.append(v, selected_shared_data)
        return origin_user_groups

    def train(self):
        """
        train
        """
        # 记录日志和结果
        log_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log",
                                str(self.share_percent) + ".txt")
        result_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid")

        # avg_num_data = self.num_data / self.args.num_users
        # self.args.local_bs = self.args.local_bs * math.ceil((self.num_share + avg_num_data) / avg_num_data)
        # print(f"local_bs {self.args.local_bs}")
        # 加载模型
        global_model = self.load_model()
        print(global_model)

        # load dataset and user groups
        train_dataset, test_dataset, user_groups = self.init_data()

        # 分发共享数据
        origin_user_groups = self.distribute(user_groups, self.eta)

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
            # print(f'\n | Global Training Round : {epoch} |\n')

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
            global_weights = average_weights(local_weights)

            # update global weights
            pre_model = copy.deepcopy(global_model.state_dict())
            global_model.load_dict(global_weights)

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
        # record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        # weight_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log",
        #                         f"weights_{self.share_percent}.txt")
        # record_log(self.args, weight_path, str(self.log_weights)[1:-1])

if __name__ == "__main__":
    args = args_parser()
    t = DataShare(args, share_percent=1, iid=False, unequal=False)
    t.train()
