import numpy as np
from flearn.utils.update import LocalUpdate
from flearn.utils.model_utils import average_weights, test_inference, is_conv, ratio_combine
from flearn.utils.options import args_parser
from flearn.utils.util import record_log, save_result
from flearn.experiments.central import CentralTraining
import copy
import os
import time

class Hybrid(CentralTraining):
    """
    对照试验2
    服务端数据参与每一轮的训练，并进行联邦平均
    """
    def __init__(self, args, share_percent=0, iid=True, unequal=False, result_dir="hybrid"):

        super(Hybrid, self).__init__(args, share_percent, iid, unequal, result_dir=result_dir)

    def server_train(self, epoch, train_dataset, user_groups, global_model, train_losses, local_weights,
                     local_losses, num_current):
        print("=== Hybrid central training ===")
        local_model = LocalUpdate(args=self.args, dataset=train_dataset,
                                  idxs=user_groups[self.args.num_users], device=self.device)

        w, loss = local_model.update_weights(
            model=copy.deepcopy(global_model), global_round=epoch, mu=self.args.client_mu)
        if loss < train_losses[0] * 3:
            # local_weights.append([int(num_current / 10), copy.deepcopy(w)])
            local_weights.append([self.num_share, copy.deepcopy(w)])
            local_losses.append(copy.deepcopy(loss))
        print("central loss: {}".format(round(loss, 4)))


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

            # loss = round(sum(local_losses) / len(local_losses), 4)
            # train_losses.append(loss)

            # 进行中心训练
            if self.share_percent > 0 and self.args.central_train == 1:
                self.server_train(epoch, train_dataset, user_groups, global_model, train_losses, local_weights, local_losses, num_current)

            # update global weights
            pre_model = copy.deepcopy(global_model.state_dict())
            global_weights = average_weights(local_weights)

            # global_weights = ratio_combine(global_weights, global_model.state_dict(), 1 / 11)

            # update global weights
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
        record_log(self.args, log_path, f"- log_weight: {self.log_weights} \n\n")
        weight_path = os.path.join(self.result_dir, "iid" if self.iid else "noniid", "log",
                                f"weights_{self.share_percent}.txt")
        record_log(self.args, weight_path, str(self.log_weights)[1:-1])

if __name__ == "__main__":
    args = args_parser()
    t = Hybrid(args, share_percent=1, iid=False, unequal=False)
    t.train()
