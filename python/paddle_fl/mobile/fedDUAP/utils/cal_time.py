"""
Cal time
"""
import matplotlib.pyplot as plt
import matplotlib
import os
import re
from utils.get_flops import get_flops
from scipy import signal

mode1 = {
    "tick": 17,
    "fig": [5, 4],
    "legend": 14,
    "label": 22
}

mode2 = {
    "tick": 13,
    "fig": [5, 4],
    "legend": None,
    "label": 18
}

mode3 = {
    "tick": 20,
    "fig": [5, 4],
    "legend": 18,
    "label": 22
}

mode = mode3

algo_name = "FedDU"
prune_name = "FedAP"
method_name = "FedDUAP"

matplotlib.rc('xtick', labelsize=mode["tick"])
matplotlib.rc('ytick', labelsize=mode["tick"])

# 获取当前目录
cwd = os.getcwd()
root_name = "FedDUAP_pd"
# 获取根目录
root_path = cwd[:cwd.find(root_name) + len(root_name)]

device_speed = 50 / 8  # MB/s
server_speed = 100 / 8  # MB/s

model_time = {
    "cnn": {"training_time": 23 / 10},
    "vgg": {"training_time": 80 / 10},
    "resnet": {"training_time": 101 / 10},
    "lenet": {"training_time": 13 / 10}
}

target_acc = 0.5
data_names = ["cifar10", "mnist", "fashionmnist", "cifar100"]
data_name = data_names[0]
models = ["cnn", "vgg", "resnet", "lenet"]
model_name = "cnn"
# base_dir = f"result/{data_name}/{model_name}"
base_dir = "result/" + data_name + "/" + model_name

# fig_dir = f"./fig/{model_name}/{data_name}_prune_cmp_p10"
fig_dir = "./fig/" + model_name + "/" + data_name + "_prune_cmp_p10"
isPDF = False

base_flop, base_size = get_flops(model_name, dataset=data_name)
training_time = model_time[model_name]["training_time"]
comm_time = base_size * 4 * 10 / 1024 / 1024 / device_speed + base_size * 4 * 10 / 1024 / 1024 / server_speed


percents = [10]
cwd = os.getcwd()
rounds = 500
base_acc = 1
weights_cord = ["LotteryTicket", "IMC-D", "IMC", "PruneFL"]

base = ["FedAvg"]

# for cifar10 on cnn vgg
name_dic1 = {
    "FedAvg": "base10_equal_l5",
    algo_name: ["central10_equal_l5_d0.900", "central10_equal_l5_d0.950",
                "central10_equal_l5_d0.990", "central10_equal_l5_d0.999"],
    "Hybrid-FL": ["hybrid10_equal_l5_weightedavg"],
    "Hybrid-FL_weighted": ["hybrid10_equal_l5_500_weightedavg"],
    "Hybrid-FL_direct": ["hybrid10_equal_l5_500_directavg"],
    "Data-sharing": ["datashare10_equal_l5_origin"],

    prune_name: ["hrank10_equal_l5_nocenter_auto_50"],
    method_name: ["hrank10_equal_l5_d0.900_auto", "hrank10_equal_l5_d0.950_auto",
                  "hrank10_equal_l5_d0.990_auto", "hrank10_equal_l5_d0.999_auto",
                  "hrank10_equal_l5_auto_500_10_decay900", "hrank10_equal_l5_auto_500_10_decay950",
                  "hrank10_equal_l5_auto_500_10_decay990", "hrank10_equal_l5_auto_500_10_decay999"],

    algo_name + "_d2": ["central10_equal_l5_d0.900", "central10_equal_l5_d0.950",
                        "central10_equal_l5_d0.990", "central10_equal_l5_d0.999"],
    algo_name + "_d3": ["central10_equal_l10_d0.900", "central10_equal_l10_d0.950",
                        "central10_equal_l10_d0.990", "central10_equal_l10_d0.999"],
    algo_name + "_d1": ["central10_equal_l2_d0.900", "central10_equal_l2_d0.950",
                        "central10_equal_l2_d0.990", "central10_equal_l2_d0.999"],
    prune_name + "_with_dense": ["hrank10_equal_l5_nocenter_auto_with_dense_50"],
    method_name + "_with_dense": ["hrank10_equal_l5_d0.900_auto_with_dense", "hrank10_equal_l5_d0.950_auto_with_dense",
                                  "hrank10_equal_l5_d0.990_auto_with_dense", "hrank10_equal_l5_d0.999_auto_with_dense"],

    "IMC": ["lotteryticket10_equal_l5_nocenter_50"],
    "PruneFL": "prunefl_nocenter",

    "FedAvg_with_degree": "base10_equal_l5_with_degree",
    algo_name + "_with_degree": ["central10_equal_l5_d0.900_with_degree", "central10_equal_l5_d0.900_with_degree",
                                 "central10_equal_l5_d0.900_with_degree", "central10_equal_l5_d0.900_with_degree"],

    "FedAvg_use_adam": "base10_equal_l5_use_adam",
    algo_name + "_use_adam": ["central10_equal_l5_d0.900_use_adam", "central10_equal_l5_d0.950_use_adam",
                              "central10_equal_l5_d0.990_use_adam", "central10_equal_l5_d0.999_use_adam"],

    "FedAvg_use_sgdm": "base10_equal_l5_use_sgdm",
    algo_name + "_use_sgdm": ["central10_equal_l5_d0.900_use_sgdm", "central10_equal_l5_d0.950_use_sgdm",
                              "central10_equal_l5_d0.990_use_sgdm", "central10_equal_l5_d0.999_use_sgdm"],

    method_name + "_use_sgdm": ["hrank10_equal_l5_d0.900_auto_use_sgdm", "hrank10_equal_l5_d0.950_auto_use_sgdm",
                                "hrank10_equal_l5_d0.990_auto_use_sgdm", "hrank10_equal_l5_d0.999_auto_use_sgdm"],

    "FedAvg_use_da": "base10_equal_l5_use_fsgdm",
    algo_name + "_use_da": ["central10_equal_l5_d0.900use_fsgdm", "central10_equal_l5_d0.950use_fsgdm",
                               "central10_equal_l5_d0.990use_fsgdm", "central10_equal_l5_d0.999use_fsgdm"],

    "FedAvg_use_globalm": "base10_equal_l5_use_globalm",
    algo_name + "_use_globalm": ["central10_equal_l5_d0.900_use_globalm", "central10_equal_l5_d0.950_use_globalm",
                                 "central10_equal_l5_d0.990_use_globalm", "central10_equal_l5_d0.999_use_globalm"]
}

name_dic2 = {
    "FedAvg": "base10_equal_l5",
    algo_name: ["central10_equal_l5_d0.900", "central10_equal_l5_d0.950",
                "central10_equal_l5_d0.990", "central10_equal_l5_d0.999"],
    "Hybrid-FL": ["hybrid10_equal_l5_weightedavg"],
    "Hybrid-FL_weighted": ["hybrid10_equal_l5_500_weightedavg"],
    "Hybrid-FL_direct": ["hybrid10_equal_l5_500_directavg"],
    "Data-sharing": ["datashare10_equal_l5_origin"],

    prune_name: ["hrank10_equal_l5_nocenter_auto_10", "hrank10_equal_l5_nocenter_auto_30",
                 "hrank10_equal_l5_nocenter_auto_50", "hrank10_equal_l5_nocenter_auto_70"],
    method_name: ["hrank10_equal_l5_d0.900_auto", "hrank10_equal_l5_d0.950_auto",
                  "hrank10_equal_l5_d0.990_auto", "hrank10_equal_l5_d0.999_auto",
                  "hrank10_equal_l5_auto_500_10_decay900", "hrank10_equal_l5_auto_500_10_decay950",
                  "hrank10_equal_l5_auto_500_10_decay990", "hrank10_equal_l5_auto_500_10_decay999",
                  ],

    algo_name + "_d2": ["central10_equal_l5_d0.900", "central10_equal_l5_d0.950",
                        "central10_equal_l5_d0.990", "central10_equal_l5_d0.999"],
    algo_name + "_d3": ["central10_equal_l10_d0.900", "central10_equal_l10_d0.950",
                        "central10_equal_l10_d0.990", "central10_equal_l10_d0.999"],
    algo_name + "_d1": ["central10_equal_l2_d0.900", "central10_equal_l2_d0.950",
                        "central10_equal_l2_d0.990", "central10_equal_l2_d0.999"],

    prune_name + "_with_dense": ["hrank10_equal_l5_nocenter_auto_with_dense_10", "hrank10_equal_l5_nocenter_auto_with_dense_30",
                             "hrank10_equal_l5_nocenter_auto_with_dense_50", "hrank10_equal_l5_nocenter_auto_with_dense_70"],
    method_name + "_with_dense": ["hrank10_equal_l5_d0.900_auto_with_dense", "hrank10_equal_l5_d0.950_auto_with_dense",
                                  "hrank10_equal_l5_d0.990_auto_with_dense", "hrank10_equal_l5_d0.999_auto_with_dense"],

    "IMC": ["lotteryticket10_equal_l5_nocenter_10", "lotteryticket10_equal_l5_nocenter_30",
            "lotteryticket10_equal_l5_nocenter_50", "lotteryticket10_equal_l5_nocenter_70"],
    "PruneFL": "prunefl_nocenter",

    "FedAvg_with_degree": "base10_equal_l5_with_degree",
    algo_name + "_with_degree": ["central10_equal_l5_d0.900_with_degree", "central10_equal_l5_d0.900_with_degree",
                                 "central10_equal_l5_d0.900_with_degree", "central10_equal_l5_d0.900_with_degree"],

    "FedAvg_use_adam": "base10_equal_l5_use_adam",
    algo_name + "_use_adam": ["central10_equal_l5_d0.900_use_adam", "central10_equal_l5_d0.950_use_adam",
                              "central10_equal_l5_d0.990_use_adam", "central10_equal_l5_d0.999_use_adam"],

    "FedAvg_use_sgdm": "base10_equal_l5_use_sgdm",
    algo_name + "_use_sgdm": ["central10_equal_l5_d0.900_use_sgdm", "central10_equal_l5_d0.950_use_sgdm",
                              "central10_equal_l5_d0.990_use_sgdm", "central10_equal_l5_d0.999_use_sgdm"],

    method_name + "_use_sgdm": ["hrank10_equal_l5_d0.900_auto_use_sgdm", "hrank10_equal_l5_d0.950_auto_use_sgdm",
                                "hrank10_equal_l5_d0.990_auto_use_sgdm", "hrank10_equal_l5_d0.999_auto_use_sgdm"],

    "FedAvg_use_da": "base10_equal_l5_use_fsgdm",
    algo_name + "_use_da": ["central10_equal_l5_d0.900use_fsgdm", "central10_equal_l5_d0.950use_fsgdm",
                               "central10_equal_l5_d0.990use_fsgdm", "central10_equal_l5_d0.999use_fsgdm"],

    "FedAvg_use_globalm": "base10_equal_l5_use_globalm",
    algo_name + "_use_globalm": ["central10_equal_l5_d0.900_use_globalm", "central10_equal_l5_d0.950_use_globalm",
                                 "central10_equal_l5_d0.990_use_globalm", "central10_equal_l5_d0.999_use_globalm"]
}

name_dic = name_dic2
if data_name == "cifar10" and (model_name == "cnn" or model_name == "vgg"):
    name_dic = name_dic1

cmp3 = ["Data-sharing", "Hybrid-FL", algo_name]

cmp4 = ["PruneFL", "IMC", prune_name]

cmp5 = ["PruneFL", "Hybrid-FL", "Data-sharing", "IMC", method_name]

cmp6 = [algo_name + "_d1", algo_name + "_d2", algo_name + "_d3"]

cmp7 = [algo_name, prune_name, method_name]

cmp8 = [algo_name, prune_name, "PruneFL", "Hybrid-FL", "Data-sharing", "IMC", method_name]

cmp9 = [prune_name, prune_name + "_with_dense"]

cmp10 = [method_name, method_name + "_with_dense"]

cmp11 = [algo_name, algo_name + "_with_degree"]

cmp12 = [algo_name, algo_name + "_use_adam"]

cmp13 = [algo_name, algo_name + "_use_sgdm"]

cmp14 = [method_name, method_name + "_use_sgdm"]

cmp15 = ["Data-sharing", "Hybrid-FL", "IMC", "PruneFL", method_name]

cmp16 = [algo_name, algo_name + "_use_da"]

cmp17 = [algo_name, algo_name + "_use_globalm"]

method = cmp4

messages = []


def get_time(filters_dic, weights_dic, name=None, percent=0):
    """
    Get time
    :param filters_dic:
    :param weights_dic:
    :param name:
    :param percent:
    :return:
    """
    if "M" not in filters_dic[0] and model_name not in ["resnet", "lenet"]:
        transfer_filters(filters_dic)
    train_time = training_time
    if "Data-sharing" in name:
        percent = 5 if percent > 5 else percent
        train_time = train_time * (int(percent) + 1)
    time = [0]
    t = train_time + comm_time
    flops = [base_flop]
    weights = [base_size]
    for i in range(1, rounds):
        if i in filters_dic:
            flop, weight = get_flops(model_name, filters_dic[i], dataset=data_name)
            flops.append(flop)
            if name not in weights_cord:
                weights.append(min(weight, weights_dic[i]))
            else:
                weights.append(weights_dic[i])
            t = train_time * flops[-1] / base_flop + comm_time * weights[-1] / base_size
        time.append(time[i - 1] + t)
    return time, flops[-1], weights[-1]


def get_time2(sizes):
    """
    get time 2
    :param sizes:
    :return:
    """
    time = [0]
    base = training_time + comm_time
    for i in range(0, len(sizes)):
        t = base * sizes[i] / base_size
        time.append(time[i] + t)
    return time

def get_time_for_prunefl(dir_name):
    """
    get time for prunefl
    :param dir_name:
    :return:
    """
    file_path = os.path.join(root_path, dir_name, "model_size.txt")
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as f:
        weights = f.read()
    if weights[0] == "[":
        weights = weights[1:]
    if weights[-1] == "]":
        weights = weights[:-1]
    weights = weights.split(",")
    weights = [float(weights[i]) for i in range(min(rounds, len(weights)))]

    train_time = training_time
    t = train_time + comm_time
    time = [0]
    for i in range(1, min(rounds, len(weights))):
        t = train_time + comm_time * weights[i - 1] / base_size
        time.append(time[i - 1] + t)
    return time, base_flop, weights[-1]


def get_loss_or_acc(file_name):
    """
    get loss or acc
    :param file_name:
    :return:
    """
    file_path = os.path.join(root_path, file_name)
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as f:
        loss = f.read()

    if loss[0] == "[":
        loss = loss[1:]
    if loss[-1] == "]":
        loss = loss[:-1]

    loss = loss.split(",")
    loss = [float(loss[i]) for i in range(min(rounds, len(loss)))]
    return loss


def get_model_size(dir_name, percent):
    """
    给定 log 的上级目录，返回模型大小
    :param dir_name:
    :param percent:
    :return:
    """
    file_path = os.path.join(root_path, dir_name, "log", "weights_" + percent + ".txt")
    print(file_path)
    if not os.path.exists(file_path):
        return [dict(), dict()]
    with open(file_path, "r") as f:
        weights = f.read()
    filters_dic = dict()
    weights_dic = dict()
    points = re.findall("\[(.*?])]", weights)
    for point in points:
        filters = re.search("\[(.*)]", point)
        filters = filters.group(1).split(",")
        filters = [int(i) for i in filters]
        nums = re.findall("\d+", point)
        rounds = int(nums[0])
        size = int(nums[1])
        filters_dic[rounds] = filters
        weights_dic[rounds] = size

    return filters_dic, weights_dic


cnn_m = [1, 3]
vgg_m = [1, 3, 6, 9, 12]


def transfer_filters(filters_dic):
    """
    transfer filters
    :param filters_dic:
    :return:
    """
    if model_name == "cnn":
        m_pos = cnn_m
    elif model_name == "vgg":
        m_pos = vgg_m
    else:
        print("unknown model")
        return -1
    for k, v in filters_dic.items():
        for pos in m_pos:
            v.insert(pos, "M")
        filters_dic[k] = v


def plot_single_line(file_name, name, percent=0):
    """
    plot single line
    :param file_name:
    :param name:
    :param percent:
    :return:
    """
    global base_acc
    dir_name = os.path.dirname(file_name)
    print(dir_name)
    filters_dic, weights_dic = get_model_size(dir_name, percent)
    print(filters_dic)
    if len(filters_dic) == 0:
        return
    if name.lower() == "prunefl":
        time, final_flop, final_weight = get_time_for_prunefl(dir_name)
    else:
        time, final_flop, final_weight = get_time(filters_dic, weights_dic, name, percent)

    loss = get_loss_or_acc(file_name)
    if len(loss) == 0:
        return

    start = loss[0]
    loss = loss[1: rounds + 1]
    loss = signal.savgol_filter(loss, 39, 3)

    loss = [round(i, 3) for i in loss]
    loss.insert(0, start)

    rounds_need = -1
    for i in range(len(loss)):
        if loss[i] >= target_acc:
            rounds_need = i
            break

    time = time[0: min(len(loss), rounds)]
    acc = max(loss)
    if "FedAvg" in name:
        base_acc = acc

    msg1 = "name: {:15}, percent: {:2d} acc: {:.4f} ({:.4f})".format(name, percent, acc, acc - base_acc)
    msg2 = "finish time： {:.1f} final_flop: {} ({:.1f}%) final parameters {} ({:.1f}%) "\
        .format(time[-1], round(final_flop / 1000000, 1),
                                                                            final_flop / base_flop * 100,
                                                                            final_weight,
                                                                            final_weight / base_size * 100)

    msg3 = "time needed: {}".format("NaN" if rounds_need <= -1 else round(time[rounds_need], 0))

    msg4 = "final_flop: {} ({:.1f}%) final parameters {} ({:.1f}%) ".format(round(final_flop / 1000000, 1),
                                                                            final_flop / base_flop * 100,
                                                                            final_weight,
                                                                            final_weight / base_size * 100)
    messages.append(msg1 + "\n " + msg3 + " " + msg4 + "\n" + msg2)

    print(messages[-1])

    print()
    if "FedAvg" in name or "PruneFL" in name:
        plt.plot(time, loss, label=name)
    else:
        plt.plot(time, loss, label=name + ":" + str(percent) + "%")


def plot_multi_lines(iid=False, loss=False):
    """
    plot multi lines
    :param iid:
    :param loss:
    :return:
    """
    iid_path = "iid" if iid else "noniid"
    loss_path = "_train_loss.txt" if loss else "_test_accuracy.txt"

    # file_name = os.path.join(base_dir, "base", iid_path, "0" + loss_path)
    # plot_single_line(file_name, "FedAvg")

    for percent in percents:
        plt.figure(figsize=[mode["fig"][0], mode["fig"][1]])
        for name in base:
            if name in name_dic:
                file_name = os.path.join(base_dir, name_dic[name], iid_path, "0" + loss_path)
                plot_single_line(file_name, name)
        for name in method:
            if name == "PruneFL":
                file_name = os.path.join(base_dir, name_dic[name], iid_path, "0" + loss_path)
                plot_single_line(file_name, name)
                continue
            if name in name_dic:
                target = None
                target_acc = 0
                for choice in name_dic[name]:
                    file_name = os.path.join(base_dir, choice, iid_path, str(percent) + loss_path)
                    choice_loss = get_loss_or_acc(file_name)
                    if len(choice_loss) == 0:
                        continue

                    choice_loss = choice_loss[0: rounds + 1]
                    choice_loss = signal.savgol_filter(choice_loss, 39, 3)

                    choice_loss = [round(i, 3) for i in choice_loss]
                    if target_acc < max(choice_loss):
                        target = choice
                        target_acc = max(choice_loss)
                if target is not None:
                    file_name = os.path.join(base_dir, target, iid_path, str(percent) + loss_path)
                    plot_single_line(file_name, name, percent)
                pass
            else:
                file_name = os.path.join(base_dir, name, iid_path, str(percent) + loss_path)
                plot_single_line(file_name, name, percent)

        loss_label = "Loss" if loss else "Accuracy"
        iid_label = "IID" if iid else "NonIID"
        # plt.xlim(0, 800)
        # plt.ylim(0, 0.5)
        plt.legend(fontsize=mode["legend"])
        plt.xlabel("Time(s)", fontsize=mode["label"])
        # plt.ylabel(f"{loss_label}", fontsize=mode["label"])
        # plt.title(f"{iid_label} {loss_label}", fontsize=mode["label"])
        plt.tight_layout()
        # if isPDF:
        #     plt.savefig(fig_dir + f'_{iid_label}_{loss_label}.pdf')
        # else:
        #     plt.show()


# plot_multi_lines(True, False)
plot_multi_lines(False, False)

for message in messages:
    print(message)