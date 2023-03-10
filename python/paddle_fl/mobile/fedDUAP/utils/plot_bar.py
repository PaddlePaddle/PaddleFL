"""
Plot bar
"""
#python 画柱状图折线图
#-*- coding: utf-8 -*-

import matplotlib.ticker as mtick
import matplotlib.pyplot as plt
import matplotlib
import os
import re
from utils.get_flops import get_flops
from scipy import signal


mode1 = {
    "tick": 17,
    "fig": [5, 4],
    "legend": 17,
    "label": 22
}

mode2 = {
    "tick": 10,
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
root_path = cwd[:cwd.find(root_name)+len(root_name)]

device_speed = 50 / 8  # MB/s
server_speed = 100 / 8  # MB/s


model_time = {
    "cnn": {"training_time": 23 / 10},
    "vgg": {"training_time": 80 / 10},
    "resnet": {"training_time": 101 / 10},
    "lenet": {"training_time": 13 / 10}
}


data_names = ["cifar10", "mnist", "fashionmnist", "cifar100"]
data_name = data_names[0]
models = ["cnn", "vgg", "resnet", "lenet"]
model_name = "vgg"
base_dir = "result/" + data_name + "/" + model_name
fig_dir = "./fig/" + model_name + "/cmp_prunerate_l5_interval30"
isPDF = False

base_flop, base_size = get_flops(model_name, dataset=data_name)
training_time = model_time[model_name]["training_time"]
comm_time = base_size * 4 * 10 / 1024 / 1024 / device_speed + base_size * 4 * 10 / 1024 / 1024 / server_speed

percents = [10]
cwd = os.getcwd()
rounds = 500
base_acc = 1
weights_cord = ["LotteryTicket", "IMC", "PruneFL"]

base = ["Avg"]
name_dic1 = {
    "Avg": "base10_equal_l5",

    "0.2": ["hrank10_equal_l5_nocenter_0.2"],
    "0.4": ["hrank10_equal_l5_nocenter_0.4"],
    "0.6": ["hrank10_equal_l5_nocenter_0.6"],
    "0.8": ["hrank10_equal_l5_nocenter_0.8"],
    "auto2": ["hrank10_equal_auto_newprune_nocenter_l5_interval30"],
    "auto3": ["hrank10_equal_l5_nocenter_auto_new_500_divided_50"],
    "AP": ["hrank10_equal_l5_nocenter_auto_50"],

    "IMC": ["lotteryticket10_equal_nocenter_withdevice_onemore_l5"],
    "PruneFL": ["prunefl_nocenter"],
}

name_dic2 = {
    "Avg": "base10_equal_l5",

    "0.2": ["hrank10_equal_l5_nocenter_0.2"],
    "0.4": ["hrank10_equal_l5_nocenter_0.4"],
    "0.6": ["hrank10_equal_l5_nocenter_0.6"],
    "0.8": ["hrank10_equal_l5_nocenter_0.8"],
    "AP": ["hrank10_equal_l5_nocenter_auto_50", "hrank10_equal_l5_nocenter_auto_10",
             "hrank10_equal_l5_nocenter_auto_50", "hrank10_equal_l5_nocenter_auto_70",
             "hrank10_equal_l5_nocenter_auto_1", "hrank10_equal_l5_nocenter_auto_3",
             "hrank10_equal_l5_nocenter_auto_5", "hrank10_equal_l5_nocenter_auto_7"],
}

name_dic = name_dic1
cmp1 = ["HRank_0.6", "HRank_0.8", "FedGMS-P"]
cmp2 = ["FedGMS-D", "LotteryTicket", "HRank_auto"]

cmp3 = ["DataSharing", "Hybrid", "FedGMS-D"]

cmptest18 = ["20%", "40%", "60%", "80%", prune_name]
cmptest19 = ["auto", "IMC", "PruneFL"]
cmptest18 = ["0.2", "0.4", "0.6", "0.8", "AP"]
method = cmptest18

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
    if "M" not in filters_dic[0]:
        transfer_filters(filters_dic)
    train_time = training_time
    if "DataSharing" in name:
        train_time = train_time * (int(percent) + 1)
    time=[0]
    t = train_time + comm_time
    flops = [base_flop]
    weights = [base_size]
    for i in range(1, rounds):
        if i in filters_dic:
            flop, weight = get_flops(model_name, filters_dic[i])
            flops.append(flop)
            if name not in weights_cord:
                weights.append(weight)
            else:
                weights.append(weights_dic[i])
            t = train_time * flops[-1] / base_flop + comm_time * weights[-1] / base_size
        time.append(time[i - 1] + t)
    return time, flops[-1], weights[-1]


def get_time2(sizes):
    """
    Get time 2
    :param sizes:
    :return:
    """
    time = [0]
    base = training_time + comm_time
    for i in range(0, len(sizes)):
        t = base * sizes[i] / base_size
        time.append(time[i] + t)
    return time


def get_loss_or_acc(file_name):
    """
    Get loss or acc
    :param file_name:
    :return:
    """
    file_path = os.path.join(root_path, file_name)
    if not os.path.exists(file_path):
        return []

    with open(file_path, "r") as f:
        loss = f.read()

    loss = loss.split(",")
    loss = [float(loss[i]) for i in range(len(loss))]
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
    Transfer filters
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
    Plot single line
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
    time, final_flop, final_weight = get_time(filters_dic, weights_dic, name, percent)

    loss = get_loss_or_acc(file_name)
    if len(loss) == 0:
        return

    start = loss[0]
    loss = loss[1: rounds + 1]
    loss = signal.savgol_filter(loss, 39, 3)

    loss = [round(i, 3) for i in loss]
    loss.insert(0, start)
    time = time[0: min(len(loss), rounds)]
    acc = max(loss)
    if "FedAvg" in name:
        base_acc = acc

    messages.append("name: {:15}, percent: {:2d} acc: {:.4f} ({:.4f})  finish time {:.1f} final_flop: {} "
                    "({:.1f}%) final parameters {} "
                    "({:.1f}%) ".format(name, percent, acc, acc - base_acc, time[-1], final_flop,
                                        final_flop / base_flop * 100, final_weight, final_weight / base_size * 100))
    print(messages[-1])

    print()
    # if "FedAvg" in name:
    #     plt.plot(time, loss, label=name)
    # else:
    #     plt.plot(time, loss, label=name + ": " + str(percent) + "%")
    return acc, time[-1], final_flop, final_weight

def plot_multi_lines(iid=False, loss=False):
    """
    Plot multi lines
    :param iid:
    :param loss:
    :return:
    """
    iid_path = "iid" if iid else "noniid"
    loss_path = "_train_loss.txt" if loss else "_test_accuracy.txt"

    # file_name = os.path.join(base_dir, "base", iid_path, "0" + loss_path)
    # plot_single_line(file_name, "FedAvg")

    messages_dic = dict()
    for percent in percents:
        for name in base:
            if name in name_dic:
                file_name = os.path.join(base_dir, name_dic[name], iid_path, "0" + loss_path)
                acc, time, final_flop, final_weight = plot_single_line(file_name, name)
                if name not in messages_dic:
                    messages_dic[name] = dict()
                    messages_dic[name]["acc"] = acc
                    messages_dic[name]["time"] = time
                    messages_dic[name]["final_flop"] = final_flop
                    messages_dic[name]["final_weight"] = final_weight
                else:
                    messages_dic[name]["acc"] += acc
                    messages_dic[name]["time"] += time
                    messages_dic[name]["final_flop"] += final_flop
                    messages_dic[name]["final_weight"] += final_weight

        for name in method:
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
                pass
            else:
                file_name = os.path.join(base_dir, name, iid_path, str(percent) + loss_path)
            acc, time, final_flop, final_weight = plot_single_line(file_name, name, percent)
            if name not in messages_dic:
                messages_dic[name] = dict()
                messages_dic[name]["acc"] = acc
                messages_dic[name]["time"] = time
                messages_dic[name]["final_flop"] = final_flop
                messages_dic[name]["final_weight"] = final_weight
            else:
                messages_dic[name]["acc"] += acc
                messages_dic[name]["time"] += time
                messages_dic[name]["final_flop"] += final_flop
                messages_dic[name]["final_weight"] += final_weight


    avg_num = len(percents)
    for name in messages_dic.keys():
        messages_dic[name]["acc"] = round((messages_dic[name]["acc"] / avg_num) * 100, 2)
        messages_dic[name]["time"] /= avg_num
        messages_dic[name]["final_flop"] /= avg_num
        messages_dic[name]["final_weight"] /= avg_num

    lx = []
    for name in base:
        if name in name_dic:
            lx.append(name)
    for name in method:
        if name in name_dic:
            lx.append(name)
        else:
            lx.append(name)

    accs = []
    times = []
    for name in lx:
        if name in name_dic and name in messages_dic:
            times.append(messages_dic[name]["time"])
            accs.append(messages_dic[name]["acc"])
        elif name in messages_dic:
            times.append(messages_dic[name]["time"])
            accs.append(messages_dic[name]["acc"])

    plot_bar(lx, accs, times, loss, iid)


def plot_bar(lx, accs, times, loss, iid):
    """
    Plot bar
    :param lx:
    :param accs:
    :param times:
    :param loss:
    :param iid:
    :return:
    """
    fig = plt.figure(figsize=[mode["fig"][0], mode["fig"][1]])
    # font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    l = [i for i in range(len(lx))]
    fmt = '%.0f%%'
    yticks = mtick.FormatStrFormatter(fmt)  # 设置百分比形式的坐标轴
    ax1 = fig.add_subplot(111)
    ax1.plot(l, accs, 'or-', label='acc')
    ax1.yaxis.set_major_formatter(yticks)
    for i, (_x, _y) in enumerate(zip(l, accs)):
        plt.text(_x - 0.5, _y, accs[i], color='black', fontsize=16, )  # 将数值显示在图形上
    ax1.legend( fontsize=mode["legend"])
    ax1.set_ylim([max(min(accs) - 2, 0), min(max(accs) + 4, 100)])
    ax1.set_ylabel('Accuracy', fontsize=mode["label"])
    plt.legend(fontsize=mode["legend"])  # 设置中文
    ax2 = ax1.twinx()  # this is the important function
    plt.bar(l, times, alpha=0.3, color='blue', label="time")
    ax2.legend( fontsize=mode["legend"])
    ax2.set_ylim([0, int(max(times) * 1.3)])  # 设置y轴取值范围
    ax2.ticklabel_format(style='sci', scilimits=(0, 0), axis='both', useMathText=True)
    plt.legend(fontsize=mode["legend"], loc="upper left")
    plt.xticks(l, lx)

    loss_label = "Loss" if loss else "Accuracy"
    iid_label = "IID" if iid else "NonIID"
    plt.tight_layout()
    if isPDF:
        plt.savefig(fig_dir + "_" + iid_label + "_" + loss_label + ".pdf")
    else:
        plt.show()


    # # plt.xlim(0, 800)
    # plt.legend()
    # plt.xlabel("# Time(s)", fontsize=18)
    # plt.ylabel(f"# {loss_label}", fontsize=18)
    # plt.title(f"{iid_label} {loss_label}", fontsize=18)
    # plt.tight_layout()
    # if isPDF:
    #     plt.savefig(fig_dir + f'{iid_label}_{loss_label}.pdf')
    # else:
    #     plt.show()



# plot_multi_lines(True, False)
plot_multi_lines(False, False)
#
# for message in messages:
#     print(message)


# font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
# a=[1228.3,3.38,63.8,0.07,0.16,6.74,1896.18]  #数据
# b=[0.12,-12.44,1.82,16.67,6.67,-6.52,4.04]
# l=[i for i in range(7)]
#
# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
#
# fmt='%.2f%%'
# yticks = mtick.FormatStrFormatter(fmt)  #设置百分比形式的坐标轴
# lx=[u'粮食',u'棉花',u'油料',u'麻类',u'糖料',u'烤烟',u'蔬菜']

# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.plot(l, b,'or-',label=u'增长率')
# ax1.yaxis.set_major_formatter(yticks)
# for i,(_x,_y) in enumerate(zip(l,b)):
#     plt.text(_x,_y,b[i],color='black',fontsize=10,)  #将数值显示在图形上
# ax1.legend(loc=1)
# ax1.set_ylim([-20, 30])
# ax1.set_ylabel('增长率')
# plt.legend(prop={'family':'SimHei','size':8})  #设置中文
# ax2 = ax1.twinx() # this is the important function
# plt.bar(l,a,alpha=0.3,color='blue',label=u'产量')
# ax2.legend(loc=2)
# ax2.set_ylim([0, 2500])  #设置y轴取值范围
# plt.legend(prop={'family':'SimHei','size':8},loc="upper left")
# plt.xticks(l,lx)
# plt.show()