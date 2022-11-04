import copy
import random
from skopt import gp_minimize
from skopt.space import Categorical
import numpy as np
from core.Genetic import Gene

O = [2 * (10 ** 5), 10 ** 5, 10 ** 3]  # Beta

def order_way(i):
    return i[0]

def Variance(pre_clients, temp, m):
    if temp is not None:
        for i in temp:
            pre_clients[int(i)] += 1

    num_class = pre_clients
    S = sum(num_class)
    if S == 0:
        return 0
    num_class = np.array(num_class) / S

    N = len(num_class)
    miu = sum(num_class) / N
    return (sum((np.array(num_class) - miu) ** 2) / N) * m


def Greedy(share_clients, job, client_per_round, job_No, client_time, Pre_clients):
    devices = share_clients[:]
    scheme = []

    tim = [(client_time[index][job_No] + Variance(copy.deepcopy(Pre_clients[job_No]), None, O[job_No]), index) for
           index in
           list(devices)]
    for i in range(client_per_round):
        tim.sort(key=order_way, reverse=False)
        d = int(tim[0][1])
        scheme.append(d)
        devices.remove(d)
        tim = [(client_time[index][job_No] + Variance(copy.deepcopy(Pre_clients[job_No]), scheme, O[job_No]), index)
               for index in
               list(devices)]
    return scheme


def Genetic(share_clients, job, client_per_round, job_No, client_time, Pre_clients):
    devices = share_clients[:]
    greedy_scheme = []
    for j in range(3):
        sh = Greedy(list(set(devices) - set(greedy_scheme)), job, client_per_round, j, client_time, Pre_clients)
        greedy_scheme.extend(sh)
    scheme = Gene(devices, Pre_clients, client_time, greedy_scheme)
    return scheme[job_No * client_per_round: (job_No + 1) * client_per_round]


def Common(share_clients, job=None, client_per_round=10, job_No=None, client_time=None, Pre_clients=None):
    return random.sample(list(share_clients), client_per_round)


def Bayesian(share_clients, job, client_per_round, job_No, client_time, Pre_clients):
    """
    :param share_clients: 当前可用的客户端编号
    :param client_per_round: 每一轮需要选择的客户端数量
    :param job_No: 任务编号
    :param client_time: 100个客户端的模拟运行时间
    :param Nclass_clients: 所有客户端的类统计[(class, number),..,]
    :param Pre_clients_class: 之前所选客户端的类统计[number,...,]
    :return:
    """
    clients = [[], [], []]  # the selected devices

    def Target(x, client_time=client_time):
        temp = []
        sum = 0
        if len(set(x)) == len(x):
            for j, J in enumerate(job):
                temp.append([client_time[c][J] for c in clients[J]] + [client_time[x[j]][J]])

                client_temp = clients[J] + [x[j]]
                sum += Variance(copy.deepcopy(Pre_clients[J]), client_temp, O[J])

            for i in range(len(job)):
                sum += max(temp[i])

        else:
            sum = 3500
        return sum

    remain = share_clients[:]
    a = []
    for k in range(len(job)):
        a.append(Categorical(remain))

    for i in range(client_per_round):
        res = gp_minimize(Target,  # the function to minimize
                          a,  # the bounds on each dimension of x
                          acq_func="EI",  # the acquisition function
                          n_calls=20,  # the number of evaluations of f
                          n_random_starts=10,  # the number of random initialization points
                          random_state=None)  # the random seed
        # clients = np.c_[clients, res.x]
        for j in job:
            clients[j].append(int(res.x[job.index(job_No)]))
        # print(res.x)
        for j in res.x:
            remain.remove(j)

        a = []
        for k in range(len(job)):
            a.append(Categorical(remain))

    v = Variance(copy.deepcopy(Pre_clients[job_No]), clients[job_No], O[job_No])
    print(res.fun, np.sum(Pre_clients, axis=1), v)
    return clients[job_No]


def Fedcs(share_clients, job, client_per_round, job_No, client_time, Pre_clients):
    T = [90, 130, 40]  # Cnn, Resnet, Alexnet
    selected = []
    temp_cl = random.sample(list(share_clients), client_per_round)
    time = [[client_time[index][job_No], index] for index in list(temp_cl)]
    time.sort(key=order_way, reverse=False)

    for i in range(10):
        if time[i][0] > T[job_No]:
            break
        selected.append(time[i][1])
    # print(T[job_No], selected)
    return selected


def DRL(model, share_clients, job, client_per_round, job_No, Pre_clients):
    """Predict Q value given state."""
    devices = share_clients[:]
    scheme = model.round_train(devices, copy.deepcopy(Pre_clients))

    return scheme[job_No * client_per_round: (job_No + 1) * client_per_round]
