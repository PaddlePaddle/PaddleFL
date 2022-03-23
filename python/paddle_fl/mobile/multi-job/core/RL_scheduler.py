# -*- coding:utf-8 -*-
import copy
import json
import pickle as pkl
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.nn as nn
from paddle.nn import LSTM, Linear
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set the Beta value
O = [2 * (10 ** 6), 2 * (10 ** 6), 10 ** 2]  # NIID
# O = [5*(10**6), 2*(10**6), 10**3]  # IID

with open('config.json') as file:
    config = json.load(file)
with open("utils/client_time_" + config["group"] + ".json", "r") as f:
    client_time = json.load(f)

# 0:Cnn, 1:Resnet, 2:Alexnet
# job_weight_num, job_data_size(KB), model_input_shape
jobs_feature = [[224874, 54.9888, 3072],
                [598186, 1402.88, 3072],
                [3274634, 420.9664, 784]]
Pre_clients = [[0 for i in range(100)] for j in range(config["total_jobs"])]


def Normalized(inputs):
    one_hot_len = 100
    inputs_non_one_hot = inputs[:, :, one_hot_len:]
    input_non_one_hot_normalized = (inputs_non_one_hot - inputs_non_one_hot.mean()) / (inputs_non_one_hot.std() + 1e-10)
    inputs[:, :, one_hot_len:] = input_non_one_hot_normalized
    return inputs

def Variance(action, pre_clients, t):
    Pre = copy.deepcopy(pre_clients)
    if action is not None:
        for i, d in enumerate(action):
            j = i // 10
            Pre[j][d] += 1

    var = []
    for i, v in enumerate(Pre):
        num_class = v
        S = sum(num_class)
        if S == 0:
            var.append(0)
        else:
            num_class = np.array(num_class) / S

            N = len(num_class)
            miu = sum(num_class) / N
            var.append(sum((np.array(num_class) - miu) ** 2) / N * t[i])
    return var


def Processing_input(devices, V):
    """
    :param devices: available devices for job j
    :param num_job: the number of jobs [0,1,2]
    :param r: the round r
    :param V: the data variance
    """
    inputs = []

    encode_device = [0 for i in range(100)]
    for i in devices:
        encode_device[i] = 1
    inputs.append(encode_device + V)
    inputs = np.array(Normalized(np.array([inputs])), dtype=np.float32)
    return inputs


def get_reward(scheme, Pre_clients, total_job, e, E, t):
    temp = []
    Reward = []

    for j, d in enumerate(scheme):
        time = []
        Pre = copy.deepcopy(Pre_clients[j // 10])

        time.append(client_time[d][j // 10])
        Pre[d] += 1
        V = Variance(None, [Pre], [O[j // 10]])

        Reward.append(-max(time) - V[0])
        temp.append((max(time), -V[0]))

    if e == E - 1:
        print(np.sum(Pre_clients, axis=1), temp)
    return Reward


class Network(nn.Layer):
    """CNN Model."""
    def __init__(self, input_size, output_shape):
        super(Network, self).__init__()
        self.rnn = LSTM(input_size=input_size, hidden_size=32, num_layers=1)
        self.fc = Linear(in_features=32, out_features=output_shape)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)
        x = self.fc(r_out[:, -1, :])
        return x


class RL(object):
    """DRL Implementation."""

    def __init__(self, input_size, output_shape, episode=300, total_job=3, C=10):
        self.lr = 0.00009
        self.C = C
        self.input_size = input_size
        self.output_shape = output_shape
        self.gamma = 0.85
        self.epsilon = 1.0
        self.epsilon_decay = 0.99999
        self.epsilon_min = 0.1
        self.convergence_min_choice_prob = 0.99

        self.total_job = total_job
        self.episode = episode


    def predict(self, devices, Pre_clients):
        """Predict Q value given state."""

        inputs = Processing_input(devices, Variance(None, Pre_clients, O))
        inputs = fluid.dygraph.to_variable(inputs)
        pro = self.model(inputs)

        scheme = []
        tensor_list = np.array(pro[0])
        while len(scheme) < self.total_job * self.C:
            ind = np.argmax(tensor_list)
            if ind not in devices:
                tensor_list[ind] = float('-inf')
                continue
            scheme.append(int(ind))
            tensor_list[ind] = float('-inf')
        return scheme

    def round_train(self, devices, Pre_clients):
        start = time.time()
        Q_mean = 0
        last_iter = False
        self.epsilon = 1.0
        for e in range(self.episode):
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if e == self.episode - 1:
                last_iter = True

            with fluid.dygraph.guard():
                self.model = Network(self.input_size, self.output_shape)
                if e == 0:
                    if config["isIID"]:
                        c = "iid"
                    else:
                        c = "niid"
                    self.model.set_state_dict(paddle.load('__cache__/model'+ c + '.pdparams'))
                self.model.train()
                inputs = Processing_input(devices, Variance(None, Pre_clients, O))
                inputs = fluid.dygraph.to_variable(inputs)

                pro = fluid.layers.log(fluid.layers.softmax(self.model(inputs)))
                pro = paddle.to_tensor(pro)

                scheme = []  # selected devices
                if np.random.rand() <= self.epsilon:
                    scheme = np.random.choice(devices, self.total_job * self.C, replace=False)
                else:
                    tensor_list = np.array(pro[0])
                    while len(scheme) < self.total_job * self.C:
                        ind = np.argmax(tensor_list)
                        if ind in devices:
                            scheme.append(ind)
                        tensor_list[ind] = float('-inf')

                SCHEME = np.array(scheme).reshape(len(scheme), 1)
                sum_pro = fluid.layers.reduce_sum(pro[0] * fluid.layers.one_hot(paddle.to_tensor(SCHEME), depth=100))

                r = get_reward(scheme, Pre_clients, self.total_job, e, self.episode, -1)

                loss = -fluid.layers.reduce_mean(sum_pro * paddle.to_tensor(np.array(r) - Q_mean))

                # apply gradient descent to update train model
                if not last_iter:
                    self.optimizer = paddle.optimizer.Adam(self.lr, parameters=self.model.parameters())
                    if e > 39 and e // 10 == 0 and self.lr > 0.9e-5:
                        self.lr *= 0.85
                        self.optimizer = paddle.optimizer.Adam(self.lr, parameters=self.model.parameters())
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.clear_grad()

            Q_mean = Q_mean * self.gamma + np.mean(r) * (1 - self.gamma)

        # self.model.save()
        if config["isIID"]:
            c = "iid"
        else:
            c = "niid"

        if not os.path.exists('../__cache__/model'):
            os.makedirs('../__cache__/model')
        updata_state_dict = self.model.state_dict()
        paddle.save(updata_state_dict, '../__cache__/model'+ c + '.pdparams')

        scheme = self.predict(devices, Pre_clients)
        end = time.time()
        print(end - start)
        return scheme