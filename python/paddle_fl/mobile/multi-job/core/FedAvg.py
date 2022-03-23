import numpy as np
import paddle
import paddle.nn as nn
import paddle.fluid as fluid
from paddle.fluid.io import DataLoader

class FedAvg(object):
    def __init__(self, BATCH_SIZE=32, EPOCH_NUM=5, local_lr=0.001, drop_r=0):
        """
        drop_r：模拟设备算力的不一致，即stragglers的比例
        """
        self.BATCH_SIZE = BATCH_SIZE
        self.EPOCH_NUM = EPOCH_NUM
        self._reset_list()
        self.local_lr = local_lr
        self.drop_r = drop_r
        # 是指上一轮的增量
        self.updates = []

    def _reset_list(self):
        """重置列表"""
        self.global_weights = []
        self.delta_list = []
        self.n_list = []
        self.loss_list = []
        self.acc_list = []

    def metrics(self, y_true, y_pred):
        """测量性能"""
        loss = fluid.layers.reduce_mean(fluid.layers.cross_entropy(input=y_pred, label=y_true))
        acc = np.mean(y_true == np.argmax(y_pred, axis=1))
        return loss, acc

    def set_global_weights(self, weights):
        """设置global weights"""
        self.global_weights = [w + 0 for w in weights]

    def client_train(self, client, train_set, MODEL, r):
        """train on client"""
        train_loader = DataLoader(dataset=train_set, batch_size=self.BATCH_SIZE)
        MODEL.prepare(optimizer=paddle.optimizer.Adam(parameters=MODEL.parameters()),
              loss=nn.CrossEntropyLoss(soft_label=True),
              metrics=paddle.metric.Accuracy())
        MODEL.fit(train_data=train_loader, epochs=self.EPOCH_NUM, batch_size=self.BATCH_SIZE)
        self.updates.append(MODEL.parameters())

    def avg(self, g_lr=1.0):
        """平均"""
        # 更新glocal_model的参数
        new_weights = list()
        for layer_tuples in zip(*self.updates):
            new_weights.append(
                np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*layer_tuples)]))
        global_weights = new_weights
        # 更新update
        self.updates = []
        # 重置列表
        self._reset_list()

        # 返回权重
        return global_weights

    def fed_eval(self, MODEL, train_set, test_set):
        """测试"""
        MODEL.prepare(optimizer=paddle.optimizer.Adam(parameters=MODEL.parameters()),
                      loss=nn.CrossEntropyLoss(soft_label=True),
                      metrics=paddle.metric.Accuracy())
        train_loader = DataLoader(dataset=train_set, batch_size=self.BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(dataset=test_set, shuffle=True)

        train_result = MODEL.evaluate(eval_data=train_loader, verbose=0)  # Loss and Accuracy
        test_result = MODEL.evaluate(eval_data=test_loader, verbose=0)
        # get the train loss and acc
        return np.sum(train_result['loss']) / len(train_set), np.sum(train_result['acc']) / len(train_set), \
               np.sum(test_result['loss']) / len(test_set), np.sum(test_result['acc']) / len(test_set)

def preprocessing(self, x):
        """预处理，如标准化"""
        data = x.astype(np.float32)
        mu, sigma = np.mean(data, 0), np.std(data, 0)
        return (data - mu) / (sigma + 1e-3)

