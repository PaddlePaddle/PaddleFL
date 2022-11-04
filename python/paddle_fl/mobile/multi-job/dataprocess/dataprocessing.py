import json
import pickle as pkl
from random import sample
import numpy as np


class Database:
    def __init__(self, dataset):
        from tensorflow.keras.datasets import mnist, cifar10, fashion_mnist
        from extra_keras_datasets import emnist
        from tensorflow.keras.utils import to_categorical
        import os

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        self.dataset = dataset
        # 不同的数据集对应的模型input_shape不同，在mode.py中更改
        if dataset == 'emnist_digital':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = emnist.load_data(type='digits')
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, 28, 28)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, 28, 28)
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_train.astype('float32')
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)

        if dataset == 'emnist_letter':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = emnist.load_data(type='letters')
            self.x_train = self.x_train.reshape(self.x_train.shape[0], 1, 28, 28)
            self.x_test = self.x_test.reshape(self.x_test.shape[0], 1, 28, 28)
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_train.astype('float32')
            self.y_train[:] = [x - 1 for x in self.y_train]
            self.y_test[:] = [x - 1 for x in self.y_test]
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)

        if dataset == 'mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist.load_data()
            self.x_train = self.x_train.reshape(60000, 1, 28, 28)
            self.x_test = self.x_test.reshape(10000, 1, 28, 28)
            self.x_train = self.x_train / 255.0
            self.x_test = self.x_test / 255.0
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_train.astype('float32')
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)


        if dataset == 'cifar10':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = cifar10.load_data()
            print(self.x_train.shape, self.x_train)
            self.x_train = (self.x_train - 127.5) / 128
            self.x_train = self.x_train.reshape(50000, 3, 32, 32)
            self.x_test = (self.x_test - 127.5) / 128
            self.x_test = self.x_test.reshape(10000, 3, 32, 32)
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_train.astype('float32')
            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)

        if dataset == 'fashion_mnist':
            (self.x_train, self.y_train), (self.x_test, self.y_test) = fashion_mnist.load_data()
            self.x_train = self.x_train.reshape(60000, 1, 28, 28)
            self.x_test = self.x_test.reshape(10000, 1, 28, 28)
            self.x_train = self.x_train.astype('float32')
            self.x_test = self.x_train.astype('float32')

            self.y_train = to_categorical(self.y_train)
            self.y_test = to_categorical(self.y_test)

    def generate_IID(self, n_train, n_test, total_clients):  # n is the number of simple on per client;
        client = []
        share_index_tr = [i for i in range(len(self.y_train))]
        share_index_te = [i for i in range(len(self.y_test))]
        # print(self.x_train[0])
        for i in range(total_clients):
            Index_tr = sample(share_index_tr, n_train)
            Index_te = sample(share_index_te, n_test)

            share_index_te = list(set(share_index_te) - set(Index_te))
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            for j in Index_tr:
                x_train.append(self.x_train[j])
                y_train.append(self.y_train[j])
            for j in Index_te:
                x_test.append(self.x_test[j])
                y_test.append(self.y_test[j])
            client.append({"x_train": np.array(x_train), "y_train": np.array(y_train), "x_test": np.array(x_test),
                           "y_test": np.array(y_test)})

        with open("../datasets/IID/" + self.dataset + '/train_split.pkl', 'wb') as outfile:
            pkl.dump(client, outfile)
        with open("../datasets/IID/" + self.dataset + '/test.pkl', 'wb') as outfile:
            pkl.dump((self.x_test, self.y_test), outfile)

    def generate_non_IID(self, class_fraction, n_train, n_test, n_class, total_clients):  # n is the number of simple on per client;
        client = []
        classified_tr = [[] for i in range(n_class)]
        classified_te = [[] for i in range(n_class)]
        a = 0
        b = 0
        # 按类排序
        for i in range(n_train * total_clients):
            classified_tr[self.y_train[i].tolist().index(1)].append(i)  # 标签所在的索引即为类别
        for i in range(n_test * total_clients):
            classified_te[self.y_test[i].tolist().index(1)].append(i)

        class_index = [i for i in range(n_class)]

        # 每一类分成20份， 共200份
        for i in range(n_class):
            temp = classified_tr[i]
            delta = round(len(temp) / 20)
            classified_tr[i] = [temp[i * delta: (i + 1) * delta] for i in range(19)]
            classified_tr[i].append(temp[19 * delta:])

            temp = classified_te[i]
            delta = round(len(temp) / 20)
            classified_te[i] = [temp[i * delta: (i + 1) * delta] for i in range(19)]
            classified_te[i].append(temp[19 * delta:])

        # 为每个客户端从十类中选两类, 即选两份
        for i in range(total_clients):
            if len(class_index) < round(class_fraction):  # ֻ只剩一类
                class_sample = [class_index[0]]
            else:
                class_sample = sample(class_index, round(class_fraction))  # 随机选两类

            sample_index_tr = []
            sample_index_te = []
            for j in class_sample:  # 选中的两类数据取出
                index = np.random.randint(0, len(classified_tr[j]))
                sample_index_tr += classified_tr[j].pop(index)
                sample_index_te += classified_te[j].pop(index)
                if not len(classified_tr[j]):  # 判断该类数据是否被取完
                    class_index.remove(j)
                # 只剩一类时，需再取一遍
                if len(class_sample) < round(class_fraction):
                    class_sample.append(j)

            a += len(sample_index_tr)
            b += len(sample_index_te)

            x_train = []
            y_train = []
            x_test = []
            y_test = []
            for j in sample_index_tr:
                x_train.append(self.x_train[j])
                y_train.append(self.y_train[j])
            for j in sample_index_te:
                x_test.append(self.x_test[j])
                y_test.append(self.y_test[j])
            client.append({"x_train": np.array(x_train), "y_train": np.array(y_train), "x_test": np.array(x_test),
                           "y_test": np.array(y_test)})

        with open("../datasets/NonIID/" + self.dataset + '/train_split.pkl', 'wb') as outfile:
            pkl.dump(client, outfile)
        with open("../datasets/NonIID/" + self.dataset + '/test.pkl', 'wb') as outfile:
            pkl.dump((self.x_test, self.y_test), outfile)
        print(len(self.x_train), len(self.x_test), a, b)


if __name__ == '__main__':
    with open('../config.json') as file:
        config = json.load(file)
    num = {"emnist_digital": [2400, 400], "emnist_letter": [1248, 208], "cifar10": [500,100], "mnist": [600, 100], "fashion_mnist": [600, 100]}

    for v in num.keys():
        db = v
        sample_on_client_train = num[db][0]  # 每个客户端拥有的样本数
        sample_on_client_test = num[db][1]

        total_clients = 100  # total clients
        n_class = 10
        fraction = 2  # (2 or 5) 非独立同分布的分数
        if db == "emnist_letter":
            n_class = 26
            fraction = 5

        database = Database(db)
        with open("../datasets/AllData/" + database.dataset + '.pkl', 'wb') as outfile:  # 未划分的数据保存起来
            pkl.dump((database.x_train, database.y_train, database.x_test, database.y_test), outfile)

        database.generate_IID(n_train=sample_on_client_train, n_test=sample_on_client_test,
                              total_clients=total_clients)  # 共100个客户端，每个客户端600条数据, 用到的参数n, total_clients, fraction
        database.generate_non_IID(class_fraction=fraction, n_train=sample_on_client_train, n_test=sample_on_client_test,
                                  n_class=n_class, total_clients=total_clients)
