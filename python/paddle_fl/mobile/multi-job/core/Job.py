import datetime
import logging
import json
import pickle as pkl
import paddle
import paddle.fluid as fluid
from core.FedAvg import FedAvg
from core.model import VGG, CnnIID, CnnNIID, LeNet, AlexNet, ResNet, CNNiid, CNNniid
from core.scheduler import Common, Greedy, Bayesian


class AllClientDataset(paddle.io.Dataset):
    def __init__(self, data_name, dir, mode):
        super(AllClientDataset, self).__init__()
        with open("datasets/" + dir + "/" + data_name + "/train_split.pkl", "rb") as f:
            train_data = pkl.load(f)
        with open("datasets/" + dir + "/" + data_name + "/test.pkl", "rb") as f:
            test_data = pkl.load(f)
        if mode == 'train':
            self.dataset = []
            for client in range(100):
                for sample in range(len(train_data[client]['x_train'])):
                    data = []
                    data.append(train_data[client]['x_train'][sample])
                    data.append(train_data[client]['y_train'][sample])
                    self.dataset.append(data)
        else:
            self.dataset = []
            for client in range(100):
                for sample in range(len(train_data[client]['x_test'])):
                    data = []
                    data.append(train_data[client]['x_test'][sample])
                    data.append(train_data[client]['y_test'][sample])
                    self.dataset.append(data)

    def __getitem__(self, index):
        data = self.dataset[index][0]
        label = self.dataset[index][1]
        return data, label

    def __len__(self):
        return len(self.dataset)

class ClientDataset(paddle.io.Dataset):
    def __init__(self, data_name, dir, client, mode='train'):
        super(ClientDataset, self).__init__()
        with open("datasets/" + dir + "/" + data_name + "/train_split.pkl", "rb") as f:
            train_data = pkl.load(f)
        with open("datasets/" + dir + "/" + data_name + "/test.pkl", "rb") as f:
            test_data = pkl.load(f)
        if mode == 'train':
            self.dataset = []
            for sample in range(len(train_data[client]['x_train'])):
                data = []
                data.append(train_data[client]['x_train'][sample])
                # print(len(data))
                data.append(train_data[client]['y_train'][sample])
                # print(len(data))
                self.dataset.append(data)
        else:
            self.dataset = []
            for sample in range(len(test_data[client]['x_test'])):
                data = []
                data.append(test_data[client]['x_test'][sample])
                data.append(test_data[client]['y_test'][sample])
                self.dataset.append(data)
    def __getitem__(self, index):
        data = self.dataset[index][0]
        label = self.dataset[index][1]
        return data, label

    def __len__(self):
        return len(self.dataset)

def start(Scheduler, model, config, job, data_name, dir):
    if config["isIID"]:
        SD = "IID"
    else:
        SD = "NIID"

    with open("simulation/" + Scheduler + "/" + SD + "/" + job + "_rounds_client" + config['group'] + ".json", "r") as f:
        rounds_client = json.load(f)
    t_begin = datetime.datetime.now()

    filehandler = logging.FileHandler(
        "results/" + Scheduler + "/" + "client_time_" + config["group"] + "/" + SD + "/" + job + '.log')
    streamhandler = logging.StreamHandler()

    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    fedAvg = FedAvg(BATCH_SIZE=config[job]["batch_size"], EPOCH_NUM=config[job]["local_epochs"],
                    drop_r=config["drop_r"])
    # 加载模型
    model_avg = model
    MODEL = paddle.Model(model_avg)
    # global training
    for r in range(config[job]["rounds"]):  # 100round随机的客户端序列len（clients）==100
        t = datetime.datetime.now()
        # # 设备调度
        clients = rounds_client[r]
        print(r)
        for client in clients:
            train_set = ClientDataset(data_name=data_name, dir=dir, client=client, mode='train')
            fedAvg.client_train(client, train_set, MODEL, r)

        # global Aggregation
        global_weight = fedAvg.avg()
        tensor_global_weight = []
        for l in global_weight:
            tensor_weight = paddle.to_tensor(l)
            tensor_global_weight.append(tensor_weight)
        # tensor_global_weight = paddle.tensor.to_tensor(global_weight)
        name = model_avg.state_dict().keys()
        global_weight_dict = dict(zip(name, tensor_global_weight))
        model_avg.set_state_dict(global_weight_dict)

        # evaluate the model
        print(config["scheduler"], ":")
        all_train_set = AllClientDataset(data_name=data_name, dir=dir, mode='train')
        all_test_set = AllClientDataset(data_name=data_name, dir=dir, mode='test')
        tr_loss, tr_acc, ts_loss, ts_acc = fedAvg.fed_eval(MODEL, train_set=all_train_set, test_set=all_test_set)
        _time = datetime.datetime.now()

        logger.info('%s:: Epoch %d: ts_acc=%f ts_loss=%f, time=%.2f, delta_time=%.2f' %
                    (job, r, ts_acc, ts_loss, (_time - t_begin).seconds, (_time - t).seconds))
    logger.removeHandler(filehandler)
    logger.removeHandler(streamhandler)

def Image_CnnEm(config):
    """
    operater: 设备调度算法
    model: 任务模型
    config:模型配置
    """
    print("Loading image data...")
    data_name = 'emnist_letter'
    if config["isIID"]:
        dir = "IID"
        model = CnnIID()
    else:
        dir = "NonIID"
        model = CnnNIID()
    Scheduler = config["scheduler"]
    start(Scheduler, model, config, "CnnEm", data_name, dir)

def Image_CnnFM(config):
    """
    operater: 设备调度算法
    model: 任务模型
    config:模型配置
    """
    data_name = 'fashion_mnist'
    print("Loading image data...")
    if config["isIID"]:
        dir = "IID"
        model = CNNiid()
    else:
        dir = "NonIID"
        model = CNNniid()
    Scheduler = config["scheduler"]
    start(Scheduler, model, config, "CnnFM", data_name, dir)


def Image_VGG(config):
    print("Loading image data...")
    data_name = 'cifar10'
    if config["isIID"]:
        dir = "IID"
    else:
        dir = "NonIID"
    model = VGG()
    Scheduler = config["scheduler"]
    start(Scheduler, model, config, "VGG", data_name, dir)

def Image_LeNet(config):
    print("Loading image data...")
    data_name = 'emnist_digital'
    if config["isIID"]:
        dir = "IID"
    else:
        dir = "NonIID"
    model = LeNet()
    Scheduler = config["scheduler"]
    start(Scheduler, model, config, "Lenet", data_name, dir)

def Image_AlexNet(config):
    print("Loading image data...")
    data_name = 'mnist'
    if config["isIID"]:
        dir = "IID"
    else:
        dir = "NonIID"
    model = AlexNet()
    Scheduler = config["scheduler"]
    start(Scheduler, model, config, "Alexnet", data_name, dir)


def Image_ResNet(config):
    print("Loading image data...")
    data_name = 'cifar10'
    if config["isIID"]:
        dir = "IID"
    else:
        dir = "NonIID"
    model = ResNet()
    Scheduler = config["scheduler"]
    start(Scheduler, model, config, "Resnet", data_name, dir)