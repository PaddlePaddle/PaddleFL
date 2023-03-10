"""
update
"""
import paddle
import paddle.nn as nn
from paddle.io import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return paddle.to_tensor(image), paddle.to_tensor(label)


class LocalUpdate(object):
    """
    Local update
    """
    def __init__(self, args, dataset, idxs, device, local_bs=None, local_ep=None, logger=None):
        self.args = args
        self.local_bs = local_bs if local_bs is not None else args.local_bs
        self.local_ep = local_ep if local_ep is not None else args.local_ep
        self.logger = logger
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.testloader = self.trainloader
        self.device = device
        # Default criterion set to NLL loss function
        self.criterion = nn.CrossEntropyLoss()

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)
        idxs_train = idxs
        # idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        # idxs_test = idxs[int(0.9*len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.local_bs, shuffle=True)
        # validloader = DataLoader(DatasetSplit(dataset, idxs_val),
        #                          batch_size=int(len(idxs_val)/10), shuffle=False)
        # testloader = DataLoader(DatasetSplit(dataset, idxs_test),
        #                         batch_size=int(len(idxs_test)/10), shuffle=False)
        return trainloader

    def update_weights(self, model, global_round, mu=0, v={}):
        """
        update weights
        :param model:
        :param global_round:
        :param mu:
        :param v:
        :return:
        """
        # Set mode to train model
        model.train()
        epoch_loss = []

        # Set optim for the local updates
        if self.args.optim == 'sgd':
            optimizer = paddle.optimizer.SGD(learning_rate=self.args.lr, parameters=model.parameters())


        for iter in range(self.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                optimizer.clear_grad()
                log_probs = model(images)
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                                            100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        return model.state_dict(), epoch_loss[-1]

    def inference(self, model):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        idx = 0
        for batch_idx, (images, labels) in enumerate(self.testloader):

            # Inference
            outputs = model(images)
            labels = paddle.reshape(labels, [labels.shape[0], -1])
            batch_loss = self.criterion(outputs, labels)
            loss += batch_loss.item()

            # Prediction
            acc = paddle.metric.accuracy(outputs, labels).item()
            correct += acc * images.shape[0]
            total += images.shape[0]
            idx += 1

        accuracy = correct / total
        return accuracy, loss / idx
