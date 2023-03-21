import copy
import paddle
import paddle.nn as nn
from paddle.io import DataLoader

from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear

def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0][1])
    total = 0
    for i in range(0, len(w)):
        total += w[i][0]
    for key in w_avg.keys():
        w_avg[key] *= w[0][0]
        for i in range(1, len(w)):
            w_avg[key] += w[i][1][key] * w[i][0]
        w_avg[key] = paddle.divide(w_avg[key], paddle.to_tensor(total * 1.0))
    return w_avg

def ratio_combine(w1, w2, ratio=0):
    """
    将两个权重进行加权平均，ratio 表示 w2 的占比
    :param w1:
    :param w2:
    :param ratio:
    :return:
    """
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] = (w2[key] - w1[key]) * ratio + w1[key]
    return w

def ratio_minus(w1, P, ratio=0):
    w = copy.deepcopy(w1)
    for key in w.keys():
        w[key] = w1[key] - P[key] * ratio
    return w

def is_fc(layer):
    return isinstance(layer, DenseLinear)

def is_conv(layer):
    return isinstance(layer, DenseConv2d)

def is_bn(layer):
    return isinstance(layer, nn.BatchNorm2D)

def is_norm_conv(layer):
    return isinstance(layer, nn.Conv2D)


def traverse_module(module, criterion, layers: list, names: list, prefix="", leaf_only=True):
    if leaf_only:
        for key, submodule in module._sub_layers.items():
            new_prefix = prefix
            if prefix != "":
                new_prefix += '.'
            new_prefix += key
            # is leaf and satisfies criterion
            if len(submodule._sub_layers.keys()) == 0 and criterion(submodule):
                layers.append(submodule)
                names.append(new_prefix)
            traverse_module(submodule, criterion, layers, names, prefix=new_prefix, leaf_only=leaf_only)
    else:
        raise NotImplementedError("Supports only leaf modules")


def test_inference(model, test_dataset):
    """ Returns the test accuracy and loss.
    """
    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    criterion = nn.CrossEntropyLoss()
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        # Inference
        outputs = model(images)
        labels = paddle.reshape(labels, [labels.shape[0], -1])
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        acc = paddle.metric.accuracy(outputs, labels).item()
        correct += acc * images.shape[0]
        total += images.shape[0]

    accuracy = correct/total
    return round(accuracy, 4), round(loss / (len(testloader)), 4)