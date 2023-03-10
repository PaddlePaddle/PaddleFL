"""
Lenet
"""
import paddle
import paddle.nn as nn
import numpy as np
import random
from flearn.models.base_model import BaseModel
from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear
from flearn.utils.model_utils import is_fc, is_conv
import paddle.nn.functional as F


class LENET(BaseModel):
    """
    LeNet
    """
    def __init__(self, in_channel = 3, dict_module = None, use_mask = True, config = None, use_batchnorm = False,
                 num_classes = 10):
        self.in_channel = in_channel
        self.num_classes = num_classes
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()

            features = nn.Sequential(
                DenseConv2d(in_channels=in_channel, out_channels=6, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=2, stride=2),

                DenseConv2d(in_channels=6, out_channels=16, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2D(kernel_size=2, stride=2)
            )
            classifier = nn.Sequential(
                DenseLinear(16 * 5 * 5, 120, use_mask=True),
                nn.ReLU(),
                DenseLinear(120, 84, use_mask=True),
                nn.ReLU(),
                DenseLinear(84, num_classes, use_mask=True),
                nn.ReLU()
            )

            dict_module["features"] = features
            dict_module["classifier"] = classifier
        super(LENET, self).__init__(nn.CrossEntropyLoss(), dict_module)

    def collect_layers(self):
        """
        Collect layers
        """
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        self.prunable_layers = [layer for layer in self.param_layers if is_conv(layer) or is_fc(layer)]
        self.prunable_layer_prefixes = [pfx for ly, pfx in zip(self.param_layers, self.param_layer_prefixes) if
                                        is_conv(ly) or is_fc(ly)]
        self.relu_layers = [m for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]
        self.relu_layers_prefixes = [k for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]

        # print(self.relu_layers, self.relu_layers_prefixes)

    def forward(self, x, mask=None):
        """
        forward
        """
        # if mask is not None:
        #     self._apply_mask(mask)
        x = self.features(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.classifier(x)
        return x

    def get_pt_model(self):
        """
        get pt model
        """
        from flearn.models.tmodels.lenet import LENETOrigin
        model = LENETOrigin(self.in_channel, self.num_classes)
        model.generate_from_pd_model(self.state_dict())
        return model

class LENETOrigin(nn.Layer):
    """
    LeNet Origin
    """
    def __init__(self, in_channel=3, num_classes=10, config=None):
        super(LENETOrigin, self).__init__()
        if config is None:
            config = [6, 16]
        self.conv1 = nn.Conv2D(in_channel, config[0], 5)
        self.pool = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(config[0], config[1], 5)
        self.fc1 = nn.Linear(config[1] * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        """
        forward
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = paddle.reshape(x, [x.shape[0], -1])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

if __name__ == "__main__":
    # 设置随机种子
    paddle.seed(777)
    np.random.seed(777)
    random.seed(777)

    model = LENET(in_channel=3, num_classes=10)
    inputs = paddle.ones((1, 3, 32, 32))
    outputs = model(inputs)
    print(outputs.shape)

    rank = [0, paddle.linspace(1, 16, num=16)]
    _, ind = model.unstructured_by_rank(rank, 0.6, 1, "cpu")

    outputs = model(inputs)
    channels = model.get_channels()
    print(channels)
    print("#################")