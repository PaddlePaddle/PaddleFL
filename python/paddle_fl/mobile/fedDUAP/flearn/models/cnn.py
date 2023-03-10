"""
cnn
"""
from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear
from flearn.models.base_model import BaseModel
import numpy as np
# import torchvision
import random
from flearn.utils.model_utils import is_conv
# import torchstat
import math
from paddle import nn, vision
from paddle.vision import transforms
import paddle

class CNN(BaseModel):
    """
    CNN
    """
    def __init__(self, in_channel = 3, num_classes = 10, dict_module = None, use_mask = True, config = None,
                 use_batchnorm = False):
        self.in_channel = in_channel
        self.num_classes = num_classes
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()
            self.batch_norm = use_batchnorm

            if config is None:
                self.config = [32, 'M', 64, 'M', 64]
            else:
                self.config = config

            features = self._make_feature_layers()
            classifier = nn.Sequential(
                DenseLinear(4 * 4 * self.config[-1], 64, use_mask=True),
                nn.ReLU(),
                DenseLinear(64, num_classes, use_mask=True)
            )

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(CNN, self).__init__(nn.CrossEntropyLoss(), dict_module)
        for k, m in self.named_sublayers():
            if isinstance(m, nn.Conv2D):
                n = m._kernel_size[0] * m._kernel_size[1] * m.out_channels
                data = m.weight.clone()
                data = paddle.normal(0, math.sqrt(2. / n), data.shape)
                m.weight.set_value(data)
                bias_data = m.bias.clone()
                bias_data.zero_()
                m.bias.set_value(bias_data)

    def _apply_mask(self, mask):
        """
        apply mask
        """
        for name, param in self.named_parameters():
            param.data *= mask[name]

    def collect_layers(self):
        """
        collect layers
        """
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_nums = [ly_id for ly_id, ly in enumerate(self.param_layers)]
        self.prunable_layers = [self.param_layers[ly_id] for ly_id in prunable_nums]
        self.prunable_layer_prefixes = [self.param_layer_prefixes[ly_id] for ly_id in prunable_nums]
        self.relu_layers = [m for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]
        self.relu_layers_prefixes = [k for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]

        # print(self.relu_layers, self.relu_layers_prefixes)

    def _make_feature_layers(self):
        """
        make feature layers
        """
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2D(kernel_size=2))
            else:
                layers.extend([DenseConv2d(in_channels, param, kernel_size=3, padding=0, use_mask=self.use_mask),
                               nn.ReLU()])
                in_channels = param

        return nn.Sequential(*layers)

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
        from flearn.models.tmodels.cnn import CNNOrigin
        model = CNNOrigin(self.in_channel, self.num_classes, self.config)
        model.generate_from_pd_model(self.state_dict())
        return model


class CNNOrigin(nn.Layer):
    """
    CNNOrigin
    """
    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
        super(CNNOrigin, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if config is None:
            self.config = [32, 'M', 64, 'M', 64]
        else:
            self.config = config

        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * self.config[-1], 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def _make_feature_layers(self):
        """
        make feature layers
        """
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2D(kernel_size=2))
            else:
                layers.extend([nn.Conv2D(in_channels, param, kernel_size=3, padding=0),
                               nn.ReLU()])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward
        """
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.classifier(x)
        return x



if __name__ == "__main__":
    a = [3, 2, 1, 5, 4]
    b = np.argsort(a)
    print(a)
    print(b)
    # 设置随机种子
    paddle.seed(777)
    np.random.seed(777)
    random.seed(777)

    a = nn.Linear(1024, 64)

    model = CNN()
    # torchstat.stat(model, (3, 32, 32))
    # FLOPs = paddle.flops(lenet, [1, 1, 28, 28], custom_ops={nn.LeakyReLU: count_leaky_relu},
    #                      print_detail=True)
    # print(FLOPs)

    print(model.get_channels())

    # # torchstat.stat(model, (3, 32, 32))
    inputs = paddle.ones((64, 3, 32, 32))
    outputs = model(inputs)
    print(outputs.shape)
    c = model.features[0].weight
    b = model.features[0]
    #
    # a = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=1)
    # a.weight.data = torch.index_select(a.weight.data, 0, torch.tensor([0, 2]))
    #
    # for k, v in a.named_parameters():
    #     print(k)
    #     print(v)
    #
    for idx, layer in enumerate(model.prunable_layers):
        print("{}: isConv {} size {}".format(idx, is_conv(layer), layer.num_weight))
    #
    # # print(a.mask is None)
    # a.mask = torch.ones((64, 3, 3, 3))
    # print(a.mask is None)
    # print(c)
    # f = [[[1, 2, 3, 4, 5],
    #      [6, 7, 8, 9, 10],
    #      [1, 2, 3, 4, 5],
    #      [1, 2, 3, 4, 5],
    #      [1, 2, 3, 4, 5]],
    #      [[1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5]],
    #      [[1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5],
    #       [1, 2, 3, 4, 5]]
    #      ]
    # e = np.array(f)
    # g = torch.index_select(torch.tensor(e), 1, torch.tensor([1, 2]))
    # print(g)
    # sub_mask = torch.tensor([[3, 2, 1, 5, 5],
    #       [1, 2, 3, 4, 5]])
    # g[2, :] = sub_mask
    # print(g)
    # sub_mask[0, 0] = 1
    # print(g)

    import copy
    params = copy.deepcopy(model.state_dict())
    for key in params.keys():
        params[key] = paddle.divide(params[key], paddle.to_tensor(2.0))
    model.load_dict(params)
    print(1)

