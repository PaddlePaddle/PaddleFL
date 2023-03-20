import paddle
import paddle.nn as nn
from flearn.models.conv2d import DenseConv2d
from flearn.models.linear import DenseLinear
from flearn.models.base_model import BaseModel
from paddle.io import DataLoader
import numpy as np
import random
# from thop import profile
# from data.cifar100.cifar100_data import get_dataset
from data.cifar10.cifar10_data import get_dataset
# from flearn.optim.prox import Prox
from flearn.utils.model_utils import test_inference
import math

class VGG11(BaseModel):
    def __init__(self, in_channel=3, num_classes=10, dict_module: dict = None, use_mask=True, config=None, use_batchnorm=False):
        self.in_channel = in_channel
        self.num_classes = num_classes
        if dict_module is None:
            self.use_mask = use_mask
            dict_module = dict()
            self.batch_norm = use_batchnorm

            if config is None:
                self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
            else:
                self.config = config

            features = self._make_feature_layers()
            classifier = nn.Sequential(
                DenseLinear(self.config[-2], 512, use_mask=use_mask),
                nn.ReLU(),
                DenseLinear(512, 512, use_mask=use_mask),
                nn.ReLU(),
                DenseLinear(512, num_classes, use_mask=use_mask)
            )

            dict_module["features"] = features
            dict_module["classifier"] = classifier

        super(VGG11, self).__init__(nn.CrossEntropyLoss(), dict_module)

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
        for name, param in self.named_parameters():
            param.data *= mask[name]

    def collect_layers(self):
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        prunable_nums = [ly_id for ly_id, ly in enumerate(self.param_layers)]
        self.prunable_layers = [self.param_layers[ly_id] for ly_id in prunable_nums]
        self.prunable_layer_prefixes = [self.param_layer_prefixes[ly_id] for ly_id in prunable_nums]
        self.relu_layers = [m for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]
        self.relu_layers_prefixes = [k for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]

        # print(self.relu_layers, self.relu_layers_prefixes)

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2D(kernel_size=2))
            else:
                layers.extend([DenseConv2d(in_channels, param, kernel_size=3, padding=1, use_mask=self.use_mask),
                               nn.ReLU()])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x, mask=None):
        # if mask is not None:
        #     self._apply_mask(mask)
        x = self.features(x)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.classifier(x)
        return x

    def get_pt_model(self):
        from flearn.models.tmodels.vgg import VGGOrigin
        model = VGGOrigin(self.in_channel, self.num_classes, self.config)
        model.generate_from_pd_model(self.state_dict())
        return model

class VGGOrigin(nn.Layer):
    def __init__(self, in_channel=3, num_classes=10, config=None, use_batchnorm=False):
        super(VGGOrigin, self).__init__()
        self.in_channel = in_channel
        self.batch_norm = use_batchnorm

        if config is None:
            self.config = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']
        else:
            self.config = config

        self.features = self._make_feature_layers()
        self.classifier = nn.Sequential(
            nn.Linear(self.config[-2], 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )

    def _make_feature_layers(self):
        layers = []
        in_channels = self.in_channel
        for param in self.config:
            if param == 'M':
                layers.append(nn.MaxPool2D(kernel_size=2))
            else:
                layers.extend([nn.Conv2D(in_channels, param, kernel_size=3, padding=1),
                               nn.ReLU()])
                in_channels = param

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), -1)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.classifier(x)
        return x


if __name__=="__main__":
    # 设置随机种子
    paddle.seed(777)
    np.random.seed(777)
    random.seed(777)

    if "gpu" in paddle.device.get_device():
        paddle.device.set_device('gpu')

    model = VGG11()

    train_dataset, test_dataset, user_groups = get_dataset(num_data=40000, num_users=100, iid=False, num_share=4000, l=2,
                                                          unequal=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=100, shuffle=True, num_workers=0, drop_last=False)

    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    learning_rate = 1e-1
    optimizer = paddle.optimizer.SGD(learning_rate=learning_rate, parameters=model.parameters())

    # 设置网络的训练参数
    total_train_step = 0
    total_test_step = 0
    epoch = 10

    model.train()
    # 开始训练
    for i in range(epoch):
        print(f"=== epoch {i} start ===")

        avg_loss = 0
        for data in train_loader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)

            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            total_train_step += 1
            if total_train_step % 10 == 0:
                model.recovery_model()
                print(loss)

        test_acc, test_loss = test_inference(model, test_dataset)
        print(test_acc)
