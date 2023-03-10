"""
resnet
"""
from copy import deepcopy
import paddle
import paddle.nn as nn
import numpy as np
import random

from paddle import vision

# from thop import profile
import paddle.nn.functional as F

from flearn.models.base_model import BaseModel

from flearn.models.linear import DenseLinear

from flearn.models.conv2d import DenseConv2d
from flearn.utils.model_utils import is_fc, is_conv

# __all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "resnext50_32x4d", "resnext101_32x8d",
#            "wide_resnet50_2", "wide_resnet101_2", "ResNet18"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return DenseConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, use_bias=True, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return DenseConv2d(in_planes, out_planes, kernel_size=1, stride=stride, use_bias=True)


def conv1x1_no_prune(in_planes, out_planes, stride=1):
    """1x1 convolution, no pruning"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=True)


class BasicBlock(nn.Layer):
    """
    Basic block
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU()

    def forward(self, x):
        """
        forward
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Layer):
    """
    bottle neck
    """
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while conventional implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        forward
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(BaseModel):
    """
    ResNet
    """
    def __init__(self, in_channel = 3, dict_module = None, block = BasicBlock, layers = (2, 2, 2, 2),
                 num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group=64,
                 replace_stride_with_dilation = None,
                 norm_layer = None):
        self.in_channel = in_channel
        self.num_classes = num_classes
        new_arch = dict_module is None
        if new_arch:
            dict_module = dict()
            if norm_layer is None:
                norm_layer = nn.BatchNorm2D
            self._norm_layer = norm_layer

            self.inplanes = 64
            self.dilation = 1
            if replace_stride_with_dilation is None:
                # each element in the tuple indicates if we should replace
                # the 2x2 stride with a dilated convolution instead
                replace_stride_with_dilation = [False, False, False]
            if len(replace_stride_with_dilation) != 3:
                raise ValueError("replace_stride_with_dilation should be None "
                                 "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
            self.groups = groups
            self.base_width = width_per_group
            dict_module["conv1"] = nn.Sequential(DenseConv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                                               use_bias=True))
            dict_module["bn1"] = norm_layer(self.inplanes)
            dict_module["relu"] = nn.ReLU()
            # dict_module["maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            dict_module["layer1"] = self._make_layer(block, 64, layers[0])
            dict_module["layer2"] = self._make_layer(block, 128, layers[1], stride=2,
                                                     dilate=replace_stride_with_dilation[0])
            dict_module["layer3"] = self._make_layer(block, 256, layers[2], stride=2,
                                                     dilate=replace_stride_with_dilation[1])
            dict_module["layer4"] = self._make_layer(block, 512, layers[3], stride=2,
                                                     dilate=replace_stride_with_dilation[2])
            # dict_module["avgpool"] = nn.AdaptiveAvgPool2d((1, 1))
            dict_module["classifier"] = DenseLinear(512 * block.expansion, num_classes)

            self.dict_module = dict_module

        super(ResNet, self).__init__(nn.CrossEntropyLoss(), dict_module)

        # if new_arch:
        #     self.reset_parameters(zero_init_residual)

    def reset_parameters(self, zero_init_residual):
        """
        resnet parameters
        """
        for m in self.modules():
            if isinstance(m, DenseConv2d) or isinstance(m, nn.Conv2D):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2D, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        make layer
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_no_prune(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """
        forward impl
        """
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.avg_pool2d(x, 4)
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.classifier(x)

        return x

    def forward(self, x):
        """
        forward
        """
        return self._forward_impl(x)

    def collect_layers(self):
        """
        collect layers
        """
        self.get_param_layers(self.param_layers, self.param_layer_prefixes)
        self.prunable_layers = [layer for layer in self.param_layers if is_conv(layer) or is_fc(layer)]
        self.prunable_layer_prefixes = [pfx for ly, pfx in zip(self.param_layers, self.param_layer_prefixes) if
                                        is_conv(ly) or is_fc(ly)]
        self.relu_layers = [m for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]
        self.relu_layers_prefixes = [k for (k, m) in self.named_sublayers() if isinstance(m, nn.ReLU)]

    @staticmethod
    def _block_to_sparse(block):
        """
        block to sparse
        """
        assert isinstance(block, BasicBlock) or isinstance(block, Bottleneck)
        new_block = deepcopy(block)
        new_block.conv1 = block.conv1.to_sparse()
        new_block.conv2 = block.conv2.to_sparse()
        if isinstance(block, Bottleneck):
            new_block.conv3 = block.conv3.to_sparse()
        return new_block

    def to_sparse(self):
        """
        to sparse
        """
        new_dict = {}
        for key, module in self.dict_module.items():
            if hasattr(module, "to_sparse"):
                new_dict[key] = module.to_sparse()
                if isinstance(module, DenseLinear):
                    new_dict[key].transpose = True
            elif isinstance(module, nn.Sequential):
                blocks = [self._block_to_sparse(block) for block in module]
                new_dict[key] = nn.Sequential(*blocks)
            else:
                new_dict[key] = deepcopy(module)
        return self.__class__(new_dict)

    def get_pt_model(self):
        """
        get pt model
        """
        from flearn.models.tmodels.resnet import resnet18
        model = resnet18(self.in_channel, self.num_classes)
        model.generate_from_pd_model(self.state_dict())
        return model


def _resnet(block, layers, num_classes, inchannel=3, **kwargs):
    """
    resnet
    """
    model = ResNet(in_channel=inchannel, dict_module=None, block=block, layers=layers, num_classes=num_classes,
                   **kwargs)
    return model


def resnet18(in_channel = 3, num_classes = 10):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, inchannel=in_channel)


def resnet34(num_classes=1000):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    """
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=1000):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    """
    return _resnet(Bottleneck, [3, 8, 36, 3], num_classes)


def resnext50_32x4d(num_classes=1000):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs = {'groups': 32,
              'width_per_group': 4}
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def resnext101_32x8d(num_classes=1000):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    """
    kwargs = {'groups': 32,
              'width_per_group': 8}
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


def wide_resnet50_2(num_classes=1000):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs = {'width_per_group': 64 * 2}
    return _resnet(Bottleneck, [3, 4, 6, 3], num_classes, **kwargs)


def wide_resnet101_2(num_classes=1000):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    """
    kwargs = {'width_per_group': 64 * 2}
    return _resnet(Bottleneck, [3, 4, 23, 3], num_classes, **kwargs)


class ResidualBlock(nn.Layer):
    """
    Residual block
    """
    def __init__(self, inchannel, outchannel1, outchannel2, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2D(inchannel, outchannel1, kernel_size=3, stride=stride, padding=1, bias_attr=True),
            nn.BatchNorm2D(outchannel1),
            nn.ReLU(),
            nn.Conv2D(outchannel1, outchannel2, kernel_size=3, stride=1, padding=1, bias_attr=True),
            nn.BatchNorm2D(outchannel2)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel2:
            self.shortcut = nn.Sequential(
                nn.Conv2D(inchannel, outchannel2, kernel_size=1, stride=stride, bias_attr=True),
                nn.BatchNorm2D(outchannel2)
            )

    def forward(self, x):
        """
        forward
        """
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet2(nn.Layer):
    """
    ResNet2
    """
    def __init__(self, ResidualBlock, in_channel=3, num_classes=10, config=None):
        super(ResNet2, self).__init__()
        if config is None:
            config = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
        self.inchannel = config[0]
        self.conv1 = nn.Sequential(
            nn.Conv2D(in_channel, config[0], kernel_size=3, stride=1, padding=1, bias_attr=True),
            nn.BatchNorm2D(config[0]),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, [config[0], config[1], config[2], config[3], config[4]], 2,
                                      stride=1)
        self.layer2 = self.make_layer(ResidualBlock, [config[4], config[5], config[6], config[7], config[8]], 2,
                                      stride=2)
        self.layer3 = self.make_layer(ResidualBlock, [config[8], config[9], config[10], config[11], config[12]], 2,
                                      stride=2)
        self.layer4 = self.make_layer(ResidualBlock, [config[12], config[13], config[14], config[15], config[16]], 2,
                                      stride=2)
        self.fc = nn.Linear(config[16], num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        """
        make layer
        """
        strides = [stride] + [1] * (num_blocks - 1)  # strides=[1,1]
        layers = []
        # for stride in strides:
        #     layers.append(block(self.inchannel, channels, stride))
        #     self.inchannel = channels
        # 让channels变成一个list
        # (self, inchannel, outchannel1, outchannel2, stride=1):
        layers.append(block(channels[0], channels[1], channels[2], strides[0]))
        layers.append(block(channels[2], channels[3], channels[4], strides[1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        forward
        """
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = paddle.reshape(out, [out.shape[0], -1])
        out = self.fc(out)
        return out


def ResNetOrigin(in_channel=3, num_classes=10, config=None):
    """
    ResNet Origin
    """
    return ResNet2(ResidualBlock, in_channel=in_channel, num_classes=num_classes, config=config)


if __name__ == "__main__":
    # 设置随机种子
    paddle.seed(777)
    np.random.seed(777)
    random.seed(777)

    model = ResNetOrigin()
    sd = model.state_dict()
    model = resnet18(in_channel=3, num_classes=10)
    inputs = paddle.ones((64, 3, 32, 32))
    outputs = model(inputs)
    # print(f"cifar10: {outputs.shape}")
    # import torchstat
    # torchstat.stat(model, (3, 32, 32))
    model = resnet18(in_channel=3, num_classes=100)
    inputs = paddle.ones((64, 3, 32, 32))
    outputs = model(inputs)
    # print(f"cifar100: {outputs.shape}")
    model = resnet18(in_channel=1, num_classes=10)
    inputs = paddle.ones((64, 1, 32, 32))
    outputs = model(inputs)
    # print(f"mnist: {outputs.shape}")

    rank = [0, paddle.linspace(1, 64, num=64)]
    _, ind = model.unstructured_by_rank(rank, 0.6, 1, "cpu")

    model = ResNetOrigin(in_channel=3, num_classes=10)
    print(model)
    inputs = paddle.ones((1, 3, 32, 32))
    # flops, params = profile(model, inputs=(inputs,))
    # print(f"original model flops: {flops} params: {params}")
    # torchstat.stat(model, (3, 32, 32))
    # prune_channels = [26, 26, 26, 26, 26, 52, 52, 52, 52, 103, 103, 103, 103, 205, 205, 205, 205]
    # model = ResNetOrigin(in_channel=3, num_classes=10, config=prune_channels)
    # flops, params = profile(model, inputs=(inputs,))
    # print(f"pruned model flops: {flops} params: {params}")
    # model = torchvision.models.resnet18()
    # flops, params = profile(model, inputs=(inputs,))
    # print(f"torch model flops: {flops} params: {params}")

