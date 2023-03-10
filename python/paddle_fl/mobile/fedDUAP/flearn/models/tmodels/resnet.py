"""
resnet
"""
import paddle
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections


__all__ = ["resnet18"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                       padding=dilation, groups=groups, bias=True, dilation=dilation)


def conv1x1_no_prune(in_planes, out_planes, stride=1):
    """1x1 convolution, no pruning"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=True)


class BasicBlock(nn.Module):
    """
    basic block
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)

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



class ResNet(nn.Module):
    """
    ResNet
    """
    def __init__(self, in_channel = 3, dict_module = None, block = BasicBlock, layers = (2, 2, 2, 2),
                 num_classes = 1000, zero_init_residual = False, groups = 1, width_per_group = 64,
                 replace_stride_with_dilation = None,
                 norm_layer = None):
        super(ResNet, self).__init__()

        dict_module = dict()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
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
        dict_module["conv1"] = nn.Sequential(nn.Conv2d(in_channel, self.inplanes, kernel_size=3, stride=1, padding=1,
                                           bias=True))
        dict_module["bn1"] = norm_layer(self.inplanes)
        dict_module["relu"] = nn.ReLU(inplace=True)
        # dict_module["maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        dict_module["layer1"] = self._make_layer(block, 64, layers[0])
        dict_module["layer2"] = self._make_layer(block, 128, layers[1], stride=2,
                                                 dilate=replace_stride_with_dilation[0])
        dict_module["layer3"] = self._make_layer(block, 256, layers[2], stride=2,
                                                 dilate=replace_stride_with_dilation[1])
        dict_module["layer4"] = self._make_layer(block, 512, layers[3], stride=2,
                                                 dilate=replace_stride_with_dilation[2])
        # dict_module["avgpool"] = nn.AdaptiveAvgPool2d((1, 1))
        dict_module["classifier"] = nn.Linear(512 * block.expansion, num_classes)

        self.dict_module = dict_module

        for module_name, module in dict_module.items():
            self.add_module(module_name, module)

        self.loss_func = nn.CrossEntropyLoss()

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
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

    def forward(self, x):
        """
        forward
        """
        return self._forward_impl(x)

    def generate_from_pd_model(self, state_pd):
        """
        generate from pd model
        """
        state_pd_to_pt = collections.OrderedDict()
        for k, v in state_pd.items():
            if "mean" in k:
                new_k = k.replace("_mean", "running_mean")
                state_pd_to_pt[new_k] = torch.tensor(v.clone().numpy())
                continue
            if "variance" in k:
                new_k = k.replace("_variance", "running_var")
                state_pd_to_pt[new_k] = torch.tensor(v.clone().numpy())
                next_k = k.replace("_variance", "num_batches_tracked")
                state_pd_to_pt[next_k] = torch.tensor(0)
                continue

            state_pd_to_pt[k] = torch.tensor(v.clone().numpy())
            if 'classifier' in k and 'weight' in k:
                state_pd_to_pt[k] = state_pd_to_pt[k].T
        self.load_state_dict(state_pd_to_pt)

def _resnet(block, layers, num_classes, inchannel=3, **kwargs):
    """
    resnet
    """
    model = ResNet(in_channel=inchannel, dict_module=None, block=block, layers=layers, num_classes=num_classes,
                   **kwargs)
    return model

def resnet18(in_channel = 3, num_classes = 10):
    """
    resnet 18
    """
    return _resnet(BasicBlock, [2, 2, 2, 2], num_classes, inchannel = in_channel)

if __name__ == "__main__":
    paddle.set_device("cpu")
    from flearn.models.resnet import resnet18 as ResNet_pd
    import collections
    model_pd = ResNet_pd()
    model_pt = resnet18()

    state_pd = model_pd.state_dict()

    state_pd_to_pt = collections.OrderedDict()
    for k, v in state_pd.items():
        if "mean" in k:
            new_k = k.replace("_mean", "running_mean")
            state_pd_to_pt[new_k] = torch.tensor(v.clone().numpy())
            continue
        if "variance" in k:
            new_k = k.replace("_variance", "running_var")
            state_pd_to_pt[new_k] = torch.tensor(v.clone().numpy())
            next_k = k.replace("_variance", "num_batches_tracked")
            state_pd_to_pt[next_k] = torch.tensor(0)
            continue

        state_pd_to_pt[k] = torch.tensor(v.clone().numpy())
        if 'classifier' in k and 'weight' in k:
            state_pd_to_pt[k] = state_pd_to_pt[k].T
    state_pt = model_pt.state_dict()
    model_pt.load_state_dict(state_pd_to_pt)



