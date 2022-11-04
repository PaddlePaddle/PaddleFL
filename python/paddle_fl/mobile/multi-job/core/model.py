import paddle
import os
import paddle.nn.functional as F
import paddle.nn as nn
from paddle.nn import Sequential, Conv2D, MaxPool2D, AvgPool2D, BatchNorm, Dropout2D, Linear, Flatten, ReLU, Softmax,Dropout
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# cifar10
class VGG(nn.Layer):
    def __init__(self):
        super(VGG, self).__init__()

        in_channels = [3, 64, 128, 256, 512, 512]
        # 定义第一个卷积块，包含两个卷积
        self.conv1_1 = Conv2D(in_channels=in_channels[0], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        self.conv1_2 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[1], kernel_size=3, padding=1, stride=1)
        # 定义第二个卷积块，包含两个卷积
        self.conv2_1 = Conv2D(in_channels=in_channels[1], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        self.conv2_2 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[2], kernel_size=3, padding=1, stride=1)
        # 定义第三个卷积块，包含三个卷积
        self.conv3_1 = Conv2D(in_channels=in_channels[2], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_2 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        self.conv3_3 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[3], kernel_size=3, padding=1, stride=1)
        # 定义第四个卷积块，包含三个卷积
        self.conv4_1 = Conv2D(in_channels=in_channels[3], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_2 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        self.conv4_3 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[4], kernel_size=3, padding=1, stride=1)
        # 定义第五个卷积块，包含三个卷积
        self.conv5_1 = Conv2D(in_channels=in_channels[4], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_2 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)
        self.conv5_3 = Conv2D(in_channels=in_channels[5], out_channels=in_channels[5], kernel_size=3, padding=1, stride=1)

        # 使用Sequential 将全连接层和relu组成一个线性结构（fc + relu）
        # 当输入为224x224时，经过五个卷积块和池化层后，特征维度变为[512x7x7]
        self.fc1 = paddle.nn.Sequential(paddle.nn.Linear(512, 4096), paddle.nn.ReLU())
        self.drop1_ratio = 0.5
        self.dropout1 = paddle.nn.Dropout(self.drop1_ratio, mode='upscale_in_train')
        # 使用Sequential 将全连接层和relu组成一个线性结构（fc + relu）
        self.fc2 = paddle.nn.Sequential(paddle.nn.Linear(4096, 4096), paddle.nn.ReLU())

        self.drop2_ratio = 0.5
        self.dropout2 = paddle.nn.Dropout(self.drop2_ratio, mode='upscale_in_train')
        self.fc3 = paddle.nn.Linear(4096, 10)

        self.relu = paddle.nn.ReLU()
        self.pool = MaxPool2D(stride=2, kernel_size=2)

    def forward(self, x):
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.pool(x)

        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.pool(x)

        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.pool(x)

        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x = self.pool(x)

        x = self.relu(self.conv5_1(x))
        x = self.relu(self.conv5_2(x))
        x = self.relu(self.conv5_3(x))
        x = self.pool(x)

        x = paddle.flatten(x, 1, -1)
        x = self.dropout1(self.relu(self.fc1(x)))
        x = self.dropout2(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
# emnist_letter
class CnnIID(nn.Layer):
    def __init__(self):
        super(CnnIID, self).__init__()
        self.conv1 = Sequential(
            Conv2D(in_channels=1, out_channels=32, kernel_size=3),
            ReLU(),
            BatchNorm(num_channels=32),
            MaxPool2D(kernel_size=2)
        )
        self.conv2 = Sequential(
            Conv2D(in_channels=32, out_channels=64, kernel_size=3),
            ReLU(),
            BatchNorm(num_channels=64),
            MaxPool2D(kernel_size=2)
        )
        self.drop1 = Dropout2D(0.25)
        self.fc = Sequential(
            Flatten(),
            Linear(in_features=1600, out_features=1568),
            Dropout(0.5),
            Linear(in_features=1568, out_features=784),
            BatchNorm(num_channels=784),
            Linear(in_features=784, out_features=26),
            Softmax()
        )

    def forward(self, x):
        # x = fluid.layers.reshape(x, (-1, 1, 28, 28))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop1(x)
        x = self.fc(x)
        return x


class CnnNIID(nn.Layer):
    def __init__(self):
        super(CnnNIID, self).__init__()
        self.conv1 = Sequential(
            Conv2D(in_channels=1, out_channels=32, kernel_size=3),
            ReLU(),
            MaxPool2D(kernel_size=2))
        self.conv2 = Sequential(
            Conv2D(in_channels=32, out_channels=64, kernel_size=3),
            ReLU(),
            MaxPool2D(kernel_size=2))
        self.conv3 = Sequential(
            Conv2D(in_channels=64, out_channels=64, kernel_size=3),
            ReLU())
        self.fc = Sequential(
            Flatten(),
            Linear(in_features=576, out_features=64),
            ReLU(),
            Linear(in_features=64, out_features=26),
            Softmax()
        )

    def forward(self, x):
        # x = fluid.layers.reshape(x, (-1, 1, 28, 28))
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x



# emnist_digital
class LeNet(nn.Layer):
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        # 创建卷积和池化层
        # 创建第1个卷积层
        self.conv1 = Conv2D(in_channels=1, out_channels=6, kernel_size=5)
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 尺寸的逻辑：池化层未改变通道数；当前通道数为6
        # 创建第2个卷积层
        self.conv2 = Conv2D(in_channels=6, out_channels=16, kernel_size=5)
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 创建第3个卷积层
        self.conv3 = Conv2D(in_channels=16, out_channels=120, kernel_size=4)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        # 输入size是[28,28]，经过三次卷积和两次池化之后，C*H*W等于120
        self.fc1 = Linear(in_features=120, out_features=64)
        # 创建全连接层，第一个全连接层的输出神经元个数为64， 第二个全连接层输出神经元个数为分类标签的类别数
        self.fc2 = Linear(in_features=64, out_features=num_classes)
    # 网络的前向计算过程
    def forward(self, x):
        # x = fluid.layers.reshape((x, (-1, 1, 28, 28)))
        x = self.conv1(x)
        # 每个卷积层使用Sigmoid激活函数，后面跟着一个2x2的池化
        x = F.sigmoid(x)
        x = self.max_pool1(x)
        x = F.sigmoid(x)
        x = self.conv2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        # 尺寸的逻辑：输入层将数据拉平[B,C,H,W] -> [B,C*H*W]
        x = paddle.reshape(x, [x.shape[0], -1])
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        return x


# Group A#######################################################################################

# Fashion-mnist model
class CNNniid(nn.Layer):
    def __init__(self):
        super(CNNniid, self).__init__()
        self.layer1 = Sequential(
            Conv2D(in_channels=1, out_channels=64, kernel_size=2),
            ReLU(),
            Dropout(0.05)
        )
        self.layer2 = Sequential(
            Conv2D(in_channels=64, out_channels=32, kernel_size=2),
            ReLU(),
            Dropout(0.05)
        )
        self.flatten = Flatten()
        self.drop = Dropout(0.05)
        self.fc = Linear(in_features=21632, out_features=10)
        self.sf = Softmax()


    def forward(self, x):
        # x = fluid.layers.reshape(x, (-1, 1, 28, 28))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        x = self.sf(x)
        return x


# Fashion-mnist model
class CNNiid(nn.Layer):
    def __init__(self):
        super(CNNiid, self).__init__()
        self.layer1 = Sequential(
            Conv2D(in_channels=1, out_channels=64, kernel_size=2),
            ReLU(),
            Dropout(0.5)
        )
        self.layer2 = Sequential(
            Conv2D(in_channels=64, out_channels=32, kernel_size=2),
            ReLU(),
            Dropout(0.5)
        )
        self.flatten = Flatten()
        self.drop = Dropout(0.5)
        self.fc = Linear(in_features=21632, out_features=10)
    def forward(self, x):
        # x = fluid.layers.reshape(x, (-1, 1, 28, 28))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.drop(x)
        x = self.fc(x)
        return x


class AlexNet(nn.Layer):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = Sequential(
            Conv2D(1, 32, kernel_size=5, stride=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Conv2D(32, 64, kernel_size=5, stride=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2)
        )
        self.flatten = Flatten()
        self.fc = Sequential(
            Linear(in_features=1024, out_features=1024),
            ReLU(),
            Dropout(0.5),
            Linear(in_features=1024, out_features=10),
            Softmax()
        )
    def forward(self, x):
        x = self.features(x)
        x= self.flatten(x)
        x = self.fc(x)
        return x



class Residual(nn.Layer):
    def __init__(self, in_channel, out_channel, use_conv1x1=False, stride=1):
        super(Residual, self).__init__()

        # 第一个卷积单元
        self.conv1 = nn.Conv2D(in_channel, out_channel, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2D(out_channel)
        self.relu = nn.ReLU()

        # 第二个卷积单元
        self.conv2 = nn.Conv2D(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2D(out_channel)

        if use_conv1x1:  # 使用1x1卷积核完成shape匹配,stride=2实现下采样
            self.skip = nn.Conv2D(in_channel, out_channel, kernel_size=1, stride=stride)
        else:
            self.skip = None

    def forward(self, x):
        # 前向计算
        # [b, c, h, w], 通过第一个卷积单元
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # 通过第二个卷积单元
        out = self.conv2(out)
        out = self.bn2(out)
        #  通过 identity 模块
        if self.skip:
            x = self.skip(x)
        #  2 条路径输出直接相加,然后输入激活函数
        output = F.relu(out + x)

        return output


# 通过build_resblock 可以一次完成2个残差模块的创建。代码如下：
def build_resblock(in_channel, out_channel, num_layers, is_first=False):
    if is_first:
        assert in_channel == out_channel
    block_list = []
    for i in range(num_layers):
        if i == 0 and not is_first:
            block_list.append(Residual(in_channel, out_channel, use_conv1x1=True, stride=2))
        else:
            block_list.append(Residual(out_channel, out_channel))
    resNetBlock = nn.Sequential(*block_list)  # 用*号可以把list列表展开为元素
    return resNetBlock


# 下面来实现ResNet18网络模型。代码如下：
class ResNet(nn.Layer):
    # 继承paddle.nn.Layer定义网络结构
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        # 初始化函数(根网络，预处理)
        # x:[b, c, h ,w]=[b,3,224,224]
        self.features = nn.Sequential(
            nn.Conv2D(in_channels=3, out_channels=64, kernel_size=7,
                      stride=2, padding=3),  # 第一层卷积,x:[b,64,112,112]
            nn.BatchNorm2D(64),  # 归一化层
            nn.ReLU(),
            nn.MaxPool2D(kernel_size=3, stride=2, padding=1)  # 最大池化，下采样,x:[b,64,56,56]
        )

        # 堆叠 4 个 Block，每个 block 包含了多个残差模块,设置步长不一样
        self.layer1 = build_resblock(64, 64, 2, is_first=True)  # x:[b,64,56,56]
        self.layer2 = build_resblock(64, 150, 2)  # x:[b,150,28,28]
        self.layer3 = build_resblock(150, 360, 2)  # x:[b,360,14,14]
        self.layer4 = build_resblock(360, 720, 2)  # x:[b,720,7,7]

        # 通过 Pooling 层将高宽降低为 1x1,[b,720,1,1]
        self.avgpool = nn.AdaptiveAvgPool2D(1)
        # 需要拉平为[b,720],不能直接输出连接线性层
        self.flatten = nn.Flatten()
        # 最后连接一个全连接层分类
        self.fc = nn.Linear(in_features=720, out_features=num_classes)

    def forward(self, inputs):
        # 前向计算函数：通过根网络
        x = self.features(inputs)
        # 一次通过 4 个模块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 通过池化层
        x = self.avgpool(x)
        # 拉平
        x = self.flatten(x)
        # 通过全连接层
        x = self.fc(x)
        return x


