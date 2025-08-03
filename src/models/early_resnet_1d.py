# models/early_resnet_1d.py
import torch
import torch.nn as nn


class ResNetBlock1D(nn.Module):
    """
    一维残差块 (Residual Block)
    包含两个卷积层、批归一化和ReLU激活，以及一个捷径连接。
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, downsample=None):
        super(ResNetBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 如果输入和输出的维度不同 (例如通道数或尺寸)，
        # 需要通过一个下采样层 (通常是 1x1 卷积) 来匹配维度。
        if self.downsample is not None:
            identity = self.downsample(x)

        # 捷径连接 (Shortcut Connection)
        out += identity
        out = self.relu(out)

        return out


class ResNet1D(nn.Module):
    """
    借鉴 ResNet 思想的一维卷积神经网络，用于时序分类。
    """

    def __init__(self, in_channels, num_classes, block=ResNetBlock1D, layers=[2, 2, 2, 2], kernel_size=7):
        super(ResNet1D, self).__init__()
        self.in_channels = 16  # 第一个卷积层后的基础通道数

        # 初始卷积层和池化层，用于初步特征提取和降维
        self.conv1 = nn.Conv1d(in_channels, self.in_channels, kernel_size=kernel_size, stride=2,
                               padding=kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 堆叠四个残差层
        self.layer1 = self._make_layer(block, 16, layers[0], kernel_size=kernel_size)
        self.layer2 = self._make_layer(block, 32, layers[1], kernel_size=kernel_size, stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], kernel_size=kernel_size, stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], kernel_size=kernel_size, stride=2)

        # 自适应平均池化，将不同长度的序列池化为长度为1的特征图
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.fc = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, kernel_size, stride=1):
        downsample = None
        # 如果需要改变输入的维度 (通道数或尺寸)，则创建一个下采样层
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, kernel_size, stride, downsample))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels, kernel_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平
        x = self.fc(x)

        return x