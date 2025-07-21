# src/models/cnn_1d.py
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(SimpleCNN1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2)

        # 需要动态计算全连接层的输入大小，或者用 Adaptive Pooling
        self.fc1 = nn.Linear(32 * 480, 128)  # 1920 -> pool -> 960 -> pool -> 480
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 480)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x