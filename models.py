# file: models.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    通道注意力模块：学习给不同通道分配不同的重要性权重。
    """

    def __init__(self, in_channels, reduction_ratio=4):
        super(ChannelAttention, self).__init__()
        # 使用自适应平均池化将每个通道的时间维度压缩为1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 使用一个小型MLP来学习通道间的非线性关系
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False),
            nn.Sigmoid()  # Sigmoid将权重缩放到0-1之间
        )

    def forward(self, x):
        # x shape: (batch, channels, length)
        b, c, _ = x.size()
        # y shape: (batch, channels) -> (batch, channels, 1)
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        # 将原始输入 x 与学习到的通道权重 y 相乘
        return x * y.expand_as(x)


class CnnGruAttentionModel(nn.Module):
    """
    端到端模型：结合了通道注意力、CNN和双向GRU。
    """

    def __init__(self, in_channels, num_classes,
                 cnn_out_channels=32, gru_hidden_size=64, gru_num_layers=2, dropout=0.5):
        super(CnnGruAttentionModel, self).__init__()

        self.channel_attention = ChannelAttention(in_channels=in_channels)

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Conv1d(16, cnn_out_channels, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # 最终的分类器，双向GRU的输出维度是 hidden_size * 2
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        x = self.channel_attention(x)
        x = self.cnn_encoder(x)
        x = x.permute(0, 2, 1)
        outputs, _ = self.gru(x)
        last_output = outputs[:, -1, :]
        logits = self.classifier(last_output)
        return logits



