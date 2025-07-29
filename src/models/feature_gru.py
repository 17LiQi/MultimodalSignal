# models/feature_gru.py
import torch
import torch.nn as nn


class FeatureGRU(nn.Module):
    def __init__(self, in_features, num_classes, **model_args):
        super(FeatureGRU, self).__init__()

        hidden_size = model_args.get('hidden_size', 64)
        num_layers = model_args.get('num_layers', 2)
        dropout = model_args.get('dropout', 0.5)

        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),  # 双向GRU的输出是隐藏状态的两倍
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x 初始形状: (batch_size, sequence_length, in_features)

        # GRU处理序列
        outputs, _ = self.gru(x)

        # 获取最后一个时间步的输出
        last_output = outputs[:, -1, :]

        # 分类
        logits = self.classifier(last_output)

        return logits