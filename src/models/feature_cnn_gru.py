# models/feature_cnn_gru.py
import torch
import torch.nn as nn


class FeatureCnnGru(nn.Module):
    def __init__(self, in_features, num_classes, **model_args):
        super(FeatureCnnGru, self).__init__()

        cnn_out_channels = model_args.get('cnn_out_channels', 32)
        gru_hidden_size = model_args.get('gru_hidden_size', 64)
        gru_num_layers = model_args.get('gru_num_layers', 2)
        dropout = model_args.get('dropout', 0.5)

        # --- CNN 特征提取器 (作用于特征序列) ---
        # 输入形状: (batch_size, in_features, sequence_length)
        # 我们需要先将 (batch, seq, feat) -> (batch, feat, seq)
        self.cnn_encoder = nn.Sequential(
            # 例如，用一个卷积核大小为3的卷积层来学习连续3个窗口的特征模式
            nn.Conv1d(in_channels=in_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=cnn_out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU()
        )

        # --- GRU 时序建模器 ---
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # --- 分类器 ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x 初始形状: (batch_size, sequence_length, in_features)

        # 1. 调整形状以适应 Conv1d
        # -> (batch_size, in_features, sequence_length)
        x = x.permute(0, 2, 1)

        # 2. 通过 CNN 提取跨时间步的特征组合
        # -> (batch_size, cnn_out_channels, sequence_length)
        x = self.cnn_encoder(x)

        # 3. 再次调整形状以适应 GRU
        # -> (batch_size, sequence_length, cnn_out_channels)
        x = x.permute(0, 2, 1)

        # 4. 通过 GRU 学习长期依赖
        outputs, _ = self.gru(x)

        # 5. 获取最后一个时间步的输出
        last_output = outputs[:, -1, :]

        # 6. 分类
        logits = self.classifier(last_output)

        return logits