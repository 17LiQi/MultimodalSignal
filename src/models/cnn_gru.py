# models/cnn_gru.py
import torch.nn as nn


class CnnGruModel(nn.Module):
    def __init__(self, in_channels, num_classes, **model_args):
        """
        CNN-GRU 混合模型，用于时序分类。
        params: 包含 cnn_out_channels, gru_hidden_size 等的字典
        """
        super(CnnGruModel, self).__init__()

        cnn_out_channels = model_args.get('cnn_out_channels', 32)
        gru_hidden_size = model_args.get('gru_hidden_size', 64)
        gru_num_layers = model_args.get('gru_num_layers', 2)
        dropout = model_args.get('dropout', 0.5)

        # --- CNN 特征提取器 ---
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

        # --- GRU 时序建模器 ---
        # CNN的输出将作为GRU的输入
        # GRU的输入形状: (seq_len, batch_size, input_size)
        self.gru = nn.GRU(
            input_size=cnn_out_channels,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,  # 让输入形状为 (batch_size, seq_len, input_size)
            bidirectional=True,  # 使用双向GRU以捕捉前后文信息
            dropout=dropout if gru_num_layers > 1 else 0
        )

        # --- 分类器 ---
        # 双向GRU的输出是隐藏状态的两倍
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x 初始形状: (batch_size, in_channels, seq_len)

        # 1. 通过 CNN 提取特征
        # 输出形状: (batch_size, cnn_out_channels, new_seq_len)
        x = self.cnn_encoder(x)

        # 2. 调整形状以适应 GRU
        # (batch_size, cnn_out_channels, new_seq_len) -> (batch_size, new_seq_len, cnn_out_channels)
        x = x.permute(0, 2, 1)

        # 3. 通过 GRU 学习时序依赖
        # 我们只关心GRU的输出，不关心最后的隐藏状态
        # outputs 形状: (batch_size, seq_len, num_directions * hidden_size)
        outputs, _ = self.gru(x)

        # 4. 获取最后一个时间步的输出作为整个序列的表示
        # last_output 形状: (batch_size, num_directions * hidden_size)
        last_output = outputs[:, -1, :]

        # 5. 通过分类器得到最终预测
        logits = self.classifier(last_output)

        return logits