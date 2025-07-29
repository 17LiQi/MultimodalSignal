# models/transformer_encoder.py
import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, **model_args):
        super(TransformerClassifier, self).__init__()

        d_model = model_args.get('d_model', 128)
        nhead = model_args.get('nhead', 8)
        num_layers = model_args.get('num_layers', 4)
        dropout = model_args.get('dropout', 0.3)

        # 1. 输入嵌入层 (用1D卷积模拟Patch Embedding)
        # 将每个时间步的 C 个通道映射到 d_model 维
        self.input_embedding = nn.Linear(in_channels, d_model)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. 分类头
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x 初始形状: (batch_size, in_channels, seq_len)

        # 调整形状以适应Linear层和Transformer
        # -> (batch_size, seq_len, in_channels)
        x = x.permute(0, 2, 1)

        # 输入嵌入
        # -> (batch_size, seq_len, d_model)
        x = self.input_embedding(x)

        # 位置编码 (需要调整形状)
        # -> (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        # -> (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)

        # Transformer Encoder 处理
        # -> (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)

        # 使用序列的平均值进行分类 (一种常见的池化策略)
        # -> (batch_size, d_model)
        x = x.mean(dim=1)

        # 分类
        logits = self.classifier(x)

        return logits