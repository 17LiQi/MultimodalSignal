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
    def __init__(self, in_features, num_classes, **model_args):
        super(TransformerClassifier, self).__init__()

        d_model = model_args.get('d_model', 128)
        nhead = model_args.get('nhead', 8)
        num_layers = model_args.get('num_layers', 4)
        dropout = model_args.get('dropout', 0.3)

        # 1. 输入嵌入层 (用1D卷积模拟Patch Embedding)
        # 将每个时间步的 C 个通道映射到 d_model 维
        self.input_embedding = nn.Linear(in_features, d_model)

        # 2. 位置编码
        self.pos_encoder = PositionalEncoding(d_model)

        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 4. 分类头
        self.classifier = nn.Linear(d_model, num_classes)

        self.d_model = d_model

    def forward(self, x):
        # x 初始形状: (batch_size, sequence_length, in_features)

        # 1. 输入嵌入
        # -> (batch_size, sequence_length, d_model)
        x = self.input_embedding(x) * math.sqrt(self.d_model)  # 遵循原始 Transformer 论文的缩放

        # 2. 位置编码
        # TransformerEncoderLayer 期望的输入是 (seq_len, batch, features)
        # 如果 batch_first=False。但因为我们设置了 batch_first=True，所以不需要 permute。
        # 然而，PositionalEncoding 是按 (seq_len, batch, features) 设计的，所以这里需要临时转换一下。
        x = x.permute(1, 0, 2)  # -> (sequence_length, batch_size, d_model)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)  # -> (batch_size, sequence_length, d_model)

        # 3. Transformer Encoder 处理
        # -> (batch_size, sequence_length, d_model)
        x = self.transformer_encoder(x)

        # 4. 池化: 使用序列的第一个时间步的输出 (CLS token 思想的简化版) 或平均池化
        x = x[:, 0, :]  # -> (batch_size, d_model)
        # 或者 x = x.mean(dim=1)

        # 5. 分类
        logits = self.classifier(x)

        return logits