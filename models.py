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


class FeatureProcessor(nn.Module):
    """
    手工特征处理器：一个简单的多层感知机(MLP)，用于提取手工特征中的高阶信息。
    """

    def __init__(self, in_features, hidden_dims, out_features, dropout=0.5):
        super(FeatureProcessor, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, in_features)
        return self.net(x)


class HybridModel(nn.Module):
    """
    混合模型：将端到端分支和手工特征分支进行融合。
    """

    def __init__(self, in_channels, num_classes, num_handcrafted_features,
                 # 参数传递给子模块
                 cnn_out_channels=32, gru_hidden_size=64, gru_num_layers=2, dropout=0.5,
                 feature_hidden_dims=[32], feature_out_dim=16):
        super(HybridModel, self).__init__()

        # --- 1. 端到端分支 ---
        # 我们直接在这里实例化 CnnGruAttentionModel
        self.raw_signal_model = CnnGruAttentionModel(
            in_channels=in_channels,
            num_classes=num_classes,  # 临时传入，后续会被替换
            cnn_out_channels=cnn_out_channels,
            gru_hidden_size=gru_hidden_size,
            gru_num_layers=gru_num_layers,
            dropout=dropout
        )

        # --- 移除原始模型的分类器，把它变成一个纯粹的特征提取器 ---
        # 我们需要知道原始分类器前的特征维度，这里是 gru_hidden_size * 2
        raw_feature_dim = gru_hidden_size * 2
        self.raw_signal_model.classifier = nn.Identity()  # 替换为一个什么都不做的层

        # --- 2. 手工特征分支 ---
        self.feature_processor = FeatureProcessor(
            in_features=num_handcrafted_features,
            hidden_dims=feature_hidden_dims,
            out_features=feature_out_dim
        )

        # --- 3. 融合后的最终分类器 ---
        # 输入维度是两个分支输出特征维度之和
        fused_feature_dim = raw_feature_dim + feature_out_dim
        self.final_classifier = nn.Sequential(
            nn.Linear(fused_feature_dim, 64),
            nn.BatchNorm1d(64),  # 在融合后加一个BN层，有助于稳定训练
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, inputs):
        # 解包输入
        x_raw, x_feat = inputs
        # x_raw shape: (batch, in_channels, seq_len)
        # x_feat shape: (batch, num_handcrafted_features)

        # 1. 通过端到端分支提取高级时序特征
        raw_features = self.raw_signal_model(x_raw)  # -> (batch, gru_hidden_size * 2)

        # 2. 通过特征分支处理手工特征
        handcrafted_features = self.feature_processor(x_feat)  # -> (batch, feature_out_dim)

        # 3. 拼接（Concatenate）两种特征
        fused_features = torch.cat([raw_features, handcrafted_features], dim=1)

        # 4. 通过最终分类器得到预测
        logits = self.final_classifier(fused_features)

        return logits