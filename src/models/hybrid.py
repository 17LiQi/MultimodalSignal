# models/hybrid_model.py
import torch
import torch.nn as nn

# 我们可以复用之前写的 CNN-GRU 和 FeatureGRU 作为模型的构建块
# from cnn_gru import CnnGruModel
# from feature_gru import FeatureGRU


class HybridModel(nn.Module):
    def __init__(self, raw_model, feature_model, num_classes, **model_args):
        """
        双流混合融合模型。

        参数:
        - raw_model (nn.Module): 处理原始信号的子模型 (如 CnnGruModel)。
        - feature_model (nn.Module): 处理高级特征序列的子模型 (如 FeatureGRU)。
        - num_classes (int): 最终分类的类别数。
        - fusion_dim (int): 融合后的特征维度。
        """
        super(HybridModel, self).__init__()
        fusion_dim = model_args.get('fusion_dim', 128)

        # 流A: 原始信号流
        self.raw_stream = raw_model
        # 移除原始模型的分类头，我们只用它来提取特征
        self.raw_stream.classifier = nn.Identity()

        # 流B: 高级特征流
        self.feature_stream = feature_model
        # 同样移除分类头
        self.feature_stream.classifier = nn.Identity()

        # 获取两个流输出的特征维度
        # 假设它们都使用双向GRU，输出维度是 hidden_size * 2
        raw_out_dim = raw_model.gru.hidden_size * 2
        feature_out_dim = feature_model.gru.hidden_size * 2

        # 融合头
        self.fusion_head = nn.Sequential(
            nn.Linear(raw_out_dim + feature_out_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, num_classes)
        )

    def forward(self, x):
        # x 是一个元组: (x_raw_tensor, x_feat_tensor)
        x_raw, x_feat = x

        # 1. 通过各自的流提取特征
        # raw_features 的形状: (batch_size, raw_out_dim)
        raw_features = self.raw_stream(x_raw)

        # feat_features 的形状: (batch_size, feature_out_dim)
        feat_features = self.feature_stream(x_feat)

        # 2. 拼接两个特征向量
        # combined_features 的形状: (batch_size, raw_out_dim + feature_out_dim)
        combined_features = torch.cat([raw_features, feat_features], dim=1)

        # 3. 通过融合头进行最终分类
        logits = self.fusion_head(combined_features)

        return logits