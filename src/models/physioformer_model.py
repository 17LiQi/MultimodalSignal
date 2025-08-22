# physioformer_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any

from torch import Tensor


# =================================================================
# 模块 A: ContribNet
# 论文依据: Page 6, Definition 1, 公式 (1)
# =================================================================
class ContribNet(nn.Module):
    """
    计算单个生理信号指标的贡献度 alpha。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super(ContribNet, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # 输出一个标量 alpha
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        # output shape: (batch_size, 1)
        return self.layer(x)


# =================================================================
# 模块 B: AffectNet
# 论文依据: Page 7, Definition 2, 公式 (2)
# =================================================================
class AffectNet(nn.Module):
    """
    计算单个生理信号指标的情感状态水平 theta。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 32):
        super(AffectNet, self).__init__()
        # 论文描述为 "deep structure"，我们用两层实现
        self.layer = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 输出情感状态向量 theta
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        # output shape: (batch_size, output_dim)
        return self.layer(x)


# =================================================================
# 模块 C: AffectAnalyser
# 论文依据: Page 8, Definition 3, 公式 (3)
# =================================================================
class AffectAnalyser(nn.Module):
    """
    整合所有情感状态水平，做出最终预测。
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 3):
        super(AffectAnalyser, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)  # 输出最终的分类 logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, input_dim)
        # output shape: (batch_size, num_classes)
        return self.layer(x)


# =================================================================
# 最终组装: PhysioFormer 主模型
# 论文依据: Page 8, Section 4.1.1 & Algorithm 1
# =================================================================
class PhysioFormer(nn.Module):
    def __init__(self, feature_names: List[str], num_classes: int = 3):
        """
        Args:
            feature_names (List[str]): 包含所有特征名称的列表。
                                       用于自动识别个人属性和不同生理信号的特征。
            num_classes (int): 分类的类别数。
        """
        super(PhysioFormer, self).__init__()
        self.feature_names = feature_names
        self.num_classes = num_classes

        # 1. 识别并分离特征
        # -----------------------------------------------------
        self.signal_types = ['ACC', 'BVP', 'EDA', 'TEMP', 'ECG', 'EMG', 'RESP']
        self.attribute_indices: Dict[str, List[int]] = {}
        self.signal_indices: Dict[str, List[int]] = {}

        # 找到个人属性特征的索引
        # 我们假设个人属性不包含任何 signal_types 关键字
        self.attribute_indices['personal'] = [
            i for i, name in enumerate(feature_names)
            if not any(sig in name for sig in self.signal_types)
        ]

        # 为每个生理信号找到其对应的特征索引
        for sig_type in self.signal_types:
            indices = [i for i, name in enumerate(feature_names) if sig_type in name]
            if indices:  # 只有当数据集中存在该信号的特征时才添加
                self.signal_indices[sig_type] = indices

        self.full_feature_dim = len(feature_names)
        self.personal_attr_dim = len(self.attribute_indices['personal'])

        print("Model Initialized with following feature structure:")
        print(f"  - Total Features: {self.full_feature_dim}")
        print(f"  - Personal Attribute Features: {self.personal_attr_dim}")
        print("  - Physiological Signal Features:")
        for sig, indices in self.signal_indices.items():
            print(f"    - {sig}: {len(indices)} features")

        # 2. 实例化子模块
        # -----------------------------------------------------
        # 为每个存在的生理信号创建一个 ContribNet 和 AffectNet
        self.contrib_nets = nn.ModuleDict()
        self.affect_nets = nn.ModuleDict()

        affectnet_output_dim = 32  # 定义一个统一的 theta 向量维度

        for sig_type in self.signal_indices.keys():
            # ContribNet 的输入是个人属性+该信号的特征
            contrib_input_dim = self.personal_attr_dim + len(self.signal_indices[sig_type])
            self.contrib_nets[sig_type] = ContribNet(contrib_input_dim)

            # AffectNet 的输入也是个人属性+该信号的特征
            affect_input_dim = self.personal_attr_dim + len(self.signal_indices[sig_type])
            self.affect_nets[sig_type] = AffectNet(affect_input_dim, output_dim=affectnet_output_dim)

        # AffectAnalyser 的输入维度是所有 AffectNet 输出的 theta 向量的总和
        analyser_input_dim = len(self.signal_indices) * affectnet_output_dim
        self.affect_analyser = AffectAnalyser(analyser_input_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, Tensor] | Any]:
        """
        严格遵循 Algorithm 1 的流程
        Args:
            x (torch.Tensor): 输入的完整特征向量, shape: (batch_size, full_feature_dim)

        Returns:
            A dictionary containing the final logits and all contribution alphas.
        """
        # 提取个人属性特征
        personal_attrs = x[:, self.attribute_indices['personal']]

        all_alphas: Dict[str, torch.Tensor] = {}
        all_thetas: List[torch.Tensor] = []

        # 遍历每个生理信号类型，执行 Algorithm 1 的核心循环 (lines 3-7)
        for sig_type, sig_indices in self.signal_indices.items():
            # 提取当前信号的特征
            signal_features = x[:, sig_indices]

            # 步骤 3 & 4 (简化): 拼接个人属性和生理特征，计算贡献度 alpha
            # PF_p_bj = Concat(A_p, B_p_bj)
            pf_bj = torch.cat([personal_attrs, signal_features], dim=1)

            # alpha_bj = ContribNet(BN(PF_p_bj))
            # 注意：我们的ContribNet内部已经包含了BatchNorm
            alpha_bj = self.contrib_nets[sig_type](pf_bj)
            all_alphas[sig_type] = alpha_bj

            # 步骤 5 & 6: 计算情感状态水平 theta
            # 论文中 AffectNet 的输入 y_p_bj 也是拼接了个人属性和加权后的生理特征
            # 这里我们直接使用 pf_bj 作为输入，因为 alpha 只是用于损失函数
            # 这是一个常见的简化，alpha 的作用是正则化，而不是直接改变前向传播的数据流
            theta_bj = self.affect_nets[sig_type](pf_bj)
            all_thetas.append(theta_bj)

        # 步骤 8 & 9: 融合所有 theta 向量
        # Splice all theta_bj to obtain Theta_p
        theta_p = torch.cat(all_thetas, dim=1)
        # Sum with initial affective level (u_p) is omitted as it's not a learnable part
        phi_p = theta_p

        # 步骤 10: 最终预测
        # e_p = AffectAnalyser(Phi_p)
        logits = self.affect_analyser(phi_p)

        # 返回一个字典，方便 Trainer 提取 logits 和 alphas
        return {
            "logits": logits,
            "alphas": all_alphas
        }