# dataset/wesad_hybrid_dataset.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadHybridDataset(Dataset):
    def __init__(self, early_data_path: Path, feature_data_path: Path, subjects: list,
                 channels_to_use: list, all_channel_names: list,
                 sequence_length=10, step=1):
        """
        为混合融合模型准备数据。
        """
        self.sequences = []
        self.raw_signal_windows = []
        self.labels = []

        # 早期融合数据的相关参数
        self.TARGET_FS = 64
        self.WINDOW_SEC = 30
        self.early_window_len = self.WINDOW_SEC * self.TARGET_FS

        for sid in subjects:
            # --- 加载特征数据并创建序列 (与 FeatureSequenceDataset 类似) ---
            x_feat_subject = np.load(feature_data_path / f'{sid}_X.npy')
            y_subject = np.load(feature_data_path / f'{sid}_y.npy')

            scaler = StandardScaler()
            x_feat_scaled = scaler.fit_transform(x_feat_subject)

            # --- 加载早期融合数据 ---
            x_early_subject = np.load(early_data_path / f'{sid}_X.npy')
            # 同样进行受试者内归一化
            channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]
            x_early_selected = x_early_subject[:, :, channel_indices]

            num_s, seq_l, num_c = x_early_selected.shape
            x_early_reshaped = x_early_selected.reshape(-1, num_c)
            early_scaler = StandardScaler()
            x_early_scaled_reshaped = early_scaler.fit_transform(x_early_reshaped)
            x_early_scaled = x_early_scaled_reshaped.reshape(num_s, seq_l, num_c)

            # 创建对齐的序列
            # feature_fusion 和 early_fusion 的窗口是按相同方式生成的
            for i in range(0, len(x_feat_scaled) - sequence_length + 1, step):
                # 提取一个特征序列
                feat_seq = x_feat_scaled[i: i + sequence_length]
                self.sequences.append(feat_seq)

                # 提取与序列 *最后一个时间步* 对应的原始信号窗口
                raw_window_idx = i + sequence_length - 1
                self.raw_signal_windows.append(x_early_scaled[raw_window_idx])

                # 提取标签
                label = y_subject[raw_window_idx]
                self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_feat_seq = self.sequences[idx]
        x_raw_win = self.raw_signal_windows[idx]
        y = self.labels[idx]

        # 转换为 PyTorch 张量
        # 特征序列 -> (seq_len, in_features)
        x_feat_tensor = torch.from_numpy(x_feat_seq).float()
        # 原始信号窗口 -> (in_channels, window_len)
        x_raw_tensor = torch.from_numpy(x_raw_win).float().permute(1, 0)

        y_tensor = torch.tensor(y, dtype=torch.long)

        # 返回一个包含两种输入的元组和标签
        return (x_raw_tensor, x_feat_tensor), y_tensor