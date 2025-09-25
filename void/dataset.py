# file: dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path

class WesadDataset(Dataset):
    def __init__(self, data_path: Path, subjects: list, channels_to_use: list,
                 all_channel_names: list, classification_mode='binary'):
        self.data_list = []
        self.labels_list = []
        self.classification_mode = classification_mode  # 'binary' or 'ternary'

        channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]

        for sid in subjects:
            x_raw = np.load(data_path / f'{sid}_X.npy')
            y_raw = np.load(data_path / f'{sid}_y.npy')  # 原始标签 (1/2/3/4)
            x_selected = x_raw[:, :, channel_indices]
            num_samples, seq_len, num_channels = x_selected.shape

            # 动态标签映射
            if self.classification_mode == 'binary':
                y = np.where(y_raw == 2, 1, 0)  # TSST=1, 其他=0
            elif self.classification_mode == 'ternary':
                y = np.where(y_raw == 1, 0, np.where(y_raw == 3, 1, np.where(y_raw == 2, 2, 0)))  # Base=0, Fun=1, TSST=2
            else:
                raise ValueError(f"Unknown classification_mode: {classification_mode}")

            # 基线归一化 (仅task=='Base', y_raw==1)
            baseline_mask = (y_raw == 1)
            if np.sum(baseline_mask) == 0:
                print(f"警告: 受试者 {sid} 无基线数据，使用整体均值/标准差作为fallback。")
                mean_baseline = np.mean(x_selected, axis=(0, 1))
                std_baseline = np.std(x_selected, axis=(0, 1)) + 1e-8
                mean_log_baseline = mean_baseline
                std_log_baseline = std_baseline
            else:
                baseline_x = x_selected[baseline_mask]
                mean_baseline = np.zeros(num_channels)
                std_baseline = np.zeros(num_channels)
                mean_log_baseline = np.zeros(num_channels)
                std_log_baseline = np.zeros(num_channels)

                for ch in range(num_channels):
                    channel_name = all_channel_names[channel_indices[ch]]
                    if channel_name == 'chest_EDA':
                        log_baseline = np.log1p(baseline_x[:, :, ch])
                        mean_log_baseline[ch] = np.mean(log_baseline)
                        std_log_baseline[ch] = np.std(log_baseline) + 1e-8
                        x_selected[:, :, ch] = (np.log1p(x_selected[:, :, ch]) - mean_log_baseline[ch]) / std_log_baseline[ch]
                    else:
                        mean_baseline[ch] = np.mean(baseline_x[:, :, ch])
                        std_baseline[ch] = np.std(baseline_x[:, :, ch]) + 1e-8
                        x_selected[:, :, ch] = (x_selected[:, :, ch] - mean_baseline[ch]) / std_baseline[ch]

            self.data_list.append(x_selected)
            self.labels_list.append(y)  # 映射后的标签

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx]).float().permute(1, 0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class HybridDataset(Dataset):
    def __init__(self, early_fusion_path: Path, feature_fusion_path: Path,
                 subjects: list, channels_to_use: list, all_channel_names: list,
                 features_to_use: list = None, classification_mode='binary'):
        self.raw_data_list = []
        self.feat_data_list = []
        self.labels_list = []  # 用于存储原始信号的标签
        self.labels_list_feat = []  # 用于存储特征数据的标签
        self.classification_mode = classification_mode

        # --- 1. 原始信号 ---
        channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]

        for sid in subjects:
            x_raw_file = early_fusion_path / f'{sid}_X.npy'
            y_raw_file = early_fusion_path / f'{sid}_y.npy'
            if not x_raw_file.exists() or not y_raw_file.exists():
                print(f"Warning: Skipping subject {sid} for raw data, file not found.")
                continue
            x_raw = np.load(x_raw_file)
            y_raw = np.load(y_raw_file)
            x_selected = x_raw[:, :, channel_indices]
            num_samples, seq_len, num_channels = x_selected.shape

            # 动态标签映射
            if self.classification_mode == 'binary':
                y = np.where(y_raw == 2, 1, 0)
            elif self.classification_mode == 'ternary':
                y = np.where(y_raw == 1, 0, np.where(y_raw == 3, 1, np.where(y_raw == 2, 2, 0)))
            else:
                raise ValueError(f"Unknown classification_mode: {classification_mode}")

            # 基线归一化
            baseline_mask = (y_raw == 1)
            if np.sum(baseline_mask) == 0:
                print(f"警告: 受试者 {sid} 无基线数据，使用整体均值/标准差作为fallback。")
                mean_baseline = np.mean(x_selected, axis=(0, 1))
                std_baseline = np.std(x_selected, axis=(0, 1)) + 1e-8
                mean_log_baseline = mean_baseline
                std_log_baseline = std_baseline
            else:
                baseline_x = x_selected[baseline_mask]
                mean_baseline = np.zeros(num_channels)
                std_baseline = np.zeros(num_channels)
                mean_log_baseline = np.zeros(num_channels)
                std_log_baseline = np.zeros(num_channels)
                for ch in range(num_channels):
                    channel_name = all_channel_names[channel_indices[ch]]
                    if channel_name == 'chest_EDA':
                        log_baseline = np.log1p(baseline_x[:, :, ch])
                        mean_log_baseline[ch] = np.mean(log_baseline)
                        std_log_baseline[ch] = np.std(log_baseline) + 1e-8
                        x_selected[:, :, ch] = (np.log1p(x_selected[:, :, ch]) - mean_log_baseline[ch]) / std_log_baseline[ch]
                    else:
                        mean_baseline[ch] = np.mean(baseline_x[:, :, ch])
                        std_baseline[ch] = np.std(baseline_x[:, :, ch]) + 1e-8
                        x_selected[:, :, ch] = (x_selected[:, :, ch] - mean_baseline[ch]) / std_baseline[ch]

            self.raw_data_list.append(x_selected)
            self.labels_list.append(y)  # 存储原始信号标签

        if not self.raw_data_list:
            raise ValueError(f"No raw data loaded for subjects: {subjects}. Check paths and data existence.")

        self.raw_data = np.concatenate(self.raw_data_list, axis=0)
        # raw_data的标签已经生成完毕，把它作为基准
        self.labels = np.concatenate(self.labels_list, axis=0)

        # --- 2. 手工特征 ---
        with open(feature_fusion_path / '_feature_names.txt', 'r') as f:
            all_feature_names = [line.strip() for line in f]
        feature_indices = [all_feature_names.index(feat_name) for feat_name in features_to_use] if features_to_use else list(range(len(all_feature_names)))

        for sid in subjects:
            x_feat_file = feature_fusion_path / f'{sid}_X.npy'
            y_feat_file = feature_fusion_path / f'{sid}_y.npy'
            if not x_feat_file.exists() or not y_feat_file.exists():
                print(f"Warning: Skipping subject {sid} for feature data, file not found.")
                continue
            x_feat_raw = np.load(x_feat_file)
            y_feat_raw = np.load(y_feat_file)
            x_feat_selected = x_feat_raw[:, feature_indices]

            # 动态标签映射
            if self.classification_mode == 'binary':
                y_feat = np.where(y_feat_raw == 2, 1, 0)
            elif self.classification_mode == 'ternary':
                y_feat = np.where(y_feat_raw == 1, 0, np.where(y_feat_raw == 3, 1, np.where(y_feat_raw == 2, 2, 0)))

            # 基线归一化
            baseline_mask = (y_feat_raw == 1)
            if np.sum(baseline_mask) == 0:
                print(f"警告: 受试者 {sid} 无基线数据，使用整体均值/标准差作为fallback。")
                mean_baseline = np.mean(x_feat_selected, axis=0)
                std_baseline = np.std(x_feat_selected, axis=0) + 1e-3
            else:
                baseline_feat = x_feat_selected[baseline_mask]
                mean_baseline = np.mean(baseline_feat, axis=0)
                std_baseline = np.std(baseline_feat, axis=0) + 1e-3
            x_feat_normalized = (x_feat_selected - mean_baseline) / std_baseline

            self.feat_data_list.append(x_feat_normalized)
            self.labels_list_feat.append(y_feat)  # 添加到新列表

        if not self.feat_data_list:
            raise ValueError(f"No feature data loaded for subjects: {subjects}. Check paths and data existence.")

        self.feat_data = np.concatenate(self.feat_data_list, axis=0)
        labels_feat_final = np.concatenate(self.labels_list_feat, axis=0)

        assert self.raw_data.shape[0] == self.feat_data.shape[0], \
            f"原始信号样本数({self.raw_data.shape[0]})和特征样本数({self.feat_data.shape[0]})不匹配!"
        # 断言 self.labels 和 labels_feat_final 是否一致
        assert np.array_equal(self.labels, labels_feat_final), \
            "原始信号标签和特征标签不匹配!"

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_raw = self.raw_data[idx]
        x_feat = self.feat_data[idx]
        y = self.labels[idx]
        x_raw_tensor = torch.from_numpy(x_raw).float().permute(1, 0)
        x_feat_tensor = torch.from_numpy(x_feat).float()
        y_tensor = torch.tensor(y, dtype=torch.long)
        return (x_raw_tensor, x_feat_tensor), y_tensor
