# file: dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter

class WesadDataset(Dataset):
    def __init__(self, data_path: Path, subjects: list, channels_to_use: list,
                 all_channel_names: list, classification_mode='binary'):
        self.data_list = []
        self.labels_list = []
        self.classification_mode = classification_mode  # 'binary' or 'ternary'

        channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]

        for sid in subjects:
            x_raw_file = data_path / f'{sid}_X.npy'
            y_raw_file = data_path / f'{sid}_y.npy'
            if not x_raw_file.exists() or not y_raw_file.exists():
                print(f"Warning: Skipping subject {sid} for data, file not found.")
                continue
            x_raw = np.load(x_raw_file)
            y_raw = np.load(y_raw_file)  # 原始标签 (1/2/3/4)
            x_selected = x_raw[:, :, channel_indices]
            num_samples, seq_len, num_channels = x_selected.shape

            # 动态标签映射
            if self.classification_mode == 'binary':
                y = np.where(y_raw == 2, 1, 0)  # TSST=1, 其他=0
            elif self.classification_mode == 'ternary':
                y = np.where(y_raw == 1, 0, np.where(y_raw == 3, 1, np.where(y_raw == 2, 2, 0)))  # Base=0, Fun=1, TSST=2
            else:
                raise ValueError(f"Unknown classification_mode: {classification_mode}")

            # 被试内归一化 (基于全部数据)
            mean_all = np.mean(x_selected, axis=(0, 1))
            std_all = np.std(x_selected, axis=(0, 1)) + 1e-8
            mean_log_all = mean_all
            std_log_all = std_all

            for ch in range(num_channels):
                channel_name = all_channel_names[channel_indices[ch]]
                if channel_name == 'chest_EDA':
                    log_data = np.log1p(x_selected[:, :, ch])
                    mean_log_all[ch] = np.mean(log_data)
                    std_log_all[ch] = np.std(log_data) + 1e-8
                    x_selected[:, :, ch] = (log_data - mean_log_all[ch]) / std_log_all[ch]
                else:
                    x_selected[:, :, ch] = (x_selected[:, :, ch] - mean_all[ch]) / std_all[ch]

            self.data_list.append(x_selected)
            self.labels_list.append(y)

        if not self.data_list:
            raise ValueError(f"No data loaded for subjects: {subjects}. Check paths and data existence.")

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

        # 将所有数据列表初始化在循环外部
        self.raw_data_list = []
        self.feat_data_list = []
        self.labels_list = []
        self.classification_mode = classification_mode

        # 获取通道和特征的索引
        channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]
        with open(feature_fusion_path / '_feature_names.txt', 'r') as f:
            all_feature_names = [line.strip() for line in f]
        feature_indices = [all_feature_names.index(feat_name) for feat_name in
                           features_to_use] if features_to_use else list(range(len(all_feature_names)))

        # --- 单一循环处理每个受试者 ---
        for sid in subjects:
            # --- 1. 加载数据和标签 ---
            x_raw_file = early_fusion_path / f'{sid}_X.npy'
            y_raw_file = early_fusion_path / f'{sid}_y.npy'
            x_feat_file = feature_fusion_path / f'{sid}_X.npy'
            y_feat_file = feature_fusion_path / f'{sid}_y.npy'

            if not all([f.exists() for f in [x_raw_file, y_raw_file, x_feat_file, y_feat_file]]):
                print(f"Warning: Skipping subject {sid}, at least one data file not found.")
                continue

            x_raw = np.load(x_raw_file)
            y_raw = np.load(y_raw_file)
            x_feat_raw = np.load(x_feat_file)
            y_feat_raw = np.load(y_feat_file)

            # --- 2. 检查和生成标签 ---
            # 关键检查：确保原始标签完全一致
            if not np.array_equal(y_raw, y_feat_raw):
                print(f"FATAL ERROR: Raw and Feature labels mismatch for subject {sid}!")
                # 在这里可以抛出异常或进行更详细的调试
                raise ValueError(f"Label mismatch for {sid}")

            # 动态标签映射 (只需做一次)
            if self.classification_mode == 'binary':
                y = np.where(y_raw == 2, 1, 0)
            elif self.classification_mode == 'ternary':
                y = np.where(y_raw == 1, 0, np.where(y_raw == 3, 1, np.where(y_raw == 2, 2, 0)))
            else:
                raise ValueError(f"Unknown classification_mode: {classification_mode}")

            # --- 3. 处理原始信号 ---
            x_selected_raw = x_raw[:, :, channel_indices]
            num_channels = x_selected_raw.shape[2]

            # 被试内归一化 (raw)
            mean_all_raw = np.mean(x_selected_raw, axis=(0, 1))
            std_all_raw = np.std(x_selected_raw, axis=(0, 1)) + 1e-8
            for ch in range(num_channels):
                channel_name = all_channel_names[channel_indices[ch]]
                if channel_name == 'chest_EDA':
                    log_data = np.log1p(x_selected_raw[:, :, ch])
                    mean_log_all = np.mean(log_data)
                    std_log_all = np.std(log_data) + 1e-8
                    x_selected_raw[:, :, ch] = (log_data - mean_log_all) / std_log_all
                else:
                    x_selected_raw[:, :, ch] = (x_selected_raw[:, :, ch] - mean_all_raw[ch]) / std_all_raw[ch]

            # --- 4. 处理手工特征 ---
            x_selected_feat = x_feat_raw[:, feature_indices]

            # 被试内归一化 (feat)
            mean_all_feat = np.mean(x_selected_feat, axis=0)
            std_all_feat = np.std(x_selected_feat, axis=0) + 1e-8
            x_feat_normalized = (x_selected_feat - mean_all_feat) / std_all_feat

            # --- 5. 添加到列表 ---
            self.raw_data_list.append(x_selected_raw)
            self.feat_data_list.append(x_feat_normalized)
            self.labels_list.append(y)

        # --- 循环结束后，进行拼接 ---
        if not self.raw_data_list:
            raise ValueError(f"No data loaded for subjects: {subjects}. Check paths and data existence.")

        self.raw_data = np.concatenate(self.raw_data_list, axis=0)
        self.feat_data = np.concatenate(self.feat_data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

        assert self.raw_data.shape[0] == self.feat_data.shape[0], \
            f"最终拼接后样本数不匹配: raw({self.raw_data.shape[0]}) vs feat({self.feat_data.shape[0]})"
        assert self.raw_data.shape[0] == len(self.labels), \
            "最终拼接后数据和标签长度不匹配"

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