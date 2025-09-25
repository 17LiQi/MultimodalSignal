# file: dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from collections import Counter

class WesadDataset(Dataset):
    def __init__(self, data_path: Path, subjects: list, channels_to_use: list,
                 all_channel_names: list, classification_mode='stress_binary'):
        self.data_list = []
        self.labels_list = []
        self.classification_mode = classification_mode  # 'stress_binary' or 'ternary'

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
            if self.classification_mode == 'stress_binary':
                y = np.where(y_raw == 2, 1, 0)  # TSST=1, 其他=0
            elif self.classification_mode == 'ternary':
                y = np.where(y_raw == 1, 0, np.where(y_raw == 3, 1, np.where(y_raw == 2, 2, 0)))  # Base=0, Fun=1, TSST=2
            else:
                raise ValueError(f"Unknown classification_mode: {classification_mode}")

            # 被试内归一化 (基于全部数据)
            mean_all = np.mean(x_selected, axis=(0, 1))
            std_all = np.std(x_selected, axis=(0, 1)) + 1e-8

            for ch in range(num_channels):
                channel_name = all_channel_names[channel_indices[ch]]
                if channel_name == 'chest_EDA':
                    log_data = np.log1p(x_selected[:, :, ch])
                    mean_log = np.mean(log_data)
                    std_log = np.std(log_data) + 1e-8
                    x_selected[:, :, ch] = (log_data - mean_log) / std_log
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
