# file: dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.preprocessing import StandardScaler


class WesadDataset(Dataset):
    def __init__(self, data_path: Path, subjects: list, channels_to_use: list, all_channel_names: list):
        self.data_list = []
        self.labels_list = []

        channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]

        for sid in subjects:
            x_raw = np.load(data_path / f'{sid}_X.npy')
            y = np.load(data_path / f'{sid}_y.npy')

            # 选择需要的通道
            x_selected = x_raw[:, :, channel_indices]

            # 受试者内归一化 (Intra-subject Normalization)
            num_samples, seq_len, num_channels = x_selected.shape
            x_reshaped = x_selected.reshape(-1, num_channels)
            scaler = StandardScaler()
            x_scaled_reshaped = scaler.fit_transform(x_reshaped)
            x_final = x_scaled_reshaped.reshape(num_samples, seq_len, num_channels)

            self.data_list.append(x_final)
            self.labels_list.append(y)

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 数据形状需要为 (C, L) for Conv1d
        x = torch.from_numpy(self.data[idx]).float().permute(1, 0)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y