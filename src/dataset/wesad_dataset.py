# src/dataset/wesad_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadEarlyFusionDataset(Dataset):
    def __init__(self, data_path, subjects, channels_to_use, all_channel_names, scaler):
        """
        data_path: processed_data/early_fusion/ 的路径
        subjects: 要加载的受试者ID列表, e.g., ['S3', 'S4']
        channels_to_use: 本次实验要使用的通道名列表, e.g., ['chest_ECG']
        all_channel_names: 预处理时保存的所有通道名列表
        scaler: 一个预先 fit 好的 StandardScaler 对象
        """
        self.data_path = data_path
        self.subjects = subjects
        self.channels_to_use = channels_to_use
        self.all_channel_names = all_channel_names

        # 将传入的 scaler 保存为类的属性
        self.scaler = scaler

        self.channel_indices = [self.all_channel_names.index(ch) for ch in self.channels_to_use]

        # 加载并拼接所有指定受试者的数据
        # (如果数据集很大，这里可以优化为懒加载，但对于WESAD来说一次性加载没问题)
        data_list = [np.load(self.data_path / f'{sid}_X.npy') for sid in self.subjects]
        labels_list = [np.load(self.data_path / f'{sid}_y.npy') for sid in self.subjects]

        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(labels_list, axis=0)

        # 仅选择本次实验需要的通道
        self.data = self.data[:, :, self.channel_indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        # 使用初始化时传入的 scaler 进行 transform
        # x.shape = (seq_len, num_channels)
        x_scaled = self.scaler.transform(x)

        # 转换为 PyTorch 张量并调整维度以适应CNN (N, C, L)
        # (C, L) -> (num_channels, seq_len)
        x_tensor = torch.from_numpy(x_scaled).float().permute(1, 0)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor

