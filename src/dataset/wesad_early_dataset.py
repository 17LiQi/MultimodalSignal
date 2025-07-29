# src/dataset/wesad_early_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadEarlyFusionDataset(Dataset):
    def __init__(self, data_path, subjects, channels_to_use, all_channel_names):
        """
        data_path: processed_data/wesad_early_fusion/ 的路径
        subjects: 要加载的受试者ID列表, e.g., ['S3', 'S4']
        channels_to_use: 本次实验要使用的通道名列表, e.g., ['chest_ECG']
        all_channel_names: 预处理时保存的所有通道名列表
        scaler: 一个预先 fit 好的 StandardScaler 对象
        """
        self.data_path = data_path
        self.subjects = subjects
        self.channels_to_use = channels_to_use
        self.all_channel_names = all_channel_names
        self.data_list = []
        self.labels_list = []
        # 将传入的 scaler 保存为类的属性
        # self.scaler = scaler

        self.channel_indices = [self.all_channel_names.index(ch) for ch in self.channels_to_use]

        for sid in self.subjects:
            x = np.load(self.data_path / f'{sid}_X.npy')
            y = np.load(self.data_path / f'{sid}_y.npy')

            # --- 受试者内归一化 ---
            # 1. 选择需要的通道
            channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]
            x_selected = x[:, :, channel_indices]

            # 2. Reshape to 2D for scaler
            num_samples, seq_len, num_channels = x_selected.shape
            x_reshaped = x_selected.reshape(-1, num_channels)

            # 3. Fit and transform *only on this subject's data*
            scaler = StandardScaler()
            x_scaled_reshaped = scaler.fit_transform(x_reshaped)

            # 4. Reshape back to 3D
            x_final = x_scaled_reshaped.reshape(num_samples, seq_len, num_channels)

            self.data_list.append(x_final)
            self.labels_list.append(y)

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 数据在初始化时已经归一化，这里直接返回
        x = self.data[idx]
        y = self.labels[idx]

        x_tensor = torch.from_numpy(x).float().permute(1, 0)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor

