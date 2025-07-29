# dataset/wesad_transformer_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadTransformerDataset(Dataset):
    def __init__(self, data_path, subjects, channels_to_use, all_channel_names, patch_size=64):
        self.data_list = []
        self.labels_list = []
        self.patch_size = patch_size

        for sid in subjects:
            x = np.load(data_path / f'{sid}_X.npy')
            y = np.load(data_path / f'{sid}_y.npy')

            # 受试者内归一化
            channel_indices = [all_channel_names.index(ch) for ch in channels_to_use]
            x_selected = x[:, :, channel_indices]

            num_s, seq_l, num_c = x_selected.shape
            x_reshaped = x_selected.reshape(-1, num_c)
            scaler = StandardScaler()
            x_scaled_reshaped = scaler.fit_transform(x_reshaped)
            x_final = x_scaled_reshaped.reshape(num_s, seq_l, num_c)

            self.data_list.append(x_final)
            self.labels_list.append(y)

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        x_window = self.data[idx]  # (seq_len, num_channels)
        y = self.labels[idx]

        # 转换为 PyTorch 张量
        # (seq_len, num_channels) -> (num_channels, seq_len)
        x_tensor = torch.from_numpy(x_window).float().permute(1, 0)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor