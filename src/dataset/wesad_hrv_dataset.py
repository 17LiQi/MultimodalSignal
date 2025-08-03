# dataset/wesad_hrv_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadHrvDataset(Dataset):
    def __init__(self, data_path, subjects):
        self.data_list = []
        self.labels_list = []

        for sid in subjects:
            x = np.load(data_path / f'{sid}_X.npy')
            y = np.load(data_path / f'{sid}_y.npy')

            # 受试者内归一化 (在 1D 序列上)
            scaler = StandardScaler()
            # fit_transform 需要 2D 输入，所以先 reshape
            x_scaled = scaler.fit_transform(x)

            self.data_list.append(x_scaled)
            self.labels_list.append(y)

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # 数据是 (seq_len,)
        x_seq = self.data[idx]
        y = self.labels[idx]

        # 增加一个通道维度以适应 CNN/Transformer 模型
        # (seq_len,) -> (1, seq_len)
        x_tensor = torch.from_numpy(x_seq).float().unsqueeze(0)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor