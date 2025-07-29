# data_utils/wesad_feature_dataset.py
import torch
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class WesadFeatureDataset(Dataset):
    def __init__(self, data_path, subjects):
        """
        data_path: processed_data/feature_fusion/ 的路径
        subjects: 要加载的受试者ID列表
        """
        self.data_list = []
        self.labels_list = []

        for sid in subjects:
            x = np.load(data_path / f'{sid}_X.npy')
            y = np.load(data_path / f'{sid}_y.npy')

            # --- 受试者内归一化  ---
            # 为当前受试者的数据独立地进行 fit 和 transform
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)

            self.data_list.append(x_scaled)
            self.labels_list.append(y)

        self.data = np.concatenate(self.data_list, axis=0)
        self.labels = np.concatenate(self.labels_list, axis=0)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        # 数据已经是归一化后的特征向量
        x_features = self.data[idx]
        y = self.labels[idx]

        x_tensor = torch.from_numpy(x_features).float()
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor