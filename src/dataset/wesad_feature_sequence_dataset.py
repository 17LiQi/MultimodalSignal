# dataset/wesad_feature_sequence_dataset.py
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadFeatureSequenceDataset(Dataset):
    def __init__(self, data_path, subjects, sequence_length=5, step=1):
        """
        将特征向量组织成序列。

        参数:
        - sequence_length (int): 每个样本包含的连续窗口数量。
        - step (int): 创建下一个序列时滑动的窗口步数。
        """
        self.sequences = []
        self.labels = []

        for sid in subjects:
            # 加载该受试者的未归一化特征数据
            x_subject = np.load(data_path / f'{sid}_X.npy')
            y_subject = np.load(data_path / f'{sid}_y.npy')

            # 对该受试者的 *全部* 数据进行受试者内归一化
            scaler = StandardScaler()
            x_subject_scaled = scaler.fit_transform(x_subject)

            # 在该受试者内部，创建重叠的序列
            for i in range(0, len(x_subject_scaled) - sequence_length + 1, step):
                # 提取一个序列的特征
                seq = x_subject_scaled[i: i + sequence_length]
                self.sequences.append(seq)

                # 标签：我们通常使用序列中最后一个时间步的标签作为整个序列的标签
                label = y_subject[i + sequence_length - 1]
                self.labels.append(label)

        # 将列表转换为Numpy数组以便索引
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 注意：这里的数据已经是 (sequence_length, in_features) 的形状
        # GRU/LSTM 模型期望的输入是 (seq_len, input_size)，所以这个形状可以直接使用
        x_seq = self.sequences[idx]
        y = self.labels[idx]

        x_tensor = torch.from_numpy(x_seq)
        y_tensor = torch.tensor(y, dtype=torch.long)

        return x_tensor, y_tensor