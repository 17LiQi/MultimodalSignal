# dataset/wesad_sequence_dataset.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class WesadSequenceDataset(Dataset):
    def __init__(self, data_path: Path, subjects: list, sequence_length=20, step=5):
        """
        加载阶段性特征序列，并创建重叠的子序列。
        """
        self.sequences = []
        self.labels = []

        all_phase_files = [f for f in data_path.glob('*_X.npy')]

        for sid in subjects:
            # 找到属于该受试者的所有阶段文件
            subject_phase_files = [f for f in all_phase_files if f.name.startswith(sid)]

            subject_data_list = []
            # 先加载并归一化该受试者的所有数据
            for x_file in subject_phase_files:
                subject_data_list.append(np.load(x_file))

            if not subject_data_list: continue

            # 对该受试者的 *全部* 数据进行受试者内归一化
            full_subject_data = np.concatenate(subject_data_list, axis=0)
            scaler = StandardScaler()
            full_subject_data_scaled = scaler.fit_transform(full_subject_data)

            # 将归一化后的数据重新按阶段切分
            processed_phases = []
            current_pos = 0
            for data_arr in subject_data_list:
                segment = full_subject_data_scaled[current_pos: current_pos + len(data_arr)]
                processed_phases.append(segment)
                current_pos += len(data_arr)

            # 现在，在每个归一化后的阶段内部，创建子序列
            for i, phase_data in enumerate(processed_phases):
                y_file = subject_phase_files[i].with_name(subject_phase_files[i].name.replace('_X.npy', '_y.npy'))
                phase_labels = np.load(y_file)

                for j in range(0, len(phase_data) - sequence_length + 1, step):
                    seq = phase_data[j: j + sequence_length]
                    # 使用子序列最后一个时间步的标签
                    label = phase_labels[j + sequence_length - 1]

                    self.sequences.append(seq)
                    self.labels.append(label)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x_seq = self.sequences[idx]
        y = self.labels[idx]

        x_tensor = torch.from_numpy(x_seq).float()
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Transformer 期望 (seq_len, batch, features)
        # 但 PyTorch 的 TransformerEncoderLayer 可以通过 batch_first=True
        # 接收 (batch, seq_len, features)，这更方便
        return x_tensor, y_tensor
