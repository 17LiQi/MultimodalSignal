# file: dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List


class WesadLosoDataset(Dataset):
    """
    PyTorch Dataset for WESAD, designed for Leave-One-Subject-Out (LOSO) cross-validation.
    It loads preprocessed data and applies normalization parameters specific to the LOSO fold.
    """

    def __init__(self, data_root: Path, subject_ids: List[str], test_subject: str):
        """
        Args:
            data_root (Path): The directory where preprocessed .npy and .npz files are stored.
            subject_ids (List[str]): A list of subject IDs to be included in this dataset split (e.g., train_ids or test_ids).
            test_subject (str): The ID of the subject left out in this fold, used to load the correct normalization parameters.
        """
        self.data_root = data_root
        self.subject_ids = subject_ids

        # Load all data for the given subjects into memory
        X_list = [np.load(self.data_root / f'{sid}_X.npy') for sid in self.subject_ids]
        y_list = [np.load(self.data_root / f'{sid}_y.npy') for sid in self.subject_ids]

        self.X_data = np.concatenate(X_list, axis=0)
        self.y_data = np.concatenate(y_list, axis=0)

        # Load the correct normalization parameters for this specific LOSO fold
        norm_params_path = self.data_root / f'norm_params_for_{test_subject}.npz'
        norm_params = np.load(norm_params_path)
        self.mean = torch.from_numpy(norm_params['mean']).float()
        self.std = torch.from_numpy(norm_params['std']).float()

        print(
            f"Loaded data for subjects {subject_ids} (Test Subject: {test_subject}). Total samples: {len(self.X_data)}")

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, idx):
        # Get a single window and its label
        x_window = self.X_data[idx]
        y_label = self.y_data[idx]

        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x_window).float()  # Shape: (L, C)
        y_tensor = torch.tensor(y_label, dtype=torch.long)

        # Apply Z-score normalization using the pre-computed stats for the fold
        x_tensor = (x_tensor - self.mean) / (self.std + 1e-8)  # Add epsilon for stability

        # PyTorch models (especially CNNs/Transformers) expect (L, C) or (C, L).
        # Transformers often expect (L, C). Let's stick with that for now.
        # If your model needs (C, L), you can transpose it here: x_tensor = x_tensor.T

        return x_tensor, y_tensor