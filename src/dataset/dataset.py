# dataset.py

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple


class PhysioFormerDataset(Dataset):
    """
    A simple PyTorch Dataset for loading the pre-processed feature vectors
    created by `preprocess_physioformer.py`.

    This dataset is straightforward because all the complex feature engineering
    has already been handled in the preprocessing script. Its only job is to
    load a feature vector and its corresponding label.
    """

    def __init__(self, data_path: Path, dataset_type: str = 'wrist'):
        """
        Args:
            data_path (Path): The path to the directory containing the processed .npy files
                              (e.g., './data/physioformer_processed').
            dataset_type (str): The type of dataset to load. Must be either 'wrist' or 'chest'.
        """
        super(PhysioFormerDataset, self).__init__()

        if dataset_type not in ['wrist', 'chest']:
            raise ValueError(f"dataset_type must be either 'wrist' or 'chest', but got {dataset_type}")

        # Construct the full paths to the data and label files
        features_file = data_path / f'X_{dataset_type}.npy'
        labels_file = data_path / f'y_{dataset_type}.npy'

        # Check if the files exist
        if not features_file.exists() or not labels_file.exists():
            raise FileNotFoundError(
                f"Could not find preprocessed data files in {data_path}. "
                f"Please run `preprocess_physioformer.py` first."
            )

        # Load the data and labels from the .npy files
        self.features = np.load(features_file)
        self.labels = np.load(labels_file)

        print(f"Successfully loaded '{dataset_type}' dataset.")
        print(f"  - Number of samples: {len(self.labels)}")
        print(f"  - Number of features: {self.features.shape[1]}")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a single sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            A tuple containing:
            - The feature vector as a FloatTensor.
            - The label as a LongTensor.
        """
        # Get the feature vector and label for the given index
        feature_vector = self.features[index]
        label = self.labels[index]

        # Convert to PyTorch tensors
        # Features should be float, and labels should be long for CrossEntropyLoss
        # Note: The paper's labels are 1, 2, 3. CrossEntropyLoss expects 0, 1, 2.
        # We subtract 1 to adjust the label range.
        return (
            torch.from_numpy(feature_vector).float(),
            torch.tensor(label - 1, dtype=torch.long)
        )