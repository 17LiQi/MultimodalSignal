# In your main.py

import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader, random_split

# Import all our modules
from src.dataset.dataset import PhysioFormerDataset
from src.models.physioformer_model import PhysioFormer
from src.trainer.Trainer import PhysioFormerTrainer
from utils.path_manager import get_path_manager


def main():
    paths = get_path_manager()
    # --- 1. Configuration ---
    # You can load this from a YAML file for more flexibility
    CONFIG = {
        'dataset_type': 'chest',  # 'wrist' or 'chest'
        'num_classes': 3,
        'class_names': ['Neutral', 'Stress', 'Amusement'],  # Adjust if using different labels
        'data_path': paths.DATA_ROOT,
        'output_root': paths.OUTPUT_ROOT,
        'trainer': {
            'learning_rate': 0.001,
            'epochs': 150,
            'lambda_reg': 0.1,  # Regularization strength
            'batch_size': 16,  # The dataset is small, so a small batch size is fine
            'early_stopping': {
                'patience': 20,
            }
        }
    }

    # --- 2. Setup Output Directory ---
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = paths.OUTPUT_ROOT / f'physioformer_run_{run_timestamp}'

    # --- 3. Load Data ---
    full_dataset = PhysioFormerDataset(
        data_path=CONFIG['data_path'],
        dataset_type=CONFIG['dataset_type']
    )

    # Simple split (you might want a more robust CV strategy for a real paper)
    train_size = int(0.7 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=CONFIG['trainer']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['trainer']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['trainer']['batch_size'], shuffle=False)

    # --- 4. Initialize Model ---
    # We need the feature names to initialize the model correctly
    feature_names_path = CONFIG['data_path'] / f'features_{CONFIG["dataset_type"]}.json'
    feature_names = pd.read_json(feature_names_path)['feature'].tolist()

    model = PhysioFormer(feature_names=feature_names, num_classes=CONFIG['num_classes'])

    # --- 5. Initialize Trainer and Start Training ---
    trainer = PhysioFormerTrainer(
        model=model,
        config=CONFIG['trainer'],
        output_dir=output_dir,
        num_classes=CONFIG['num_classes'],
        class_names=CONFIG['class_names']
    )

    trainer.train(train_loader, val_loader)

    # --- 6. Final Evaluation on Test Set ---
    trainer.evaluate(test_loader, is_test_set=True)


if __name__ == '__main__':
    main()