# src/main.py
import time

import numpy as np
import torch
import pandas as pd
from datetime import datetime

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import compute_class_weight
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Import all our modules
from src.dataset.dataset import PhysioFormerDataset
from src.models.physioformer_model import PhysioFormer
from src.trainer.Trainer import PhysioFormerTrainer
from trainer.custom_loss import PhysioFormerLoss
from utils.early_stopping import EarlyStopping
from utils.path_manager import get_path_manager


def main():
    paths = get_path_manager()
    # --- 1. Configuration ---
    CONFIG = {
        'dataset_type': 'chest',  # 'wrist' or 'chest'
        'num_classes': 3,
        'class_names': ['Neutral', 'Stress', 'Amusement'],  # Adjust if using different labels
        'data_path': paths.DATA_ROOT,
        'output_root': paths.OUTPUT_ROOT,
        'trainer': {
            'learning_rate': 0.0005,
            'epochs': 180,
            'lambda_reg': 0.1,  # Regularization strength
            'batch_size': 32,  # The dataset is small, so a small batch size is fine
            'early_stopping': {
                'patience': 25,
            }
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- 2. Setup Directories & Logging ---
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    main_output_dir = paths.OUTPUT_ROOT / f'physioformer_loso_sequential_{run_timestamp}'
    main_output_dir.mkdir(parents=True, exist_ok=True)

    main_log_file = main_output_dir / 'cv_summary.txt'

    def log_main(message):
        print(message)
        with open(main_log_file, 'a') as f:
            f.write(message + '\n')

    log_main("====== Starting Sequential-Training Leave-One-Subject-Out CV Run ======")
    log_main(f"All results will be saved in: {main_output_dir}")

    # --- 3. Load Full Dataset and Subject Groups ---
    full_dataset = PhysioFormerDataset(
        data_path=CONFIG['data_path'],
        dataset_type=CONFIG['dataset_type']
    )
    groups = np.load(CONFIG['data_path'] / f'subjects_{CONFIG["dataset_type"]}.npy')
    feature_names_path = CONFIG['data_path'] / f'features_{CONFIG["dataset_type"]}.json'
    feature_names = pd.read_json(feature_names_path)['feature'].tolist()

    # --- 4. Setup LOSO Cross-Validator ---
    logo = LeaveOneGroupOut()
    all_fold_results = []

    unique_subjects = np.unique(groups)
    subject_id_map = {i: f"S{i + 2 if i < 10 else i + 3}" for i in unique_subjects}  # Map index back to S_ID

    # --- 5. Start the LOSO Loop ---
    for fold, (train_val_indices, test_indices) in enumerate(logo.split(X=np.arange(len(full_dataset)), groups=groups)):

        test_subject_idx = np.unique(groups[test_indices])[0]
        test_subject_id = subject_id_map.get(test_subject_idx, f"Unknown_{test_subject_idx}")
        fold_output_dir = main_output_dir / f'fold_{fold}_test_on_{test_subject_id}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        log_main(
            f"\n----- FOLD {fold + 1}/{logo.get_n_splits(groups=groups)} | Testing on Subject {test_subject_id} -----")

        # --- Data Splitting and Loader Creation ---
        # A. Create Test Loader
        test_dataset = Subset(full_dataset, test_indices)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['trainer']['batch_size'], shuffle=False)

        # B. Split remaining data into Train and Validation
        np.random.shuffle(train_val_indices)  # Shuffle before splitting
        val_size = int(0.15 * len(train_val_indices))
        val_indices = train_val_indices[:val_size]
        train_indices = train_val_indices[val_size:]

        val_dataset = Subset(full_dataset, val_indices)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['trainer']['batch_size'], shuffle=False)

        # C. CRITICAL CHANGE: Create a dictionary of DataLoaders for each training subject
        train_subject_loaders = {}
        train_groups = groups[train_indices]
        for train_subject_idx in np.unique(train_groups):
            # Find indices for this specific training subject
            subject_specific_indices = train_indices[np.where(train_groups == train_subject_idx)[0]]
            subject_dataset = Subset(full_dataset, subject_specific_indices)
            train_subject_loaders[train_subject_idx] = DataLoader(
                subject_dataset,
                batch_size=CONFIG['trainer']['batch_size'],
                shuffle=True,
                drop_last=True  # Important to avoid batch_size=1 errors
            )

            # 1. Get all labels from the training indices
            train_labels = full_dataset.labels[train_indices]

            # 2. Calculate weights using scikit-learn's utility
            # This handles cases where a class might be missing in a fold
            class_weights = compute_class_weight(
                class_weight='balanced',
                classes=np.unique(train_labels),
                y=train_labels
            )

            # 3. Convert to a PyTorch tensor
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

            log_main(f"Applying class weights for this fold: {np.round(class_weights, 2)}")

            # --- Setup for this fold's training ---
            model = PhysioFormer(feature_names=feature_names, num_classes=CONFIG['num_classes'])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['trainer']['learning_rate'], weight_decay=1e-3)

            # 4. Pass the calculated weights to our custom loss function
            # We need to modify the loss class to accept weights
            criterion = PhysioFormerLoss(
                lambda_reg=CONFIG['trainer']['lambda_reg'],
                class_weights=class_weights_tensor  # <-- Pass weights here
            )
        early_stopping = EarlyStopping(
            patience=CONFIG['trainer']['early_stopping']['patience'],
            checkpoint_path=fold_output_dir / 'best_model.pt',
        )

        # --- Manual Training Loop to implement sequential training ---
        log_main("--- Starting Sequential Training for this fold ---")
        for epoch in range(CONFIG['trainer']['epochs']):
            epoch_start_time = time.time()
            model.train()

            # Use tqdm for the outer epoch loop
            epoch_progress = tqdm(total=len(train_indices), desc=f"Epoch {epoch + 1}/{CONFIG['trainer']['epochs']}")

            # SEQUENTIAL TRAINING: Loop through each training subject's dataloader
            training_subjects = list(train_subject_loaders.keys())
            np.random.shuffle(training_subjects)  # Shuffle subject order each epoch

            for subject_idx in training_subjects:
                loader = train_subject_loaders[subject_idx]
                for inputs, labels in loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    optimizer.zero_grad()
                    model_output = model(inputs)
                    loss = criterion(model_output, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_progress.update(inputs.size(0))  # Update progress bar
                    epoch_progress.set_postfix(loss=loss.item())

            epoch_progress.close()

            # --- Validation after each epoch ---
            model.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    model_output = model(inputs)
                    loss = criterion(model_output, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            log_main(f"Epoch {epoch + 1} | Time: {time.time() - epoch_start_time:.2f}s | Val Loss: {avg_val_loss:.4f}")

            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                log_main("Early stopping triggered!")
                break

        # --- Final Evaluation for this fold ---
        log_main(f"Loading best model for fold {fold + 1}")
        model.load_state_dict(torch.load(early_stopping.checkpoint_path))

        # We need a temporary trainer object just to use its evaluate method
        eval_trainer = PhysioFormerTrainer(model, CONFIG['trainer'], fold_output_dir, CONFIG['num_classes'],
                                           CONFIG['class_names'])
        test_results = eval_trainer.evaluate(test_loader, is_test_set=True)
        all_fold_results.append(test_results)

    # --- Summarize CV Results ---
    log_main("\n\n====== FINAL Cross-Validation Summary ======")
    all_accuracies = [res['accuracy'] for res in all_fold_results]
    all_f1_scores = [res['f1_score_weighted'] for res in all_fold_results]

    mean_acc = np.mean(all_accuracies)
    std_acc = np.std(all_accuracies)
    mean_f1 = np.mean(all_f1_scores)
    std_f1 = np.std(all_f1_scores)

    summary_text = (
        f"Average Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})\n"
        f"Average Weighted F1-score: {mean_f1:.4f} (+/- {std_f1:.4f})\n\n"
        f"Individual Fold Accuracies:\n{all_accuracies}"
    )
    log_main(summary_text)


if __name__ == '__main__':
    main()