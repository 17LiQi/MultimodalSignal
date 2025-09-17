# file: main_train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from tqdm import tqdm
import time
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import our custom modules
from dataset import WesadLosoDataset
from model import PlaceholderModel

# ======================================================================================
# --- SECTION 1: PATH & EXPERIMENT CONFIGURATION (All settings are here) ---
# ======================================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
PREPROCESSED_DATA_PATH = PROJECT_ROOT / "data"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Ensure directories exist
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
(OUTPUT_PATH / "checkpoints").mkdir(exist_ok=True)

CONFIG = {
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 64,
    "early_stopping_patience": 10,
    "model_params": {
        "num_channels": 24,  # This will be calculated automatically later
        "num_classes": 3,
        "window_length": 1920  # 30s * 64Hz
    }
}

CLASS_NAMES = ['Neutral', 'Amusement', 'Stress']


# ======================================================================================
# --- SECTION 2: TRAINER UTILITIES (Trainer, EarlyStopping, Loss) ---
# All training-related tools are consolidated here for simplicity.
# ======================================================================================

class EarlyStopping:
    def __init__(self, patience=7, path='checkpoint.pt', trace_func=print):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class ModelTrainer:
    def __init__(self, model, config, output_dir, class_names):
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.class_names = class_names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            path=self.output_dir / 'checkpoints' / 'best_model.pt'
        )

    def train(self, train_loader, val_loader):
        print(f"\n--- Starting Training on {self.device} ---")
        for epoch in range(self.config['epochs']):
            self.model.train()
            total_train_loss = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']} [Train]")
            for inputs, labels in train_pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()
                train_pbar.set_postfix(loss=loss.item())

            avg_train_loss = total_train_loss / len(train_loader)

            val_loss, val_acc, _ = self.evaluate(val_loader, is_validation=True)

            print(
                f"Epoch {epoch + 1} Summary: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping triggered!")
                break

        print("--- Training Finished. Loading best model. ---")
        self.model.load_state_dict(torch.load(self.early_stopping.path))

    def evaluate(self, data_loader, is_validation=False):
        self.model.eval()
        total_loss = 0
        all_preds, all_labels = [], []
        desc = "Validating" if is_validation else "Testing"

        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc=desc):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        return avg_loss, accuracy, f1

    def final_test(self, test_loader):
        print("\n--- Running Final Evaluation on Test Set ---")
        test_loss, test_acc, test_f1 = self.evaluate(test_loader)

        print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.2f}% | Test F1-score: {test_f1:.4f}")
        return test_acc, test_f1


# ======================================================================================
# --- SECTION 3: MAIN EXECUTION SCRIPT (LOSO Cross-Validation) ---
# ======================================================================================

def main():
    # Get all subject IDs
    subject_ids = sorted([f"S{i}" for i in range(2, 18) if i != 12])

    # Load channel names to configure the model correctly
    try:
        channel_names = np.load(PREPROCESSED_DATA_PATH / 'channel_names.npy')
        CONFIG['model_params']['num_channels'] = len(channel_names)
        print(f"Found {len(channel_names)} channels: {channel_names.tolist()}")
    except FileNotFoundError:
        print("Error: 'channel_names.npy' not found. Please run preprocess.py first.")
        return

    all_test_accuracies = []
    all_test_f1_scores = []

    for test_subject in subject_ids:
        fold_output_dir = OUTPUT_PATH / f'fold_test_{test_subject}'
        fold_output_dir.mkdir(exist_ok=True)
        print(f"\n===== Starting LOSO Fold: Testing on {test_subject} =====")

        # 1. Split subject IDs for this fold
        train_subjects = [sid for sid in subject_ids if sid != test_subject]
        # We'll use one of the training subjects as a validation set
        val_subject = train_subjects.pop()

        # 2. Create Datasets and DataLoaders
        train_dataset = WesadLosoDataset(PREPROCESSED_DATA_PATH, train_subjects, test_subject)
        val_dataset = WesadLosoDataset(PREPROCESSED_DATA_PATH, [val_subject], test_subject)
        test_dataset = WesadLosoDataset(PREPROCESSED_DATA_PATH, [test_subject], test_subject)

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

        # 3. Initialize model and trainer for this fold
        model = PlaceholderModel(**CONFIG['model_params'])
        trainer = ModelTrainer(model, CONFIG, fold_output_dir, CLASS_NAMES)

        # 4. Train and evaluate
        trainer.train(train_loader, val_loader)
        test_acc, test_f1 = trainer.final_test(test_loader)

        all_test_accuracies.append(test_acc)
        all_test_f1_scores.append(test_f1)

        print(f"===== Fold {test_subject} Finished. Accuracy: {test_acc * 100:.2f}%, F1: {test_f1:.4f} =====")

    # --- Final Summary of LOSO Cross-Validation ---
    mean_acc = np.mean(all_test_accuracies)
    std_acc = np.std(all_test_accuracies)
    mean_f1 = np.mean(all_test_f1_scores)
    std_f1 = np.std(all_test_f1_scores)

    summary_msg = (
        f"\n\n====== FINAL Cross-Validation Summary ======\n"
        f"Average Accuracy over {len(subject_ids)} folds: {mean_acc * 100:.2f}% (± {std_acc * 100:.2f}%)\n"
        f"Average Weighted F1-score over {len(subject_ids)} folds: {mean_f1:.4f} (± {std_f1:.4f})\n"
        f"=========================================="
    )
    print(summary_msg)

    # Save the final results to a file
    with open(OUTPUT_PATH / "final_summary.txt", "w") as f:
        f.write(summary_msg)


if __name__ == '__main__':
    main()