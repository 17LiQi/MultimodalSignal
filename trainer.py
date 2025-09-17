# trainer.py

import torch
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from typing import Dict, List

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from src.models.physioformer_model import PhysioFormer
from src.trainer.custom_loss import PhysioFormerLoss
from utils.early_stopping import EarlyStopping


class PhysioFormerTrainer:
    def __init__(self, model: PhysioFormer, config: Dict, output_dir: Path, num_classes: int, class_names: List[str]):
        """
        Trainer for the PhysioFormer model.

        Args:
            model (PhysioFormer): The PhysioFormer model instance.
            config (Dict): A dictionary containing trainer configurations (lr, epochs, etc.).
            output_dir (Path): Directory to save logs, models, and results.
            num_classes (int): The number of classes for classification.
            class_names (List[str]): A list of class names for reporting (e.g., ['Neutral', 'Stress', 'Amusement']).
        """
        self.model = model
        self.config = config
        self.output_dir = output_dir
        self.num_classes = num_classes
        self.class_names = class_names

        # Create output directories if they don't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.output_dir / 'training_log.txt'

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"PhysioFormer Training log starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Setup device, optimizer, and loss function
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.criterion = PhysioFormerLoss(lambda_reg=self.config.get('lambda_reg', 0.1))

        # Initialize Early Stopping
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping']['patience'],
            checkpoint_path=self.output_dir / 'best_model.pt',
            log_func=self._log
        )

    def _log(self, message: str):
        """Logs a message to the console and the log file."""
        tqdm.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train(self, train_loader, val_loader):
        self._log("\n--- Starting Training ---")
        self._log(f"Device: {self.device}")

        for epoch in range(self.config['epochs']):
            epoch_start_time = time.time()

            # --- Training Phase ---
            self.model.train()
            total_train_loss = 0.0
            train_preds, train_labels = [], []

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.config['epochs']} [Training]")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                model_output = self.model(inputs)
                loss = self.criterion(model_output, labels)

                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

                # For calculating training accuracy
                preds = torch.argmax(model_output['logits'], dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                progress_bar.set_postfix(loss=loss.item())

            # --- Validation Phase ---
            val_results = self.evaluate(val_loader, is_test_set=False)

            avg_train_loss = total_train_loss / len(train_loader)
            train_acc = accuracy_score(train_labels, train_preds)

            epoch_duration = time.time() - epoch_start_time
            log_msg = (
                f"Epoch {epoch + 1}/{self.config['epochs']} | Time: {epoch_duration:.2f}s | "
                f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_results['loss']:.4f} | Val Acc: {val_results['accuracy'] * 100:.2f}%"
            )
            self._log(log_msg)

            # --- Early Stopping Check ---
            self.early_stopping(val_results['loss'], self.model)
            if self.early_stopping.early_stop:
                self._log("Early stopping triggered!")
                break

        self._log("--- Training Finished ---")
        self._log(f"Loading best model weights from: {self.early_stopping.checkpoint_path}")
        self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path))

    def evaluate(self, data_loader, is_test_set: bool = True) -> Dict:
        """
        Evaluates the model on a given dataset.

        Args:
            data_loader: The DataLoader for the dataset to evaluate.
            is_test_set (bool): If True, will print a detailed report and save results.

        Returns:
            A dictionary containing performance metrics.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        desc = "Testing" if is_test_set else "Validating"
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc=desc):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                model_output = self.model(inputs)
                loss = self.criterion(model_output, labels)
                total_loss += loss.item()

                preds = torch.argmax(model_output['logits'], dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(data_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        results = {
            "loss": avg_loss,
            "accuracy": accuracy,
            "f1_score_weighted": f1,
            "predictions": np.array(all_preds),
            "true_labels": np.array(all_labels)
        }

        if is_test_set:
            self._log("\n===== Final Test Set Performance =====")
            self._log(f"\nOverall Accuracy: {accuracy:.4f}")
            self._log(f"Weighted F1-score: {f1:.4f}\n")

            # --- Classification Report ---
            report = classification_report(
                results["true_labels"],
                results["predictions"],
                target_names=self.class_names,
                labels=np.arange(len(self.class_names)),
                digits=4,
                zero_division=0  # Avoid warnings for classes with no support
            )
            self._log("--- Classification Report ---")
            self._log(report)

            # --- Confusion Matrix ---
            self._log("--- Confusion Matrix ---")
            cm = confusion_matrix(results["true_labels"], results["predictions"])
            self._log(str(cm))
            self.plot_confusion_matrix(cm)

        return results

    def plot_confusion_matrix(self, cm):
        """Saves a plot of the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names, yticklabels=self.class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')

        save_path = self.output_dir / 'confusion_matrix.png'
        plt.savefig(save_path)
        plt.close()
        self._log(f"Confusion matrix plot saved to: {save_path}")