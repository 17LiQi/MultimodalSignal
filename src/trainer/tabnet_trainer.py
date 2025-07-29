# trainer/tabnet_trainer.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


class TabNetTrainer:
    """一个专门用于 TabNet 模型的训练器。"""

    def __init__(self, model_wrapper, config, fold_output_dir: Path):
        self.model_wrapper = model_wrapper
        self.config = config
        self.fold_dir = fold_output_dir
        self.log_file = self.fold_dir / 'training_log.txt'

        with open(self.log_file, 'w') as f:
            f.write(f"Log for TabNet run starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    def _log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train_and_evaluate(self, train_dataset, val_dataset, test_dataset, test_subject=None):
        # TabNet 直接使用 NumPy 数组，所以我们从 Dataset 对象中提取它们
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_val, y_val = val_dataset.data, val_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels

        checkpoint_path = self.fold_dir / "best_model"

        self._log("\n--- 开始训练 TabNet 模型 ---")
        self.model_wrapper.fit(
            X_train, y_train, X_val, y_val,
            max_epochs=self.config.trainer.epochs,
            patience=self.config.trainer.early_stopping.patience,
            batch_size=self.config.trainer.batch_size,
            checkpoint_path=checkpoint_path
        )
        self._log("--- 训练完成 ---")

        # 保存训练好的模型
        self.model_wrapper.load_best_model(checkpoint_path)

        self._log(f"\n--- 开始在测试集 {test_subject} 上进行评估 ---")
        preds = self.model_wrapper.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        self.plot_confusion_matrix(y_test, preds, filename="test_confusion_matrix.png")
        self._log(f"测试集结果: Accuracy={acc:.4f}, F1-score={f1:.4f}")

        return acc, f1

    def plot_confusion_matrix(self, true_labels, pred_labels, filename="confusion_matrix.png"):
        try:
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Neutral', 'Amusement', 'Stress'],
                        yticklabels=['Neutral', 'Amusement', 'Stress'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            cm_path = self.fold_dir / filename
            plt.savefig(cm_path)
            plt.close()
        except Exception as e:
            self._log(f"保存混淆矩阵失败: {str(e)}")
