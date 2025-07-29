# trainer/ml_trainer.py
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MLTrainer:
    """一个通用的机器学习模型训练器 (适用于XGBoost, RandomForest等)。"""

    def __init__(self, model, config, fold_output_dir: Path):
        self.model = model
        self.config = config
        self.fold_dir = fold_output_dir
        self.log_file = self.fold_dir / 'training_log.txt'

    def _log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train_and_evaluate(self, train_dataset, val_dataset, test_dataset, test_subject_id):
        # 直接从Dataset对象中获取Numpy数据
        X_train, y_train = train_dataset.data, train_dataset.labels
        X_val, y_val = val_dataset.data, val_dataset.labels
        X_test, y_test = test_dataset.data, test_dataset.labels

        self._log("\n--- 开始训练 XGBoost 模型 ---")
        self.model.fit(X_train, y_train, X_val, y_val)
        self._log("--- 训练完成 ---")

        self._log(f"\n--- 开始在测试集 {test_subject_id} 上进行评估 ---")
        preds = self.model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')

        self.plot_confusion_matrix(y_test, preds, filename="test_confusion_matrix.png")
        self._log(f"测试集结果: Accuracy={acc:.4f}, F1-score={f1:.4f}")

        return acc, f1

    def plot_confusion_matrix(self, true_labels, pred_labels, filename="cm.png"):
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
        self._log(f"混淆矩阵已保存至: {cm_path}")