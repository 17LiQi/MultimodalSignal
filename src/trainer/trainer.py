import torch
import time
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from utils.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from threading import Lock

# 初始化 tqdm 的全局锁
tqdm.set_lock(Lock())


class Trainer:
    def __init__(self, model, config, output_path: Path):
        self.model = model
        self.config = config
        self.output_path = output_path

        # 创建本次运行的独立输出目录
        self.run_dir = self.output_path / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.run_dir / 'training_log.txt'

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.lr = config.trainer.learning_rate
        self.epochs = config.trainer.epochs
        self.batch_size = config.trainer.batch_size

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)  # 移动损失函数到 GPU

        self.early_stopping = None
        if config.trainer.early_stopping.enabled:
            self.early_stopping = EarlyStopping(
                patience=config.trainer.early_stopping.patience,
                delta=config.trainer.early_stopping.delta,
                checkpoint_path=self.run_dir / 'best_model.pt',
                verbose=True,
                mode=config.trainer.early_stopping.mode
            )

        # 记录总训练开始时间
        self.total_start_time = time.time()

    def _log(self, message):
        """记录日志到文件和控制台"""
        tqdm.write(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train(self, train_loader, val_loader):
        self._log("--- 开始训练 ---")
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0

            # 使用线程安全的 tqdm，设置 position=0
            with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Training]", position=0,
                      leave=True) as progress_bar:
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    train_loss += loss.item()

                    progress_bar.set_postfix(loss=loss.item())

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # 验证
            val_loss, val_acc, val_f1 = self.evaluate(val_loader, epoch)
            log_msg = (f"Epoch {epoch + 1}/{self.epochs} | "
                       f"耗时: {epoch_duration:.2f}s | "
                       f"训练损失: {train_loss / len(train_loader):.4f} | "
                       f"验证损失: {val_loss:.4f} | "
                       f"验证Acc: {val_acc:.4f} | "
                       f"验证F1: {val_f1:.4f}")
            self._log("\n" + log_msg)

            # 调用早停
            if self.early_stopping:
                score_to_monitor = val_loss if self.early_stopping.mode == 'min' else val_acc
                self.early_stopping(score_to_monitor, self.model)
                if self.early_stopping.early_stop:
                    self._log("触发早停")
                    break

        # 训练结束后，加载最佳模型并输出总训练时长
        if self.early_stopping and self.early_stopping.early_stop:
            self._log(f"加载性能最佳的模型权重从: {self.early_stopping.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path))

        total_end_time = time.time()
        total_duration = total_end_time - self.total_start_time
        self._log(f"--- 训练完成 --- 总训练时长: {total_duration:.2f}秒")

    def evaluate(self, data_loader, epoch=0, is_test=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        desc = "Testing" if is_test else f"Epoch {epoch + 1}/{self.epochs} [Validation]"
        # 使用线程安全的 tqdm，设置 position=1
        with tqdm(data_loader, desc=desc, position=1, leave=True) as progress_bar:
            with torch.no_grad():
                for inputs, labels in data_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)

                    _, preds = torch.max(outputs, 1)
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        loss = total_loss / len(data_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        if is_test:
            self.plot_confusion_matrix(all_labels, all_preds)
            log_msg = f"测试集结果: Loss={loss:.4f}, Accuracy={acc:.4f}, F1-score={f1:.4f}"
            self._log(log_msg)

        return loss, acc, f1

    def plot_confusion_matrix(self, true_labels, pred_labels):
        try:
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Neutral', 'Amusement', 'Stress'],
                        yticklabels=['Neutral', 'Amusement', 'Stress'])
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('Confusion Matrix')
            cm_path = self.run_dir / 'confusion_matrix.png'
            plt.savefig(cm_path)
            plt.close()
            self._log(f"混淆矩阵已保存至: {cm_path}")
        except Exception as e:
            self._log(f"保存混淆矩阵失败: {str(e)}")
