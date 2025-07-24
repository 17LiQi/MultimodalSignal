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
    def __init__(self, model, config, fold_output_dir: Path):
        self.model = model
        self.config = config
        self.fold_dir = fold_output_dir
        self.log_file = self.fold_dir / 'training_log.txt'

        # 确保日志文件存在且为空
        with open(self.log_file, 'w') as f:
            f.write(f"Training log for run starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

        # --- 直接使用传入的折叠目录 ---
        self.fold_dir = fold_output_dir  # 重命名 run_dir 为 fold_dir 更清晰
        self.log_file = self.fold_dir / 'training_log.txt'

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 设置超参数
        self.lr = config.trainer.learning_rate
        self.epochs = config.trainer.epochs
        self.batch_size = config.trainer.batch_size

        # 设置优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)  # 移动损失函数到 GPU

        self.early_stopping = None
        # --- 保存最佳模型到折叠目录 ---
        if config.trainer.early_stopping.enabled:
            self.early_stopping = EarlyStopping(
                patience=config.trainer.early_stopping.patience,
                delta=config.trainer.early_stopping.delta,
                checkpoint_path=self.fold_dir / 'best_model.pt',  # <-- 2. 路径更新
                verbose=True,
                log_func=self._log
            )

        # 记录总训练开始时间
        self.total_start_time = time.time()

    def _log(self, message):
        """记录日志到文件和控制台"""
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train(self, train_loader, val_loader):
        best_val_acc = 0  # 用于追踪最佳验证准确率，以保存对应的混淆矩阵

        # 外层循环只负责迭代 epoch，不使用 tqdm
        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0

            # --- 最简化的进度条 ---
            # 1. 在每个 epoch 的开始，为 train_loader 创建一个简单的进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Training]")

            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # --- 简化验证 ---
            # 3. 调用 evaluate，它将不再显示自己的进度条
            val_loss, val_acc, val_f1, val_preds, val_labels = self.evaluate(val_loader, is_val=True)
            # 只有当验证集表现提升时，才更新并保存验证集的混淆矩阵
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # self._log(f"  - 新的最佳验证准确率: {val_acc:.4f}。正在更新验证集混淆矩阵...")
                # self.plot_confusion_matrix(val_labels, val_preds, filename="validation_cm.png")

            # 4. 在 epoch 结束后写入完整的统计日志
            log_msg = (f"Epoch {epoch + 1}/{self.epochs} | "
                       f"耗时: {epoch_duration:.2f}s | "
                       f"训练损失: {train_loss / len(train_loader):.4f} | "
                       f"验证损失: {val_loss:.4f} | "
                       f"验证Acc: {val_acc:.4f} | "
                       f"验证F1: {val_f1:.4f}")
            self._log(log_msg)

            if self.early_stopping:
                score_to_monitor = val_loss if self.early_stopping.mode == 'min' else val_acc
                self.early_stopping(score_to_monitor, self.model)
                if self.early_stopping.early_stop:
                    self._log("触发早停")
                    break

        # 训练结束后，加载最佳模型并输出总训练时长
        if self.early_stopping and self.early_stopping.early_stop:
            self._log(f"加载性能最佳的模型权重从: {self.early_stopping.checkpoint_path}")
            try:
                self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path, weights_only=True))
            except TypeError:
                self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path))

        total_end_time = time.time()
        total_duration = total_end_time - self.total_start_time
        self._log(f"--- 训练完成 --- 总训练时长: {total_duration:.2f}秒")

    def evaluate(self, data_loader, is_test=False, is_val=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        # 直接在无梯度的模式下迭代数据加载器
        with torch.no_grad():
            for inputs, labels in data_loader:  # 直接迭代，不使用 tqdm
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

        # 如果是测试集评估，则直接绘制并保存测试混淆矩阵
        if is_test:
            self.plot_confusion_matrix(all_labels, all_preds, filename="test_confusion_matrix.png")
            self._log(f"\n--- 最终测试结果 ---")
            self._log(f"测试损失: {loss:.4f} | 测试Acc: {acc:.4f} | 测试F1: {f1:.4f}")

        # 如果是验证，返回预测和标签供 train 方法判断
        if is_val:
            return loss, acc, f1, all_preds, all_labels

        return loss, acc, f1


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
