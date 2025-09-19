import torch
import time
import numpy as np
from sklearn.utils import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class EarlyStopping:
    def __init__(self, patience=7, delta=0, checkpoint_path='checkpoint.pt', verbose=False, log_func=None):
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.log_func = log_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.log_func:
                self.log_func(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.checkpoint_path)

class Trainer:
    def __init__(self, model, fold_output_dir: Path, config):
        self.model = model
        self.fold_dir = fold_output_dir
        self.config = config

        # 确保输出目录存在
        self.fold_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.fold_dir / 'training_log.txt'

        # 初始化日志文件
        with open(self.log_file, 'w') as f:
            f.write(f"Training log for run starting at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 50 + "\n")

        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # 从 config 中提取参数
        self.epochs = config['trainer']['epochs']
        self.learning_rate = config['trainer']['learning_rate']
        self.patience = config['trainer']['early_stopping']['patience']
        self.weight_decay = config['trainer']['weight_decay']
        self.use_class_weights = config['trainer'].get('use_class_weights', False)

        # 设置优化器和损失函数
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

        # 设置学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',  # 监控 val_loss
            factor=0.1,  # 学习率衰减为 10%
            patience=3
        )

        # 动态设置类别加权损失
        class_weights = None
        if self.use_class_weights and hasattr(config['trainer'], 'train_dataset'):
            self._log("计算类别权重...")
            try:
                train_dataset = config['trainer']['train_dataset']
                class_weights = compute_class_weight(
                    'balanced',
                    classes=np.unique(train_dataset.y_data),
                    y=train_dataset.y_data
                )
                class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
                self._log(f"已启用类别加权损失，权重为: {class_weights.cpu().numpy()}")
            except Exception as e:
                self._log(f"警告: 类别权重计算失败 - {e}。将使用标准损失函数。")
                class_weights = None

        if class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights).to(self.device)

        # 设置早停
        self.early_stopping = None
        if config['trainer']['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.patience,
                delta=config['trainer']['early_stopping']['delta'],
                checkpoint_path=self.fold_dir / 'best_model.pt',
                verbose=True,
                log_func=self._log
            )

        # 记录总训练开始时间
        self.total_start_time = time.time()

    def _log(self, message):
        """记录日志到文件和控制台"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')

    def train(self, train_loader, val_loader):
        best_val_acc = 0  # 追踪最佳验证准确率

        for epoch in range(self.epochs):
            epoch_start_time = time.time()
            self.model.train()
            train_loss = 0.0

            # 训练进度条
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{self.epochs} [Training]")

            for inputs, labels in progress_bar:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                batch_size = labels.size(0)
                train_loss += loss.item() * batch_size
                progress_bar.set_postfix(loss=loss.item())

            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time

            # 验证
            val_loss, val_acc, val_f1, val_preds, val_labels = self.evaluate(val_loader, is_val=True)
            self.scheduler.step(val_loss)

            # 保存最佳验证准确率的混淆矩阵
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # self.plot_confusion_matrix(val_labels, val_preds, filename="validation_cm.png")

            # 记录日志
            log_msg = (f"Epoch {epoch + 1}/{self.epochs} | "
                       f"耗时: {epoch_duration:.2f}s | "
                       f"训练损失: {train_loss / len(train_loader.dataset):.4f} | "
                       f"验证损失: {val_loss:.4f} | "
                       f"验证Acc: {val_acc:.4f} | "
                       f"验证F1: {val_f1:.4f}")
            self._log(log_msg)

            # 早停检查
            if self.early_stopping:
                score_to_monitor = val_loss
                self.early_stopping(score_to_monitor, self.model)
                if self.early_stopping.early_stop:
                    self._log("触发早停")
                    break

        # 加载最佳模型
        if self.early_stopping and self.early_stopping.early_stop:
            self._log(f"加载性能最佳的模型权重从: {self.early_stopping.checkpoint_path}")
            self.model.load_state_dict(torch.load(self.early_stopping.checkpoint_path, weights_only=True))

        total_end_time = time.time()
        total_duration = total_end_time - self.total_start_time
        self._log(f"--- 训练完成 --- 总训练时长: {total_duration:.2f}秒")

    def evaluate(self, data_loader, is_test=False, is_val=False):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        loss = total_loss / len(data_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')

        if is_test:
            self.plot_confusion_matrix(all_labels, all_preds, filename="test_confusion_matrix.png")
            self._log(f"\n--- 最终测试结果 ---")
            self._log(f"测试损失: {loss:.4f} | 测试Acc: {acc:.4f} | 测试F1: {f1:.4f}")

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
            self._log(f"混淆矩阵已保存至: {cm_path}")
        except Exception as e:
            self._log(f"保存混淆矩阵失败: {str(e)}")