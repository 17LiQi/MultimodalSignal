# utils/early_stopping.py
import numpy as np
import torch


class EarlyStopping:
    """早停法以防止过拟合，并在验证损失不再改善时停止训练。"""

    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt', verbose=True, mode='min', log_func=print):
        """
        参数:
            patience (int): 验证性能在多少个epoch内没有提升就停止训练
            delta (float): 性能提升的最小阈值
            checkpoint_path (Path): 保存最佳模型权重的路径
            verbose (bool): 是否打印早停相关信息
            mode (str): 'min' 或 'max'。'min'表示监控指标越小越好 (如loss)，'max'反之 (如accuracy)
        """
        self.patience = patience
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose
        self.mode = mode
        self.log_func = log_func

        self.counter = 0
        self.best_score = np.Inf if self.mode == 'min' else -np.Inf
        self.early_stop = False

    def __call__(self, current_score, model):
        """
        在每个验证epoch后调用此方法。
        """
        score_improved = False
        if self.mode == 'min':
            if current_score < self.best_score - self.delta:
                self.best_score = current_score
                score_improved = True
        else:  # mode == 'max'
            if current_score > self.best_score + self.delta:
                self.best_score = current_score
                score_improved = True

        if score_improved:
            self.counter = 0
            if self.verbose:
                self.log_func(f'  - EarlyStopping: Val score improved ({self.best_score:.6f}). Saving model...')
            torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.verbose:
                self.log_func(f'  - EarlyStopping: Counter {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True