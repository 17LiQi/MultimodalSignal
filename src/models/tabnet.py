# models/tabnet_model.py
from pathlib import Path

import torch
from pytorch_tabnet.tab_model import TabNetClassifier


class TabNetModelWrapper:
    """
    对 pytorch-tabnet 的 TabNetClassifier 进行封装，使其接口更统一。
    """

    def __init__(self, in_features, num_classes, device='cuda', scheduler_params=None, **model_args):
        self.device = device

        # 定义优化器和学习率调度器
        optimizer_fn = torch.optim.Adam
        scheduler_fn = torch.optim.lr_scheduler.ReduceLROnPlateau

        if scheduler_params is None:
            scheduler_params = {"mode": "min", "patience": 5, "factor": 0.1, "verbose": True}

        self.model = TabNetClassifier(
            input_dim=in_features,
            output_dim=num_classes,
            optimizer_fn=optimizer_fn,
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
            device_name=device,
            **model_args  # 允许从配置中传入其他 TabNet 特定参数
        )

    def fit(self, X_train, y_train, X_val, y_val, max_epochs, patience, batch_size, checkpoint_path:  Path):
        """
        训练模型。TabNet 的 fit 方法集成了训练和验证循环。
        """
        # TabNet 需要一个验证集来进行早停
        eval_set = [(X_val, y_val)]

        self.model.fit(
            X_train=X_train, y_train=y_train,
            eval_set=eval_set,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            eval_metric=['accuracy', 'logloss'],

        )
        save_model_path = self.model.save_model(str(checkpoint_path))
        print(f"保存模型到 {save_model_path}")

    def load_best_model(self, checkpoint_path: Path):
        """加载已保存的最佳模型。"""
        # .zip 文件的实际路径
        model_file_path = str(checkpoint_path) + ".zip"
        if Path(model_file_path).exists():
            print(f"正在从 {model_file_path} 加载 TabNet 模型...")
            self.model.load_model(model_file_path)
        else:
            print(f"警告: 找不到 TabNet 模型文件 {model_file_path}。将使用最后一次迭代的模型。")

    def predict(self, X_test):
        return self.model.predict(X_test)