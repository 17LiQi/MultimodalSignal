# models/xgboost_model.py
import xgboost as xgb
import numpy as np

class XGBoostModel:
    """XGBoost模型的简单封装。"""
    def __init__(self, num_classes, params=None, early_stopping_rounds=10):
        if params is None:
            # XGBoost的一些强大的默认参数
            params = {
                'objective': 'multi:softprob', # 输出概率
                'num_class': num_classes,
                'eta': 0.1, # 学习率
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'eval_metric': 'mlogloss',
                'use_label_encoder': False,
                'seed': 42
            }
        self.params = params
        self.model = xgb.XGBClassifier(
            **self.params,
            early_stopping_rounds=early_stopping_rounds  # 现在这是合法的
        )

    def fit(self, X_train, y_train, X_val, y_val):
        # XGBoost可以利用验证集进行早停
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False # 在训练期间不打印过多信息
        )

    def predict(self, X_test):
        return self.model.predict(X_test)