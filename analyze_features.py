# file: analyze_features.py
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns


def analyze_feature_importance():
    FEATURE_FUSION_PATH = Path('./data/chest_feature')
    subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]

    # 1. 加载所有数据
    all_X = []
    all_y = []
    for sid in subject_ids:
        all_X.append(np.load(FEATURE_FUSION_PATH / f'{sid}_X.npy'))
        all_y.append(np.load(FEATURE_FUSION_PATH / f'{sid}_y.npy'))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    # mask = np.isin(y, [1, 2, 3])
    # X = X[mask]
    # y = y[mask]

    # 将标签从 [1,2,3] 转换为 [0,1,2] 以符合XGBoost期望
    y = y - 1

    with open(FEATURE_FUSION_PATH / '_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]

    print("数据加载完成，总样本数:", X.shape[0])

    # --- 2. 训练XGBoost模型 (三分类) ---
    print("\n--- 训练三分类模型以获取总体特征重要性 ---")
    model_all = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss')
    model_all.fit(X, y)

    # 3. 获取并可视化特征重要性
    importances_all = model_all.feature_importances_
    df_importances_all = pd.DataFrame({'feature': feature_names, 'importance': importances_all})
    df_importances_all = df_importances_all.sort_values('importance', ascending=False)

    print("总体特征重要性排名:")
    print(df_importances_all)

    # 可视化三分类总体特征重要性
    plt.figure(figsize=(16, 12))  # 更大的图像尺寸以呈现更多细节
    sns.barplot(x='importance', y='feature', data=df_importances_all)
    plt.title('Feature Importance for Three-Class Classification (Neutral vs. Amusement vs. Stress)')
    plt.tight_layout()
    plt.savefig('three_class_feature_importance.png')
    print("\n三分类特征重要性图已保存至 three_class_feature_importance.png")

    # --- 4. 训练XGBoost模型 (二分类: Neutral vs. Amusement) ---
    print("\n--- 训练二分类模型以寻找区分Amusement的关键特征 ---")
    mask = (y == 0) | (y == 1)  # 0: Neutral, 1: Amusement
    X_binary = X[mask]
    y_binary = y[mask]
    # 将标签映射到 0 和 1
    y_binary = np.where(y_binary == 1, 1, 0)

    model_binary = xgb.XGBClassifier(eval_metric='logloss')
    model_binary.fit(X_binary, y_binary)

    importances_binary = model_binary.feature_importances_
    df_importances_binary = pd.DataFrame({'feature': feature_names, 'importance': importances_binary})
    df_importances_binary = df_importances_binary.sort_values('importance', ascending=False)

    print("区分 Neutral vs. Amusement 的特征重要性排名:")
    print(df_importances_binary)

    # 可视化二分类特征重要性
    plt.figure(figsize=(16, 12))  # 更大的图像尺寸以呈现更多细节
    sns.barplot(x='importance', y='feature', data=df_importances_binary)
    plt.title('Feature Importance for Discriminating Neutral vs. Amusement')
    plt.tight_layout()
    plt.savefig('amusement_feature_importance.png')
    print("\n二分类特征重要性图已保存至 amusement_feature_importance.png")


if __name__ == '__main__':
    analyze_feature_importance()