# file: explore_feature_distributions.py
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# --- 配置区域 ---
FEATURE_FUSION_PATH = Path('./data/chest_feature')
# 根据您的预处理配置，将数字标签映射回可读的字符串
# 原始标签: 1: Base, 2: TSST, 3: Fun, 4: Medi1/Medi2
LABEL_INT_TO_STR_MAP = {1: 'baseline', 2: 'stress', 3: 'amusement', 4: 'baseline'}  # 将Medi合并到baseline
STATE_ORDER = ['baseline', 'amusement', 'stress']  # 保证绘图顺序一致


# --- 1. 数据加载与准备 ---
def load_and_prepare_data():
    """
    加载所有受试者的特征和标签数据，并整合成一个Pandas DataFrame。
    """
    subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]

    all_X = []
    all_y = []
    for sid in subject_ids:
        try:
            all_X.append(np.load(FEATURE_FUSION_PATH / f'{sid}_X.npy'))
            all_y.append(np.load(FEATURE_FUSION_PATH / f'{sid}_y.npy'))
        except FileNotFoundError:
            print(f"警告: 未找到 {sid} 的数据，已跳过。")
            continue

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    with open(FEATURE_FUSION_PATH / '_feature_names.txt', 'r') as f:
        feature_names = [line.strip() for line in f]

    print(f"数据加载完成，总样本数: {X.shape[0]}")

    # 转换为Pandas DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['label_int'] = y
    df['label'] = df['label_int'].map(LABEL_INT_TO_STR_MAP)

    # 检查是否有未成功映射的标签
    if df['label'].isnull().any():
        print("警告: 存在未能映射的标签值！")
        print(df[df['label'].isnull()]['label_int'].unique())

    return df


# --- 2. 可视化函数 ---

def plot_univariate_distributions(df: pd.DataFrame):
    """
    为每个特征绘制小提琴图，以对比其在不同情绪状态下的分布。
    """
    print("\n正在生成单特征分布图...")
    feature_names = df.select_dtypes(include=np.number).columns.drop('label_int').tolist()
    n_features = len(feature_names)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(40, 12 * n_rows))  # 尺寸加倍
    axes = axes.flatten()

    for i, feature in enumerate(feature_names):
        sns.violinplot(x='label', y=feature, data=df, order=STATE_ORDER, ax=axes[i])
        sns.stripplot(x='label', y=feature, data=df, order=STATE_ORDER, ax=axes[i], color='k', alpha=0.1, size=2)
        axes[i].set_title(f'Distribution of {feature}', fontsize=28)
        axes[i].set_xlabel('Condition', fontsize=24)
        axes[i].set_ylabel('Value', fontsize=24)

    for i in range(n_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.suptitle('Univariate Feature Distributions Across States', fontsize=40)
    plt.savefig('feature_distributions_violin.png', dpi=300)
    plt.close()
    print("单特征分布图已保存至 feature_distributions_violin.png")


def plot_bivariate_relationships(df: pd.DataFrame):
    """
    使用配对图可视化最重要的几个特征之间的关系。
    """
    print("\n正在生成多特征关系对比图...")
    # 基于您的XGBoost结果，我们挑选最重要的几个特征
    # 总体重要性: 'RESP_RRV_SDNN', 'EDA_Tonic_Slope', 'EDA_SCR_Peaks_N'
    # 区分Amusement: 'RESP_RRV_SDNN', 'RESP_Rate_Mean', 'EMG_Amplitude_Mean'
    top_features = ['RESP_RRV_SDNN', 'EDA_Tonic_Slope', 'EMG_Amplitude_Mean', 'EDA_SCR_Peaks_N', 'HRV_RMSSD']

    # 确保所有选择的特征都在DataFrame中
    top_features = [f for f in top_features if f in df.columns]
    if not top_features:
        print("警告: top_features 列表中的特征均未在数据中找到，跳过配对图绘制。")
        return

    # 自定义调色板: 使用更加鲜明和对比强烈的颜色
    custom_palette = {'baseline': '#0072B2', 'amusement': '#009E73', 'stress': '#D55E00'}  # 蓝色, 绿色, 橙色

    pairplot_fig = sns.pairplot(df, vars=top_features, hue='label', hue_order=STATE_ORDER, palette=custom_palette,
                                corner=True, plot_kws={'alpha': 0.7}, diag_kind='kde')
    pairplot_fig.fig.set_size_inches(32, 24)  # 尺寸加倍
    pairplot_fig.fig.suptitle('Pairwise Relationships of Top Features by State', y=1.03, fontsize=32)
    plt.savefig('feature_pairplot.png', dpi=300)
    plt.close()
    print("多特征关系对比图已保存至 feature_pairplot.png")



def plot_multivariate_projection(df: pd.DataFrame):
    """
    使用PCA和t-SNE降维，并可视化所有特征在2D空间中的分布。
    """
    print("\n正在生成PCA和t-SNE降维投影图...")
    feature_names = df.select_dtypes(include=np.number).columns.drop('label_int').tolist()

    # 准备数据
    X_scaled = StandardScaler().fit_transform(df[feature_names])
    labels = df['label']

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df_pca['label'] = labels
    exp_var_ratio = pca.explained_variance_ratio_

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    df_tsne = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
    df_tsne['label'] = labels

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(40, 18))  # 尺寸加倍

    # PCA Plot
    sns.scatterplot(x='PC1', y='PC2', hue='label', hue_order=STATE_ORDER, data=df_pca, ax=axes[0], s=100, alpha=0.7)
    axes[0].set_title('2D PCA Projection of Features', fontsize=32)
    axes[0].set_xlabel(f'Principal Component 1 ({exp_var_ratio[0]:.2%})', fontsize=24)
    axes[0].set_ylabel(f'Principal Component 2 ({exp_var_ratio[1]:.2%})', fontsize=24)
    axes[0].grid(True)

    # t-SNE Plot
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='label', hue_order=STATE_ORDER, data=df_tsne, ax=axes[1], s=100, alpha=0.7)
    axes[1].set_title('2D t-SNE Projection of Features', fontsize=32)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=24)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=24)
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.suptitle('Multivariate Projections of Feature Space', fontsize=40)
    plt.savefig('feature_projections_2D.png', dpi=300)
    plt.close()
    print("PCA和t-SNE降维投影图已保存至 feature_projections_2D.png")


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    # 步骤1: 加载并准备数据
    main_df = load_and_prepare_data()

    if not main_df.empty:
        # 步骤2: 运行所有可视化分析
        plot_univariate_distributions(main_df)
        plot_bivariate_relationships(main_df)
        plot_multivariate_projection(main_df)
        print("\n所有可视化探索已完成！")