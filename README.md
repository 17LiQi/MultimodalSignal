
# PhysioFormer 论文复现项目

## 1. 项目概述

项目当前分支基于 PyTorch 框架，旨在**严格复现**论文 *《PhysioFormer: Integrating Multimodal Physiological Signals and Symbolic Regression for Explainable Affective State Prediction》* 中提出的情感状态预测模型。该论文声称在公开的WESAD数据集上取得了超过99%的惊人准确率。

本复现项目的核心目标是：
*   **忠实实现**论文描述的预处理流程，特别是基于**特征工程 (Feature Engineering)** 的方法。
*   精确构建论文中定义的 `ContribNet`, `AffectNet`, `AffectAnalyser` 三大核心模型模块。
*   实现论文中包含自定义正则化项的复合损失函数。
*   在一个严谨、排除了常见数据泄露风险的流程下，**验证并评估 `PhysioFormer` 架构的真实性能**。

> **注意**: 本项目不包含论文中用于事后分析的“符号回归”部分，因为它不参与模型的训练和预测，对复现其核心性能没有影响。

## 2. 数据集说明

*   **名称**: WESAD (WEarable Stress and Affect Detection)
*   **简介**: 一个公开的、用于压力和情绪检测的可穿戴设备生理信号数据集。包含15名受试者在三种核心状态（**1-中性/基线, 2-压力, 3-娱乐**）下的多模态数据。
*   **传感器数据**:
    *   **胸部 (RespiBAN)**: ECG, EDA, EMG, RESP, TEMP, ACC (均为 700Hz)。
    *   **腕部 (Empatica E4)**: ACC (32Hz), BVP (64Hz), EDA (4Hz), TEMP (4Hz)。

### 数据集资源

*   **详细文档**: [WESAD官方说明](https://archive.ics.uci.edu/dataset/468/wesad+wearable+stress+affect+detection)
*   **下载地址**: [官方云存储](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx)

> **重要**: 下载后，请将包含 `S2`, `S3` 等子文件夹的 `WESAD` 文件夹放置在项目根目录下。

## 3. 项目结构

```
.
├── configs/              # 存放所有实验的配置文件 (.yaml)
├── data/                 # 存放预处理后的数据 (physioformer_processed/)
├── output/               # 保存所有实验结果 (日志, 模型权重, 混淆矩阵)
├── src/
│   ├── dataset/          # PyTorch Dataset 类 (dataset.py)
│   ├── models/           # 模型定义 (physioformer_model.py)
│   ├── trainer/          # 训练器类 (trainer.py)
│   ├── utils/            # 辅助函数 (early_stopping.py, cvxEDA.py 等)
│   ├── preprocess.py     # 核心数据预处理脚本
│   └── main.py           # 主训练/评估脚本
└── requirements.txt      # Python 依赖
```

## 4. 环境配置

*   **必要工具**: Conda
*   **硬件要求**: 推荐使用支持 CUDA 的 NVIDIA GPU

### 步骤 1: 创建 Conda 虚拟环境

```bash
conda create -n emo-env python=3.10 -y
conda activate emo-env
```

### 步骤 2: 安装 Python 依赖包

本项目依赖于多个科学计算库，其中 `cvxEDA` 需要特别处理。

```bash
# (可选) 配置国内镜像源以加速下载
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 requirements.txt 中的大部分依赖
pip install -r requirements.txt
```

### 步骤 3: 【关键】安装 `cvxEDA` 和 `cvxpy`

`cvxEDA` 库未发布在PyPI，需要我们手动集成。

1.  **复制核心文件**:
    *   在其他地方（例如，使用 `git clone https://github.com/lciti/cvxEDA.git`）获取 `cvxEDA` 的源代码。
    *   找到其中的 `cvxeda.py` 文件。
    *   将这个 **`cvxeda.py` 文件**复制到本项目的 `src/utils/` 目录下。

2.  **安装 `git` (如果尚未安装)**:
    部分依赖可能需要从GitHub直接安装，请确保您的系统中已安装 [Git](https://git-scm.com/downloads)。

## 5. 运行说明

本项目采用**两步走**的流程：首先进行数据预处理，然后运行模型训练。

### 步骤 1: 数据预处理

该步骤是复现成功的**关键**。它会将原始的 `.pkl` 文件严格按照论文描述，转换为包含大量特征工程的 `.npy` 数据文件。

**核心流程**:
1.  **重采样**: 将所有生理信号统一重采样到 64Hz。
2.  **窗口化**: 使用30秒**无重叠**窗口进行数据切片。
3.  **特征提取**: 对每个窗口，使用 `NeuroKit2` 和 `cvxEDA` 等库提取数十种统计学和生理学特征。
4.  **整合**: 将提取的动态特征与被试者的静态个人属性（年龄、性别等）合并。
5.  **归一化**: 对最终的特征矩阵进行全局标准化。

**运行命令**:
```bash
# 确保你在目录根目录下或直接运行 preprocess.py 脚本。
python preprocess.py
```

**产出**:
此脚本运行一次即可。它会在 `data` 目录下生成 `X_wrist.npy`, `y_wrist.npy`, `X_chest.npy`, `y_chest.npy` 等文件。

> **注意**: 如果您修改了 `preprocess.py` 中的任何逻辑，请务必**删除** `data` 文件夹并重新运行此脚本。

### 步骤 2: 运行模型训练与评估

所有实验都通过主脚本 `main.py` 启动。

**运行命令**:
```bash
# 确保你在 src 目录下或直接运行 main.py 脚本。
cd src
python main.py
```

**自定义实验**:
*   **切换数据集**: 打开 `main.py`，在 `CONFIG` 字典中修改 `dataset_type: 'wrist'` 为 `'chest'` 来训练胸部数据集。
*   **调整超参数**: 您可以在 `main.py` 的 `CONFIG` 字典中轻松调整学习率 (`learning_rate`)、批大小 (`batch_size`)、正则化强度 (`lambda_reg`) 等。

**结果查看**:
*   **实时进度**: 训练过程的损失和准确率将实时显示在终端。
*   **详细结果**: 每次运行都会在 `output/` 目录下创建一个带时间戳的文件夹（例如 `output/physioformer_run_20250821_180000/`）。
*   **文件夹内容**:
    *   `training_log.txt`: 详细的逐轮训练日志。
    *   `best_model.pt`: 在验证集上性能最佳的模型权重。
    *   `confusion_matrix.png`: 测试集结果的可视化混淆矩阵。
    *   **最终报告**: `training_log.txt` 的末尾包含了测试集上的**最终分类报告 (Classification Report)**。

## 6. 常见问题

1.  **`cvxEDA` 报错或无法导入**
    *   **解决方法**:
        1.  确保 `cvxeda.py` 文件已**正确复制**到 `src/utils/` 目录下。
        2.  确保 `cvxpy` 已成功安装 (`pip install cvxpy`)。
        3.  确保 `main.py` 顶部的 `import` 语句能正确找到它（`from utils.cvxEDA import cvxEDA`）。

2.  **`num_workers > 0` 时报错或卡住**
    *   这是 PyTorch 在 Windows 上的常见问题。
    *   **解决方法**: 在 `main.py` 中创建 `DataLoader` 时，确保 `num_workers=0`。

3.  **CUDA 相关错误 (e.g., "CUDA out of memory")**
    *   **解决方法**: 您的 GPU 显存不足。请在 `main.py` 的 `CONFIG` 字典中减小 `batch_size` 的值（例如从16减到8或4）。