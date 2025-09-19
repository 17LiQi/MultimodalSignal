# 面向鲁棒跨被试情感识别的智能融合模型

## 1. 项目概述

本项目旨在解决多模态生理信号情感识别领域中的一个核心挑战：**模型的跨被试泛化能力（Cross-Subject Generalization）**。

当前领域内大量研究报告了极高的识别准确率，但我们的系统性文献审计发现，这些结果大多是在存在 **“被试依赖”数据泄露风险** 的评估方法下取得的。当采用严格的 **“留一被试交叉验证（Leave-One-Subject-Out, LOSO）”** 范式时，许多先进模型的真实性能会大幅下降。

本项目基于 PyTorch 框架，提出并验证了一种**基于通道注意力机制的智能融合模型（CnnGruAttentionModel）**。该模型旨在取代传统的“暴力拼接”或“信息孤岛”式融合策略，通过在特征层面动态学习不同生理信号通道的重要性，显著提升模型在严格跨被试评估下的**性能**和**鲁棒性**。

**本项目的主要贡献：**
*   **范式批判与基线建立**：通过对大量文献的审计，揭示了“被试内”评估的普遍性，并建立了一个严格遵守LOSO范式的、性能可靠的新基线。
*   **智能融合模型**：提出了`CnnGruAttentionModel`，实验证明，相比传统融合方法，该模型能更有效地利用多模札信息，将WESAD数据集上的跨被试平均识别准确率**从约75%提升至82.44%**。
*   **开源与可复现性**：提供了一个**精简、轻量级、完全可复现**的研究代码库，方便社区进行验证和进一步探索。

## 2. 数据集说明

*   **名称**: WESAD (WEarable Stress and Affect Detection)
*   **简介**: 一个公开的、用于压力和情绪检测的可穿戴设备生理信号数据集。包含15名受试者在三种核心状态（中性、压力、娱乐）下的多模态数据。
*   **传感器数据**:
    *   **胸部 (RespiBAN)**: ECG, EDA, EMG, RESP, TEMP, ACC (均为 700Hz)。
    *   **腕部 (Empatica E4)**: ACC (32Hz), BVP (64Hz), EDA (4Hz), TEMP (4Hz)。
*   **验证机制**: 包含每个阶段精确的起止时间戳和主观问卷（PANAS, STAI等）。

### 数据集资源

*   **详细文档**: [WESAD官方说明](https://archive.ics.uci.edu/dataset/468/wesad+wearable+stress+and+affect+detection)
*   **数据预览**: [交互式可视化数据预览](https://kristofvl.github.io/wesadviz/)
*   **下载地址**: [官方云存储](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx)

> **注意**: 下载后，请将包含 `S2`, `S3` 等子文件夹的 `WESAD` 文件夹放置在根目录下。
## 3. 项目结构

本项目采用扁平化的精简结构，聚焦于核心逻辑。
```
.
├── WESAD/
├── data/
├── output/
│   └── ... (所有实验结果)
├── preprocess.py         # 1. 数据预处理脚本
├── dataset.py            # PyTorch Dataset 定义
├── models.py             # 模型定义 (含通道注意力模型)
├── trainer.py            # 训练器与早停逻辑
└── main.py               # 2. 主训练/评估脚本
```

## 4. 环境配置

```bash
# 1. 创建 Conda 虚拟环境
conda create -n emo-env python=3.10 -y
conda activate emo-env

# 2. 安装依赖 (推荐使用国内镜像源)
pip install torch numpy pandas scikit-learn tqdm pyts matplotlib seaborn
```
*对于GPU用户，请根据您的CUDA版本从PyTorch官网获取对应的安装命令。*

## 5. 快速开始 (两步复现核心成果)

### 步骤 1: 数据预处理

此脚本负责将原始的 `.pkl` 文件转换为所有模型统一使用的、重采样到64Hz的窗口化数据。**只需运行一次。**

```bash
python preprocess.py
```
*   **输入**: `./data/WESAD/`
*   **输出**: `./processed_data/wesad_early_fusion/`

### 步骤 2: 运行核心实验

所有实验均通过主脚本 `main.py` 启动。您可以通过修改文件顶部的全局参数来控制实验。

```python
# file: main.py

# --- 1. 全局参数直接定义 ---
# 实验设置
MODEL_TO_USE = 'cnn_gru_attention'  # 'cnn_gru' (基线) 或 'cnn_gru_attention' (我们的模型)
RUN_NAME = f"{MODEL_TO_USE}"

# 消融实验在这里修改:
CHANNELS_TO_USE = ['chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_Resp', 'wrist_BVP', 'wrist_EDA']

# ... 其他训练参数 ...
```

**运行训练与评估：**
```bash
python main.py
```
该脚本将**自动执行完整**的“留一被试交叉验证（LOSOCV）”流程。

## 6. 结果查看

*   **实时进度**: 终端会显示每个LOSOCV折叠的训练进度。
*   **详细结果**: 每次运行都会在 `./data/` 目录下创建一个唯一的文件夹，例如 `runs/cnn_gru_attention/run_20250916_103000/`。
*   **核心产出**:
    *   `cv_summary.txt`: 包含了本次交叉验证的**最终平均性能**（准确率和F1分数）、标准差以及每个折叠的详细结果。
    *   `fold_test_on_S*/`: 每个折叠的子文件夹内，都保存了该折叠的最佳模型权重（`best_model.pt`）、详细训练日志（`training_log.txt`）和测试集混淆矩阵。