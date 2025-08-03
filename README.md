# 多模态生理信号压力识别系统

## 1. 项目概述

本项目基于 PyTorch 框架，旨在探索和实现先进的多模态生理信号融合模型，用于高精度地识别人体的压力状态。项目采用模块化和配置驱动的设计，支持对不同模型架构（CNN, GRU, ResNet, Transformer等）、不同融合策略（早期、中期、混合）以及不同生理信号组合的系统性消融实验。

**核心研究问题**:
*   不同生理信号模态对压力识别的贡献度。
*   端到端学习（早期融合）与基于专家特征的（中期融合）方法的性能对比。
*   先进深度学习模型（如Transformer）在生理时序信号分类任务上的有效性。

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

## 3. 项目结构示意

```
.
├── configs/              # 存放所有实验的配置文件 (.yaml)
├── data/                 # 存放预处理后的数据
├── data_utils/           # PyTorch Dataset 类
├── models/               # 模型定义 (.py)
├── trainer/              # 训练器类
├── utils/                # 辅助函数 (路径管理, 早停等)
├── output/               # 保存所有实验结果 (日志, 模型权重, 混淆矩阵)
├── preprocess.py         # 数据预处理脚本
├── main.py               # 主训练/评估脚本
└── requirements.txt      # Python 依赖
```

## 4. 环境配置

*   **必要工具**: Conda 包管理工具
*   **硬件要求**: 推荐使用支持 CUDA 的 NVIDIA GPU

### 步骤 1: 创建 Conda 虚拟环境

```bash
conda create -n emo-env python=3.10 -y
conda activate emo-env
```

### 步骤 2: 安装依赖包

为加速下载，建议配置国内镜像源。

```bash
# 配置清华镜像源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

# 安装依赖
pip install -r requirements.txt
```

## 5. 运行说明

本项目采用**两步走**的流程：首先进行数据预处理，然后运行模型训练。

### 步骤 1: 数据预处理

该步骤会将原始的 `.pkl` 文件转换为模型可以直接读取的、窗口化的 `.npy` 数据。

**运行命令**:
```bash
python preprocess_old.py
```

**产出**:
此脚本运行一次即可，它会在 `src/data/processed_data/` 目录下生成两个关键子文件夹：
*   `wesad_early_fusion/`: 存放为早期融合准备的、重采样到 64Hz 的原始信号段。
*   `wesad_feature_fusion/`: 存放为中期融合准备的、使用 `NeuroKit2` 提取的高级生理特征向量。

> **注意**: 如果您修改了 `preprocess.py` 中的任何逻辑（如窗口大小、步长等），请务必**删除** `processed_data` 文件夹并重新运行此脚本，以确保数据的一致性。

### 步骤 2: 运行实验

所有实验都通过主脚本 `main.py` 启动，并通过修改配置文件来控制。

**核心流程**:
1.  **选择一个实验**: 打开 `configs/` 目录，选择一个您想运行的实验配置文件，例如 `exp_early_transformer.yaml`。
2.  **配置 `main.py`**: 打开 `main.py` 文件，找到 `if __name__ == '__main__':` 部分。
3.  **修改加载的配置文件名**: 将 `exp_conf = OmegaConf.load(...)` 这一行中的文件名修改为您选择的实验配置文件名。

    ```python
    # main.py
    try:
        base_conf = OmegaConf.load('configs/base_config.yaml')

        # *************************在这里切换配置**************************
        exp_conf = OmegaConf.load('configs/early_transformer.yaml')
        # ***************************************************************

        config = OmegaConf.merge(base_conf, exp_conf)
        print(f"配置已加载, 当前配置:\n{OmegaConf.to_yaml(exp_conf)}")
    except Exception as e:
        print(f"错误: 无法加载配置文件。 {e}")
        exit()
    ```
4.  **运行训练**:
    ```bash
    python main.py
    ```

**结果查看**:
*   **实时进度**: 训练过程的实时进度将显示在终端。
*   **详细结果**: 每次运行都会在 `output/` 目录下创建一个带时间戳的文件夹，例如 `output/wesad_early_fusion/transformer/run_20250722_103000/`。
*   **内部结构**: 该文件夹下会为**每一次**留一法交叉验证的折叠（例如 `fold_test_on_S2/`）保存详细的训练日志 (`training_log.txt`)、最佳模型权重 (`best_model.pt`) 和测试集的混淆矩阵 (`test_confusion_matrix.png`)。
*   **最终总结**: 运行结束后，主运行目录下会生成一个 `cv_summary.txt` 文件，包含了所有折叠的性能汇总和最终的平均准确率/F1分数。

## 6. 如何进行新的实验 (示例)

假设您想对比**7通道CNN-GRU**和**3通道Transformer**的性能：

1.  **运行实验 A (CNN-GRU)**:
    *   在 `main.py` 中，设置 `exp_conf = OmegaConf.load('configs/7channel_cnngru.yaml')`。
    *   运行 `python main.py`。
    *   在 `output/` 目录下查看结果。
2.  **运行实验 B (Transformer)**:
    *   在 `main.py` 中，设置 `exp_conf = OmegaConf.load('configs/early_transformer.yaml')`。
    *   运行 `python main.py`。
    *   在 `output/` 目录下查看结果。
3.  **对比结果**: 比较两次运行生成的 `cv_summary.txt` 文件中的最终平均性能。

## 7. 常见问题

1.  **无法安装 PyTorch 或只能安装 CPU 版**
    *   **解决方法**: 确认您的 CUDA 版本，并尝试从 PyTorch 官网获取对应的安装命令，或使用国内镜像提供的预编译包。例如，针对 CUDA 12.1:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ```

2.  **OSError: [WinError 126] 找不到指定的模块** (例如 `fbgemm.dll`)
    *   这通常是由于缺少 Visual C++ 运行库等底层依赖。
    *   **解决方法**:
        1.  下载并安装最新版的 [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)。
        2.  (推荐) 使用 [Dependencies](https://github.com/lucasg/Dependencies) 工具分析缺失的 `.dll` 文件，并手动安装对应的库。

3.  **`num_workers > 0` 时报错或变慢**
    *   这是 PyTorch 在 Windows 上的一个常见问题。
    *   **解决方法**: 在 `configs/base_config.yaml` 文件中，将 `trainer.num_workers` 的值设置为 `0`。这将禁用多进程数据加载，但能保证程序的稳定性。