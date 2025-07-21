# 多模态信号融合压力判别系统说明文档

## 项目概述
本项目基于PyTorch框架，实现多模态生理信号融合模型，用于辅助判别人体压力状态。


## 数据集说明
- **名称**：WESAD（Wearable Stress and Affect Detection）
- **作者**：Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, Kristof Van Laerhoven
- **简介**：公开的可穿戴设备生理信号数据集，包含15名受试者的多模态生理数据，通过三种情感状态（中性、压力、娱乐）的标注，为压力与情绪研究提供了标准化数据支持。
- **传感器数据**：
  - 胸部（chest）和腕部（wrist）双设备采集
  - 包含血容量脉搏、心电图、皮肤电活动、肌电图、呼吸、体温、三轴加速度等信号
- **验证机制**：受试者在每个实验阶段后填写调查问卷，确保数据标注有效性


## 数据集资源
- **详细文档**：[WESAD官方说明](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
- **数据预览**：[交互式可视化数据预览](https://kristofvl.github.io/wesadviz/)
- **下载地址**：[Sciebo云存储](https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx)


## 环境要求
- **必要工具**：conda 包管理工具
- **硬件要求**：带GPU的主机（推荐NVIDIA GPU，支持CUDA加速）


## 环境配置步骤

### 1. 创建虚拟环境
```bash
conda create -n emo-env python=3.10.18 -y
conda activate emo-env  
```

### 2. 安装依赖包
#### 方法1：临时使用镜像源
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
```

#### 方法2：配置全局镜像源
```bash
pip config set global.index-url https:https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn

pip install -r requirements.txt
```

## 环境验证

为确保依赖安装正确，可运行项目提供的依赖验证脚本：

### 步骤
1. 激活项目虚拟环境：
```bash
conda activate emo-env
```
2. 运行验证脚本
```bash
python check_env.py
```

## 备注
- 若依赖安装失败，可尝试替换为其他镜像源（如阿里源：`https://mirrors.aliyun.com/pypi/simple/` ）
- 首次运行前请确保数据集已下载并放置在项目指定路径（建议在项目根目录下）

## 常见问题
1. 无法安装Pytorch或只能安装CPU版
- 解决方法:
- 尝试使用其他镜像源，如阿里源：`https://mirrors.aliyun.com/pypi/simple/` 或者官网下载
- 或直接从指定源下载安装包安装
```bash
pip install https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.4.0+cu121-cp310-cp310-win_amd64.whl 
```

2. Pytorch 找不到指定的模块, 这可能是由于镜像源问题导致的, 如:
```bash
oserror: [winerror 126] 找不到指定的模块: error loading "\miniconda3\envs\emo-env\lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies. 
```
- 解决方法: 
- 下载 dll 文件依赖分析工具 Dependencies: https://github.com/lucasg/Dependencies/releases/tag/v1.11.1
- 选择合适的版本,如 x64_Release.zip
- 运行 DependenciesGui.exe 点击 File -> Open -> 选择 dll 文件 -> 点击 Analyze ,根据缺失的文件提示单独安装

