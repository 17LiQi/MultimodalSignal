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
- **GitHub地址**: [GitHub-WJMatthew/WESAD](https://github.com/WJMatthew/WESAD)
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
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com
```

#### 方法2：配置全局镜像源
```bash
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
pip config set install.trusted-host mirrors.aliyun.com

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
- 若依赖安装失败，可尝试替换为其他镜像源（如清华源：`https://pypi.tuna.tsinghua.edu.cn/simple`）
- 首次运行前请确保数据集已下载并放置在项目指定路径（建议在项目根目录下）
