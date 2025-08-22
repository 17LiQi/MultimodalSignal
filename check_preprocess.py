# test_siamese_dataset.py
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, random_split
from dataset.wesad_siamese_dataset import WesadSiameseDataset
from src.utils.path_manager import get_path_manager

def test_dataset_output():
    """
    一个独立的函数，用于测试 WesadSiameseDataset 的输出是否正确。
    """
    print("--- 开始测试 WesadSiameseDataset ---")

    # --- 1. 设置路径和参数 ---
    try:
        # 确保这个路径指向您 preprocess_siamese.py 的输出目录
        path = get_path_manager()
        DATA_PATH = path.DATA_ROOT
        if not DATA_PATH.exists():
            print(f"错误: 找不到数据路径 '{DATA_PATH}'。请先运行 preprocess_siamese.py。")
            return

        # 检查核心文件是否存在
        required_files = ['bvp_windows.npy', 'eda_windows.npy', 'labels.npy']
        for f in required_files:
            if not (DATA_PATH / f).exists():
                print(f"错误: 找不到必需文件 '{f}'。")
                return
    except Exception as e:
        print(f"路径设置出错: {e}")
        return

    # --- 2. 实例化数据集并进行划分 ---
    # 论文中提到 85% 训练, 15% 验证。我们先模拟这个划分。
    print("\n步骤1: 实例化并划分数据集...")
    full_dataset_indices = np.arange(len(np.load(DATA_PATH / 'labels.npy')))

    train_size = int(0.85 * len(full_dataset_indices))
    val_size = len(full_dataset_indices) - train_size
    train_indices, val_indices = random_split(full_dataset_indices, [train_size, val_size])

    # 我们只用训练集的索引来创建我们的测试实例
    train_dataset = WesadSiameseDataset(DATA_PATH, train_indices)

    print(f"数据集成功实例化。训练集样本数: {len(train_dataset)}")
    assert len(train_dataset) > 0, "数据集为空！"
    print("... PASS\n")

    # --- 3. 取出一个样本并验证 ---
    print("步骤2: 从数据集中获取一个样本并检查其结构...")
    # Dataset 的 __getitem__ 会被调用
    inputs, labels = train_dataset[0]

    print("成功获取样本。")
    print("\n--- 样本结构分析 ---")

    # --- a. 验证输入 (Inputs) ---
    print("\n[输入部分 (Inputs)]")
    print(f"类型: {type(inputs)}")
    assert isinstance(inputs, tuple) and len(
        inputs) == 4, f"输入部分应为一个包含4个元素的元组，但得到的是 {type(inputs)}，长度为 {len(inputs) if hasattr(inputs, '__len__') else 'N/A'}"
    print("... PASS: 类型为元组，长度为4\n")

    anchor_bvp, anchor_eda, pair_bvp, pair_eda = inputs

    print("锚点 BVP (anchor_bvp):")
    print(f"  - 类型: {type(anchor_bvp)}")
    print(f"  - 数据类型 (dtype): {anchor_bvp.dtype}")
    print(f"  - 形状 (shape): {anchor_bvp.shape}")
    assert isinstance(anchor_bvp, torch.Tensor), "anchor_bvp 不是 PyTorch 张量"
    assert anchor_bvp.shape == (3, 1920), f"anchor_bvp 形状应为 (3, 1920)，但得到的是 {anchor_bvp.shape}"
    print("... PASS\n")

    print("锚点 EDA (anchor_eda):")
    print(f"  - 形状 (shape): {anchor_eda.shape}")
    assert anchor_eda.shape == (1, 120), f"anchor_eda 形状应为 (1, 120)，但得到的是 {anchor_eda.shape}"
    print("... PASS\n")

    print("配对 BVP (pair_bvp):")
    print(f"  - 形状 (shape): {pair_bvp.shape}")
    assert pair_bvp.shape == (3, 1920), f"pair_bvp 形状应为 (3, 1920)，但得到的是 {pair_bvp.shape}"
    print("... PASS\n")

    print("配对 EDA (pair_eda):")
    print(f"  - 形状 (shape): {pair_eda.shape}")
    assert pair_eda.shape == (1, 120), f"pair_eda 形状应为 (1, 120)，但得到的是 {pair_eda.shape}"
    print("... PASS\n")

    # --- b. 验证标签 (Labels) ---
    print("[标签部分 (Labels)]")
    print(f"类型: {type(labels)}")
    assert isinstance(labels, tuple) and len(
        labels) == 3, f"标签部分应为一个包含3个元素的元组，但得到的是 {type(labels)}"
    print("... PASS: 类型为元组，长度为3\n")

    anchor_emotion, pair_emotion, siamese_label = labels

    print("锚点情感标签 (anchor_emotion):")
    print(f"  - 类型: {type(anchor_emotion)}")
    print(f"  - 数据类型 (dtype): {anchor_emotion.dtype}")
    print(f"  - 形状 (shape): {anchor_emotion.shape}")
    print(f"  - 值: {anchor_emotion.item()}")
    assert isinstance(anchor_emotion, torch.Tensor), "anchor_emotion 不是张量"
    assert anchor_emotion.dtype == torch.long, "anchor_emotion 数据类型应为 torch.long"
    assert anchor_emotion.item() in [0, 1, 2, 3], "情感标签值超出范围 [0, 3]"
    print("... PASS\n")

    print("配对情感标签 (pair_emotion):")
    print(f"  - 值: {pair_emotion.item()}")
    assert pair_emotion.item() in [0, 1, 2, 3], "情感标签值超出范围 [0, 3]"
    print("... PASS\n")

    print("孪生标签 (siamese_label):")
    print(f"  - 类型: {type(siamese_label)}")
    print(f"  - 数据类型 (dtype): {siamese_label.dtype}")
    print(f"  - 值: {siamese_label.item()}")
    assert siamese_label.dtype == torch.float, "孪生标签数据类型应为 torch.float"
    assert siamese_label.item() in [0.0, 1.0], "孪生标签值必须是 0.0 或 1.0"

    # 交叉验证孪生标签的逻辑
    if anchor_emotion.item() == pair_emotion.item():
        assert siamese_label.item() == 1.0, "逻辑错误: 情感标签相同，但孪生标签不为1.0"
    else:
        assert siamese_label.item() == 0.0, "逻辑错误: 情感标签不同，但孪生标签不为0.0"
    print("... PASS: 孪生标签逻辑正确\n")

    print("\n---  所有测试通过  ---")
    print("WesadSiameseDataset 能够正确加载数据并生成符合多任务学习预期的样本。")


if __name__ == '__main__':
    test_dataset_output()