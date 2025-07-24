# main.py
from collections import Counter

import yaml
import pickle
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

# --- 从项目中导入自定义模块 ---
from dataset.wesad_dataset import WesadEarlyFusionDataset
from models.cnn_1d import SimpleCNN1D
from models.resnet_1d import ResNet1D
from trainer.trainer import Trainer
from utils.path_manager import get_path_manager  # 导入您自己的路径管理器


def run_fold(config, test_subject, all_subjects, paths ,run_output_dir):
    """
    运行单次留一法交叉验证的折叠 (fold)。
    这个函数封装了针对一个特定测试受试者的完整“训练-验证-测试”流程。

    参数:
    - config (OmegaConf): 本次运行的完整配置对象。
    - test_subject (str): 当前折叠中用作测试的受试者ID (e.g., 'S2')。
    - all_subjects (list): 数据集中所有受试者的ID列表。
    - paths: 从 path_manager 获取的路径对象。

    返回:
    - test_acc (float): 本次折叠在测试集上的准确率。
    - test_f1 (float): 本次折叠在测试集上的加权F1分数。
    """

    # --- 1. 设置路径 ---
    # 定义预处理数据的根目录
    processed_path = paths.DATA_ROOT
    # 在主运行目录下，为此折叠的输出创建子目录
    fold_output_dir = run_output_dir / f'fold_test_on_{test_subject}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== 开始处理折叠: 测试集=[{test_subject}] | 输出至: {fold_output_dir} =====")

    # --- 2. 动态数据集划分 ---
    print(f"\n===== 开始处理折叠: 测试集=[{test_subject}] =====")
    # 从所有受试者中移除当前测试的受试者，剩下的用于训练和验证
    train_val_subjects = [s for s in all_subjects if s != test_subject]

    # 使用 scikit-learn 中的 train_test_split 将剩余受试者划分为训练集和验证集
    # random_state 保证了每次运行时，对于同一个 test_subject，划分结果都相同，便于复现
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=config.dataset.val_split_ratio,
        random_state=config.seed
    )
    print(f"训练集 ({len(train_subjects)}): {train_subjects}")
    print(f"验证集 ({len(val_subjects)}): {val_subjects}")

    # --- 验证训练集的类别分布 ---
    print("--- 验证训练集类别分布 ---")
    all_train_labels = []
    data_folder_for_check = paths.DATA_ROOT / config.dataset.name
    for sid in train_subjects:
        y = np.load(data_folder_for_check / f'{sid}_y.npy')
        all_train_labels.append(y)

    all_train_labels = np.concatenate(all_train_labels, axis=0)
    label_counts = Counter(all_train_labels)

    print(f"当前折叠 (测试集: {test_subject}) 的训练集标签分布:")
    print(f"  - 类别 0 (Neutral): {label_counts.get(0, 0)} 个样本")
    print(f"  - 类别 1 (Amusement): {label_counts.get(1, 0)} 个样本")
    print(f"  - 类别 2 (Stress): {label_counts.get(2, 0)} 个样本")
    if label_counts.get(1, 0) < label_counts.get(0, 0) * 0.5 and label_counts.get(1, 0) < label_counts.get(2, 0) * 0.5:
        print("\033[93m警告: 类别 1 (Amusement) 样本数量显著偏少，可能导致模型无法有效学习该类别。\033[0m")
    print("-" * 30)

    # --- 3. 准备数据加载所需的资源 ---
    data_folder = processed_path / config.dataset.name

    # 这是一个预定义的全通道列表，用于从12通道数据中按名称索引我们需要的通道
    # 最佳实践是在 preprocess.py 中生成并保存这个列表为 .json 文件，然后在这里加载
    all_channels = ['chest_ACC_x', 'chest_ACC_y', 'chest_ACC_z',
                    'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_Resp',
                    'wrist_ACC_x', 'wrist_ACC_y', 'wrist_ACC_z',
                    'wrist_BVP', 'wrist_EDA']

    # 加载为本次交叉验证折叠预计算的 全部 归一化参数
    norm_params_path = data_folder / f'norm_params_for_{test_subject}.npz'
    norm_params = np.load(norm_params_path)
    full_mean_vec = norm_params['mean']
    full_std_vec = norm_params['std']

    # 根据 config 中指定的通道，筛选出本次实验所需的均值和标准差
    # a. 获取所需通道在 all_channels 列表中的索引
    channel_indices = [all_channels.index(ch) for ch in config.dataset.channels_to_use]

    # b. 使用这些索引来从完整的均值/标准差向量中切片出需要的部分
    mean_for_this_run = full_mean_vec[channel_indices]
    std_for_this_run = full_std_vec[channel_indices]

    # --- 4. 归一化处理 ---
    # 加载为本次交叉验证折叠预先计算好的归一化参数 (均值和标准差)
    # 这确保了我们只使用训练集的信息来归一化所有数据
    # 创建一个新的、只包含本次实验所需统计信息的 scaler 对象
    scaler = StandardScaler()
    scaler.mean_ = mean_for_this_run
    scaler.scale_ = std_for_this_run

    # --- 5. 创建 PyTorch Dataset 和 DataLoader 实例 ---
    # 根据配置选择使用哪个 Dataset 类
    if config.dataset.name == "early_fusion":
        DatasetClass = WesadEarlyFusionDataset
    # elif config.dataset.name == "feature_fusion":
    #     DatasetClass = WesadFeatureDataset # (未来扩展)
    else:
        raise ValueError(f"未知的数据集名称: {config.dataset.name}")

    # 实例化训练、验证和测试数据集
    train_dataset = DatasetClass(data_folder, train_subjects, config.dataset.channels_to_use, all_channels, scaler)
    val_dataset = DatasetClass(data_folder, val_subjects, config.dataset.channels_to_use, all_channels, scaler)
    test_dataset = DatasetClass(data_folder, [test_subject], config.dataset.channels_to_use, all_channels, scaler)

    # 创建 DataLoader
    # 启用 pin_memory 提高 GPU 数据传输效率
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        pin_memory=torch.cuda.is_available()  # 启用 pin_memory 如果 GPU 可用
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    # --- 6. 根据配置创建模型 ---
    if config.model.name == "cnn_1d":
        model = SimpleCNN1D(
            in_channels=len(config.dataset.channels_to_use),  # 输入通道数由配置动态决定
            num_classes=config.model.params.num_classes
        )
    elif config.model.name == "resnet_1d":  # <--- 新增的逻辑分支
        model = ResNet1D(
            in_channels=len(config.dataset.channels_to_use),
            num_classes=config.model.params.num_classes
        )
    else:
        raise ValueError(f"未知模型: {config.model.name}")

    # 支持 GPU(如果可用)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # --- 7. 创建训练器并启动训练和评估流程 ---
    # <-- 2. 将折叠的输出目录传递给 Trainer

    trainer = Trainer(model, config, fold_output_dir)
    trainer.train(train_loader, val_loader)

    print(f"--- 开始在测试集 {test_subject} 上进行评估 ---")

    # 测试集评估
    test_loss, test_acc, test_f1 = trainer.evaluate(test_loader, is_test=True)

    # 记录测试结果到单独文件
    fold_result_file = fold_output_dir  / 'test_results.txt'
    with open(fold_result_file, 'w') as f:
        f.write(f"Test Subject: {test_subject}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1-score: {test_f1:.4f}\n")

    return test_acc, test_f1


if __name__ == '__main__':
    # --- A. 加载配置 ---
    # 使用 OmegaConf 加载基础配置和指定的实验配置
    # 这种方式可以轻松地通过修改 'ecg_cnn.yaml' 来运行不同的实验
    base_conf = OmegaConf.load('configs/base_config.yaml')
    exp_conf = OmegaConf.load('configs/3channel_resnet.yaml')
    try:
        # 合并配置，实验配置会覆盖基础配置中的同名参数
        config = OmegaConf.merge(base_conf, exp_conf)
        print(f"配置已加载, 当前配置:{exp_conf}")
    except FileNotFoundError as e:
        print(f"错误: 无法找到配置文件。请确保 {base_conf} 和 {exp_conf} 存在。")
        exit()

    # --- B. 初始化 ---
    # 获取路径管理器
    paths = get_path_manager()

    # 设置随机种子以保证实验可复现
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True  # 确保 GPU 计算可复现
        torch.backends.cudnn.benchmark = False  # 禁用优化以保证可复现性

    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = paths.OUTPUT_ROOT / config.dataset.name / config.model.name / f'run_{run_timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"====== 本次交叉验证运行结果将保存至: {run_output_dir} ======")

    all_subjects = config.dataset.all_subjects
    all_fold_accs = []
    all_fold_f1s = []

    # --- C. 执行留一法交叉验证 (LOSOCV) 循环 ---
    print("====== 开始留一法交叉验证 (LOSOCV) ======")
    for subject_to_test in all_subjects:
        # <-- 3. 将主运行目录传递给 run_fold
        acc, f1 = run_fold(config, subject_to_test, all_subjects, paths, run_output_dir)
        all_fold_accs.append(acc)
        all_fold_f1s.append(f1)

    # --- D. 汇总并打印最终结果 ---
    print("\n\n====== 留一法交叉验证全部完成 ======")
    # 保存所有折叠的汇总结果
    summary_file = run_output_dir / 'cv_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"实验配置: {config.dataset.name} - {config.model.name} - Channels: {config.dataset.channels_to_use}\n")
        f.write("\n每个折叠的详细结果:\n")
        for i, sid in enumerate(all_subjects):
            f.write(f"  - 测试 {sid}: Accuracy = {all_fold_accs[i]:.4f}, F1-score = {all_fold_f1s[i]:.4f}\n")
        f.write("\n最终平均性能:\n")
        f.write(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}\n")
        f.write(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}\n")
    print(f"交叉验证汇总结果已保存至: {summary_file}")