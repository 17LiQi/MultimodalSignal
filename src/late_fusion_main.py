# late_fusion_main.py
import torch
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

# --- 从项目中导入模块 ---
from utils.path_manager import get_path_manager
from dataset.wesad_early_dataset import WesadEarlyFusionDataset
from models.early_cnn_gru import CnnGruModel
from trainer.trainer import Trainer  # 复用标准的 PyTorch 训练器


def train_single_expert(config, train_subjects, val_subjects, channels, paths, fold_output_dir):
    """
    在一个数据折叠内部，训练一个单模态专家模型。
    现在接收 fold_output_dir 以保存该专家模型的日志和权重。
    """

    # 为这个专家模型创建一个独立的子目录
    channel_name = "_".join(channels)
    expert_output_dir = fold_output_dir / f"expert_{channel_name}"
    expert_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 创建数据集
    data_folder = paths.DATA_ROOT / 'wesad_early_fusion'
    train_ds = WesadEarlyFusionDataset(data_folder, train_subjects, channels, config.dataset.all_channel_names)
    val_ds = WesadEarlyFusionDataset(data_folder, val_subjects, channels, config.dataset.all_channel_names)

    # 2. 创建模型
    model_params = dict(config.model.get('params', {}))  # 获取结构超参数

    model = CnnGruModel(
        in_channels=len(channels),
        num_classes=config.model.num_classes,  # 从顶层获取
        **model_params  # 安全地解包
    )

    # 3. 训练模型
    trainer = Trainer(model, config, expert_output_dir, train_ds)
    train_loader = DataLoader(train_ds, batch_size=config.trainer.batch_size, shuffle=True,
                              num_workers=config.trainer.num_workers)
    val_loader = DataLoader(val_ds, batch_size=config.trainer.batch_size, shuffle=False,
                            num_workers=config.trainer.num_workers)

    print(f"  - 正在为通道 {channels} 训练专家模型 (结果保存至 {expert_output_dir})...")
    trainer.train(train_loader, val_loader)

    # 返回训练好的模型 (已加载最佳权重)
    return trainer.model


def run_late_fusion_fold(config, test_subject, all_subjects, paths, run_output_dir):
    """
    运行单次晚期融合的交叉验证折叠。
    """
    # 1. 创建该折叠的输出目录
    fold_output_dir = run_output_dir / f'fold_test_on_{test_subject}'
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n===== 开始处理折叠: 测试集=[{test_subject}] =====")

    # 2. 划分训练/验证受试者
    train_val_subjects = [s for s in all_subjects if s != test_subject]
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=config.dataset.val_split_ratio, random_state=config.seed
    )

    expert_models = []
    # 3. 在当前折叠内部，为每个专家模态训练一个模型
    for channels in config.fusion.expert_channels:
        # 传递 config, subjects, channels, paths, fold_output_dir
        model = train_single_expert(config, train_subjects, val_subjects, channels, paths, fold_output_dir)
        expert_models.append(model)

    # 4. 在测试集上获取所有专家的预测概率
    print(f"  - 在测试集 {test_subject} 上进行评估...")

    data_folder = paths.DATA_ROOT / 'wesad_early_fusion'
    all_probas = []  # 存储每个模型的预测概率
    true_labels = None  # 我们只需要获取一次真实标签
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 循环创建每个专家专用的测试集和加载器
    for i, model in enumerate(expert_models):
        expert_channels = config.fusion.expert_channels[i]

        # 为当前专家创建一个只包含其所需通道的测试集
        test_ds_expert = WesadEarlyFusionDataset(
            data_folder,
            [test_subject],
            expert_channels,  # 传入正确的通道
            config.dataset.all_channel_names
        )

        # 第一次循环时，获取真实标签
        if true_labels is None:
            true_labels = test_ds_expert.labels

        test_loader_expert = DataLoader(test_ds_expert, batch_size=config.trainer.batch_size, shuffle=False)

        model.eval()
        probas = []
        with torch.no_grad():
            for inputs, _ in test_loader_expert:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probas.append(torch.softmax(outputs, dim=1).cpu().numpy())

        all_probas.append(np.concatenate(probas, axis=0))

    # 5. 进行融合
    if true_labels is None:
        print("警告: 未能获取测试集标签，跳过此折叠的评估。")
        return 0.0, 0.0

    if config.fusion.strategy == 'average':
        final_probas = np.mean(all_probas, axis=0)
        final_preds = np.argmax(final_probas, axis=1)
    else:
        raise NotImplementedError(f"融合策略 '{config.fusion.strategy}' 尚未实现。")

    # 6. 计算该折叠的性能
    acc = accuracy_score(true_labels, final_preds)
    f1 = f1_score(true_labels, final_preds, average='weighted')

    print(f"  - 折叠 {test_subject} 融合结果: Accuracy = {acc:.4f}, F1-score = {f1:.4f}")
    return acc, f1


def main():
    # --- A. 加载配置 ---
    try:
        base_conf = OmegaConf.load('configs/base_config.yaml')
        exp_conf = OmegaConf.load('configs/late_fusion_gru.yaml')
        config = OmegaConf.merge(base_conf, exp_conf)
        print("已加载晚期融合实验配置。")
    except Exception as e:
        print(f"错误: 无法加载配置文件。 {e}")
        exit()

    # --- B. 初始化 ---
    paths = get_path_manager()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建本次运行的主输出目录，与 main.py 逻辑一致
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 目录名包含数据集、模型(这里是'late_fusion')和时间戳
    run_output_dir = paths.OUTPUT_ROOT / 'wesad_late_fusion' / f'{config.model.name}' / f'run_{run_timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"====== 本次交叉验证运行结果将保存至: {run_output_dir} ======")

    all_subjects = config.dataset.all_subjects
    results = []

    # --- C. 执行留一法交叉验证 (LOSOCV) 循环 ---
    print("\n====== 开始晚期融合的留一法交叉验证 ======")
    for subject_to_test in all_subjects:
        acc, f1 = run_late_fusion_fold(config, subject_to_test, all_subjects, paths, run_output_dir)
        results.append({'subject': subject_to_test, 'accuracy': acc, 'f1_score': f1})

    # --- D. 汇总并打印/保存最终结果 (与 main.py 逻辑一致) ---
    print("\n\n====== 晚期融合交叉验证全部完成 ======")

    all_fold_accs = [r['accuracy'] for r in results]
    all_fold_f1s = [r['f1_score'] for r in results]

    summary_file = run_output_dir / 'cv_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"实验配置:\n{OmegaConf.to_yaml(exp_conf)}\n")
        f.write("\n每个折叠的详细结果:\n")
        for res in results:
            f.write(f"  - 测试 {res['subject']}: Accuracy = {res['accuracy']:.4f}, F1-score = {res['f1_score']:.4f}\n")
        f.write("\n最终平均性能:\n")
        f.write(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}\n")
        f.write(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}\n")

    print(f"交叉验证汇总结果已保存至: {summary_file}")
    print("\n--- 最终平均性能 ---")
    print(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}")
    print(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}")


if __name__ == '__main__':
    main()