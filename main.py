import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from fontTools.varLib.avarPlanner import WEIGHTS
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import WesadDataset
from models import CnnGruModel, CnnGruAttentionModel  # 导入模型
from trainer import Trainer  # 导入训练器

# --- 1. 全局参数直接定义 ---
# 实验设置
MODEL_TO_USE = 'cnn_gru_attention'  # 'cnn_gru' / 'cnn_gru_attention'
RUN_NAME = f"{MODEL_TO_USE}"  # 用于命名输出文件夹
SEED = 42

# 数据集参数
PROCESSED_DATA_PATH = Path('./data')
ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]
# 消融实验在这里修改:
# CHANNELS_TO_USE = ['chest_ACC_x', 'chest_ACC_y', 'chest_ACC_z', 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_Resp']
CHANNELS_TO_USE = [ 'chest_ECG', 'chest_EDA', 'chest_EMG', 'chest_Resp','wrist_BVP', 'wrist_EDA']

# 训练参数
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_WORKERS = 0
PATIENCE = 20
WEIGHTS_DECAY = 1e-4

# 模型参数
NUM_CLASSES = 3
MODEL_PARAMS = {
    'cnn_out_channels': 32,
    'gru_hidden_size': 64,
    'gru_num_layers': 2,
    'dropout': 0.5
}


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- 创建输出目录 ---
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = Path('./output') / RUN_NAME / f'run_{run_timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"====== 运行结果将保存至: {run_output_dir} ======")

    # 加载所有通道的名称
    with open(PROCESSED_DATA_PATH / '_channel_names.txt', 'r') as f:
        all_channel_names = [line.strip() for line in f]

    # --- 执行LOSOCV循环 ---
    results = []
    for subject_to_test in ALL_SUBJECTS:  # 移除 tqdm
        print(f"Processing subject: {subject_to_test}")  # 添加简单的打印提示
        fold_output_dir = run_output_dir / f'fold_test_on_{subject_to_test}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)

        train_val_subjects = [s for s in ALL_SUBJECTS if s != subject_to_test]
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.2, random_state=SEED)

        # 创建数据集
        train_ds = WesadDataset(PROCESSED_DATA_PATH, train_subjects, CHANNELS_TO_USE, all_channel_names)
        val_ds = WesadDataset(PROCESSED_DATA_PATH, val_subjects, CHANNELS_TO_USE, all_channel_names)
        test_ds = WesadDataset(PROCESSED_DATA_PATH, [subject_to_test], CHANNELS_TO_USE, all_channel_names)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # 创建模型
        if MODEL_TO_USE == 'cnn_gru':
            model = CnnGruModel(in_channels=len(CHANNELS_TO_USE), num_classes=NUM_CLASSES, **MODEL_PARAMS)
        elif MODEL_TO_USE == 'cnn_gru_attention':
            model = CnnGruAttentionModel(in_channels=len(CHANNELS_TO_USE), num_classes=NUM_CLASSES, **MODEL_PARAMS)
        else:
            raise ValueError(f"未知模型: {MODEL_TO_USE}")

        # 创建训练器并开始训练
        config_dict = {
            'trainer': {
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'early_stopping': {'enabled': True, 'patience': PATIENCE, 'delta': 0},
                'weight_decay': WEIGHTS_DECAY
            }
        }
        trainer = Trainer(model, fold_output_dir, config_dict)
        trainer.train(train_loader, val_loader)
        _, test_acc, test_f1 = trainer.evaluate(test_loader, is_test=True)
        results.append({'subject': subject_to_test, 'accuracy': test_acc, 'f1_score': test_f1})

    # --- 汇总结果 ---
    print("\n\n====== 留一法交叉验证全部完成 ======")

    # 提取所有折叠的结果
    all_fold_accs = [r['accuracy'] for r in results]
    all_fold_f1s = [r['f1_score'] for r in results]

    summary_file = run_output_dir / 'cv_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("实验配置:\n")
        f.write(f"MODEL_TO_USE: {MODEL_TO_USE}\n")
        f.write(f"RUN_NAME: {RUN_NAME}\n")
        f.write(f"SEED: {SEED}\n")
        f.write(f"CHANNELS_TO_USE: {CHANNELS_TO_USE}\n")
        f.write(f"EPOCHS: {EPOCHS}\n")
        f.write(f"BATCH_SIZE: {BATCH_SIZE}\n")
        f.write(f"LEARNING_RATE: {LEARNING_RATE}\n")
        f.write(f"NUM_WORKERS: {NUM_WORKERS}\n")
        f.write(f"PATIENCE: {PATIENCE}\n")
        f.write(f"NUM_CLASSES: {NUM_CLASSES}\n")
        f.write(f"MODEL_PARAMS: {MODEL_PARAMS}\n")
        f.write("\n每个折叠的详细结果:\n")
        for res in results:
            f.write(f"  - 测试 {res['subject']}: Accuracy = {res['accuracy']:.4f}, F1-score = {res['f1_score']:.4f}\n")
        f.write("\n最终平均性能:\n")
        f.write(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}\n")
        f.write(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}\n")

    print(f"交叉验证汇总结果已保存至: {summary_file}")
    # 同时在终端打印最终结果
    print("\n--- 最终平均性能 ---")
    print(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}")
    print(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}")


if __name__ == '__main__':
    main()