# file: main.py
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import WesadDataset
from models import CnnGruAttentionModel
from trainer import Trainer
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")


# --- 配置区 ---
USE_HIERARCHICAL_CLASSIFICATION = False # True: 运行分层分类; False: 运行标准二分类

if USE_HIERARCHICAL_CLASSIFICATION:
    RUN_NAME = 'hierarchical_binary'
    # --- 第一层模型 (M1: Stress vs Non-Stress) 配置 ---
    M1_CHANNELS_TO_USE = ['chest_ECG', 'chest_EDA', 'chest_Resp']
    M1_MODEL_PARAMS = {
        'cnn_out_channels': 32,
        'gru_hidden_size': 64,
        'gru_num_layers': 2,
        'dropout': 0.5
    }

    # --- 第二层模型 (M2: Fun vs Base) 配置 ---
    M2_CHANNELS_TO_USE = ['chest_ECG', 'chest_EDA', 'chest_Resp'] # 可以尝试不同组合
    M2_MODEL_PARAMS = {
        'cnn_out_channels': 32,
        'gru_hidden_size': 32, # 可以使用更小的模型
        'gru_num_layers': 1,
        'dropout': 0.5
    }
else:
    # --- 标准二分类实验配置 ---
    RUN_NAME = 'simple_binary'
    CLASSIFICATION_MODE = 'stress_binary'
    NUM_CLASSES = 2
    MODEL_TO_USE = 'cnn_gru_attention'
    CHANNELS_TO_USE = ['chest_ECG', 'chest_EDA', 'chest_Resp']
    MODEL_PARAMS = {
        'cnn_gru_attention': {
            'cnn_out_channels': 32,
            'gru_hidden_size': 64,
            'gru_num_layers': 2,
            'dropout': 0.5
        }
    }

# --- 通用实验配置 ---
PROCESSED_DATA_PATH = Path('./data')
EARLY_DATA_PATH = PROCESSED_DATA_PATH / 'chest_raw'
SEED = 42
NUM_WORKERS = 0
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 20
WEIGHTS_DECAY = 1e-4
ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = Path('./output') / RUN_NAME / f'run_{run_timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"====== 运行结果将保存至: {run_output_dir} ======")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"====== 使用设备: {device} ======")

    with open(EARLY_DATA_PATH / '_channel_names.txt', 'r') as f:
        all_channel_names = [line.strip() for line in f]

    # --- 根据总开关选择执行路径 ---
    if USE_HIERARCHICAL_CLASSIFICATION:
        run_hierarchical_experiment(run_output_dir, device, all_channel_names)
    else:
        run_simple_experiment(run_output_dir, device, all_channel_names)


def run_simple_experiment(run_output_dir, device, all_channel_names):
    """执行标准的、非分层的分类实验"""
    print("\n" + "=" * 80)
    print(f"开始执行标准二分类实验 (模式: {CLASSIFICATION_MODE})")
    print("=" * 80)

    results = []
    for subject_to_test in ALL_SUBJECTS:
        print(f"\n--- 处理折叠: 测试受试者 {subject_to_test} ---")
        fold_output_dir = run_output_dir / f'fold_test_on_{subject_to_test}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        train_val_subjects = [s for s in ALL_SUBJECTS if s != subject_to_test]
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.2, random_state=SEED)

        train_ds = WesadDataset(EARLY_DATA_PATH, train_subjects, CHANNELS_TO_USE, all_channel_names,
                                classification_mode=CLASSIFICATION_MODE)
        val_ds = WesadDataset(EARLY_DATA_PATH, val_subjects, CHANNELS_TO_USE, all_channel_names,
                              classification_mode=CLASSIFICATION_MODE)
        test_ds = WesadDataset(EARLY_DATA_PATH, [subject_to_test], CHANNELS_TO_USE, all_channel_names,
                               classification_mode=CLASSIFICATION_MODE)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        model = CnnGruAttentionModel(in_channels=len(CHANNELS_TO_USE), num_classes=NUM_CLASSES,
                                     **MODEL_PARAMS[MODEL_TO_USE])

        config_dict = {'trainer': {'epochs': EPOCHS, 'learning_rate': LEARNING_RATE,
                                   'early_stopping': {'enabled': True, 'patience': PATIENCE, 'delta': 0},
                                   'weight_decay': WEIGHTS_DECAY}}
        trainer = Trainer(model, fold_output_dir, config_dict)
        trainer.train(train_loader, val_loader)
        _, test_acc, test_f1 = trainer.evaluate(test_loader, is_test=True)
        results.append({'subject': subject_to_test, 'accuracy': test_acc, 'f1_score': test_f1})

    # --- 汇总结果 ---
    print("\n\n====== 留一法交叉验证全部完成 ======")
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


def run_hierarchical_experiment(run_output_dir, device, all_channel_names):
    """执行分层分类实验"""
    print("\n" + "=" * 80)
    print("开始执行分层分类实验")
    print("=" * 80)

    all_final_preds = []
    all_final_true_labels = []

    for subject_to_test in ALL_SUBJECTS:
        fold_start_time = datetime.now()
        print(f"\n--- 处理折叠: 测试受试者 {subject_to_test} | 开始时间: {fold_start_time.strftime('%H:%M:%S')} ---")

        fold_output_dir = run_output_dir / f'fold_test_on_{subject_to_test}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        train_val_subjects = [s for s in ALL_SUBJECTS if s != subject_to_test]
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.2, random_state=SEED)

        # --- 训练 M1: Stress vs Non-Stress ---
        print("\n--- 阶段 1: 训练 压力 vs 非压力 分类器 (M1) ---")
        m1_train_ds = WesadDataset(EARLY_DATA_PATH, train_subjects, M1_CHANNELS_TO_USE, all_channel_names,
                                   classification_mode='stress_binary')
        m1_val_ds = WesadDataset(EARLY_DATA_PATH, val_subjects, M1_CHANNELS_TO_USE, all_channel_names,
                                 classification_mode='stress_binary')
        m1_train_loader = DataLoader(m1_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        m1_val_loader = DataLoader(m1_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        model_m1 = CnnGruAttentionModel(in_channels=len(M1_CHANNELS_TO_USE), num_classes=2, **M1_MODEL_PARAMS)
        trainer_m1 = Trainer(model_m1, fold_output_dir / 'model_m1', {
            'trainer': {'epochs': EPOCHS, 'learning_rate': LEARNING_RATE,
                        'early_stopping': {'enabled': True, 'patience': PATIENCE, 'delta': 0},
                        'weight_decay': WEIGHTS_DECAY}})
        trainer_m1.train(m1_train_loader, m1_val_loader)

        # --- 训练 M2: Fun vs Base ---
        print("\n--- 阶段 2: 训练 娱乐 vs 基线 分类器 (M2) ---")
        m2_train_ds = WesadDataset(EARLY_DATA_PATH, train_subjects, M2_CHANNELS_TO_USE, all_channel_names,
                                   classification_mode='amusement_binary')
        m2_val_ds = WesadDataset(EARLY_DATA_PATH, val_subjects, M2_CHANNELS_TO_USE, all_channel_names,
                                 classification_mode='amusement_binary')
        if len(m2_train_ds) == 0 or len(m2_val_ds) == 0:
            print("警告: 训练集或验证集在 amusement_binary 模式下没有数据，跳过此折叠。")
            continue
        m2_train_loader = DataLoader(m2_train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        m2_val_loader = DataLoader(m2_val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        model_m2 = CnnGruAttentionModel(in_channels=len(M2_CHANNELS_TO_USE), num_classes=2, **M2_MODEL_PARAMS)
        trainer_m2 = Trainer(model_m2, fold_output_dir / 'model_m2', {
            'trainer': {'epochs': EPOCHS, 'learning_rate': LEARNING_RATE,
                        'early_stopping': {'enabled': True, 'patience': PATIENCE, 'delta': 0},
                        'weight_decay': WEIGHTS_DECAY}})
        trainer_m2.train(m2_train_loader, m2_val_loader)

        # --- 添加: 阶段 3.1: 单独评估第一层 M1 的二分类准确率 ---
        print("\n--- 阶段 3.1: 评估第一层 M1 (压力 vs 非压力) 的准确率 ---")
        m1_test_ds = WesadDataset(EARLY_DATA_PATH, [subject_to_test], M1_CHANNELS_TO_USE, all_channel_names,
                                  classification_mode='stress_binary')
        m1_test_loader = DataLoader(m1_test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        _, m1_acc, m1_f1 = trainer_m1.evaluate(m1_test_loader, is_test=True)
        print(f"第一层 M1 在 {subject_to_test} 上的准确率: Accuracy = {m1_acc:.4f}, F1 = {m1_f1:.4f}")

        # --- 阶段 3.2: 分层评估 (三分类) ---
        print(f"\n--- 阶段 3.2: 在测试受试者 {subject_to_test} 上进行分层评估 ---")
        all_eval_channels = list(set(M1_CHANNELS_TO_USE + M2_CHANNELS_TO_USE))
        test_ds_ternary_full = WesadDataset(EARLY_DATA_PATH, [subject_to_test], all_eval_channels, all_channel_names,
                                            classification_mode='ternary')
        if len(test_ds_ternary_full) == 0:
            print(f"警告: 测试受试者 {subject_to_test} 没有可用于三分类评估的数据。")
            continue
        test_loader_ternary = DataLoader(test_ds_ternary_full, batch_size=BATCH_SIZE, shuffle=False,
                                         num_workers=NUM_WORKERS)
        model_m1 = trainer_m1.model
        model_m2 = trainer_m2.model
        model_m1.eval()
        model_m2.eval()
        fold_true_labels = test_ds_ternary_full.labels
        fold_preds = []
        with torch.no_grad():
            for inputs_full, _ in test_loader_ternary:
                inputs_full = inputs_full.to(device)
                m1_channel_indices = [all_eval_channels.index(ch) for ch in M1_CHANNELS_TO_USE]
                inputs_m1 = inputs_full[:, m1_channel_indices, :]
                m2_channel_indices = [all_eval_channels.index(ch) for ch in M2_CHANNELS_TO_USE]
                inputs_m2 = inputs_full[:, m2_channel_indices, :]
                m1_preds = torch.argmax(model_m1(inputs_m1), dim=1)
                m2_preds = torch.argmax(model_m2(inputs_m2), dim=1)
                final_preds_batch = [2 if p1 == 1 else p2.item() for p1, p2 in zip(m1_preds, m2_preds)]
                fold_preds.extend(final_preds_batch)
        all_final_preds.extend(fold_preds)
        all_final_true_labels.extend(fold_true_labels)
        print(f"\n折叠 {subject_to_test} 评估完成。")

if __name__ == '__main__':
    main()
