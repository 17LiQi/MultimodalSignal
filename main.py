# file: main.py
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import WesadDataset, HybridDataset
from models import CnnGruAttentionModel, HybridModel
from trainer import Trainer
import pandas as pd
import warnings

warnings.filterwarnings("ignore", message="Initializing zero-element tensors is a no-op")

# --- 配置区 ---
USE_HYBRID = False  # True: 使用手工特征 (hybrid模型, chest_raw_align + chest_feature); False: 使用原始信号 (cnn_gru_attention模型, chest_raw)
CLASSIFICATION_MODE = 'ternary'  # 'binary' 2类 or 'ternary' 3类
NUM_CLASSES = 2 if CLASSIFICATION_MODE == 'binary' else 3

# 数据路径配置 (根据模式动态设置)
PROCESSED_DATA_PATH = Path('./data')
EARLY_DATA_PATH = PROCESSED_DATA_PATH / 'chest_raw_align' if USE_HYBRID else PROCESSED_DATA_PATH / 'chest_raw'
FEATURE_DATA_PATH = PROCESSED_DATA_PATH / 'chest_feature'
MODEL_TO_USE = 'cnn_gru_attention'  # 'cnn_gru_attention' or 'hybrid'

# 通道和特征配置
# CHANNELS_TO_USE = ['chest_ECG', 'chest_EDA']
CHANNELS_TO_USE = ['chest_ECG', 'chest_EDA', 'chest_Resp']

ALL_AVAILABLE_FEATURES = [
    'HRV_RMSSD', 'HRV_SDNN', 'HRV_LFHF', 'HRV_HF', 'HRV_SampEn',
    'EDA_SCR_Peaks_N', 'EDA_Tonic_Slope',
    'RESP_Rate_Mean', 'RESP_RRV_SDNN',
    'EMG_Amplitude_Mean'
]
FEATURES_TO_USE = [
    'RESP_RRV_SDNN',
    'EDA_SCR_Peaks_N',
    'HRV_RMSSD']
NUM_HANDCRAFTED_FEATURES = len(FEATURES_TO_USE) if FEATURES_TO_USE is not None else len(ALL_AVAILABLE_FEATURES)


RUN_NAME = MODEL_TO_USE
# 模型参数
SEED = 42
NUM_WORKERS = 0
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 20
WEIGHTS_DECAY = 1e-4

MODEL_PARAMS = {
    'cnn_gru_attention': {
        'cnn_out_channels': 32,
        'gru_hidden_size': 64,
        'gru_num_layers': 2,
        'dropout': 0.5
    },
    # 'hybrid': {
    #     'cnn_out_channels': 32,
    #     'gru_hidden_size': 64,
    #     'gru_num_layers': 2,
    #     'dropout': 0.5,
    #     'feature_hidden_dims': [32],
    #     'feature_out_dim': 16
    # }
    'hybrid': {
        'cnn_out_channels': 32,
        'gru_hidden_size': 64,
        'gru_num_layers': 2,
        'dropout': 0.5,
        'feature_hidden_dims': [64],
        'feature_out_dim': 32
    }
}

ALL_SUBJECTS = [f"S{i}" for i in range(2, 18) if i != 12]


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = Path('./output') / RUN_NAME / f'run_{run_timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"====== 运行结果将保存至: {run_output_dir} ======")

    with open(EARLY_DATA_PATH / '_channel_names.txt', 'r') as f:
        all_channel_names = [line.strip() for line in f]

    results = []
    for subject_to_test in ALL_SUBJECTS:
        print(f"Processing subject: {subject_to_test}")
        fold_output_dir = run_output_dir / f'fold_test_on_{subject_to_test}'
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        train_val_subjects = [s for s in ALL_SUBJECTS if s != subject_to_test]
        train_subjects, val_subjects = train_test_split(train_val_subjects, test_size=0.2, random_state=SEED)

        if MODEL_TO_USE == 'hybrid':
            train_ds = HybridDataset(
                early_fusion_path=EARLY_DATA_PATH,
                feature_fusion_path=FEATURE_DATA_PATH,
                subjects=train_subjects,
                channels_to_use=CHANNELS_TO_USE,
                all_channel_names=all_channel_names,
                features_to_use=FEATURES_TO_USE,
                classification_mode=CLASSIFICATION_MODE
            )
            val_ds = HybridDataset(
                early_fusion_path=EARLY_DATA_PATH,
                feature_fusion_path=FEATURE_DATA_PATH,
                subjects=val_subjects,
                channels_to_use=CHANNELS_TO_USE,
                all_channel_names=all_channel_names,
                features_to_use=FEATURES_TO_USE,
                classification_mode=CLASSIFICATION_MODE
            )
            test_ds = HybridDataset(
                early_fusion_path=EARLY_DATA_PATH,
                feature_fusion_path=FEATURE_DATA_PATH,
                subjects=[subject_to_test],
                channels_to_use=CHANNELS_TO_USE,
                all_channel_names=all_channel_names,
                features_to_use=FEATURES_TO_USE,
                classification_mode=CLASSIFICATION_MODE
            )
        else:
            train_ds = WesadDataset(
                EARLY_DATA_PATH, train_subjects, CHANNELS_TO_USE, all_channel_names,
                classification_mode=CLASSIFICATION_MODE
            )
            val_ds = WesadDataset(
                EARLY_DATA_PATH, val_subjects, CHANNELS_TO_USE, all_channel_names,
                classification_mode=CLASSIFICATION_MODE
            )
            test_ds = WesadDataset(
                EARLY_DATA_PATH, [subject_to_test], CHANNELS_TO_USE, all_channel_names,
                classification_mode=CLASSIFICATION_MODE
            )

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        current_model_params = MODEL_PARAMS[MODEL_TO_USE]

        # ----------- 创建模型实例 -----------------
        if MODEL_TO_USE == 'cnn_gru_attention':
            model = CnnGruAttentionModel(in_channels=len(CHANNELS_TO_USE), num_classes=NUM_CLASSES,
                                         **current_model_params)
        elif MODEL_TO_USE == 'hybrid':
            num_handcrafted_features = len(FEATURES_TO_USE) if FEATURES_TO_USE is not None else 10
            model = HybridModel(in_channels=len(CHANNELS_TO_USE),
                                num_classes=NUM_CLASSES,
                                num_handcrafted_features=num_handcrafted_features,
                                **current_model_params)
        else:
            raise ValueError(f"未知模型: {MODEL_TO_USE}")

        config_dict = {
            'trainer': {
                'epochs': EPOCHS,
                'learning_rate': LEARNING_RATE,
                'early_stopping': {'enabled': True, 'patience': PATIENCE, 'delta': 0},
                'weight_decay': WEIGHTS_DECAY,
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
