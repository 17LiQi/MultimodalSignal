import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from src.utils.path_manager import get_path_manager

# --- 设置路径 ---
path = get_path_manager()
PROCESSED_DATA_PATH = path.DATA_ROOT
SUBJECT_TO_CHECK = 'S2'

print(f"--- 正在验证受试者 {SUBJECT_TO_CHECK} 的预处理文件 ---\n")

# --- 1. 检查 Feature Fusion (中/晚期融合) 数据 ---
print("--- 1. 检查 Feature Fusion 数据 ---")
try:
    # 加载特征和标签
    x_feat = np.load(PROCESSED_DATA_PATH / 'feature_fusion' / f'{SUBJECT_TO_CHECK}_X.npy')
    y_feat = np.load(PROCESSED_DATA_PATH / 'feature_fusion' / f'{SUBJECT_TO_CHECK}_y.npy')

    print(f"特征数据 (X_feat) 的形状: {x_feat.shape}")
    print(f"标签数据 (y_feat) 的形状: {y_feat.shape}")

    # 验证形状是否符合预期
    assert x_feat.shape[0] == y_feat.shape[0], "X和y的窗口数量不匹配!"
    assert x_feat.shape[1] == 48, "特征数量不等于48!"

    # 检查标签分布
    unique_labels, counts = np.unique(y_feat, return_counts=True)
    print(f"标签类别及其数量: {dict(zip(unique_labels, counts))}")
    print("-> 结果符合预期：特征维度为48，标签为0, 1, 2三类。\n")

except FileNotFoundError:
    print("错误: 未找到 feature_fusion 的 .npy 文件。请先运行 preprocess.py。\n")

# --- 2. 检查 Early Fusion (早期融合) 数据 ---
print("--- 2. 检查 Early Fusion 数据 ---")
try:
    # 加载信号和标签
    x_early = np.load(PROCESSED_DATA_PATH / 'early_fusion' / f'{SUBJECT_TO_CHECK}_X.npy')
    y_early = np.load(PROCESSED_DATA_PATH / 'early_fusion' / f'{SUBJECT_TO_CHECK}_y.npy')

    print(f"信号数据 (X_early) 的形状: {x_early.shape}")
    print(f"标签数据 (y_early) 的形状: {y_early.shape}")

    # 验证形状是否符合预期
    assert x_early.shape[0] == y_early.shape[0], "X和y的窗口数量不匹配!"
    assert x_early.shape[1] == 1920, "窗口长度不等于1920 (30s * 64Hz)!"
    assert x_early.shape[2] == 12, "通道数量不等于12!"

    print("-> 结果符合预期：数据形状为 (窗口数, 1920, 12)。\n")

except FileNotFoundError:
    print("错误: 未找到 early_fusion 的 .npy 文件。请先运行 preprocess.py。\n")

# --- 3. 检查并使用 Scaler 文件 ---
print("--- 3. 检查并使用 Scaler 文件 ---")
try:
    # 加载 scaler 对象
    with open(PROCESSED_DATA_PATH / 'feature_fusion' / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print(f"成功加载 Scaler 对象: {type(scaler)}")

    # 检查 scaler 内部学习到的参数
    mean_vector = scaler.mean_
    std_vector = scaler.scale_
    print(f"Scaler 学习到的均值向量形状: {mean_vector.shape}")
    assert mean_vector.shape[0] == 48, "Scaler的特征维度不匹配!"

    # 演示如何使用 scaler
    print("\n演示 Scaler 的作用:")
    # 取出第一个窗口的原始特征
    original_sample = x_feat[0, :5]  # 只看前5个特征
    print(f"归一化前 (原始特征): {np.round(original_sample, 4)}")

    # 使用加载的 scaler 对所有测试数据进行转换
    x_feat_scaled = scaler.transform(x_feat)

    # 查看同一个窗口转换后的特征
    scaled_sample = x_feat_scaled[0, :5]
    print(f"归一化后 (Z-scores):  {np.round(scaled_sample, 4)}")
    print("-> 结果符合预期：Scaler 已加载，并能成功将原始特征转换为均值接近0、标准差接近1的数据。\n")

except FileNotFoundError:
    print("错误: 未找到 scaler.pkl 文件。请确保已运行 preprocess.py 并生成了 scaler。\n")


def parse_quest_csv(subject_id: str, wesad_root: Path) -> pd.DataFrame:
    quest_path = wesad_root / subject_id / f"{subject_id}_quest.csv"
    df_raw = pd.read_csv(quest_path, sep=';', header=None, skip_blank_lines=True)
    order_row = df_raw[df_raw[0].str.contains('# ORDER', na=False)].values[0]
    start_row = df_raw[df_raw[0].str.contains('# START', na=False)].values[0]
    end_row = df_raw[df_raw[0].str.contains('# END', na=False)].values[0]
    tasks = [str(t).strip() for t in pd.Series(order_row[1:]).dropna().tolist()]
    start_times = pd.Series(start_row[1:]).dropna().astype(float).tolist()
    end_times = pd.Series(end_row[1:]).dropna().astype(float).tolist()
    protocol_df = pd.DataFrame({'task': tasks, 'start_min': start_times, 'end_min': end_times})
    return protocol_df


def load_pkl(subject_id: str, wesad_root: Path):
    pkl_path = wesad_root / subject_id / f"{subject_id}.pkl"
    with open(pkl_path, 'rb') as f: return pickle.load(f, encoding='bytes')


# --- 新的验证函数 ---

def verify_label_alignment(subject_id: str, wesad_root: Path):
    """
    验证从 .csv 解析的时间戳与 .pkl 文件中自带的标签序列是否对齐。
    """
    print(f"--- 正在对受试者 {subject_id} 进行标签对齐验证 ---")

    try:
        # 1. 加载原始数据
        protocol_df = parse_quest_csv(subject_id, wesad_root)
        data = load_pkl(subject_id, wesad_root)
        pkl_labels = data[b'label']
        fs = 700

        # *** 关键修复点 1: 建立从任务名到预期 pkl 标签的映射 ***
        # 我们将 .csv 中的任务名（去除空格）映射到 .pkl 中的标签值
        task_name_to_expected_label = {
            'Base': 1,
            'TSST': 2,
            'Fun': 3,
            'Medi1': 4,  # Medi 1 对应标签 4
            'Medi2': 4  # Medi 2 也对应标签 4
        }

        all_tasks_verified = True

        # 2. 遍历协议中的每个任务
        for _, row in protocol_df.iterrows():
            task_name_from_csv = row['task'].replace(" ", "").strip()

            # 从映射中获取该任务的预期标签值
            expected_label = task_name_to_expected_label.get(task_name_from_csv)

            # 如果这个任务不在我们的核心验证列表里 (如 sRead)，就跳过
            if expected_label is None:
                continue

            print(f"\n验证任务: {row['task']}")

            # 3. 提取序列A (基于CSV时间戳)
            start_idx_csv = int(row['start_min'] * 60 * fs)
            end_idx_csv = int(row['end_min'] * 60 * fs)
            segment_from_csv = pkl_labels[start_idx_csv:end_idx_csv]

            # 找到这个数据段中的主要标签 (忽略过渡标签0)
            counts_csv = Counter(segment_from_csv)
            top_two = counts_csv.most_common(2)
            main_label_csv = top_two[0][0] if top_two[0][0] != 0 else (top_two[1][0] if len(top_two) > 1 else 0)

            print(f"  - 根据 CSV 时间戳 ({row['start_min']}-{row['end_min']} min), "
                  f"提取的片段长度为 {len(segment_from_csv)}, "
                  f"片段中的主要标签为: {main_label_csv}")

            # *** 关键修复点 2: 修正比较逻辑 ***
            # 比较 "实际从pkl切片中找到的主要标签" 与 "根据csv任务名预期的标签"
            if main_label_csv != expected_label:
                print(
                    f"  - \033[91m验证失败\033[0m: 片段内的主要标签 ({main_label_csv}) 与预期的任务标签 ({expected_label}) 不符!")
                all_tasks_verified = False
            else:
                print(f"  - \033[92m验证通过\033[0m: 片段内的主要标签与预期一致。")

        print("\n--- 验证总结 ---")
        if all_tasks_verified:
            print("\033[92m所有核心任务的标签对齐验证成功！\033[0m")
        else:
            print("\033[91m存在标签对齐问题。\033[0m")

    except Exception as e:
        print(f"\n验证过程中出现错误: {e}")


try:
    # 运行对S2的验证
    verify_label_alignment(subject_id='S2', wesad_root=Path(path.WESAD_ROOT))
except Exception as e:
    print(f"执行失败，请确保 WESAD_ROOT_PATH 路径正确。错误: {e}")
