# file: preprocess.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from tqdm import tqdm

# --- 1. 参数直接定义 ---
WESAD_ROOT = Path('./WESAD')
OUTPUT_PATH = Path('./data')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# 信号和窗口化参数
CHEST_FS = 700
WRIST_FS = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}
TARGET_FS = 64
WINDOW_SEC = 30
# 我们将使用所有胸部和腕部信号
CHEST_CHANNELS = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
WRIST_CHANNELS = ['ACC', 'BVP', 'EDA', 'TEMP']

# 标签映射 (2:stress, 1:amusement, 0:baseline/neutral/meditation)
LABEL_MAP = {1: 0, 2: 2, 3: 1, 4: 0}


# --- 2. 辅助函数 ---
def parse_quest_csv(subject_id: str, wesad_root: Path) -> pd.DataFrame:
    """动态解析指定受试者的 _quest.csv 文件，以获取任务顺序和时间。"""
    quest_path = wesad_root / subject_id / f"{subject_id}_quest.csv"
    df_raw = pd.read_csv(quest_path, sep=';', header=None, skip_blank_lines=True)

    order_row = df_raw[df_raw[0].str.contains('# ORDER', na=False)].values[0]
    start_row = df_raw[df_raw[0].str.contains('# START', na=False)].values[0]
    end_row = df_raw[df_raw[0].str.contains('# END', na=False)].values[0]

    tasks = pd.Series(order_row[1:]).dropna().tolist()
    start_times = pd.Series(start_row[1:]).dropna().astype(float).tolist()
    end_times = pd.Series(end_row[1:]).dropna().astype(float).tolist()

    if not (len(tasks) == len(start_times) == len(end_times)):
        raise ValueError(f"为受试者 {subject_id} 解析出的任务、开始、结束时间长度不匹配!")

    protocol_df = pd.DataFrame({
        'task': tasks,
        'start_min': start_times,
        'end_min': end_times
    })
    return protocol_df


def load_pkl(subject_id: str, wesad_root: Path):
    """加载指定受试者的 pkl 文件。"""
    pkl_path = wesad_root / subject_id / f"{subject_id}.pkl"
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            return data
    except FileNotFoundError:
        print(f"警告: 无法找到文件 {pkl_path}")
        return None


def _get_windows(data_len, protocol_df, original_fs):
    """根据协议生成窗口索引，适配统一的 TARGET_FS。"""
    windows = []
    labels = []
    task_to_label_map = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}

    for _, row in protocol_df.iterrows():
        task = row['task'].replace(" ", "").strip()
        original_label = task_to_label_map.get(task)
        if original_label is None:
            continue
        label = LABEL_MAP.get(original_label)
        if label is None:
            continue

        start_idx = int(row['start_min'] * 60 * original_fs)
        end_idx = int(row['end_min'] * 60 * original_fs)
        window_samples = int(WINDOW_SEC * original_fs)
        step_samples = window_samples // 2  # 50% 重叠

        if end_idx > data_len:
            print(f"警告: 任务 '{row['task']}' 的结束索引 ({end_idx}) 超出数据长度 ({data_len})。将截断至数据末尾。")
            end_idx = data_len

        for i in range(start_idx, end_idx - window_samples + 1, step_samples):
            windows.append((i, i + window_samples))
            labels.append(label)

    return windows, np.array(labels)


# --- 3. 主预处理函数 ---
def run_preprocessing():
    subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
    print(f"--- 开始预处理 {len(subject_ids)} 位受试者 ---")

    # 定义所有通道名称，用于保存
    all_channel_names = [f"chest_{c}_{ax}" for c in ['ACC'] for ax in 'xyz'] + \
                        [f"chest_{c}" for c in ['ECG', 'EDA', 'EMG', 'Resp', 'Temp']] + \
                        [f"wrist_{c}_{ax}" for c in ['ACC'] for ax in 'xyz'] + \
                        [f"wrist_{c}" for c in ['BVP', 'EDA', 'TEMP']]

    with open(OUTPUT_PATH / '_channel_names.txt', 'w') as f:
        for name in all_channel_names:
            f.write(f"{name}\n")

    for sid in tqdm(subject_ids, desc="Preprocessing Subjects"):
        # 1. 加载数据和协议
        data = load_pkl(sid, WESAD_ROOT)
        if data is None:
            continue

        protocol_df = parse_quest_csv(sid, WESAD_ROOT)

        # 2. 重采样所有信号到 TARGET_FS
        resampled_signals = {}
        for dev, channels, fs_dict in [
            ('chest', CHEST_CHANNELS, {c: CHEST_FS for c in CHEST_CHANNELS}),
            ('wrist', WRIST_CHANNELS, WRIST_FS)
        ]:
            for channel in channels:
                original_signal = data[b'signal'][dev.encode()][channel.encode()]
                fs = fs_dict[channel]
                duration = len(original_signal) / fs
                new_len = int(duration * TARGET_FS)

                if original_signal.ndim > 1:  # 处理多轴信号（如 ACC）
                    resampled_axes = [signal.resample(original_signal[:, i], new_len) for i in
                                      range(original_signal.shape[1])]
                    resampled_signals[f"{dev}_{channel}"] = np.column_stack(resampled_axes)
                else:
                    resampled_signals[f"{dev}_{channel}"] = signal.resample(original_signal, new_len)

        # 3. 生成窗口索引
        target_windows, y = _get_windows(len(resampled_signals['chest_ECG']), protocol_df, TARGET_FS)
        if not target_windows:
            print(f"警告: 受试者 {sid} 未生成任何窗口，跳过。")
            continue

        # 4. 拼接所有通道
        all_resampled_data = []
        for dev, channels in [('chest', CHEST_CHANNELS), ('wrist', WRIST_CHANNELS)]:
            for channel in channels:
                key = f"{dev}_{channel}"
                s = resampled_signals[key]
                if s.ndim == 1:
                    s = s.reshape(-1, 1)
                all_resampled_data.append(s)

        concatenated_signals = np.concatenate(all_resampled_data, axis=1)

        # 5. 根据窗口切片
        window_len_samples = int(WINDOW_SEC * TARGET_FS)
        X = np.zeros((len(target_windows), window_len_samples, len(all_channel_names)))
        for i, (start, end) in enumerate(target_windows):
            X[i, :, :] = concatenated_signals[start:end, :]

        # 6. 保存
        np.save(OUTPUT_PATH / f'{sid}_X.npy', X)
        np.save(OUTPUT_PATH / f'{sid}_y.npy', y)
        print(f"  - 受试者 {sid}: 已生成 {X.shape[0]} 个窗口，形状为 {X.shape[1:]}")

    print(f"\n预处理完成！结果已保存至: {OUTPUT_PATH}")


if __name__ == '__main__':
    run_preprocessing()