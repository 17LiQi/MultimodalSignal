# file: preprocess.py
import pickle
import scipy.signal as signal
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress
from tqdm import tqdm
import neurokit2 as nk
import warnings

# --- 1. 参数直接定义 ---
WESAD_ROOT = Path('./WESAD')
OUTPUT_PATH = Path('./data')
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

ORIGINAL_CHEST_FS = 700
# PROCESS_TARGETS = ['raw', 'raw-align', 'feature']
# PROCESS_TARGETS = ['raw']
PROCESS_TARGETS = ['raw-align', 'feature']
RAW_FS = 128
RAW_WINDOW_SEC = 60
RAW_STRIDE_SEC = 10
FEATURE_FS = 128
FEATURE_WINDOW_SEC = 60
FEATURE_STRIDE_SEC = 10
CHEST_CHANNELS = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
TASK_TO_LABEL_MAP = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}  # 原始标签

if 'raw' in PROCESS_TARGETS:
    RAW_PATH = OUTPUT_PATH / 'chest_raw'
    RAW_PATH.mkdir(parents=True, exist_ok=True)
if 'raw-align' in PROCESS_TARGETS:
    RAW_ALIGN_PATH = OUTPUT_PATH / 'chest_raw_align'
    RAW_ALIGN_PATH.mkdir(parents=True, exist_ok=True)
if 'feature' in PROCESS_TARGETS:
    FEATURE_PATH = OUTPUT_PATH / 'chest_feature'
    FEATURE_PATH.mkdir(parents=True, exist_ok=True)

# --- 2. 辅助函数 ---
def parse_quest_csv(subject_id: str, wesad_root: Path) -> pd.DataFrame:
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
    protocol_df = pd.DataFrame({'task': tasks, 'start_min': start_times, 'end_min': end_times})
    if subject_id in ['S2', 'S6'] and 'Base' in protocol_df['task'].values:
        base_idx = protocol_df[protocol_df['task'] == 'Base'].index[0]
        start_min = protocol_df.loc[base_idx, 'start_min']
        end_min = protocol_df.loc[base_idx, 'end_min']
        protocol_df.loc[base_idx, 'start_min'] = (start_min + end_min) / 2
    return protocol_df

def load_pkl(subject_id: str, wesad_root: Path):
    pkl_path = wesad_root / subject_id / f"{subject_id}.pkl"
    try:
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
            return data
    except FileNotFoundError:
        print(f"警告: 无法找到文件 {pkl_path}")
        return None

def resample_signal(signal_data, original_fs, target_fs):
    if signal_data.ndim > 1:
        resampled = np.column_stack([signal.resample(signal_data[:, i], int(len(signal_data) * (target_fs / original_fs))) for i in range(signal_data.shape[1])])
    else:
        resampled = signal.resample(signal_data, int(len(signal_data) * (target_fs / original_fs)))
    return resampled

def extract_handcrafted_features(window_data_dict, fs):
    features = {}
    warnings.filterwarnings('ignore', category=UserWarning, module='neurokit2')
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='neurokit2')
    ecg_signal = window_data_dict.get('chest_ECG')
    try:
        ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs)
        _, ecg_info = nk.ecg_peaks(ecg_cleaned, sampling_rate=fs)
        rpeaks = ecg_info["ECG_R_Peaks"]
        hrv_df = nk.hrv(rpeaks, sampling_rate=fs, show=False)
        features['HRV_RMSSD'] = hrv_df['HRV_RMSSD'].iloc[0]
        features['HRV_SDNN'] = hrv_df['HRV_SDNN'].iloc[0]
        features['HRV_LFHF'] = hrv_df['HRV_LFHF'].iloc[0]
        features['HRV_HF'] = hrv_df['HRV_HF'].iloc[0]
        features['HRV_SampEn'] = hrv_df['HRV_SampEn'].iloc[0]
    except Exception as e:
        print("HRV extraction failed:", e)
        for k in ['HRV_RMSSD', 'HRV_SDNN', 'HRV_LFHF', 'HRV_HF', 'HRV_SampEn']:
            features[k] = np.nan
    eda_signal = window_data_dict.get('chest_EDA')
    try:
        eda_target_fs = min(16, fs)
        downsampled_eda = signal.resample(eda_signal, int(len(eda_signal) * (eda_target_fs / fs)))
        eda_df, info = nk.eda_process(downsampled_eda, sampling_rate=eda_target_fs)
        features['EDA_SCR_Peaks_N'] = len(info["SCR_Peaks"])
        tonic_signal = eda_df['EDA_Tonic'].values
        time_axis = np.arange(len(tonic_signal))
        slope, _, _, _, _ = linregress(time_axis, tonic_signal)
        features['EDA_Tonic_Slope'] = slope
    except Exception as e:
        print(f"EDA extraction failed: {e}")
        features['EDA_SCR_Peaks_N'] = np.nan
        features['EDA_Tonic_Slope'] = np.nan
    resp_signal = window_data_dict.get('chest_Resp')
    try:
        rsp_df, _ = nk.rsp_process(resp_signal, sampling_rate=fs)
        features['RESP_Rate_Mean'] = rsp_df['RSP_Rate'].mean()
        features['RESP_RRV_SDNN'] = rsp_df['RSP_Rate'].std()
    except Exception:
        features['RESP_Rate_Mean'] = np.nan
        features['RESP_RRV_SDNN'] = np.nan
    emg_signal = window_data_dict.get('chest_EMG')
    if emg_signal is not None:
        features['EMG_Amplitude_Mean'] = np.mean(np.abs(emg_signal))
    else:
        features['EMG_Amplitude_Mean'] = np.nan
    return features

# --- 3. 主预处理流程 ---
def run_preprocessing():
    subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
    all_channel_names = [f"chest_{c}_{ax}" for c in ['ACC'] for ax in 'xyz'] + \
                        [f"chest_{c}" for c in ['ECG', 'EDA', 'EMG', 'Resp', 'Temp']]
    if 'raw' in PROCESS_TARGETS:
        with open(RAW_PATH / '_channel_names.txt', 'w') as f:
            for name in all_channel_names: f.write(f"{name}\n")
    if 'raw-align' in PROCESS_TARGETS:
        with open(RAW_ALIGN_PATH / '_channel_names.txt', 'w') as f:
            for name in all_channel_names: f.write(f"{name}\n")
    feature_names_saved = False

    for sid in tqdm(subject_ids, desc="Preprocessing Subjects"):
        data = load_pkl(sid, WESAD_ROOT)
        if data is None:
            continue
        protocol_df = parse_quest_csv(sid, WESAD_ROOT)
        chest_data = data[b'signal'][b'chest']
        chest_data_str_keys = {k.decode('utf-8'): v for k, v in chest_data.items()}
        resampled_signals = {}
        if 'raw' in PROCESS_TARGETS or 'raw-align' in PROCESS_TARGETS:
            for channel in CHEST_CHANNELS:
                original_signal = chest_data_str_keys[channel]
                resampled_signals[f"raw_{channel}"] = resample_signal(original_signal, ORIGINAL_CHEST_FS, RAW_FS)
        if 'feature' in PROCESS_TARGETS:
            for channel in CHEST_CHANNELS:
                original_signal = chest_data_str_keys[channel]
                resampled_signals[f"feature_{channel}"] = resample_signal(original_signal, ORIGINAL_CHEST_FS, FEATURE_FS)

        feature_windows = []
        feature_labels = []
        raw_windows = []
        raw_labels = []

        for _, row in protocol_df.iterrows():
            task = row['task'].replace(" ", "").strip()
            label = TASK_TO_LABEL_MAP.get(task)  # 保存原始标签 (1/2/3/4)
            if label is None:
                continue

            start_idx_orig = int(row['start_min'] * 60 * ORIGINAL_CHEST_FS)
            end_idx_orig = int(row['end_min'] * 60 * ORIGINAL_CHEST_FS)

            if 'feature' in PROCESS_TARGETS:
                start_idx_feature = int(start_idx_orig * (FEATURE_FS / ORIGINAL_CHEST_FS))
                end_idx_feature = int(end_idx_orig * (FEATURE_FS / ORIGINAL_CHEST_FS))
                window_samples_feature = int(FEATURE_WINDOW_SEC * FEATURE_FS)
                stride_samples_feature = int(FEATURE_STRIDE_SEC * FEATURE_FS)
                for i in range(start_idx_feature, end_idx_feature - window_samples_feature + 1, stride_samples_feature):
                    window_start = i
                    window_end = i + window_samples_feature
                    window_feat_dict = {}
                    for channel in CHEST_CHANNELS:
                        signal_segment = resampled_signals[f"feature_{channel}"][window_start:window_end]
                        window_feat_dict[f"chest_{channel}"] = signal_segment.flatten()
                    feature_windows.append(window_feat_dict)
                    feature_labels.append(label)

            if 'raw' in PROCESS_TARGETS or 'raw-align' in PROCESS_TARGETS:
                start_idx_raw = int(start_idx_orig * (RAW_FS / ORIGINAL_CHEST_FS))
                end_idx_raw = int(end_idx_orig * (RAW_FS / ORIGINAL_CHEST_FS))
                window_samples_raw = int(RAW_WINDOW_SEC * RAW_FS)
                stride_samples_raw = int(RAW_STRIDE_SEC * RAW_FS)
                for i in range(start_idx_raw, end_idx_raw - window_samples_raw + 1, stride_samples_raw):
                    window_start = i
                    window_end = i + window_samples_raw
                    window_raw_list = []
                    for channel in CHEST_CHANNELS:
                        signal_segment = resampled_signals[f"raw_{channel}"][window_start:window_end]
                        if signal_segment.ndim == 1:
                            signal_segment = signal_segment.reshape(-1, 1)
                        window_raw_list.append(signal_segment)
                    concatenated_window = np.concatenate(window_raw_list, axis=1)
                    raw_windows.append(concatenated_window)
                    raw_labels.append(label)

        if 'feature' in PROCESS_TARGETS and feature_windows:
            all_windows_feat = [extract_handcrafted_features(w, FEATURE_FS) for w in feature_windows]
            df_feat = pd.DataFrame(all_windows_feat)
            if not feature_names_saved and not df_feat.empty:
                feature_names = df_feat.columns.tolist()
                with open(FEATURE_PATH / '_feature_names.txt', 'w') as f:
                    for name in feature_names:
                        f.write(f"{name}\n")
                feature_names_saved = True
            X_feat = df_feat.fillna(0).replace([np.inf, -np.inf], 0).values
            y_feat = np.array(feature_labels)  # 保存原始标签
            np.save(FEATURE_PATH / f'{sid}_X.npy', X_feat)
            np.save(FEATURE_PATH / f'{sid}_y.npy', y_feat)
            tqdm.write(f"  - {sid} (feature): Saved {len(y_feat)} windows. Feat shape: {X_feat.shape}")

        if 'raw' in PROCESS_TARGETS and raw_windows:
            X_raw = np.array(raw_windows)
            y_raw = np.array(raw_labels)  # 保存原始标签
            np.save(RAW_PATH / f'{sid}_X.npy', X_raw)
            np.save(RAW_PATH / f'{sid}_y.npy', y_raw)
            tqdm.write(f"  - {sid} (raw): Saved {len(y_raw)} windows. Raw shape: {X_raw.shape}")

        if 'raw-align' in PROCESS_TARGETS:
            if 'feature' not in PROCESS_TARGETS:
                print(f"警告: 'raw-align' 需要 'feature' 来对齐窗口，但 'feature' 未在 PROCESS_TARGETS 中。跳过 {sid} 的 raw-align。")
                continue
            num_feature_windows = len(feature_labels)
            num_raw_windows = len(raw_windows)
            if num_raw_windows == num_feature_windows:
                X_raw_align = np.array(raw_windows)
            elif num_raw_windows < num_feature_windows:
                extra = num_feature_windows - num_raw_windows
                X_raw_align = np.concatenate([np.array(raw_windows)] + [np.array([raw_windows[-1]])] * extra)
            else:
                X_raw_align = np.array(raw_windows[:num_feature_windows])
            y_raw_align = np.array(feature_labels)  # 使用feature标签，确保对齐
            np.save(RAW_ALIGN_PATH / f'{sid}_X.npy', X_raw_align)
            np.save(RAW_ALIGN_PATH / f'{sid}_y.npy', y_raw_align)
            tqdm.write(f"  - {sid} (raw-align): Saved {len(y_raw_align)} windows (aligned to feature). Raw-align shape: {X_raw_align.shape}")

    print("\nPreprocessing complete.")

if __name__ == '__main__':
    run_preprocessing()