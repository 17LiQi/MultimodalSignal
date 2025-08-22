import pickle
import warnings
from collections import Counter
from pathlib import Path

import neurokit2 as nk
import numpy as np
import pandas as pd
from scipy.signal import resample, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from utils.path_manager import get_path_manager

try:
    from utils.cvxEDA import cvxEDA
    print("Using cvxEDA library for EDA processing.")
    CVXEDA_AVAILABLE = True
except ImportError:
    print("Warning: cvxEDA library not found. EDA processing will be limited.")
    CVXEDA_AVAILABLE = False

SUBJECT_ATTRIBUTES = {
    'S2': {'age': 27, 'gender': 0, 'height': 175, 'weight': 80, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S3': {'age': 27, 'gender': 0, 'height': 173, 'weight': 69, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S4': {'age': 25, 'gender': 0, 'height': 175, 'weight': 90, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S5': {'age': 35, 'gender': 0, 'height': 189, 'weight': 80, 'smoker': 0, 'sport': 0, 'coffee': 1, 'ill': 0},
    'S6': {'age': 27, 'gender': 0, 'height': 170, 'weight': 66, 'smoker': 1, 'sport': 0, 'coffee': 1, 'ill': 0},
    'S7': {'age': 28, 'gender': 0, 'height': 184, 'weight': 74, 'smoker': 0, 'sport': 1, 'coffee': 0, 'ill': 0},
    'S8': {'age': 27, 'gender': 1, 'height': 172, 'weight': 64, 'smoker': 0, 'sport': 1, 'coffee': 1, 'ill': 0},
    'S9': {'age': 26, 'gender': 0, 'height': 181, 'weight': 75, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 1},
    'S10': {'age': 28, 'gender': 0, 'height': 178, 'weight': 76, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S11': {'age': 26, 'gender': 1, 'height': 171, 'weight': 54, 'smoker': 0, 'sport': 0, 'coffee': 1, 'ill': 0},
    'S13': {'age': 28, 'gender': 0, 'height': 181, 'weight': 82, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S14': {'age': 27, 'gender': 0, 'height': 180, 'weight': 80, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S15': {'age': 28, 'gender': 0, 'height': 186, 'weight': 83, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S16': {'age': 24, 'gender': 0, 'height': 184, 'weight': 69, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
    'S17': {'age': 29, 'gender': 1, 'height': 165, 'weight': 55, 'smoker': 0, 'sport': 0, 'coffee': 0, 'ill': 0},
}

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """Applies a Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def get_basic_stats(signal, prefix):
    """Calculates mean, std, min, max for a signal."""
    return {
        f'{prefix}_mean': np.mean(signal),
        f'{prefix}_std': np.std(signal),
        f'{prefix}_min': np.min(signal),
        f'{prefix}_max': np.max(signal),
    }

def extract_features_from_window(window_data, fs_dict):
    """
    Main feature extraction function, as per PhysioFormer Paper, Page 11.
    """
    features = {}

    # --- ACC Features ---
    acc_x = window_data['ACC_x']
    acc_y = window_data['ACC_y']
    acc_z = window_data['ACC_z']
    features.update(get_basic_stats(acc_x, 'ACC_x'))
    features.update(get_basic_stats(acc_y, 'ACC_y'))
    features.update(get_basic_stats(acc_z, 'ACC_z'))
    net_acc = np.sqrt(acc_x ** 2 + acc_y ** 2 + acc_z ** 2)
    features.update(get_basic_stats(net_acc, 'Net_ACC'))

    # --- BVP Features (Wrist) ---
    if 'BVP' in window_data:
        try:
            bvp_win = window_data['BVP']

            # CRITICAL FIX 1: Use ppg_findpeaks for BVP signals
            ppg_peaks_info = nk.ppg_findpeaks(bvp_win, sampling_rate=fs_dict['BVP'])
            peaks = ppg_peaks_info['PPG_Peaks']

            # Check if enough peaks were found to calculate HRV
            if len(peaks) < 2:
                raise ValueError("Not enough PPG peaks found to calculate HRV.")

            # CRITICAL FIX 2: Calculate HRV from the correct PPG peaks
            hrv_features = nk.hrv_time(peaks, sampling_rate=fs_dict['BVP'], show=False)

            features['BVP_HRV_RMSSD'] = hrv_features['HRV_RMSSD'].iloc[0]
            features['BVP_HRV_SDNN'] = hrv_features['HRV_SDNN'].iloc[0]

            # Peak frequency (simplified, no change)
            fft_vals = np.fft.rfft(bvp_win)
            fft_freq = np.fft.rfftfreq(len(bvp_win), 1.0 / fs_dict['BVP'])
            features['BVP_peak_freq'] = fft_freq[np.argmax(np.abs(fft_vals))]

        except Exception as e:
            # print(f"  - BVP processing failed on a window: {e}") # Uncomment for debugging
            features['BVP_HRV_RMSSD'] = np.nan
            features['BVP_HRV_SDNN'] = np.nan
            features['BVP_peak_freq'] = np.nan

    # --- EDA Features ---
    if 'EDA' in window_data:
        eda_win = window_data['EDA']
        eda_filtered = butter_lowpass_filter(eda_win, 1.0, fs_dict['EDA'], order=4)

        # 使用我们新定义的布尔标志来判断
        if CVXEDA_AVAILABLE:
            try:
                # 直接调用导入的 cvxEDA (驼峰) 函数
                _, phasic, _, _, _, _, _ = cvxEDA(eda_filtered, 1.0 / fs_dict['EDA'])
                tonic = eda_filtered - phasic
                features.update(get_basic_stats(phasic, 'EDA_phasic'))
                features.update(get_basic_stats(tonic, 'EDA_tonic'))
            except Exception as e:
                print(f"  - cvxEDA failed on a window. Error: {e}. Using fallback.")
                features.update(get_basic_stats(np.zeros_like(eda_win), 'EDA_phasic'))
                features.update(get_basic_stats(np.zeros_like(eda_win), 'EDA_tonic'))
        else:
            # Fallback if cvxEDA is not installed
            features.update(get_basic_stats(eda_filtered, 'EDA_filtered'))

    # --- TEMP Features ---
    if 'TEMP' in window_data:
        temp_win = window_data['TEMP']
        features.update(get_basic_stats(temp_win, 'TEMP'))
        # Slope of temperature
        time_axis = np.arange(len(temp_win))
        slope, _ = np.polyfit(time_axis, temp_win, 1)
        features['TEMP_slope'] = slope

    # --- ECG Features (Chest) ---
    if 'ECG' in window_data:
        try:
            ecg_win = window_data['ECG']
            ecg_peaks_info = nk.ecg_findpeaks(ecg_win, sampling_rate=fs_dict['ECG'])
            peaks = ecg_peaks_info['ECG_R_Peaks']

            if len(peaks) < 2:
                raise ValueError("Not enough R-peaks found to calculate HRV.")

            hrv_features = nk.hrv_time(peaks, sampling_rate=fs_dict['ECG'], show=False)
            features['ECG_HRV_RMSSD'] = hrv_features['HRV_RMSSD'].iloc[0]
            features['ECG_HRV_SDNN'] = hrv_features['HRV_SDNN'].iloc[0]
        except Exception:
            features['ECG_HRV_RMSSD'] = np.nan
            features['ECG_HRV_SDNN'] = np.nan

    # --- EMG & RESP Features (Chest) ---
    for signal_name in ['EMG', 'RESP']:
        if signal_name in window_data:
            signal_win = window_data[signal_name]
            signal_filtered = butter_lowpass_filter(signal_win, 2.0, fs_dict[signal_name], order=4)
            features.update(get_basic_stats(signal_filtered, signal_name))

    return features

def preprocess_physioformer(wesad_root: Path, output_path: Path):
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    TARGET_FS = 64
    all_subject_data = []
    subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
    original_fs = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4, 'ECG': 700, 'EMG': 700, 'RESP': 700, 'label': 700}

    for sid in tqdm(subject_ids, desc="Processing subjects"):
        with open(wesad_root / sid / f"{sid}.pkl", 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        wrist_signals = data[b'signal'][b'wrist']
        chest_signals = data[b'signal'][b'chest']
        labels_raw = data[b'label'].flatten()

        # --- STEP 1: RESAMPLE ONLY PHYSIOLOGICAL SIGNALS ---
        resampled_signals = {}

        # Resample wrist data
        acc_3d = wrist_signals[b'ACC']
        for i, axis in enumerate(['ACC_x', 'ACC_y', 'ACC_z']):
            num_samples = int(len(acc_3d[:, i]) * TARGET_FS / original_fs['ACC'])
            resampled_signals[axis] = resample(acc_3d[:, i], num_samples)

        for signal in ['BVP', 'EDA', 'TEMP']:
            original_signal = wrist_signals[signal.encode()].flatten()
            num_samples = int(len(original_signal) * TARGET_FS / original_fs[signal])
            resampled_signals[signal] = resample(original_signal, num_samples)

        # Resample chest data
        for signal in ['ECG', 'EMG', 'RESP']:
            key = b'Resp' if signal == 'RESP' else signal.encode()
            original_signal = chest_signals[key].flatten()
            num_samples = int(len(original_signal) * TARGET_FS / original_fs[signal])
            resampled_signals[signal] = resample(original_signal, num_samples)

        # --- STEP 2: ALIGN ALL RESAMPLED SIGNALS ---
        min_len_resampled = min(len(s) for s in resampled_signals.values())
        for signal_name in resampled_signals:
            resampled_signals[signal_name] = resampled_signals[signal_name][:min_len_resampled]

        # --- STEP 3: NON-OVERLAPPING WINDOWING ---
        win_sec = 30
        win_len = win_sec * TARGET_FS  # 1920 samples
        step_len = win_len  # Non-overlapping

        fs_dict_for_extraction = {key: TARGET_FS for key in original_fs}

        # Iterate based on the aligned, resampled signals
        for i in range(0, min_len_resampled - win_len, step_len):

            # --- CRITICAL FIX: Find corresponding label window in ORIGINAL labels ---
            # Calculate the start and end time in seconds
            start_time_sec = i / TARGET_FS
            end_time_sec = (i + win_len) / TARGET_FS

            # Convert time back to original 700Hz label indices
            label_start_idx = int(start_time_sec * original_fs['label'])
            label_end_idx = int(end_time_sec * original_fs['label'])

            label_window = labels_raw[label_start_idx:label_end_idx]

            if len(label_window) == 0:
                continue

            label = Counter(label_window).most_common(1)[0][0]

            if label not in [1, 2, 3]:
                continue

            # Extract feature window from resampled signals
            window_data = {}
            for signal_name in resampled_signals:
                window_data[signal_name] = resampled_signals[signal_name][i: i + win_len]

            features = extract_features_from_window(window_data, fs_dict_for_extraction)
            features['subject'] = sid
            features['label'] = label
            all_subject_data.append(features)

    # --- STEP 4: Combine, Normalize, and Save ---
    # (The rest of the script remains unchanged)
    df = pd.DataFrame(all_subject_data).dropna()
    # Replace NaN with mean for robustness
    # df.fillna(df.mean(), inplace=True) # Optional, but good practice

    static_df = pd.DataFrame.from_dict(SUBJECT_ATTRIBUTES, orient='index').reset_index().rename(
        columns={'index': 'subject'})
    final_df = pd.merge(df, static_df, on='subject')

    # --- CRITICAL CHANGE: SAVE SUBJECT IDs ---
    subject_ids_per_sample = final_df['subject'].values
    # Convert subject IDs like 'S2', 'S3' to integers 0, 1, ... for easier indexing
    subject_map = {sid: i for i, sid in enumerate(pd.unique(subject_ids_per_sample))}
    subject_indices = np.array([subject_map[sid] for sid in subject_ids_per_sample])

    # Create Wrist and Chest specific datasets
    wrist_features = [col for col in final_df.columns if any(p in col for p in ['ACC', 'BVP', 'EDA', 'TEMP', 'Net_ACC'])]
    static_features = list(SUBJECT_ATTRIBUTES['S2'].keys())
    wrist_cols = static_features + wrist_features
    wrist_df = final_df[wrist_cols + ['label']]
    y_wrist = wrist_df['label'].values
    X_wrist_df = wrist_df.drop('label', axis=1)

    # --- SAVE SUBJECT INDICES FOR WRIST ---
    # We need to align the subject indices with the final data after dropping NaN
    final_wrist_indices = wrist_df.index
    wrist_subject_indices = subject_indices[final_wrist_indices]

    np.save(output_path / 'subjects_wrist.npy', wrist_subject_indices)  # <-- NEW
    print(f"Wrist subjects saved with shape {wrist_subject_indices.shape}")

    chest_features = [col for col in final_df.columns if any(p in col for p in ['ACC', 'ECG', 'EMG', 'RESP', 'Net_ACC'])]
    chest_cols = static_features + chest_features
    chest_df = final_df[chest_cols + ['label']]
    y_chest = chest_df['label'].values
    X_chest_df = chest_df.drop('label', axis=1)

    final_chest_indices = chest_df.index
    chest_subject_indices = subject_indices[final_chest_indices]

    np.save(output_path / 'subjects_chest.npy', chest_subject_indices)
    print(f"Chest subjects saved with shape {chest_subject_indices.shape}")

    # Final Global Normalization and Saving
    output_path.mkdir(parents=True, exist_ok=True)

    scaler_wrist = StandardScaler()
    X_wrist = scaler_wrist.fit_transform(X_wrist_df)
    np.save(output_path / 'X_wrist.npy', X_wrist)
    np.save(output_path / 'y_wrist.npy', y_wrist)
    with open(output_path / 'scaler_wrist.pkl', 'wb') as f:
        pickle.dump(scaler_wrist, f)
    pd.DataFrame(X_wrist_df.columns, columns=['feature']).to_json(output_path / 'features_wrist.json')
    print(f"Wrist dataset created with shape {X_wrist.shape}")

    scaler_chest = StandardScaler()
    X_chest = scaler_chest.fit_transform(X_chest_df)
    np.save(output_path / 'X_chest.npy', X_chest)
    np.save(output_path / 'y_chest.npy', y_chest)
    with open(output_path / 'scaler_chest.pkl', 'wb') as f:
        pickle.dump(scaler_chest, f)
    pd.DataFrame(X_chest_df.columns, columns=['feature']).to_json(output_path / 'features_chest.json')
    print(f"Chest dataset created with shape {X_chest.shape}")

    print("Preprocessing complete.")

if __name__ == '__main__':
    paths = get_path_manager()
    WESAD_ROOT_PATH = paths.WESAD_ROOT
    OUTPUT_PATH = paths.DATA_ROOT

    if not WESAD_ROOT_PATH.exists():
        print(f"Error: WESAD path does not exist: {WESAD_ROOT_PATH}")
        print("Please download the WESAD dataset and update the WESAD_ROOT_PATH variable in this script.")
    else:
        preprocess_physioformer(WESAD_ROOT_PATH, OUTPUT_PATH)