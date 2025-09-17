# file: preprocess.py
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
from tqdm import tqdm

# --- 1. CONFIGURATION (All settings are here) ---
PROJECT_ROOT = Path(__file__).resolve().parent
WESAD_ROOT = PROJECT_ROOT / "WESAD"
OUTPUT_ROOT = PROJECT_ROOT / "data"

# Signal Processing Settings
WINDOW_SEC = 30  # How many seconds in each window
TARGET_FS = 64  # The uniform sampling rate to resample all signals to

# We can keep the smart stepping for efficiency
TASK_STEPS_SEC = {
    'Base': 15,
    'TSST': 5,
    'Fun': 5,
    'Medi 1': 15,
    'Medi 2': 15,
    # Add other tasks if needed, with a default fallback
}
DEFAULT_STEP_SEC = WINDOW_SEC / 2  # 50% overlap for any task not in the dict

# Label Mapping (0: neutral, 1: amusement, 2: stress)
LABEL_MAP = {1: 0, 3: 1, 2: 2, 4: 0}  # Baseline(1) and Meditation(4) -> 0 (neutral)


# --- 2. HELPER FUNCTIONS ---

def parse_quest_csv(subject_id: str, wesad_root: Path) -> pd.DataFrame:
    # (Your existing, excellent quest parsing function)
    quest_path = wesad_root / subject_id / f"{subject_id}_quest.csv"
    # ... (the rest of your function is perfect)
    df_raw = pd.read_csv(quest_path, sep=';', header=None, skip_blank_lines=True)

    order_row = df_raw[df_raw[0].str.contains('# ORDER', na=False)].values[0]
    start_row = df_raw[df_raw[0].str.contains('# START', na=False)].values[0]
    end_row = df_raw[df_raw[0].str.contains('# END', na=False)].values[0]

    tasks = pd.Series(order_row[1:]).dropna().tolist()
    start_times = pd.Series(start_row[1:]).dropna().astype(float).tolist()
    end_times = pd.Series(end_row[1:]).dropna().astype(float).tolist()

    return pd.DataFrame({'task': tasks, 'start_min': start_times, 'end_min': end_times})


def load_pkl(subject_id: str, wesad_root: Path):
    pkl_path = wesad_root / subject_id / f"{subject_id}.pkl"
    if not pkl_path.exists():
        print(f"Warning: PKL file not found for {subject_id}")
        return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


# --- 3. MAIN PREPROCESSING LOGIC ---

class WesadPreprocessor:
    def __init__(self):
        self.wesad_root = WESAD_ROOT
        self.output_path = OUTPUT_ROOT
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.CHEST_FS = 700
        self.WRIST_FS = {'ACC': 32, 'BVP': 64, 'EDA': 4, 'TEMP': 4}

        # Define all channels we want to use
        self.CHEST_CHANNELS = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp', 'Temp']
        self.WRIST_CHANNELS = ['ACC', 'BVP', 'EDA', 'TEMP']

        warnings.filterwarnings('ignore', category=FutureWarning)

    def process_all_subjects(self):
        subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]

        print("--- Step 1: Processing each subject and saving individual data files ---")
        for sid in tqdm(subject_ids, desc="Processing Subjects"):
            self._process_and_save_subject(sid)

        print("\n--- Step 2: Generating normalization parameters for each LOSO fold ---")
        for test_subject in tqdm(subject_ids, desc="Generating Scalers"):
            train_subjects = [s for s in subject_ids if s != test_subject]

            # Concatenate all training data for this fold
            X_train_fold_list = [np.load(self.output_path / f'{sid}_X.npy') for sid in train_subjects]
            X_train_fold = np.concatenate(X_train_fold_list, axis=0)

            # Reshape for channel-wise normalization: (N, L, C) -> (N*L, C)
            data_reshaped = X_train_fold.reshape(-1, X_train_fold.shape[2])

            # Calculate and save mean and std vectors
            mean_vec = np.mean(data_reshaped, axis=0)
            std_vec = np.std(data_reshaped, axis=0)

            norm_params_path = self.output_path / f'norm_params_for_{test_subject}.npz'
            np.savez(norm_params_path, mean=mean_vec, std=std_vec)

        print(f"\nPreprocessing complete! All files saved in: {self.output_path}")

    def _process_and_save_subject(self, subject_id):
        data = load_pkl(subject_id, self.wesad_root)
        if data is None: return

        protocol_df = parse_quest_csv(subject_id, self.wesad_root)

        # 1. Resample all signals to the target frequency
        resampled_signals = {}
        all_signals_data = data[b'signal']

        # Chest signals
        for channel in self.CHEST_CHANNELS:
            original = all_signals_data[b'chest'][channel.encode()]
            new_len = int((len(original) / self.CHEST_FS) * TARGET_FS)
            if original.ndim > 1:
                resampled_axes = [signal.resample(original[:, i], new_len) for i in range(original.shape[1])]
                resampled_signals[f"chest_{channel}"] = np.column_stack(resampled_axes)
            else:
                resampled_signals[f"chest_{channel}"] = signal.resample(original, new_len)

        # Wrist signals
        for channel in self.WRIST_CHANNELS:
            original = all_signals_data[b'wrist'][channel.encode()]
            fs = self.WRIST_FS[channel]
            new_len = int((len(original) / fs) * TARGET_FS)
            if original.ndim > 1:
                resampled_axes = [signal.resample(original[:, i], new_len) for i in range(original.shape[1])]
                resampled_signals[f"wrist_{channel}"] = np.column_stack(resampled_axes)
            else:
                resampled_signals[f"wrist_{channel}"] = signal.resample(original, new_len)

        # 2. Concatenate all resampled signals into one big array
        concatenated_signals_list = []
        self.channel_names = []  # Store the final order of channels

        # Define the final order of signals
        signal_order = [f"chest_{c}" for c in self.CHEST_CHANNELS] + [f"wrist_{c}" for c in self.WRIST_CHANNELS]

        for key in signal_order:
            s = resampled_signals[key]
            if s.ndim == 1: s = s.reshape(-1, 1)
            concatenated_signals_list.append(s)

            if s.shape[1] > 1:
                for i in range(s.shape[1]): self.channel_names.append(f"{key}_{'xyz'[i]}")
            else:
                self.channel_names.append(key)

        concatenated_signals = np.concatenate(concatenated_signals_list, axis=1)

        # 3. Generate windows based on the protocol
        windows, labels = self._get_windows_from_protocol(len(concatenated_signals), protocol_df)

        window_len_samples = int(WINDOW_SEC * TARGET_FS)
        X_subject = np.zeros((len(windows), window_len_samples, concatenated_signals.shape[1]))

        for i, (start, end) in enumerate(windows):
            X_subject[i] = concatenated_signals[start:end, :]

        np.save(self.output_path / f'{subject_id}_X.npy', X_subject)
        np.save(self.output_path / f'{subject_id}_y.npy', labels)
        # Also save channel names for reference
        if "S2" in subject_id:  # Only save once
            np.save(self.output_path / 'channel_names.npy', np.array(self.channel_names))

    def _get_windows_from_protocol(self, data_len, protocol_df):
        windows, labels = [], []
        task_to_label_map = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}

        for _, row in protocol_df.iterrows():
            task = row['task'].replace(" ", "").strip()
            original_label = task_to_label_map.get(task)
            if original_label is None: continue

            label = LABEL_MAP.get(original_label)
            if label is None: continue

            start_idx = int(row['start_min'] * 60 * TARGET_FS)
            end_idx = int(row['end_min'] * 60 * TARGET_FS)

            step_sec = TASK_STEPS_SEC.get(row['task'], DEFAULT_STEP_SEC)
            step_samples = int(step_sec * TARGET_FS)
            window_samples = int(WINDOW_SEC * TARGET_FS)

            if end_idx > data_len: end_idx = data_len

            for i in range(start_idx, end_idx - window_samples + 1, step_samples):
                windows.append((i, i + window_samples))
                labels.append(label)

        return windows, np.array(labels)


if __name__ == '__main__':
    preprocessor = WesadPreprocessor()
    preprocessor.process_all_subjects()