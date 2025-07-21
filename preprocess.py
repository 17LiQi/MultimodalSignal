import pickle
import numpy as np
import pandas as pd
from src.utils.path_manager import get_path_manager
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings


# --- 1. 辅助函数 ---

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


# --- 2. 核心预处理类 ---

class WesadPreprocessor:
    def __init__(self, wesad_root, output_root):
        self.wesad_root = Path(wesad_root)
        self.output_root = Path(output_root)

        # 定义信号源和采样率
        self.CHEST_FS = 700
        self.WRIST_FS = {'ACC': 32, 'BVP': 64, 'EDA': 4}

        # 定义我们感兴趣的通道 (已舍弃 TEMP)
        self.CHEST_CHANNELS = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp']
        self.WRIST_CHANNELS = ['ACC', 'BVP', 'EDA']

        # 定义窗口化策略
        self.WINDOW_SEC = 30
        self.TARGET_FS = 64
        self.TASK_STEPS_SEC = {'Base': 15, 'TSST': 5, 'Fun': 15, 'sRead': 5, 'Medi 1': 15, 'Medi 2': 15, 'fRead': 5,
                               'bRead': 5}

        # 定义标签映射 (2:stress, 1:amusement, 0:baseline/neutral)
        self.LABEL_MAP = {1: 0, 2: 2, 3: 1, 4: 0}

        # 创建输出目录
        self.early_fusion_path = self.output_root / 'early_fusion'
        self.feature_fusion_path = self.output_root / 'feature_fusion'
        self.early_fusion_path.mkdir(parents=True, exist_ok=True)
        self.feature_fusion_path.mkdir(parents=True, exist_ok=True)

        warnings.filterwarnings('ignore', category=FutureWarning)

    # 在 WesadPreprocessor 类中，找到 _get_windows 方法并替换它

    def _get_windows(self, data_len, protocol_df, original_fs):
        """根据协议和可变步长策略生成窗口索引。"""
        windows = []
        labels = []
        task_to_label_map = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}  # *** 修改点1: 字典键也去除空格 ***

        for _, row in protocol_df.iterrows():
            # *** 修改点2: 对从DataFrame中读取的任务名也去除空格和多余字符 ***
            task = row['task'].replace(" ", "").strip()

            # 我们只处理基线、压力、娱乐和冥想这几个核心状态
            original_label = task_to_label_map.get(task)
            if original_label is None:
                continue  # 跳过 sRead, fRead 等

            # 使用我们的映射来获取最终标签
            label = self.LABEL_MAP.get(original_label)
            if label is None:
                continue

            start_idx = int(row['start_min'] * 60 * original_fs)
            end_idx = int(row['end_min'] * 60 * original_fs)

            # 查找任务对应的步长，如果任务名不在字典中，提供一个默认值
            step_sec = self.TASK_STEPS_SEC.get(row['task'], self.WINDOW_SEC / 2)  # *** 修改点3: 使用原始任务名(带空格)来查找步长 ***
            step_samples = int(step_sec * original_fs)
            window_samples = int(self.WINDOW_SEC * original_fs)

            # 增加一个安全检查，防止索引越界
            if end_idx > data_len:
                print(f"警告: 任务 '{row['task']}' 的结束索引 ({end_idx}) 超出数据长度 ({data_len})。将截断至数据末尾。")
                end_idx = data_len

            for i in range(start_idx, end_idx - window_samples + 1, step_samples):
                windows.append((i, i + window_samples))
                labels.append(label)

        return windows, np.array(labels)
    def _extract_features(self, window_data):
        """为一个窗口数据提取统计特征。"""
        features = []
        # 处理多轴信号 (如ACC)
        if window_data.ndim > 1:
            for i in range(window_data.shape[1]):
                axis_data = window_data[:, i]
                features.extend([np.mean(axis_data), np.std(axis_data), np.min(axis_data), np.max(axis_data)])
        else:
            features.extend([np.mean(window_data), np.std(window_data), np.min(window_data), np.max(window_data)])
        return features

    def process_subject(self, subject_id):
        """处理单个受试者的数据。"""
        print(f"--- 正在处理受试者: {subject_id} ---")

        data = load_pkl(subject_id, self.wesad_root)
        if data is None: return None, None, None, None

        protocol_df = parse_quest_csv(subject_id, self.wesad_root)

        # 1. 为中/晚期融合 (feature_fusion) 提取特征
        chest_windows, labels = self._get_windows(len(data[b'label']), protocol_df, self.CHEST_FS)

        feature_windows = []
        for start, end in chest_windows:
            window_features = []
            # 胸部特征
            for channel in self.CHEST_CHANNELS:
                signal_data = data[b'signal'][b'chest'][channel.encode()][start:end]
                window_features.extend(self._extract_features(signal_data))

            # 腕部特征 (需要根据各自采样率计算窗口索引)
            for channel in self.WRIST_CHANNELS:
                fs = self.WRIST_FS[channel]
                w_start = int(start * fs / self.CHEST_FS)
                w_end = int(end * fs / self.CHEST_FS)
                signal_data = data[b'signal'][b'wrist'][channel.encode()][w_start:w_end]
                window_features.extend(self._extract_features(signal_data))

            feature_windows.append(window_features)

        X_feat = np.array(feature_windows)
        y_feat = labels
        print(f"为 [feature_fusion] 生成了 {X_feat.shape[0]} 个窗口，每个窗口有 {X_feat.shape[1]} 个特征。")

        # 2. 为早期融合 (early_fusion) 准备重采样后的信号
        resampled_signals = {}
        all_channels = self.CHEST_CHANNELS + self.WRIST_CHANNELS

        # 先重采样所有需要的信号
        for dev, channels, fs_dict in [('chest', self.CHEST_CHANNELS, {c: self.CHEST_FS for c in self.CHEST_CHANNELS}),
                                       ('wrist', self.WRIST_CHANNELS, self.WRIST_FS)]:
            for channel in channels:
                original_signal = data[b'signal'][dev.encode()][channel.encode()]
                fs = fs_dict[channel]
                duration = len(original_signal) / fs
                new_len = int(duration * self.TARGET_FS)

                if original_signal.ndim > 1:
                    resampled_axes = [signal.resample(original_signal[:, i], new_len) for i in
                                      range(original_signal.shape[1])]
                    resampled_signals[f"{dev}_{channel}"] = np.column_stack(resampled_axes)
                else:
                    resampled_signals[f"{dev}_{channel}"] = signal.resample(original_signal, new_len)

        # 使用重采样后的时间轴生成窗口
        target_windows, y_early = self._get_windows(len(resampled_signals['chest_ECG']), protocol_df, self.TARGET_FS)

        window_len_samples = int(self.WINDOW_SEC * self.TARGET_FS)
        num_channels = sum(
            resampled_signals[f"{dev}_{c}"].shape[1] if resampled_signals[f"{dev}_{c}"].ndim > 1 else 1 for
            dev, chans, _ in [('chest', self.CHEST_CHANNELS, {}), ('wrist', self.WRIST_CHANNELS, {})] for c in chans)

        X_early = np.zeros((len(target_windows), window_len_samples, num_channels))

        # 拼接信号
        all_resampled_data = []
        channel_names = []
        for dev, channels in [('chest', self.CHEST_CHANNELS), ('wrist', self.WRIST_CHANNELS)]:
            for channel in channels:
                key = f"{dev}_{channel}"
                s = resampled_signals[key]
                if s.ndim == 1: s = s.reshape(-1, 1)
                all_resampled_data.append(s)

                if s.shape[1] > 1:  # for ACC
                    for i in range(s.shape[1]): channel_names.append(f"{key}_{'xyz'[i]}")
                else:
                    channel_names.append(key)

        concatenated_signals = np.concatenate(all_resampled_data, axis=1)

        for i, (start, end) in enumerate(target_windows):
            X_early[i, :, :] = concatenated_signals[start:end, :]

        print(f"为 [early_fusion] 生成了 {X_early.shape[0]} 个窗口，形状为 {X_early.shape[1:]}。")

        return X_feat, y_feat, X_early, y_early

    def run(self):
        """处理所有受试者并为每次LOSOCV折叠保存独立的Scaler。"""
        subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]

        # -----------------------------------------------------------------
        # 第一步: 像之前一样，先处理并保存每个受试者的窗口化数据
        # -----------------------------------------------------------------
        for sid in subject_ids:
            X_f, y_f, X_e, y_e = self.process_subject(sid)
            if X_f is not None:
                # 保存每个受试者的预处理数据
                np.save(self.feature_fusion_path / f'{sid}_X.npy', X_f)
                np.save(self.feature_fusion_path / f'{sid}_y.npy', y_f)
                np.save(self.early_fusion_path / f'{sid}_X.npy', X_e)
                np.save(self.early_fusion_path / f'{sid}_y.npy', y_e)

        print("\n所有受试者的独立数据文件已保存。")
        print("--- 现在开始为每一次留一法交叉验证生成并保存Scaler ---")

        # -----------------------------------------------------------------
        # 第二步: 循环N次，每次留出一个受试者，用其余的来fit并保存scaler
        # -----------------------------------------------------------------
        for test_subject in subject_ids:
            train_subjects = [s for s in subject_ids if s != test_subject]

            # --- 为 Feature Fusion 数据生成 Scaler ---
            try:
                # 加载当前折叠所需的所有训练数据
                X_train_list = [np.load(self.feature_fusion_path / f'{sid}_X.npy') for sid in train_subjects]
                X_train_fold = np.concatenate(X_train_list, axis=0)

                # 创建、拟合 Scaler
                scaler_feat = StandardScaler()
                scaler_feat.fit(X_train_fold)

                # 以测试受试者的名字来命名并保存
                scaler_path = self.feature_fusion_path / f'scaler_for_{test_subject}.pkl'
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler_feat, f)
                print(f"已生成并保存: {scaler_path.name} (基于 {len(train_subjects)} 个训练受试者)")

            except Exception as e:
                print(f"为 {test_subject} 生成 feature fusion scaler 时出错: {e}")

            # 对于早期融合数据，归一化通常在DataLoader中按通道进行，
            # 所以我们计算并保存均值和标准差向量，而不是一个Scaler对象。
            try:
                X_train_list_early = [np.load(self.early_fusion_path / f'{sid}_X.npy') for sid in train_subjects]
                X_train_fold_early = np.concatenate(X_train_list_early, axis=0)

                # (N, L, C) -> (N*L, C)
                data_reshaped = X_train_fold_early.reshape(-1, X_train_fold_early.shape[2])

                mean_vec = np.mean(data_reshaped, axis=0)
                std_vec = np.std(data_reshaped, axis=0)

                # 保存均值和标准差向量
                norm_params_path = self.early_fusion_path / f'norm_params_for_{test_subject}.npz'
                np.savez(norm_params_path, mean=mean_vec, std=std_vec)
                print(f"已生成并保存: {norm_params_path.name} (基于 {len(train_subjects)} 个训练受试者)")

            except Exception as e:
                print(f"为 {test_subject} 生成 early fusion norm params 时出错: {e}")

        print("\n预处理完成！所有数据和交叉验证所需的Scaler均已生成。")


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    paths = get_path_manager()
    # WESAD数据集根目录
    WESAD_ROOT_PATH = paths.WESAD_ROOT
    # 定保存预处理后数据的目录
    DATA_PATH = paths.DATA_ROOT
    print("WESAD 数据集根目录:", WESAD_ROOT_PATH)
    print("预处理数据目录:", DATA_PATH)


    preprocessor = WesadPreprocessor(wesad_root=WESAD_ROOT_PATH, output_root=DATA_PATH)
    preprocessor.run()