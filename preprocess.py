# preprocess.py
import pickle
import numpy as np
import pandas as pd
from src.utils.path_manager import get_path_manager
from pathlib import Path
from scipy import signal
from sklearn.preprocessing import StandardScaler
import warnings
import neurokit2 as nk


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
        self.feature_fusion_path = self.output_root / 'wesad_feature_fusion'
        self.early_fusion_path = self.output_root / 'wesad_early_fusion'

        # 定义信号源和采样率
        self.CHEST_FS = 700
        self.WRIST_FS = {'ACC': 32, 'BVP': 64, 'EDA': 4}

        # 定义我们感兴趣的通道 (已舍弃 TEMP)
        self.CHEST_CHANNELS = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp']
        self.WRIST_CHANNELS = ['ACC', 'BVP', 'EDA']

        # 定义窗口化策略
        self.WINDOW_SEC = 30
        self.TARGET_FS = 64
        self.TASK_STEPS_SEC = {
            'Base': 15,  # 长时任务，稀疏采样
            'TSST': 5,  # 短时任务/核心任务，密集采样
            'Fun': 5,  # 修改为 5，密集采样
            'sRead': 5,
            'Medi 1': 15,  # 长时任务，稀疏采样
            'Medi 2': 15,  # 长时任务，稀疏采样
            'fRead': 5,
            'bRead': 5
        }

        # 定义标签映射 (2:stress, 1:amusement, 0:baseline/neutral)
        self.LABEL_MAP = {1: 0, 2: 2, 3: 1, 4: 0}

        # 创建输出目录
        (self.output_root / 'wesad_early_fusion').mkdir(parents=True, exist_ok=True)
        (self.output_root / 'wesad_feature_fusion').mkdir(parents=True, exist_ok=True)

        # 抑制 NeuroKit2 的警告
        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning, module='neurokit2')  # 抑制 NeuroKit2 的 UserWarning
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='neurokit2')  # 抑制 NeuroKit2 的 RuntimeWarning

    def _get_windows(self, data_len, protocol_df, original_fs):
        """根据协议和可变步长策略生成窗口索引。"""
        windows = []
        labels = []
        task_to_label_map = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}  # 字典键也去除空格

        for _, row in protocol_df.iterrows():
            # 对从DataFrame中读取的任务名也去除空格和多余字符
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
            step_sec = self.TASK_STEPS_SEC.get(row['task'], self.WINDOW_SEC / 2)  # 使用原始任务名(带空格)来查找步长 ***
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

    def _extract_features_nk(self, window_data_dict):
        """
        使用 NeuroKit2 为一个30秒窗口提取高级特征。
        接收一个包含不同模态原始信号段的字典。
        """
        all_features = []

        # --- 1. HRV/PRV 特征 (从 ECG 或 BVP) ---
        # 优先使用 ECG，如果不可用则使用 BVP
        ecg_signal = window_data_dict.get('chest_ECG')
        bvp_signal = window_data_dict.get('wrist_BVP')

        hrv_features = {}
        # np.full 创建一个用 NaN 填充的数组，用于处理错误
        nan_placeholder = np.full(10, np.nan)  # 假设提取10个HRV特征

        try:
            if ecg_signal is not None:
                hrv_df = nk.hrv(ecg_signal, sampling_rate=self.CHEST_FS, show=False)
            elif bvp_signal is not None:
                hrv_df = nk.hrv(bvp_signal, sampling_rate=self.WRIST_FS['BVP'], show=False)
            else:
                hrv_df = None

            if hrv_df is not None:
                # 选择关键特征
                hrv_features = hrv_df[['HRV_RMSSD', 'HRV_SDNN', 'HRV_pNN50',
                                       'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF',
                                       'HRV_SampEn', 'HRV_MeanNN', 'HRV_MedianNN']].iloc[0].values
            else:
                hrv_features = nan_placeholder
        except Exception as e:
            # NeuroKit2 在信号质量差时可能报错，我们需要捕获它
            # print(f"警告: HRV/PRV 分析失败 - {e}")
            hrv_features = nan_placeholder

        all_features.extend(hrv_features)

        # --- 2. EDA 特征 ---
        # 我们将融合胸部和腕部的EDA特征
        for key in ['chest_EDA', 'wrist_EDA']:
            eda_signal = window_data_dict.get(key)
            eda_features = np.full(4, np.nan)  # SCR_Peaks_N, SCR_Peaks_Amplitude_Mean, EDA_Phasic_Mean, EDA_Tonic_Mean
            if eda_signal is not None:
                try:
                    fs = self.CHEST_FS if 'chest' in key else self.WRIST_FS['EDA']
                    eda_df, _ = nk.eda_process(eda_signal, sampling_rate=fs)
                    peaks_df = nk.eda_peaks(eda_df["EDA_Phasic"], sampling_rate=fs, method="neurokit")
                    eda_features = [
                        peaks_df[1]['SCR_Peaks_N'],
                        np.nan_to_num(np.mean(peaks_df[0]['SCR_Amplitude'])),
                        np.nan_to_num(np.mean(eda_df['EDA_Phasic'])),
                        np.nan_to_num(np.mean(eda_df['EDA_Tonic']))
                    ]
                except Exception as e:
                    # print(f"警告: {key} 分析失败 - {e}")
                    pass  # 使用 NaN 占位符
            all_features.extend(eda_features)

        # --- 3. 呼吸 (RESP) 特征 ---
        resp_signal = window_data_dict.get('chest_Resp')
        resp_features = np.full(3, np.nan)  # RSP_Rate_Mean, RRV_RMSSD, RRV_SDNN
        if resp_signal is not None:
            try:
                rsp_df, _ = nk.rsp_process(resp_signal, sampling_rate=self.CHEST_FS)
                rsp_rate = nk.rsp_rate(rsp_df, sampling_rate=self.CHEST_FS)
                rrv_df = nk.hrv_time_domain(rsp_df, sampling_rate=self.CHEST_FS)
                resp_features = [np.nan_to_num(np.mean(rsp_rate['RSP_Rate'])), rrv_df['RRV_RMSSD'].iloc[0],
                                 rrv_df['RRV_SDNN'].iloc[0]]
            except Exception as e:
                # print(f"警告: RESP 分析失败 - {e}")
                pass
        all_features.extend(resp_features)

        # --- 4. ACC 特征 ---
        # 简单地使用信号能量作为身体活动指标
        for key in ['chest_ACC', 'wrist_ACC']:
            acc_signal = window_data_dict.get(key)
            acc_energy = np.nan
            if acc_signal is not None:
                acc_energy = np.mean(np.sqrt(np.sum(acc_signal ** 2, axis=1)))
            all_features.append(acc_energy)

        # --- 5. EMG 特征 ---
        emg_signal = window_data_dict.get('chest_EMG')
        emg_energy = np.nan
        if emg_signal is not None:
            emg_energy = np.mean(np.abs(emg_signal))
        all_features.append(emg_energy)

        return all_features

    def process_subject(self, subject_id):
        """处理单个受试者的数据。"""
        print(f"--- 正在处理受试者: {subject_id} ---")

        data = load_pkl(subject_id, self.wesad_root)
        if data is None: return None, None, None, None

        protocol_df = parse_quest_csv(subject_id, self.wesad_root)

        # 1. 为中/晚期融合 (feature_fusion) 提取特征
        chest_windows, labels = self._get_windows(len(data[b'label']), protocol_df, self.CHEST_FS)

        if not chest_windows:  # 增加一个检查，如果没有任何窗口生成
            return None, None, None, None

        feature_windows = []
        for start, end in chest_windows:
            window_data = {}
            # 收集这个窗口内所有模态的原始信号段
            for channel in self.CHEST_CHANNELS:
                window_data[f'chest_{channel}'] = data[b'signal'][b'chest'][channel.encode()][start:end]
            for channel in self.WRIST_CHANNELS:
                fs = self.WRIST_FS[channel]
                w_start = int(start * fs / self.CHEST_FS)
                w_end = int(end * fs / self.CHEST_FS)
                window_data[f'wrist_{channel}'] = data[b'signal'][b'wrist'][channel.encode()][w_start:w_end]

            # 调用新的特征提取函数
            window_features = self._extract_features_nk(window_data)
            feature_windows.append(window_features)

        # 清理可能产生的NaN/Inf值
        X_feat = pd.DataFrame(feature_windows).fillna(0).replace([np.inf, -np.inf], 0).values
        y_feat = labels

        if X_feat.shape[0] > 0:
            print(f"为 [feature_fusion] 生成了 {X_feat.shape[0]} 个窗口，每个窗口有 {X_feat.shape[1]} 个高级特征。")

        # 2. 为早期融合 (wesad_early_fusion) 准备重采样后的信号
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

        print(f"为 [wesad_early_fusion] 生成了 {X_early.shape[0]} 个窗口，形状为 {X_early.shape[1:]}。")

        return X_feat, y_feat, X_early, y_early

    def run(self):
        subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
        print("--- 开始生成所有受试者的窗口化 .npy 数据文件 ---")

        for sid in subject_ids:
            # 提取特征和信号
            X_f, y_f, X_e, y_e = self.process_subject(sid)

            if X_f is not None and y_f is not None:
                np.save(self.output_root / 'wesad_feature_fusion' / f'{sid}_X.npy', X_f)
                np.save(self.output_root / 'wesad_feature_fusion' / f'{sid}_y.npy', y_f)

            if X_e is not None and y_e is not None:
                np.save(self.output_root / 'wesad_early_fusion' / f'{sid}_X.npy', X_e)
                np.save(self.output_root / 'wesad_early_fusion' / f'{sid}_y.npy', y_e)

        print("\n预处理完成！所有数据已保存为未归一化的 .npy 文件。")

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