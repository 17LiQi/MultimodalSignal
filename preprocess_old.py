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

        # --- 分段特征创建输出目录 ---
        self.segmented_feature_path = self.output_root / 'feature_segmented'
        self.segmented_feature_path.mkdir(parents=True, exist_ok=True)

        # --- HRV/PRV 特征创建输出目录 ---
        self.hrv_sequence_path = self.output_root / 'hrv_sequence'
        self.hrv_sequence_path.mkdir(parents=True, exist_ok=True)

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

    def _extract_segmented_features(self, window_data_dict):
        """
        为一个30秒宏观窗口提取多尺度分段统计特征。
        """
        final_feature_sequences = []

        # 定义每个模态的微观窗口大小 (秒)
        micro_window_configs = {
            'chest_ECG': 3,
            'wrist_BVP': 3,
            'chest_EDA': 5,
            'wrist_EDA': 5,
            'chest_Resp': 6,
            'chest_EMG': 2,
            'chest_ACC': 2,
            'wrist_ACC': 2,
        }

        # 定义每个模态的原始采样率
        fs_map = {**self.WRIST_FS, **{c: self.CHEST_FS for c in self.CHEST_CHANNELS}}

        # 定义我们想要拼接到的最长序列长度 (来自30秒/2秒 = 15个子窗口)
        max_seq_len = int(self.WINDOW_SEC / min(micro_window_configs.values()))

        # 遍历我们感兴趣的模态
        for key in ['chest_ECG', 'wrist_BVP', 'chest_EDA', 'wrist_EDA', 'chest_Resp', 'chest_EMG', 'chest_ACC',
                    'wrist_ACC']:
            signal_segment = window_data_dict.get(key)

            if signal_segment is None: continue

            micro_win_sec = micro_window_configs[key]
            fs = fs_map[key.split('_')[1]]
            micro_win_samples = int(micro_win_sec * fs)

            fs_map = {**self.WRIST_FS, **{c: self.CHEST_FS for c in self.CHEST_CHANNELS}}

            # 我们仍然需要 max_seq_len 来确定目标形状
            max_seq_len = int(self.WINDOW_SEC / min(micro_window_configs.values()))

            for key in ['chest_ECG', 'wrist_BVP', 'chest_EDA', 'wrist_EDA', 'chest_Resp', 'chest_EMG', 'chest_ACC',
                        'wrist_ACC']:
                signal_segment = window_data_dict.get(key)

                # 定义每个特征集的维度 (ACC是3轴x3个统计量=9，其余是1轴x3个=3)
                num_features_per_sub_window = 9 if 'ACC' in key else 3

                # 1. 创建一个目标形状的全零矩阵
                target_shape = (max_seq_len, num_features_per_sub_window)
                padded_features = np.zeros(target_shape, dtype=np.float32)

                if signal_segment is None or len(signal_segment) == 0:
                    final_feature_sequences.append(padded_features)
                    continue

                micro_win_sec = micro_window_configs[key]
                fs = fs_map[key.split('_')[1]]
                micro_win_samples = int(micro_win_sec * fs)

                if micro_win_samples == 0:
                    final_feature_sequences.append(padded_features)
                    continue

                num_sub_windows = int(len(signal_segment) / micro_win_samples)

                sub_window_features = []
                for i in range(num_sub_windows):
                    start = i * micro_win_samples
                    end = start + micro_win_samples
                    sub_window = signal_segment[start:end]

                    # 如果子窗口为空，则跳过
                    if sub_window.shape[0] == 0: continue

                    mean = np.mean(sub_window, axis=0)
                    std = np.std(sub_window, axis=0)
                    rms = np.sqrt(np.mean(sub_window ** 2, axis=0))

                    sub_window_features.append(np.concatenate([np.atleast_1d(f) for f in [mean, std, rms]]))

                # 如果成功提取了任何特征
                if sub_window_features:
                    features_array = np.array(sub_window_features)
                    # 2. 将计算出的特征填充到零矩阵的左上角
                    # 这可以安全地处理 features_array 为空的情况
                    padded_features[:features_array.shape[0], :] = features_array

                final_feature_sequences.append(padded_features)

            # 3. 沿特征维度拼接所有模态的序列
            return np.concatenate(final_feature_sequences, axis=1)

    def _generate_hrv_sequences(self, data, protocol_df):
        """
        从原始ECG信号中提取HRV(RR间期)序列，并进行窗口化。
        """
        ecg_signal = data[b'signal'][b'chest'][b'ECG'].flatten()

        # 1. 使用 NeuroKit2 提取 R 波峰
        #    `robust=True` 会使用更稳健但稍慢的算法
        _, rpeaks_info = nk.ecg_peaks(ecg_signal, sampling_rate=self.CHEST_FS, correct_artifacts=True)
        rpeaks_indices = rpeaks_info['ECG_R_Peaks']

        # 2. 计算 RR 间期 (单位：毫秒)
        rr_intervals = np.diff(rpeaks_indices) * 1000.0 / self.CHEST_FS
        rr_timestamps = rpeaks_indices[1:] / self.CHEST_FS  # 每个RR间期的时间戳 (秒)

        # 3. 对不规则的 RR 间期序列进行插值，得到均匀采样的时间序列
        #    4Hz 是 HRV 分析中常用的插值频率
        INTERP_FS = 4
        duration_sec = len(ecg_signal) / self.CHEST_FS
        time_uniform = np.arange(0, duration_sec, 1 / INTERP_FS)
        rr_interpolated = np.interp(time_uniform, rr_timestamps, rr_intervals)

        # 4. 使用协议进行窗口化
        #    窗口化是在插值后的信号上进行的，所以采样率是 INTERP_FS
        #    HRV分析通常需要更长的窗口，我们这里使用60秒
        HRV_WINDOW_SEC = 60
        windows, labels = [], []
        task_to_label_map = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}

        for _, row in protocol_df.iterrows():
            task = row['task'].replace(" ", "").strip()
            original_label = task_to_label_map.get(task)
            if original_label is None: continue
            label = self.LABEL_MAP.get(original_label)
            if label is None: continue

            # 时间单位是秒，直接乘以插值频率得到索引
            start_idx = int(row['start_min'] * 60 * INTERP_FS)
            end_idx = int(row['end_min'] * 60 * INTERP_FS)

            step_sec = self.TASK_STEPS_SEC.get(row['task'], HRV_WINDOW_SEC / 2)
            step_samples = int(step_sec * INTERP_FS)
            window_samples = int(HRV_WINDOW_SEC * INTERP_FS)

            if end_idx > len(rr_interpolated): end_idx = len(rr_interpolated)

            for i in range(start_idx, end_idx - window_samples + 1, step_samples):
                window = rr_interpolated[i: i + window_samples]
                windows.append(window)
                labels.append(label)

        return np.array(windows), np.array(labels)

    def process_subject(self, subject_id, process_type='all'):
        """
        处理单个受试者的数据。
        process_type (str): 'early', 'feature', 'segmented', 或 'all'
        """
        print(f"--- 正在处理受试者: {subject_id} ---")

        data = load_pkl(subject_id, self.wesad_root)
        if data is None: return None, None, None, None, None, None

        protocol_df = parse_quest_csv(subject_id, self.wesad_root)

        # 根据 process_type 执行不同的逻辑分支
        X_feat, y_feat, X_early, y_early, X_seg, y_seg = None, None, None, None, None, None

        if process_type in ['feature', 'segmented', 'all']:
            # 这两种都需要基于原始信号的窗口
            chest_windows, labels = self._get_windows(len(data[b'label']), protocol_df, self.CHEST_FS)
            if not chest_windows:  # 增加一个检查，如果没有任何窗口生成
                return None, None, None, None, None, None

            if process_type in ['feature', 'all']:
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

                    # 调用特征提取函数
                    window_features = self._extract_features_nk(window_data)
                    feature_windows.append(window_features)

                # 清理可能产生的NaN/Inf值
                X_feat = pd.DataFrame(feature_windows).fillna(0).replace([np.inf, -np.inf], 0).values
                y_feat = labels

                if X_feat.shape[0] > 0:
                    print(f"  - [feature_fusion] 已生成 {X_feat.shape[0]} 个窗口, 特征数 {X_feat.shape[1]}")

            if process_type in ['segmented', 'all']:
                segmented_windows = []
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

                    # 调用分段特征提取函数
                    segmented_features = self._extract_segmented_features(window_data)
                    segmented_windows.append(segmented_features)

                X_seg = np.array(segmented_windows)
                y_seg = labels

                if X_seg.shape[0] > 0:
                    print(f"  - [feature_segmented] 已生成 {X_seg.shape[0]} 个窗口, 形状为 {X_seg.shape[1:]}")

        if process_type in ['early', 'all']:
            # 为早期融合 (wesad_early_fusion) 准备重采样后的信号
            resampled_signals = {}
            all_channels = self.CHEST_CHANNELS + self.WRIST_CHANNELS

            # 先重采样所有需要的信号
            for dev, channels, fs_dict in [
                ('chest', self.CHEST_CHANNELS, {c: self.CHEST_FS for c in self.CHEST_CHANNELS}),
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
            target_windows, y_early = self._get_windows(len(resampled_signals['chest_ECG']), protocol_df,
                                                        self.TARGET_FS)

            window_len_samples = int(self.WINDOW_SEC * self.TARGET_FS)
            num_channels = sum(
                resampled_signals[f"{dev}_{c}"].shape[1] if resampled_signals[f"{dev}_{c}"].ndim > 1 else 1 for
                dev, chans, _ in [('chest', self.CHEST_CHANNELS, {}), ('wrist', self.WRIST_CHANNELS, {})] for c in
                chans)

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

            if X_early.shape[0] > 0:
                print(f"  - [early_fusion] 已生成 {X_early.shape[0]} 个窗口, 形状为 {X_early.shape[1:]}")

                # --- 根据 process_type 执行不同的逻辑分支 ---
        if process_type in ['hrv_sequence', 'all']:
            X_hrv, y_hrv = self._generate_hrv_sequences(data, protocol_df)
            np.save(self.hrv_sequence_path / f'{subject_id}_X.npy', X_hrv)
            np.save(self.hrv_sequence_path / f'{subject_id}_y.npy', y_hrv)
            print(f"  - [hrv_sequence] 已生成 {X_hrv.shape[0]} 个窗口, 形状为 {X_hrv.shape[1:]}")

        return X_feat, y_feat, X_early, y_early, X_seg, y_seg

    def run(self, process_type='all'):
        """运行预处理，接受 process_type 参数控制模式"""
        subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
        print(f"--- 开始预处理, 模式: {process_type} ---")

        for sid in subject_ids:
            # 提取特征和信号
            X_f, y_f, X_e, y_e, X_s, y_s = self.process_subject(sid, process_type)

            if process_type in ['feature', 'all'] and X_f is not None and y_f is not None:
                np.save(self.feature_fusion_path / f'{sid}_X.npy', X_f)
                np.save(self.feature_fusion_path / f'{sid}_y.npy', y_f)

            if process_type in ['early', 'all'] and X_e is not None and y_e is not None:
                np.save(self.early_fusion_path / f'{sid}_X.npy', X_e)
                np.save(self.early_fusion_path / f'{sid}_y.npy', y_e)

            if process_type in ['segmented', 'all'] and X_s is not None and y_s is not None:
                np.save(self.segmented_feature_path / f'{sid}_X.npy', X_s)
                np.save(self.segmented_feature_path / f'{sid}_y.npy', y_s)

        print(f"\n预处理模式 '{process_type}' 已完成！")


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    process_types = ['all', 'early', 'feature', 'segmented', 'hrv_sequence']
    selected_type_index = 4
    process_type = process_types[selected_type_index]

    paths = get_path_manager()
    WESAD_ROOT_PATH = paths.WESAD_ROOT
    DATA_PATH = paths.DATA_ROOT

    preprocessor = WesadPreprocessor(wesad_root=WESAD_ROOT_PATH, output_root=DATA_PATH)

