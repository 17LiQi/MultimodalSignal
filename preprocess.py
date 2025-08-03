import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def parse_readme(subject_id: str, wesad_root: Path) -> dict:
    """从 SX_readme.txt 中解析年龄和性别。"""
    readme_path = wesad_root / subject_id / f"{subject_id}_readme.txt"
    attributes = {}
    with open(readme_path, 'r') as f:
        for line in f:
            if 'Age:' in line:
                attributes['age'] = int(line.split(':')[1].strip())
            if 'Gender:' in line:
                # 将性别编码为数值: male=0, female=1
                attributes['gender'] = 1 if 'female' in line.lower() else 0
    return attributes


# --- 2. 核心预处理类 ---

class WesadPreprocessor:
    def __init__(self, wesad_root, output_root):
        self.wesad_root = Path(wesad_root)
        self.output_root = Path(output_root)

        # 定义新的输出目录
        self.sequence_feature_path = self.output_root / 'wesad_sequence_features'
        self.sequence_feature_path.mkdir(parents=True, exist_ok=True)

        # 定义信号源和采样率
        self.CHEST_FS = 700
        self.WRIST_FS = {'ACC': 32, 'BVP': 64, 'EDA': 4}

        # 定义通道
        self.CHEST_CHANNELS = ['ACC', 'ECG', 'EDA', 'EMG', 'Resp']
        self.WRIST_CHANNELS = ['ACC', 'BVP', 'EDA']

        # 定义新的窗口化策略：短窗口，无重叠
        self.WINDOW_SEC = 5
        self.STEP_SEC = 5  # 无重叠

        # 标签映射 (与之前相同)
        self.LABEL_MAP = {1: 0, 2: 2, 3: 1, 4: 0}

        warnings.filterwarnings('ignore', category=FutureWarning)
        warnings.filterwarnings('ignore', category=UserWarning, module='neurokit2')
        warnings.filterwarnings('ignore', category=RuntimeWarning, module='neurokit2')

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

        hrv_features = np.full(10, np.nan)  # 默认占位符

        # np.full 创建一个用 NaN 填充的数组，用于处理错误
        try:
            # 检查信号段是否足够长
            if ecg_signal is not None and len(ecg_signal) > 10:  # 至少有10个点
                # 使用 nk.hrv，它内部已经有相当不错的异常处理
                hrv_df = nk.hrv(ecg_signal, sampling_rate=self.CHEST_FS, show=False)
            elif bvp_signal is not None and len(bvp_signal) > 10:
                hrv_df = nk.hrv(bvp_signal, sampling_rate=self.WRIST_FS['BVP'], show=False)
            else:
                hrv_df = None

            if hrv_df is not None and not hrv_df.isnull().all().all():
                hrv_features = hrv_df[['HRV_RMSSD', 'HRV_SDNN', 'HRV_pNN50',
                                       'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF',
                                       'HRV_SampEn', 'HRV_MeanNN', 'HRV_MedianNN']].iloc[0].values
        except Exception:
            pass  # 保持为 NaN

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
        for key, expected_dims in [('chest_ACC', 3), ('wrist_ACC', 3), ('chest_EMG', 1)]:
            signal_data = window_data_dict.get(key)

            if signal_data is None or signal_data.shape[0] < 5:  # 如果信号为空或太短
                energy = np.nan
            else:
                if 'ACC' in key:
                    energy = np.mean(np.sqrt(np.sum(signal_data ** 2, axis=1)))
                else:  # EMG
                    energy = np.mean(np.abs(signal_data))

            all_features.append(energy)

        return all_features

    def process_subject(self, subject_id):
        print(f"--- 正在处理受试者: {subject_id} ---")
        data = load_pkl(subject_id, self.wesad_root)
        if data is None: return

        protocol_df = parse_quest_csv(subject_id, self.wesad_root)
        attributes = parse_readme(subject_id, self.wesad_root)

        task_to_label_map = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}

        # 遍历每个实验阶段
        for _, row in protocol_df.iterrows():
            task_name = row['task'].replace(" ", "").strip()
            original_label = task_to_label_map.get(task_name)
            if original_label is None: continue

            label = self.LABEL_MAP.get(original_label)
            if label is None: continue

            print(f"  - 正在处理阶段: {task_name} (标签: {label})")

            start_sec, end_sec = row['start_min'] * 60, row['end_min'] * 60

            phase_features = []

            # 窗口化循环现在使用统一的 self.STEP_SEC ***
            for win_start_sec in np.arange(start_sec, end_sec - self.WINDOW_SEC + 1, self.STEP_SEC):
                win_end_sec = win_start_sec + self.WINDOW_SEC

                window_data_dict = {}
                start_idx_chest = int(win_start_sec * self.CHEST_FS)
                end_idx_chest = int(win_end_sec * self.CHEST_FS)
                for channel in self.CHEST_CHANNELS:
                    window_data_dict[f'chest_{channel}'] = data[b'signal'][b'chest'][channel.encode()][
                                                           start_idx_chest:end_idx_chest]
                for channel in self.WRIST_CHANNELS:
                    fs = self.WRIST_FS[channel]
                    start_idx_wrist = int(win_start_sec * fs)
                    end_idx_wrist = int(win_end_sec * fs)
                    window_data_dict[f'wrist_{channel}'] = data[b'signal'][b'wrist'][channel.encode()][
                                                           start_idx_wrist:end_idx_wrist]

                phys_features = self._extract_features_nk(window_data_dict)
                attr_features = [attributes.get('age', 30), attributes.get('gender', 0)]
                final_feature_vector = np.concatenate([phys_features, attr_features])
                phase_features.append(final_feature_vector)

            if phase_features:
                X_phase = np.array(phase_features)
                y_phase = np.full(X_phase.shape[0], label)

                # 使用更健壮的 NaN/Inf 清理策略
                # 使用 pandas DataFrame 来进行更智能的填充
                df_features = pd.DataFrame(X_phase)
                # a. 先用前一个有效值填充 (forward fill)
                df_features = df_features.fillna(method='ffill')
                # b. 再用后一个有效值填充 (backward fill), 以处理开头的 NaN
                df_features = df_features.fillna(method='bfill')
                # c. 如果整个列都是 NaN (例如某个受试者完全没有某个信号)，则用0填充
                df_features = df_features.fillna(0)
                # d. 处理可能存在的 inf 值
                df_features.replace([np.inf, -np.inf], 0, inplace=True)

                X_phase_cleaned = df_features.values

                save_path_X = self.sequence_feature_path / f'{subject_id}_{task_name}_X.npy'
                save_path_y = self.sequence_feature_path / f'{subject_id}_{task_name}_y.npy'
                np.save(save_path_X, X_phase_cleaned)
                np.save(save_path_y, y_phase)
                print(
                    f"    -> 已保存 {X_phase_cleaned.shape[0]} 个窗口, 特征数 {X_phase_cleaned.shape[1]} 到 {save_path_X.name}")

    def run(self):
        subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]
        for sid in tqdm(subject_ids, desc="正在处理所有受试者"):
            self.process_subject(sid)
        print("\n预处理完成！")


# --- 3. 主程序入口 ---
if __name__ == '__main__':
    paths = get_path_manager()
    WESAD_ROOT_PATH = paths.WESAD_ROOT
    DATA_PATH = paths.DATA_ROOT

    preprocessor = WesadPreprocessor(wesad_root=WESAD_ROOT_PATH, output_root=DATA_PATH)
    preprocessor.run()
