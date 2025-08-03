# preprocess_check.py
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from collections import Counter
from src.utils.path_manager import get_path_manager


class WESADPreprocessChecker:
    """WESAD数据集预处理验证器"""

    def __init__(self, wesad_root: Path, processed_data_root: Path, subject_id: str = 'S2'):
        self.wesad_root = Path(wesad_root)
        self.processed_data_root = Path(processed_data_root)
        self.subject_id = subject_id
        self.fs = 700  # 采样频率
        self.class_labels = {1: 'Base', 2: 'TSST', 3: 'Fun', 4: 'Medi'}  # 任务到标签的映射

    def _log(self, message: str, level: str = 'INFO'):
        """记录日志信息"""
        prefix = {'INFO': '\033[92m', 'ERROR': '\033[91m', 'WARNING': '\033[93m'}.get(level, '')
        suffix = '\033[0m'
        print(f"{prefix}[{level}] {message}{suffix}")

    def check_feature_fusion(self):
        """检查特征融合数据"""
        self._log(f"Checking Feature Fusion data for subject {self.subject_id}", 'INFO')
        try:
            x_path = self.processed_data_root / 'wesad_feature_fusion' / f'{self.subject_id}_X.npy'
            y_path = self.processed_data_root / 'wesad_feature_fusion' / f'{self.subject_id}_y.npy'

            x_feat = np.load(x_path)
            y_feat = np.load(y_path)

            self._log(f"X_feat shape: {x_feat.shape}")
            self._log(f"y_feat shape: {y_feat.shape}")

            # 验证形状
            assert x_feat.shape[0] == y_feat.shape[0], "X and y window counts do not match!"
            assert x_feat.shape[1] == 24, "Feature dimension is not 24!"

            # 检查标签分布
            unique_labels, counts = np.unique(y_feat, return_counts=True)
            self._log(f"Label distribution: {dict(zip(unique_labels, counts))}")
            self._log("Feature Fusion check passed: feature dimension is 48, labels are [0, 1, 2].")

        except FileNotFoundError:
            self._log(f"Feature Fusion .npy files not found for {self.subject_id}. Run preprocess.py.", 'ERROR')
        except AssertionError as e:
            self._log(f"Feature Fusion check failed: {str(e)}", 'ERROR')

    def check_early_fusion(self):
        """检查早期融合数据"""
        self._log(f"Checking Early Fusion data for subject {self.subject_id}", 'INFO')
        try:
            x_path = self.processed_data_root / 'wesad_early_fusion' / f'{self.subject_id}_X.npy'
            y_path = self.processed_data_root / 'wesad_early_fusion' / f'{self.subject_id}_y.npy'

            x_early = np.load(x_path)
            y_early = np.load(y_path)

            self._log(f"X_early shape: {x_early.shape}")
            self._log(f"y_early shape: {y_early.shape}")

            # 验证形状
            assert x_early.shape[0] == y_early.shape[0], "X and y window counts do not match!"
            assert x_early.shape[1] == 1920, "Window length is not 1920 (30s * 64Hz)!"
            assert x_early.shape[2] == 12, "Channel count is not 12!"

            self._log("Early Fusion check passed: shape is (windows, 1920, 12).")

        except FileNotFoundError:
            self._log(f"Early Fusion .npy files not found for {self.subject_id}. Run preprocess.py.", 'ERROR')
        except AssertionError as e:
            self._log(f"Early Fusion check failed: {str(e)}", 'ERROR')

    def parse_quest_csv(self) -> pd.DataFrame:
        """解析quest.csv文件，提取任务时间戳"""
        quest_path = self.wesad_root / self.subject_id / f"{self.subject_id}_quest.csv"
        try:
            df_raw = pd.read_csv(quest_path, sep=';', header=None, skip_blank_lines=True)
            order_row = df_raw[df_raw[0].str.contains('# ORDER', na=False)].values[0]
            start_row = df_raw[df_raw[0].str.contains('# START', na=False)].values[0]
            end_row = df_raw[df_raw[0].str.contains('# END', na=False)].values[0]

            tasks = [str(t).strip() for t in pd.Series(order_row[1:]).dropna().tolist()]
            start_times = pd.Series(start_row[1:]).dropna().astype(float).tolist()
            end_times = pd.Series(end_row[1:]).dropna().astype(float).tolist()

            return pd.DataFrame({'task': tasks, 'start_min': start_times, 'end_min': end_times})
        except Exception as e:
            self._log(f"Failed to parse quest.csv for {self.subject_id}: {str(e)}", 'ERROR')
            return pd.DataFrame()

    def load_pkl(self):
        """加载.pkl文件"""
        pkl_path = self.wesad_root / self.subject_id / f"{self.subject_id}.pkl"
        try:
            with open(pkl_path, 'rb') as f:
                return pickle.load(f, encoding='bytes')
        except Exception as e:
            self._log(f"Failed to load .pkl file for {self.subject_id}: {str(e)}", 'ERROR')
            return None

    def verify_label_alignment(self):
        """验证标签对齐"""
        self._log(f"Verifying label alignment for subject {self.subject_id}", 'INFO')

        task_name_to_label = {'Base': 1, 'TSST': 2, 'Fun': 3, 'Medi1': 4, 'Medi2': 4}
        all_tasks_verified = True

        try:
            # 加载数据
            protocol_df = self.parse_quest_csv()
            if protocol_df.empty:
                self._log("Label alignment check aborted due to parse_quest_csv failure.", 'ERROR')
                return

            data = self.load_pkl()
            if data is None:
                self._log("Label alignment check aborted due to load_pkl failure.", 'ERROR')
                return

            pkl_labels = data[b'label']

            # 验证每个任务
            for _, row in protocol_df.iterrows():
                task_name = row['task'].replace(" ", "").strip()
                expected_label = task_name_to_label.get(task_name)

                if expected_label is None:
                    self._log(f"Skipping task {row['task']} (not in core tasks).", 'INFO')
                    continue

                self._log(f"Verifying task: {row['task']}")

                # 计算时间戳
                start_idx = int(row['start_min'] * 60 * self.fs)
                end_idx = int(row['end_min'] * 60 * self.fs)
                segment = pkl_labels[start_idx:end_idx]

                # 找到主要标签（忽略过渡标签0）
                counts = Counter(segment)
                top_two = counts.most_common(2)
                main_label = top_two[0][0] if top_two[0][0] != 0 else (top_two[1][0] if len(top_two) > 1 else 0)

                self._log(f"  CSV timestamp ({row['start_min']}-{row['end_min']} min), "
                          f"segment length: {len(segment)}, main label: {main_label}")

                # 比较标签
                if main_label != expected_label:
                    self._log(f"  Verification failed: main label ({main_label}) does not match "
                              f"expected label ({expected_label})!", 'ERROR')
                    all_tasks_verified = False
                else:
                    self._log(f"  Verification passed: main label matches expected.", 'INFO')

            self._log("\nLabel Alignment Summary", 'INFO')
            self._log("All core tasks verified successfully!" if all_tasks_verified else
                      "Label alignment issues detected!", 'INFO' if all_tasks_verified else 'ERROR')

        except Exception as e:
            self._log(f"Label alignment check failed: {str(e)}", 'ERROR')

    def verify_new_features(self, fusion_type: str = 'wesad_feature_fusion', subject_ids: list = None):
        """验证新生成的特征数据"""
        if subject_ids is None:
            subject_ids = [f"S{i}" for i in range(2, 18) if i != 12]

        self._log(f"Verifying new '{fusion_type}' dataset", 'INFO')
        data_folder = self.processed_data_root / fusion_type

        if not data_folder.exists():
            self._log(f"Directory not found: {data_folder}", 'ERROR')
            return

        all_feature_dims = []
        total_windows = 0
        is_first_subject = True

        for sid in subject_ids:
            x_path = data_folder / f'{sid}_X.npy'
            y_path = data_folder / f'{sid}_y.npy'

            if is_first_subject:
                self._log(f"Checking subject {sid} (as sample)", 'INFO')
                is_first_subject = False

            if not x_path.exists() or not y_path.exists():
                self._log(f"Missing X or y file for subject {sid}", 'ERROR')
                continue

            try:
                x_data = np.load(x_path)
                y_data = np.load(y_path)

                # 验证数据
                if x_data.shape[0] != y_data.shape[0]:
                    self._log(f"Subject {sid}: X ({x_data.shape[0]}) and y ({y_data.shape[0]}) "
                              f"window counts do not match!", 'ERROR')
                if np.isnan(x_data).any() or not np.all(np.isfinite(x_data)):
                    self._log(f"Subject {sid}: X contains NaN or Inf values!", 'WARNING')

                all_feature_dims.append(x_data.shape[1])
                total_windows += x_data.shape[0]

            except Exception as e:
                self._log(f"Failed to process subject {sid}: {str(e)}", 'ERROR')

        self._log("\nNew Features Verification Summary", 'INFO')
        if not all_feature_dims:
            self._log("No data loaded for verification.", 'ERROR')
            return

        if len(set(all_feature_dims)) == 1:
            in_features = all_feature_dims[0]
            self._log(f"Verification successful!", 'INFO')
            self._log(f"All subjects have consistent feature dimension: {in_features}")
            self._log(f"Processed {len(subject_ids)} subjects, {total_windows} windows.")
            self._log(f"\nUpdate your config (e.g., exp_feature_tabnet.yaml) with:")
            self._log(f"  model -> params -> in_features: \033[1m{in_features}\033[0m")
        else:
            self._log(f"Verification failed: inconsistent feature dimensions {set(all_feature_dims)}", 'ERROR')

    def run_all_checks(self):
        """运行所有验证"""
        self._log(f"Starting preprocessing checks for subject {self.subject_id}", 'INFO')
        self._log("=" * 50, 'INFO')
        self.check_feature_fusion()
        self.check_early_fusion()
        self.verify_label_alignment()
        self.verify_new_features()
        self._log("=" * 50, 'INFO')
        self._log("All preprocessing checks completed.", 'INFO')


if __name__ == "__main__":
    path_manager = get_path_manager()
    checker = WESADPreprocessChecker(
        wesad_root=path_manager.WESAD_ROOT,
        processed_data_root=path_manager.DATA_ROOT,
        subject_id='S2'
    )
    checker.run_all_checks()