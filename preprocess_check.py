# file: check_preprocess.py
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from collections import Counter

# ======================================================================================
# --- 1. CONFIGURATION ---
# ======================================================================================
# 与 preprocess.py 保持一致，以便进行交叉验证
PROJECT_ROOT = Path(__file__).resolve().parent
WESAD_RAW_PATH = PROJECT_ROOT / "WESAD"
PREPROCESSED_DATA_PATH = PROJECT_ROOT / "data"
# 用于检查的示例被试ID
SUBJECT_TO_CHECK = 'S2'


# ======================================================================================
# --- 2. THE CHECKER CLASS ---
# ======================================================================================

class PreprocessChecker:
    """验证由 preprocess.py 生成的数据的完整性、形状和逻辑。"""

    def __init__(self, raw_data_root: Path, processed_data_root: Path, subject_id: str):
        self.raw_root = raw_data_root
        self.processed_root = processed_data_root
        self.sid = subject_id

    def _log(self, message: str, level: str = 'INFO'):
        """漂亮的彩色日志打印"""
        color_map = {'INFO': '\033[92m', 'ERROR': '\033[91m', 'WARNING': '\033[93m', 'HEADER': '\033[95m'}
        prefix = color_map.get(level, '')
        suffix = '\033[0m'
        print(f"{prefix}[{level.upper()}] {message}{suffix}")

    def check_file_existence(self):
        """检查所有必需的文件是否存在。"""
        self._log(f"--- 1. 检查文件存在性 (以 {self.sid} 为例) ---", 'HEADER')
        files_ok = True

        # 检查通道顺序文件
        self.channel_order_path = self.processed_root / 'channel_order.txt'
        if not self.channel_order_path.exists():
            self._log(f"channel_order.txt 未找到!", 'ERROR')
            files_ok = False

        # 检查单个被试的数据文件
        self.subject_X_path = self.processed_root / self.sid / 'X.npy'
        self.subject_y_path = self.processed_root / self.sid / 'y.npy'
        if not self.subject_X_path.exists() or not self.subject_y_path.exists():
            self._log(f"{self.sid} 的 X.npy 或 y.npy 文件未找到!", 'ERROR')
            files_ok = False

        # 检查为该被试准备的归一化参数文件
        self.norm_params_path = self.processed_root / f'norm_params_for_test_{self.sid}.npz'
        if not self.norm_params_path.exists():
            self._log(f"为 {self.sid} 作为测试集时的归一化文件未找到!", 'ERROR')
            files_ok = False

        if files_ok:
            self._log("所有必需的文件均已找到。", 'INFO')
        else:
            self._log("关键文件缺失，请先运行 preprocess.py。", 'ERROR')
        return files_ok

    def check_data_shape_and_content(self):
        """检查数据形状、通道对齐和内容。"""
        self._log(f"--- 2. 检查数据形状与内容 (以 {self.sid} 为例) ---", 'HEADER')
        shape_ok = True
        try:
            # 加载所有数据
            with open(self.channel_order_path, 'r') as f:
                channel_names = [line.strip() for line in f.readlines()]
            X_data = np.load(self.subject_X_path)
            y_data = np.load(self.subject_y_path)
            norm_params = np.load(self.norm_params_path)
            mean_vec = norm_params['mean']
            std_vec = norm_params['std']

            self._log(f"X.npy shape: {X_data.shape}")
            self._log(f"y.npy shape: {y_data.shape}")
            self._log(f"检测到的通道数: {len(channel_names)}")

            # 检查维度匹配
            if X_data.shape[0] != y_data.shape[0]:
                self._log("X 和 y 的窗口数量不匹配!", 'ERROR')
                shape_ok = False

            if X_data.shape[2] != len(channel_names):
                self._log("X 的通道数与 channel_order.txt 中记录的数量不匹配!", 'ERROR')
                shape_ok = False

            if len(mean_vec) != len(channel_names) or len(std_vec) != len(channel_names):
                self._log("归一化参数的维度与通道数不匹配!", 'ERROR')
                shape_ok = False

            # 检查内容
            if np.isnan(X_data).any():
                self._log("X.npy 文件中包含 NaN 值!", 'WARNING')
            if not np.all(np.isfinite(X_data)):
                self._log("X.npy 文件中包含 Inf 值!", 'WARNING')

            if shape_ok:
                self._log("数据形状和通道对齐检查通过。", 'INFO')
            return shape_ok

        except Exception as e:
            self._log(f"检查数据时发生意外错误: {e}", 'ERROR')
            return False

    def check_label_distribution_and_mapping(self):
        """检查标签分布和映射是否符合预期。"""
        self._log(f"--- 3. 检查标签分布与映射 (以 {self.sid} 为例) ---", 'HEADER')
        label_ok = True
        try:
            y_data = np.load(self.subject_y_path)
            label_counts = Counter(y_data)

            self._log(f"标签分布: {dict(label_counts)}")

            # WESAD 预处理后，标签应为 0, 1, 2
            expected_labels = {0, 1, 2}
            actual_labels = set(label_counts.keys())

            if not actual_labels.issubset(expected_labels):
                self._log(f"发现预期之外的标签: {actual_labels - expected_labels}", 'ERROR')
                label_ok = False

            if not actual_labels:
                self._log(f"标签文件为空!", 'ERROR')
                label_ok = False

            if label_ok:
                self._log("标签分布和映射检查通过 (应为 0:neutral, 1:amusement, 2:stress)。", 'INFO')
            return label_ok

        except Exception as e:
            self._log(f"检查标签时发生意外错误: {e}", 'ERROR')
            return False

    def run_all_checks(self):
        """按顺序运行所有检查。"""
        self._log(f"===== 开始对预处理数据进行全面检查 (以 {SUBJECT_TO_CHECK} 为例) =====", 'HEADER')

        if not self.check_file_existence():
            return  # 如果文件都不存在，后续检查无意义

        self.check_data_shape_and_content()
        self.check_label_distribution_and_mapping()

        self._log(f"===== 所有检查完成 =====", 'HEADER')


# ======================================================================================
# --- 3. 主程序入口 ---
# ======================================================================================
if __name__ == "__main__":
    checker = PreprocessChecker(
        raw_data_root=WESAD_RAW_PATH,
        processed_data_root=PREPROCESSED_DATA_PATH,
        subject_id=SUBJECT_TO_CHECK
    )
    checker.run_all_checks()