# file: check_preprocess.py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# ======================================================================================
# --- 1. CONFIGURATION ---
# ======================================================================================
# 与 preprocess.py, main.py, dataset.py 保持一致
PROJECT_ROOT = Path(__file__).resolve().parent
WESAD_RAW_PATH = PROJECT_ROOT / "WESAD"
PREPROCESSED_DATA_PATH = PROJECT_ROOT / "data"
EARLY_FUSION_PATH = PREPROCESSED_DATA_PATH / "chest_raw_align"  # HybridDataset 使用的路径
FEATURE_FUSION_PATH = PREPROCESSED_DATA_PATH / "chest_feature"
SUBJECT_TO_CHECK = 'S16'
CLASSIFICATION_MODE = 'binary'  # 'binary' 或 'ternary'，与 main.py 同步

# 标签映射，与 dataset.py 保持一致
LABEL_INT_TO_STR_MAP = {
    1: 'baseline',  # Base
    2: 'stress',    # TSST
    3: 'amusement', # Fun
    4: 'baseline'   # Medi1/2 合并到 baseline
}
EXPECTED_LABELS = {'binary': {0, 1}, 'ternary': {0, 1, 2}}  # 预期映射后的标签


# ======================================================================================
# --- 2. THE CHECKER CLASS ---
# ======================================================================================

class PreprocessChecker:
    """验证由 preprocess.py 生成的数据的完整性、形状和标签一致性。"""

    def __init__(self, raw_data_root: Path, early_fusion_path: Path, feature_fusion_path: Path, subject_id: str, classification_mode: str):
        self.raw_root = raw_data_root
        self.early_fusion_path = early_fusion_path
        self.feature_fusion_path = feature_fusion_path
        self.sid = subject_id
        self.classification_mode = classification_mode

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

        # 检查通道和特征名文件
        self.channel_order_path = self.early_fusion_path / '_channel_names.txt'
        self.feature_names_path = self.feature_fusion_path / '_feature_names.txt'
        if not self.channel_order_path.exists():
            self._log(f"_channel_names.txt 未找到 in {self.early_fusion_path}!", 'ERROR')
            files_ok = False
        if not self.feature_names_path.exists():
            self._log(f"_feature_names.txt 未找到 in {self.feature_fusion_path}!", 'ERROR')
            files_ok = False

        # 检查 raw-align 数据文件
        self.raw_X_path = self.early_fusion_path / f'{self.sid}_X.npy'
        self.raw_y_path = self.early_fusion_path / f'{self.sid}_y.npy'
        if not self.raw_X_path.exists() or not self.raw_y_path.exists():
            self._log(f"{self.sid} 的 raw-align X.npy 或 y.npy 文件未找到!", 'ERROR')
            files_ok = False

        # 检查 feature 数据文件
        self.feature_X_path = self.feature_fusion_path / f'{self.sid}_X.npy'
        self.feature_y_path = self.feature_fusion_path / f'{self.sid}_y.npy'
        if not self.feature_X_path.exists() or not self.feature_y_path.exists():
            self._log(f"{self.sid} 的 feature X.npy 或 y.npy 文件未找到!", 'ERROR')
            files_ok = False

        if files_ok:
            self._log("所有必需的文件均已找到。", 'INFO')
        else:
            self._log("关键文件缺失，请检查 preprocess.py 的输出。", 'ERROR')
        return files_ok

    def check_data_shape_and_content(self):
        """检查数据形状、通道/特征对齐和内容完整性。"""
        self._log(f"--- 2. 检查数据形状与内容 (以 {self.sid} 为例) ---", 'HEADER')
        shape_ok = True
        try:
            # 加载通道和特征名
            with open(self.channel_order_path, 'r') as f:
                channel_names = [line.strip() for line in f]
            with open(self.feature_names_path, 'r') as f:
                feature_names = [line.strip() for line in f]

            # 加载数据
            raw_X = np.load(self.raw_X_path)
            raw_y = np.load(self.raw_y_path)
            feature_X = np.load(self.feature_X_path)
            feature_y = np.load(self.feature_y_path)

            self._log(f"raw-align X shape: {raw_X.shape}")
            self._log(f"raw-align y shape: {raw_y.shape}")
            self._log(f"feature X shape: {feature_X.shape}")
            self._log(f"feature y shape: {feature_y.shape}")
            self._log(f"检测到的通道数: {len(channel_names)}")
            self._log(f"检测到的特征数: {len(feature_names)}")

            # 检查样本数匹配
            if raw_X.shape[0] != raw_y.shape[0]:
                self._log("raw-align X 和 y 的窗口数量不匹配!", 'ERROR')
                shape_ok = False
            if feature_X.shape[0] != feature_y.shape[0]:
                self._log("feature X 和 y 的窗口数量不匹配!", 'ERROR')
                shape_ok = False
            if raw_X.shape[0] != feature_X.shape[0]:
                self._log("raw-align 和 feature 的样本数不匹配!", 'ERROR')
                shape_ok = False

            # 检查通道/特征维度
            if raw_X.shape[2] != len(channel_names):
                self._log("raw-align X 的通道数与 _channel_names.txt 不匹配!", 'ERROR')
                shape_ok = False
            if feature_X.shape[1] != len(feature_names):
                self._log("feature X 的特征数与 _feature_names.txt 不匹配!", 'ERROR')
                shape_ok = False

            # 检查数据完整性
            if np.isnan(raw_X).any() or np.isnan(feature_X).any():
                self._log("X 数据中包含 NaN 值!", 'WARNING')
            if not np.all(np.isfinite(raw_X)) or not np.all(np.isfinite(feature_X)):
                self._log("X 数据中包含 Inf 值!", 'WARNING')

            if shape_ok:
                self._log("数据形状和内容检查通过。", 'INFO')
            return shape_ok

        except Exception as e:
            self._log(f"检查数据时发生意外错误: {e}", 'ERROR')
            return False

    def check_label_distribution_and_mapping(self):
        """检查标签分布、映射和 raw-align/feature 一致性。"""
        self._log(f"--- 3. 检查标签分布与映射 (以 {self.sid} 为例) ---", 'HEADER')
        label_ok = True
        try:
            # 加载标签
            raw_y = np.load(self.raw_y_path)
            feature_y = np.load(self.feature_y_path)

            # 检查原始标签分布
            # 检查原始标签分布
            raw_label_counts = Counter(raw_y)
            feature_label_counts = Counter(feature_y)
            self._log(f"raw-align 原始标签分布: {dict(raw_label_counts)}")
            self._log(f"feature 原始标签分布: {dict(feature_label_counts)}")

            # 检查原始标签是否一致
            if not np.array_equal(raw_y, feature_y):
                self._log("raw-align 和 feature 的原始标签不匹配!", 'ERROR')
                label_ok = False
                # 打印不匹配的索引
                diff_indices = np.where(raw_y != feature_y)[0]
                self._log(f"不匹配的索引: {diff_indices[:10]} (显示前10个)", 'ERROR')
                self._log(f"raw-align 标签示例: {raw_y[diff_indices[:5]]}", 'ERROR')
                self._log(f"feature 标签示例: {feature_y[diff_indices[:5]]}", 'ERROR')

            # 应用 dataset.py 的标签映射逻辑
            if self.classification_mode == 'binary':
                mapped_raw_y = np.where(raw_y == 2, 1, 0)
                mapped_feature_y = np.where(feature_y == 2, 1, 0)
            elif self.classification_mode == 'ternary':
                mapped_raw_y = np.where(raw_y == 1, 0, np.where(raw_y == 3, 1, np.where(raw_y == 2, 2, 0)))
                mapped_feature_y = np.where(feature_y == 1, 0, np.where(feature_y == 3, 1, np.where(feature_y == 2, 2, 0)))
            else:
                raise ValueError(f"Unknown classification_mode: {self.classification_mode}")

            # 检查映射后标签分布
            mapped_raw_counts = Counter(mapped_raw_y)
            mapped_feature_counts = Counter(mapped_feature_y)
            self._log(f"映射后 raw-align 标签分布: {dict(mapped_raw_counts)}")
            self._log(f"映射后 feature 标签分布: {dict(mapped_feature_counts)}")

            # 验证映射后标签是否符合预期
            expected_labels = EXPECTED_LABELS[self.classification_mode]
            if not set(mapped_raw_y).issubset(expected_labels):
                self._log(f"raw-align 映射后发现预期之外的标签: {set(mapped_raw_y) - expected_labels}", 'ERROR')
                label_ok = False
            if not set(mapped_feature_y).issubset(expected_labels):
                self._log(f"feature 映射后发现预期之外的标签: {set(mapped_feature_y) - expected_labels}", 'ERROR')
                label_ok = False

            # 检查映射后标签一致性
            if not np.array_equal(mapped_raw_y, mapped_feature_y):
                self._log("映射后 raw-align 和 feature 标签不匹配!", 'ERROR')
                label_ok = False
                diff_indices = np.where(mapped_raw_y != mapped_feature_y)[0]
                self._log(f"映射后不匹配的索引: {diff_indices[:10]} (显示前10个)", 'ERROR')
                self._log(f"映射后 raw-align 标签示例: {mapped_raw_y[diff_indices[:5]]}", 'ERROR')
                self._log(f"映射后 feature 标签示例: {mapped_feature_y[diff_indices[:5]]}", 'ERROR')

            if label_ok:
                self._log(f"标签分布和映射检查通过 ({self.classification_mode} 模式)。", 'INFO')
            return label_ok

        except Exception as e:
            self._log(f"检查标签时发生意外错误: {e}", 'ERROR')
            return False

    def run_all_checks(self):
        """按顺序运行所有检查。"""
        self._log(f"===== 开始对预处理数据进行全面检查 (以 {SUBJECT_TO_CHECK} 为例) =====", 'HEADER')

        if not self.check_file_existence():
            return  # 如果文件缺失，后续检查无意义

        shape_ok = self.check_data_shape_and_content()
        label_ok = self.check_label_distribution_and_mapping()

        if shape_ok and label_ok:
            self._log("所有检查通过，数据正常。", 'INFO')
        else:
            self._log("检查未通过，请根据错误日志修复 preprocess.py 或数据文件。", 'ERROR')
        self._log(f"===== 所有检查完成 =====", 'HEADER')


# ======================================================================================
# --- 3. 主程序入口 ---
# ======================================================================================
if __name__ == "__main__":
    checker = PreprocessChecker(
        raw_data_root=WESAD_RAW_PATH,
        early_fusion_path=EARLY_FUSION_PATH,
        feature_fusion_path=FEATURE_FUSION_PATH,
        subject_id=SUBJECT_TO_CHECK,
        classification_mode=CLASSIFICATION_MODE
    )
    checker.run_all_checks()