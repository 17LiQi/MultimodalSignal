# main.py

import numpy as np
from omegaconf import OmegaConf
from datetime import datetime
import torch

from utils.path_manager import get_path_manager  # 导入您自己的路径管理器
from utils.runner import run_one_fold


def main():
    # --- A. 加载配置 ---
    # 使用 OmegaConf 加载基础配置和指定的实验配置
    # 这种方式可以轻松地通过修改 'early_ecg_cnn.yaml' 来运行不同的实验
    # --- A. 加载配置 ---
    try:
        base_conf = OmegaConf.load('configs/base_config.yaml')

        # *************************在这里切换配置**************************
        exp_conf = OmegaConf.load('configs/early_transformer.yaml')
        # ***************************************************************

        config = OmegaConf.merge(base_conf, exp_conf)
        print(f"配置已加载, 当前配置:\n{OmegaConf.to_yaml(exp_conf)}")
    except Exception as e:
        print(f"错误: 无法加载配置文件。 {e}")
        exit()

    # --- B. 初始化 ---
    paths = get_path_manager()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # 创建本次运行的主输出目录
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_output_dir = paths.OUTPUT_ROOT / config.dataset.name / config.model.name / f'run_{run_timestamp}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"====== 本次交叉验证运行结果将保存至: {run_output_dir} ======")

    # --- C. 执行留一法交叉验证 (LOSOCV) 循环 ---
    all_subjects = config.dataset.all_subjects
    results = []
    print("====== 开始留一法交叉验证 (LOSOCV) ======")
    for subject_to_test in all_subjects:
        acc, f1 = run_one_fold(config, subject_to_test, all_subjects, paths, run_output_dir)
        results.append({'subject': subject_to_test, 'accuracy': acc, 'f1_score': f1})

    # --- D. 汇总并打印/保存最终结果 ---
    print("\n\n====== 留一法交叉验证全部完成 ======")

    # 提取所有折叠的结果
    all_fold_accs = [r['accuracy'] for r in results]
    all_fold_f1s = [r['f1_score'] for r in results]

    summary_file = run_output_dir / 'cv_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"实验配置:\n{OmegaConf.to_yaml(exp_conf)}\n")
        f.write("\n每个折叠的详细结果:\n")
        for res in results:
            f.write(f"  - 测试 {res['subject']}: Accuracy = {res['accuracy']:.4f}, F1-score = {res['f1_score']:.4f}\n")
        f.write("\n最终平均性能:\n")
        f.write(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}\n")
        f.write(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}\n")

    print(f"交叉验证汇总结果已保存至: {summary_file}")
    # 同时在终端打印最终结果
    print("\n--- 最终平均性能 ---")
    print(f"平均准确率 (Accuracy): {np.mean(all_fold_accs):.4f} ± {np.std(all_fold_accs):.4f}")
    print(f"平均 F1 分数 (Weighted F1-score): {np.mean(all_fold_f1s):.4f} ± {np.std(all_fold_f1s):.4f}")

if __name__ == '__main__':
    main()