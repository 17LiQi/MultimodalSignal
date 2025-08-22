# utils/report_utils.py
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def save_results(results: dict, output_dir: Path):
    """
    接收评估结果字典，生成并保存所有报告和图表。
    """
    y_true = results['labels']
    y_pred = results['preds']

    # 1. 保存原始预测结果，以便未来分析
    np.save(output_dir / 'predictions.npy', y_pred)
    np.save(output_dir / 'true_labels.npy', y_true)

    # 2. 生成并保存分类报告
    class_names = ['Neutral', 'Stress', 'Amusement', 'Meditation']
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    report_path = output_dir / 'classification_report.txt'
    with open(report_path, 'w') as f:
        f.write(f"===== Final Test Set Performance =====\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n")
        f.write(f"Weighted F1-score: {f1:.4f}\n\n")
        f.write("--- Classification Report ---\n")
        f.write(report)
    print(f"详细分类报告已保存至: {report_path}")

    # 3. 生成并保存混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    cm_path = output_dir / 'confusion_matrix.png'
    plt.savefig(cm_path)
    plt.close()
    print(f"混淆矩阵已保存至: {cm_path}")