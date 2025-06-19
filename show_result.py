import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_results(csv_path):
    results = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "image": row["image"],
                "true_label": row["true_label"],
                "predicted_label": row["predicted_label"],
                "confidence_ratio": float(-1 if row["confidence_ratio"] == "inf" else row["confidence_ratio"]),
                "distance": float(row["distance"]),
            })
    return results

def compute_fbeta(tp, fp, fn, beta=0.5):
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    fbeta = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-6)
    return precision, recall, fbeta

def find_best_confidence_ratio_threshold(pos_csv="positive_results.csv", neg_csv="negative_results.csv", beta=0.5):
    # 加载正样本（系统内身份）和负样本（系统外身份）
    pos_results = load_results(pos_csv)
    neg_results = load_results(neg_csv)

    # 获取所有可能的阈值范围
    all_confidences = [r["confidence_ratio"] for r in pos_results + neg_results if r["confidence_ratio"] >= 0]
    log_min = np.log(np.min(all_confidences))
    log_max = np.log(np.max(all_confidences))
    log_thresholds = np.linspace(log_min, log_max, num=1000)
    thresholds = np.exp(log_thresholds)

    best_fbeta = -1
    best_threshold = None
    best_metrics = None

    precisions = []
    recalls = []

    for threshold in thresholds:
        TP = FP = FN = TN = 0

        # 正样本判断
        for r in pos_results:
            if r["confidence_ratio"] >= threshold:
                if r["predicted_label"] == r["true_label"]:
                    TP += 1
                else:
                    FN += 1  # confidence够但识错了
            else:
                FN += 1  # 被拒识

        # 负样本判断
        for r in neg_results:
            if r["confidence_ratio"] >= threshold:
                FP += 1  # 错误地接受
            else:
                TN += 1  # 拒识成功

        precision, recall, fbeta = compute_fbeta(TP, FP, FN, beta=beta)
        precisions.append(precision)
        recalls.append(recall)

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
            best_metrics = (TP, FP, FN, TN, precision, recall)

    # 输出最佳结果
    TP, FP, FN, TN, precision, recall = best_metrics
    print(f"最佳置信度阈值: {best_threshold:.4f}")
    print(f"F{beta}-score: {best_fbeta:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    # 绘制 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', label='PR Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)  # 显式设置 x 轴从 0 到 1
    plt.ylim(0, 1)  # 显式设置 y 轴从 0 到 1
    plt.tight_layout()
    plt.show()

    plot_confidence_ratio_distribution(pos_results, neg_results)

    return best_threshold, best_fbeta


def find_best_distance_threshold(pos_csv="positive_results.csv", neg_csv="negative_results.csv", beta=0.5):
    # 加载正样本（系统内身份）和负样本（系统外身份）
    pos_results = load_results(pos_csv)
    neg_results = load_results(neg_csv)

    # 获取所有可能的阈值范围
    all_confidences = [r["distance"] for r in pos_results + neg_results if r["distance"] >= 0]
    thresholds = np.linspace(min(all_confidences), max(all_confidences), num=1000)

    best_fbeta = -1
    best_threshold = None
    best_metrics = None

    precisions = []
    recalls = []

    for threshold in thresholds:
        TP = FP = FN = TN = 0

        # 正样本判断
        for r in pos_results:
            if r["distance"] <= threshold:
                if r["predicted_label"] == r["true_label"]:
                    TP += 1
                else:
                    FN += 1  # confidence够但识错了
            else:
                FN += 1  # 被拒识

        # 负样本判断
        for r in neg_results:
            if r["distance"] <= threshold:
                FP += 1  # 错误地接受
            else:
                TN += 1  # 拒识成功

        precision, recall, fbeta = compute_fbeta(TP, FP, FN, beta=beta)
        precisions.append(precision)
        recalls.append(recall)

        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
            best_metrics = (TP, FP, FN, TN, precision, recall)

    # 输出最佳结果
    TP, FP, FN, TN, precision, recall = best_metrics
    print(f"最佳置信度阈值: {best_threshold:.4f}")
    print(f"F{beta}-score: {best_fbeta:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}")

    # 绘制 PR 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, color='blue', label='PR Curve')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, 1)  # 显式设置 x 轴从 0 到 1
    plt.ylim(0, 1)  # 显式设置 y 轴从 0 到 1
    plt.tight_layout()
    plt.show()

    plot_distance_distribution(pos_results, neg_results)

    return best_threshold, best_fbeta


def plot_distance_distribution(pos_results, neg_results):
    pos_distances = [r["distance"] for r in pos_results]
    neg_distances = [r["distance"] for r in neg_results]

    plt.figure(figsize=(10, 6))

    # KDE 曲线
    sns.kdeplot(pos_distances, label='Positive Samples', color='blue', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(neg_distances, label='Negative Samples', color='red', fill=True, alpha=0.3, linewidth=2)

    # 可选：加直方图
    plt.hist(pos_distances, bins=50, color='blue', alpha=0.2, label='Positive Histogram', density=True)
    plt.hist(neg_distances, bins=50, color='red', alpha=0.2, label='Negative Histogram', density=True)

    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title("Distance Distribution of Positive and Negative Samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_confidence_ratio_distribution(pos_results, neg_results):
    pos_confidence_ratio = [np.log(r["confidence_ratio"]) for r in pos_results if r["confidence_ratio"] > 0]
    neg_confidence_ratio = [np.log(r["confidence_ratio"]) for r in neg_results if r["confidence_ratio"] > 0]

    plt.figure(figsize=(10, 6))

    # KDE 曲线
    sns.kdeplot(pos_confidence_ratio, label='Positive Samples', color='blue', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(neg_confidence_ratio, label='Negative Samples', color='red', fill=True, alpha=0.3, linewidth=2)

    # 可选：加直方图
    plt.hist(pos_confidence_ratio, bins=50, color='blue', alpha=0.2, label='Positive Histogram', density=True)
    plt.hist(neg_confidence_ratio, bins=50, color='red', alpha=0.2, label='Negative Histogram', density=True)

    plt.xlabel("Log Confidence Ratio")
    plt.ylabel("Density")
    plt.title("Log Confidence Ratio Distribution of Positive and Negative Samples")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    find_best_confidence_ratio_threshold()
    # find_best_distance_threshold()