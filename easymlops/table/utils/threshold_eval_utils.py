# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calc_precision_recall_at_thresholds(y_true, y_pred, bins=10):
    """
    计算不同预测阈值下的累计样本量、precision、recall、F1。

    Args:
        y_true: 真实标签，1表示正样本，0表示负样本
        y_pred: 预测概率值或预测分数
        bins: 分箱数量，默认为10

    Returns:
        DataFrame: 包含以下列
            - threshold: 阈值（从高到低）
            - cumulative_samples: 累计样本量（预测值>=阈值的样本数）
            - tp: 真阳性数量（预测为正且实际为正）
            - fp: 假阳性数量（预测为正但实际为负）
            - fn: 假阴性数量（预测为负但实际为正）
            - precision: 精确率 = TP / (TP + FP)
            - recall: 召回率 = TP / (TP + FN)
            - f1: F1分数 = 2 * precision * recall / (precision + recall)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.6, 0.4])
        >>> result = calc_precision_recall_at_thresholds(y_true, y_pred, bins=5)
        >>> print(result)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    unique_preds = np.unique(y_pred)
    if len(unique_preds) <= bins:
        thresholds = sorted(unique_preds, reverse=True)
    else:
        thresholds = np.percentile(y_pred, np.linspace(100, 0, bins + 1))
        thresholds = np.unique(thresholds)[::-1]

    results = []

    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))

        cumulative_samples = tp + fp

        if cumulative_samples == 0:
            precision = 0.0
        else:
            precision = tp / cumulative_samples

        actual_positives = tp + fn
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = tp / actual_positives

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        results.append({
            "threshold": threshold,
            "cumulative_samples": cumulative_samples,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)


def calc_roc_at_thresholds(y_true, y_pred, bins=10):
    """
    计算不同预测阈值下的ROC指标。

    Args:
        y_true: 真实标签，1表示正样本，0表示负样本
        y_pred: 预测概率值或预测分数
        bins: 分箱数量，默认为10

    Returns:
        DataFrame: 包含以下列
            - threshold: 阈值（从高到低）
            - fpr: 假阳性率 = FP / (FP + TN)
            - tpr: 真阳性率 = TP / (TP + FN)
            - tnr: 真阴性率 = TN / (TN + FP)
            - fnr: 假阴性率 = FN / (FN + TP)

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.6, 0.4])
        >>> result = calc_roc_at_thresholds(y_true, y_pred, bins=5)
        >>> print(result)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    unique_preds = np.unique(y_pred)
    if len(unique_preds) <= bins:
        thresholds = sorted(unique_preds, reverse=True)
    else:
        thresholds = np.percentile(y_pred, np.linspace(100, 0, bins + 1))
        thresholds = np.unique(thresholds)[::-1]

    results = []

    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        results.append({
            "threshold": threshold,
            "fpr": fpr,
            "tpr": tpr,
            "tnr": tnr,
            "fnr": fnr
        })

    return pd.DataFrame(results)


def plot_roc_curve(y_true, y_preds_dict, bins=10, figsize=(10, 8), title="ROC Curve", save_path=None):
    """
    绘制多个模型的ROC曲线。

    Args:
        y_true: 真实标签，1表示正样本，0表示负样本
        y_preds_dict: 字典，key为模型名称，value为预测概率数组
                    例如: {"Model_A": y_pred_a, "Model_B": y_pred_b}
        bins: 分箱数量，默认为10
        figsize: 图像大小，默认为(10, 8)
        title: 图像标题，默认为"ROC Curve"
        save_path: 保存路径，如果为None则不保存

    Returns:
        dict: 每个模型的AUC值

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        >>> y_preds = {
        ...     "Model_A": np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.6, 0.4]),
        ...     "Model_B": np.array([0.7, 0.2, 0.6, 0.5, 0.4, 0.3, 0.75, 0.25, 0.5, 0.35])
        ... }
        >>> aucs = plot_roc_curve(y_true, y_preds, bins=5)
        >>> print(aucs)
    """
    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    auc_results = {}

    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

    for idx, (model_name, y_pred) in enumerate(y_preds_dict.items()):
        result_df = calc_roc_at_thresholds(y_true, y_pred, bins=bins)

        fprs = result_df['fpr'].values
        tprs = result_df['tpr'].values

        sort_idx = np.argsort(fprs)
        fprs_sorted = fprs[sort_idx]
        tprs_sorted = tprs[sort_idx]

        fprs_sorted = np.concatenate([[0], fprs_sorted, [1]])
        tprs_sorted = np.concatenate([[0], tprs_sorted, [1]])

        auc = np.trapz(tprs_sorted, fprs_sorted)

        color = colors[idx % len(colors)]
        plt.plot(fprs_sorted, tprs_sorted, label=f'{model_name} (AUC={auc:.4f})', linewidth=2, color=color)

        auc_results[model_name] = auc

    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return auc_results


def plot_pr_curve(y_true, y_preds_dict, bins=10, figsize=(10, 8), title="P-R Curve", save_path=None):
    """
    绘制多个模型的P-R曲线。

    Args:
        y_true: 真实标签，1表示正样本，0表示负样本
        y_preds_dict: 字典，key为模型名称，value为预测概率数组
                    例如: {"Model_A": y_pred_a, "Model_B": y_pred_b}
        bins: 分箱数量，默认为10
        figsize: 图像大小，默认为(10, 8)
        title: 图像标题，默认为"P-R Curve"
        save_path: 保存路径，如果为None则不保存

    Returns:
        dict: 每个模型的最佳F1分数和对应的阈值

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        >>> y_preds = {
        ...     "Model_A": np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.6, 0.4]),
        ...     "Model_B": np.array([0.7, 0.2, 0.6, 0.5, 0.4, 0.3, 0.75, 0.25, 0.5, 0.35])
        ... }
        >>> best_scores = plot_pr_curve(y_true, y_preds, bins=5)
        >>> print(best_scores)
    """
    plt.figure(figsize=figsize)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    best_results = {}

    for idx, (model_name, y_pred) in enumerate(y_preds_dict.items()):
        result_df = calc_precision_recall_at_thresholds(y_true, y_pred, bins=bins)

        recalls = result_df['recall'].values
        precisions = result_df['precision'].values

        sort_idx = np.argsort(recalls)
        recalls_sorted = recalls[sort_idx]
        precisions_sorted = precisions[sort_idx]

        recalls_sorted = np.concatenate([[0], recalls_sorted, [1]])
        precisions_sorted = np.concatenate([[1], precisions_sorted, [0]])

        color = colors[idx % len(colors)]
        plt.plot(recalls_sorted, precisions_sorted, label=model_name, linewidth=2, color=color)

        best_f1_idx = result_df['f1'].idxmax()
        best_f1 = result_df.loc[best_f1_idx, 'f1']
        best_threshold = result_df.loc[best_f1_idx, 'threshold']
        best_recall = result_df.loc[best_f1_idx, 'recall']
        best_precision = result_df.loc[best_f1_idx, 'precision']

        plt.scatter([best_recall], [best_precision], s=100, zorder=5, color=color, marker='o')
        plt.annotate(f'F1={best_f1:.3f}\nth={best_threshold:.3f}',
                    xy=(best_recall, best_precision),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=9, color=color)

        best_results[model_name] = {
            'best_f1': best_f1,
            'best_threshold': best_threshold,
            'best_precision': best_precision,
            'best_recall': best_recall
        }

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

    return best_results


def calc_precision_recall_at_quantiles(y_true, y_pred, quantiles=10):
    """
    计算不同分位数阈值下的累计样本量、precision、recall、F1。

    与 calc_precision_recall_at_thresholds 不同的是，此函数基于分位数
    划分阈值，确保每个阈值区间的样本量大致相等。

    Args:
        y_true: 真实标签，1表示正样本，0表示负样本
        y_pred: 预测概率值或预测分数
        quantiles: 分位数数量，默认为10

    Returns:
        DataFrame: 包含阈值、累计样本量、precision、recall、F1

    Example:
        >>> import numpy as np
        >>> np.random.seed(42)
        >>> y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
        >>> y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.6, 0.4])
        >>> result = calc_precision_recall_at_quantiles(y_true, y_pred, quantiles=5)
        >>> print(result)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    if len(y_true) == 0:
        raise ValueError("y_true and y_pred cannot be empty")

    percentiles = np.linspace(100, 0, quantiles + 1)
    thresholds = np.percentile(y_pred, percentiles)
    thresholds = np.unique(thresholds)[::-1]

    results = []

    for threshold in thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)

        tp = np.sum((y_pred_binary == 1) & (y_true == 1))
        fp = np.sum((y_pred_binary == 1) & (y_true == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true == 0))

        cumulative_samples = tp + fp

        if cumulative_samples == 0:
            precision = 0.0
        else:
            precision = tp / cumulative_samples

        actual_positives = tp + fn
        if actual_positives == 0:
            recall = 0.0
        else:
            recall = tp / actual_positives

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)

        results.append({
            "threshold": threshold,
            "cumulative_samples": cumulative_samples,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    return pd.DataFrame(results)
