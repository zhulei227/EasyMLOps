# -*- coding: utf-8 -*-
"""
类不平衡重采样工具。

由于 easymlops 的模型 pipe 在构造时绑定 y（如 LGBMClassification(y=y_train)），
行数会发生变化的重采样必须在构建 pipeline 之前完成。本工具返回对齐的 (x, y)，
用重采样结果构造模型 pipe 即可：

    x_rs, y_rs = resample(x_train, y_train, method="oversample")
    pipe.pipe(LGBMClassification(y=y_rs, use_faster_predictor=False))
"""
import numpy as np
import pandas as pd


def resample(x, y, method="oversample", random_state=42, smote_params=None):
    """
    对不平衡样本重采样，返回行数对齐的 (x, y)。

    Args:
        x: 特征数据 DataFrame
        y: 标签 Series（与 x 行对齐）
        method: 采样方式：
            - "oversample": 少数类随机过采样至多数类规模（纯 pandas 实现）
            - "undersample": 多数类随机欠采样至少数类规模（纯 pandas 实现）
            - "smote": SMOTE 合成采样，需安装 imblearn（懒加载）
        random_state: 随机种子
        smote_params: 传给 imblearn SMOTE 的参数 dict（method="smote" 时生效）

    Returns:
        (DataFrame, Series): 重采样后的特征与标签

    Example:
        >>> x_rs, y_rs = resample(x_train, y_train, method="smote")
        >>> LGBMClassification(y=y_rs, use_faster_predictor=False)
    """
    assert method in ("oversample", "undersample", "smote"), "method must be oversample/undersample/smote"
    y = pd.Series(y).reset_index(drop=True)
    x = pd.DataFrame(x).reset_index(drop=True)
    assert len(x) == len(y)
    rng = np.random.RandomState(random_state)

    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            raise ImportError("method='smote' requires imblearn, please: pip install imblearn")
        smote_params = dict() if smote_params is None else dict(smote_params)
        smote_params.setdefault("random_state", random_state)
        x_rs, y_rs = SMOTE(**smote_params).fit_resample(x, y)
        return pd.DataFrame(x_rs, columns=x.columns), pd.Series(y_rs, name=y.name)

    counts = y.value_counts()
    majority_label = counts.index[0]
    minority_label = counts.index[-1]
    if method == "oversample":
        minority_idx = np.where(y.values == minority_label)[0]
        n_add = int(counts[majority_label] - counts[minority_label])
        add_idx = rng.choice(minority_idx, size=n_add, replace=True)
        sample_idx = np.concatenate([np.arange(len(y)), add_idx])
    else:  # undersample
        majority_idx = np.where(y.values == majority_label)[0]
        keep_majority = rng.choice(majority_idx, size=int(counts[minority_label]), replace=False)
        minority_idx = np.where(y.values == minority_label)[0]
        sample_idx = np.concatenate([keep_majority, minority_idx])

    rng.shuffle(sample_idx)
    x_rs = x.iloc[sample_idx].reset_index(drop=True)
    y_rs = y.iloc[sample_idx].reset_index(drop=True)
    return x_rs, y_rs
