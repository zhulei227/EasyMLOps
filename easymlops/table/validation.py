# -*- coding: utf-8 -*-
"""
分组交叉验证评估工具（v0.2.7 新增，源自 Kaggle: group-aware CV scheme）。

适用于"同一实体的样本必须落在同一折"的场景（用户级/设备级/门店级防泄漏）。
"""
import numpy as np
import pandas as pd

dataframe_type = pd.DataFrame


def group_kfold_eval(build_model, x: dataframe_type, y, groups, cv: int = 5,
                     metric_fn=None, task: str = "regression",
                     random_state: int = 42) -> dict:
    """
    GroupKFold 交叉验证评估。

    Args:
        build_model: callable(fold_y: pd.Series) -> 模型 pipe
                     （返回已绑定折标签的未训练模型，如
                       lambda fold_y: LGBMRegression(y=fold_y, use_faster_predictor=False)）
        x: 特征 DataFrame
        y: 目标变量
        groups: 分组列（与 x 行对齐，同一组值不会同时出现在训练与验证折）
        cv: 折数
        metric_fn: callable(y_true, y_pred_values) -> float（默认回归=RMSE，二分类=AUC）
        task: "regression" / "classification"（决定默认 metric 与预测提取方式）
        random_state: 保留参数位（GroupKFold 本身无 shuffle）

    Returns:
        dict: {"scores": [每折得分], "mean": 均值, "std": 标准差}

    Example:
        >>> from easymlops.table.validation import group_kfold_eval
        >>> result = group_kfold_eval(
        ...     build_model=lambda fold_y: LGBMRegression(y=fold_y, use_faster_predictor=False),
        ...     x=train_df, y=y_train, groups=user_ids, cv=5)
        >>> print(result["mean"], result["std"])
    """
    from sklearn.model_selection import GroupKFold

    x = x.reset_index(drop=True)
    y = pd.Series(y).reset_index(drop=True)
    groups = pd.Series(groups).reset_index(drop=True)
    if len(x) != len(y) or len(x) != len(groups):
        raise Exception("x/y/groups must be aligned (same length)")

    def default_metric(y_true, y_pred_values):
        if task == "classification":
            from sklearn.metrics import roc_auc_score
            return roc_auc_score(y_true, y_pred_values)
        return float(np.sqrt(np.mean((y_pred_values - y_true) ** 2)))

    metric = metric_fn or default_metric
    splitter = GroupKFold(n_splits=cv)
    scores = []
    for train_idx, valid_idx in splitter.split(x, y, groups):
        model = build_model(y.iloc[train_idx])
        model.fit(x.iloc[train_idx].reset_index(drop=True))
        pred = model.transform(x.iloc[valid_idx].reset_index(drop=True))
        if task == "classification":
            # 二分类: 取类别 1 的概率列; 其余取首列
            values = None
            for col in (1, "1"):
                if col in pred.columns:
                    values = pred[col].to_numpy(dtype=float)
                    break
            if values is None:
                values = pred.iloc[:, 0].to_numpy(dtype=float)
        else:
            values = pred.iloc[:, 0].to_numpy(dtype=float)
        scores.append(float(metric(y.iloc[valid_idx].to_numpy(), values)))
    return {"scores": scores, "mean": float(np.mean(scores)), "std": float(np.std(scores))}
