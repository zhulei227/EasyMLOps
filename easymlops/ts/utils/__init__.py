# -*- coding: utf-8 -*-
"""
TS 评估与切分工具(纯 numpy/pandas, 零依赖)。
"""
import numpy as np
import pandas as pd


def calc_mape(y_true, y_pred):
    """MAPE(忽略 y_true=0 的点)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def calc_smape(y_true, y_pred):
    """对称 MAPE, 取值 [0, 2]"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    mask = denominator != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]))


def calc_mase(y_true, y_pred, y_train, m=1):
    """
    MASE: 模型 MAE / 训练集 naive 预测(m 步滞后差分)的 MAE。
    MASE < 1 说明优于训练集 naive 基线。
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if len(y_train) <= m:
        return float("nan")
    naive_mae = float(np.mean(np.abs(y_train[m:] - y_train[:-m])))
    if naive_mae == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred))) / naive_mae


def split_time_series(data, time_col=None, group_cols=None, test_size=0.2):
    """
    按时间切分时序数据(防泄漏): 组内按时间排序后, 前 (1-test_size) 为 train,
    尾部 test_size 为 test; 同一实体的样本不会跨越 train/test 的时间边界。

    Args:
        data: DataFrame
        time_col: 时间列名; None 时假定 data 已按时间有序
        group_cols: 分组列名(1个或多个); None 时整体按单序列切分
        test_size: 测试集比例(0, 1)

    Returns:
        (train_df, test_df): 保留原始 index
    """
    data = pd.DataFrame(data)
    if not 0 < test_size < 1:
        raise Exception("test_size must be in (0, 1)")
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols or [])

    def split_one(view):
        if time_col is not None and time_col in view.columns:
            view = view.sort_values(time_col, kind="stable")
        cut = max(1, int(len(view) * (1 - test_size)))
        return view.iloc[:cut], view.iloc[cut:]

    if not group_cols:
        train, test = split_one(data)
        return train, test

    train_parts, test_parts = [], []
    for _, view in data.groupby(group_cols, dropna=False, sort=False):
        tr, te = split_one(view)
        train_parts.append(tr)
        test_parts.append(te)
    train = pd.concat(train_parts) if train_parts else data.iloc[:0]
    test = pd.concat(test_parts) if test_parts else data.iloc[:0]
    return train, test


def calc_adf(series):
    """
    ADF 单位根检验(懒加载 statsmodels)。H0: 存在单位根(非平稳)。

    Returns:
        dict: {"stat", "pvalue", "stationary"} stationary = pvalue < 0.05
    """
    from statsmodels.tsa.stattools import adfuller
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    stat, pvalue, _, _, _, _ = adfuller(values)
    return {"stat": float(stat), "pvalue": float(pvalue), "stationary": bool(pvalue < 0.05)}


def calc_kpss(series):
    """
    KPSS 平稳性检验(懒加载 statsmodels)。H0: 平稳(与 ADF 方向相反)。

    Returns:
        dict: {"stat", "pvalue", "stationary"} stationary = pvalue >= 0.05
    """
    import warnings
    from statsmodels.tsa.stattools import kpss
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = kpss(values, regression="c", nlags="auto")
    stat, pvalue = float(result[0]), float(result[1])
    return {"stat": stat, "pvalue": pvalue, "stationary": bool(pvalue >= 0.05)}


def calc_acf_pacf(series, nlags=10):
    """
    自相关/偏自相关(懒加载 statsmodels), 用于定 ARIMA 的 q/p 与 seasonal_period。

    Returns:
        (acf_values, pacf_values): numpy 数组, 下标 0 为 lag 0
    """
    from statsmodels.tsa.stattools import acf, pacf
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    acf_values = acf(values, nlags=nlags)
    pacf_values = pacf(values, nlags=nlags)
    return np.asarray(acf_values), np.asarray(pacf_values)


def walk_forward_eval(data, fit_predict_func, target_col, time_col=None,
                      group_cols=None, min_train_size=10, horizon=1):
    """
    滚动回测: 组内按时间展开, 每次用 [0, start) 拟合、预测 [start, start+horizon),
    start 按 horizon 步进; 收集全部折叠的真实值/预测值。

    Args:
        data: DataFrame
        fit_predict_func: callable(train_df, test_df) -> 长度=len(test_df) 的预测序列
        target_col: 真实值列名(用于对齐 y_true)
        time_col: 时间列名; None 时假定已按时间有序
        group_cols: 分组列; None 时整体单序列回测
        min_train_size: 首折最小训练样本数
        horizon: 每折预测步长

    Returns:
        DataFrame: [fold, *group_cols, y_true, y_pred], 每个预测点一行
    """
    data = pd.DataFrame(data)
    group_cols = [group_cols] if isinstance(group_cols, str) else list(group_cols or [])
    if min_train_size < 1:
        raise Exception("min_train_size must be >= 1")
    if horizon < 1:
        raise Exception("horizon must be >= 1")

    def iter_views():
        if not group_cols:
            view = data
            if time_col is not None and time_col in view.columns:
                view = view.sort_values(time_col, kind="stable")
            yield (None,), view
            return
        for raw_key, view in data.groupby(group_cols, dropna=False, sort=False):
            if time_col is not None and time_col in view.columns:
                view = view.sort_values(time_col, kind="stable")
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            yield key, view

    records = []
    for key, view in iter_views():
        start = min_train_size
        fold = 0
        while start < len(view):
            test_view = view.iloc[start:start + horizon]
            train_view = view.iloc[:start]
            preds = np.asarray(fit_predict_func(train_view, test_view), dtype=float)
            if len(preds) != len(test_view):
                raise Exception("fit_predict_func returned {} preds, expect {}".format(
                    len(preds), len(test_view)))
            for j in range(len(test_view)):
                record = {"fold": fold, "y_true": float(test_view[target_col].iloc[j]),
                          "y_pred": float(preds[j])}
                for c, k in zip(group_cols, key):
                    record[c] = k
                records.append(record)
            start += horizon
            fold += 1
    return pd.DataFrame(records)


def calc_sample_entropy(series, m=2, r=None):
    """
    样本熵: 序列复杂度, 越大越不规则(随机噪声 > 周期信号)。纯 numpy 实现。

    Args:
        series: 一维序列
        m: 模板长度, 默认 2
        r: 相似度容差, 默认 0.2 * 序列标准差
    """
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(values)
    if n <= m + 1:
        return float("nan")
    if r is None:
        r = 0.2 * float(np.std(values, ddof=0))
    if r == 0:
        return float("nan")

    def _count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            template = values[i:i + template_len]
            rest = values[i + 1:n - template_len + 1]
            if len(rest) == 0:
                continue
            distances = np.max(np.abs(rest - template), axis=1) if rest.ndim > 1 \
                else np.abs(rest - template[:len(rest)])
            count += int(np.sum(distances <= r))
        return count

    # 向量化: 构造长度 m 与 m+1 的模板矩阵比较
    def _phi(length):
        emb = np.asarray([values[i:i + length] for i in range(n - length + 1)])
        count = 0
        for i in range(len(emb) - 1):
            dist = np.max(np.abs(emb[i + 1:] - emb[i]), axis=1)
            count += int(np.sum(dist <= r))
        return count

    denom = _phi(m)
    numer = _phi(m + 1)
    if denom == 0 or numer == 0:
        return float("nan")
    return float(-np.log(numer / denom))


def calc_hurst(series, min_lag=2, max_lag=20):
    """
    Hurst 指数(R/S 重标极差分析简化版): ~0.5 随机游走, <0.5 均值回复, >0.5 趋势持续。
    纯 numpy 实现。

    对每个窗口长度 lag: 将序列切为若干段, 计算各段累积离差的极差 R 与标准差 S 之比
    并取均值, 拟合 log(R/S) ~ log(lag) 的斜率即为 H。
    """
    values = pd.Series(series).dropna().to_numpy(dtype=float)
    n = len(values)
    if n < 2 * min_lag:
        return float("nan")
    upper = min(max_lag if max_lag is not None else n // 4, n // 2)
    if upper <= min_lag:
        return float("nan")
    log_lags, log_rs = [], []
    for lag in range(int(min_lag), int(upper) + 1):
        n_chunks = n // lag
        chunk = values[:n_chunks * lag].reshape(n_chunks, lag)
        dev = np.cumsum(chunk - chunk.mean(axis=1, keepdims=True), axis=1)
        ranges = dev.max(axis=1) - dev.min(axis=1)
        stds = chunk.std(axis=1, ddof=0)
        valid = stds > 0
        if not valid.any():
            continue
        rs = float(np.mean(ranges[valid] / stds[valid]))
        if rs > 0:
            log_lags.append(np.log(lag))
            log_rs.append(np.log(rs))
    if len(log_lags) < 2:
        return float("nan")
    reg = np.polyfit(log_lags, log_rs, 1)
    return float(reg[0])
