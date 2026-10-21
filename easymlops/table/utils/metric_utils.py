# -*- coding: utf-8 -*-
"""
诊断评估工具：PSI(分布漂移)、KS(区分度)、Lift(提升度)、评分卡刻度转换。

均为纯 numpy/pandas 实现，无额外依赖，可直接用于离线分析与上线后监控。
"""
import numpy as np
import pandas as pd

_EPS = 1e-6


def calc_psi(expected, actual, bins=10):
    """
    计算两个分布之间的 PSI（Population Stability Index）。

    以 expected 的分位数作为分箱边界，比较 expected 与 actual 落在各箱中的占比。
    经验判断标准：PSI < 0.1 分布稳定；0.1~0.25 需关注；> 0.25 分布显著漂移。

    Args:
        expected: 基准分布（如训练集分数），一维数组或 Series
        actual: 待比较分布（如线上分数），一维数组或 Series
        bins: 分箱数量，默认10

    Returns:
        float: PSI 值，越小分布越接近

    Example:
        >>> calc_psi(train_scores, online_scores)   # < 0.1 视为稳定
    """
    expected = pd.to_numeric(pd.Series(expected), errors="coerce").dropna().to_numpy(dtype=float)
    actual = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy(dtype=float)
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    # clip 到边界范围内，保证两分布使用同一分箱
    expected_clipped = np.clip(expected, edges[0], edges[-1])
    actual_clipped = np.clip(actual, edges[0], edges[-1])
    expected_dist = np.histogram(expected_clipped, bins=edges)[0] / len(expected)
    actual_dist = np.histogram(actual_clipped, bins=edges)[0] / len(actual)
    expected_dist = np.where(expected_dist <= _EPS, _EPS, expected_dist)
    actual_dist = np.where(actual_dist <= _EPS, _EPS, actual_dist)
    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))


def calc_ks(y_true, y_score):
    """
    计算 KS（Kolmogorov-Smirnov）统计量，衡量分数对正负样本的区分能力。

    Args:
        y_true: 真实标签（0/1），一维数组或 Series
        y_score: 预测分数/概率（越大越可能为正），一维数组或 Series

    Returns:
        dict: {"ks": KS值, "ks_threshold": 达到最大KS的分数阈值}

    Example:
        >>> calc_ks(y_test, y_prob["1"])   # 风控场景 KS 通常 0.3 以上可用
    """
    y_true = np.asarray(pd.Series(y_true).reset_index(drop=True), dtype=float)
    y_score = np.asarray(pd.Series(y_score).reset_index(drop=True), dtype=float)
    assert len(y_true) == len(y_score) and len(y_true) > 0
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return {"ks": float("nan"), "ks_threshold": float("nan")}
    thresholds = np.unique(y_score)
    pos_cdf = np.array([np.mean(pos <= t) for t in thresholds])
    neg_cdf = np.array([np.mean(neg <= t) for t in thresholds])
    ks_idx = int(np.argmax(np.abs(pos_cdf - neg_cdf)))
    return {"ks": float(abs(pos_cdf[ks_idx] - neg_cdf[ks_idx])),
            "ks_threshold": float(thresholds[ks_idx])}


def calc_lift(y_true, y_score, bins=10):
    """
    按分数从高到低等频分箱，计算每箱的 badrate / lift / 累积捕获率。

    Args:
        y_true: 真实标签（0/1），一维数组或 Series
        y_score: 预测分数/概率（越大越可能为正），一维数组或 Series
        bins: 分箱数量，默认10

    Returns:
        DataFrame: 按箱序号(1为分数最高箱)排列，含 samples/bads/bad_rate/lift/cum_gain 列
    """
    y_true = np.asarray(pd.Series(y_true).reset_index(drop=True), dtype=float)
    y_score = np.asarray(pd.Series(y_score).reset_index(drop=True), dtype=float)
    assert len(y_true) == len(y_score) and len(y_true) > 0
    # 按分数降序排列后等频切分（rank 保证并列分数均匀分布到相邻箱）
    order = np.argsort(-y_score, kind="stable")
    ranks = np.empty(len(y_score), dtype=int)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # 分箱边界按 rank 等频划分
    bin_ids = np.minimum(((ranks - 1) * bins) // len(y_score), bins - 1)
    total_bad = y_true.sum()
    rows = []
    for b in range(bins):
        mask = bin_ids == b
        samples = int(mask.sum())
        bads = int(y_true[mask].sum())
        bad_rate = bads / samples if samples > 0 else float("nan")
        lift = (bad_rate / (total_bad / len(y_true))) if samples > 0 and total_bad > 0 else float("nan")
        rows.append({"bin": b + 1, "samples": samples, "bads": bads, "bad_rate": bad_rate, "lift": lift})
    result = pd.DataFrame(rows)
    result["cum_gain"] = result["bads"].cumsum() / total_bad if total_bad > 0 else float("nan")
    return result


def prob2score(prob, base_score=600, pdo=20, base_odds=1.0 / 19):
    """
    将坏样本概率转换为评分卡刻度分数（分数越高风险越低）。

    刻度定义：基准分 base_score 对应坏样本几率(base_odds)，
    几率每翻一倍(变坏)分数下降 pdo 分。

    Args:
        prob: 坏样本概率，一维数组或 Series
        base_score: 基准分数，默认600
        pdo: Points to Double the Odds，默认20
        base_odds: 基准坏样本几率(bad/good)，默认1/19（对应20:1good:bad）

    Returns:
        与 prob 同长度的 numpy 数组：评分卡分数

    Example:
        >>> prob2score(0.05)   # 低风险 -> 高分
        >>> prob2score(0.5)    # 高风险 -> 低分
    """
    prob = np.asarray(pd.Series(prob).reset_index(drop=True), dtype=float)
    odds = prob / (1 - prob)  # 坏样本几率
    odds = np.clip(odds, _EPS, None)
    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)
    return offset - factor * np.log(odds)
