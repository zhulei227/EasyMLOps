# -*- coding: utf-8 -*-
"""
分布漂移监控 Pipe：对比 transform 数据与 fit 基准数据的逐列 PSI。

不改变数据（透传），可直接挂在主链，也可作为 branch pipe 挂载：

    table.pipe(DriftMonitorPipe(psi_threshold=0.25))          # 主链透传
    pipe.set_branch_pipe(DriftMonitorPipe())                  # 支线监控（不影响输出）

fit 时记录基准分箱分布；每次批量 transform 时计算逐列 PSI 并记录：
    - pipe.last_psi_report_: {col: psi} 最近一次批量结果
    - pipe.psi_history_:     历次批量结果列表（不随 save/load 持久化）
    - pipe.last_drift_cols_: 最近一次 PSI 超过阈值的列

单条 transform_single 不计算 PSI（无统计意义），仅透传。
"""
import numpy as np
import pandas as pd

from easymlops.table.core import TablePipeObjectBase, dataframe_type, dict_type


def calc_psi_by_dist(expected_dist, actual_dist):
    """对已归一化的两个离散分布计算 PSI（平滑至 _EPS 以上）。"""
    _eps = 1e-6
    expected = np.asarray(expected_dist, dtype=float)
    actual = np.asarray(actual_dist, dtype=float)
    assert len(expected) == len(actual)
    expected = np.where(expected <= _eps, _eps, expected)
    actual = np.where(actual <= _eps, _eps, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


class DriftMonitorPipe(TablePipeObjectBase):
    """
    逐列 PSI 漂移监控（透传）。

    Example:
        >>> table.pipe(FixInput()).pipe(DriftMonitorPipe(bins=10, psi_threshold=0.25))
        >>> table.fit(x_train)
        >>> table.transform(x_online)
        >>> print(table[DriftMonitorPipe.__name__].last_psi_report_)
    """

    def __init__(self, cols=None, bins=10, psi_threshold=0.25, **kwargs):
        """
        Args:
            cols: 监控的数值列；None 表示 fit 时自动选择全部数值列
            bins: PSI 分箱数量，默认10
            psi_threshold: 漂移判定阈值，默认0.25
        """
        super().__init__(**kwargs)
        self.cols = cols
        self.bins = bins
        self.psi_threshold = psi_threshold
        self.bin_edges_ = {}
        self.expected_dist_ = {}
        self.expected_null_frac_ = {}
        self.last_psi_report_ = {}
        self.last_drift_cols_ = []
        self.psi_history_ = []

    def _column_dist(self, col, s: dataframe_type):
        """
        计算列 col 在数据 s 上基于基准分箱的分布。

        Returns:
            (finite_dist, null_frac): 有限值在各箱的占比数组 与 空值占比
        """
        values = pd.to_numeric(s[col], errors="coerce")
        null_frac = float(values.isna().mean())
        edges = self.bin_edges_[col]
        finite = values.dropna().to_numpy(dtype=float)
        if len(edges) < 2 or len(finite) == 0:
            return np.array([1.0]) if len(finite) > 0 else np.array([]), null_frac
        clipped = np.clip(finite, edges[0], edges[-1])
        dist = np.histogram(clipped, bins=edges)[0] / len(values)
        return dist, null_frac

    def udf_fit(self, s: dataframe_type, **kwargs):
        if self.cols is None:
            self.cols = [col for col in s.columns
                         if "int" in str(s[col].dtype).lower() or "float" in str(s[col].dtype).lower()]
        if type(self.cols) not in (list, tuple):
            self.cols = [self.cols]
        for col in self.cols:
            values = pd.to_numeric(s[col], errors="coerce")
            self.expected_null_frac_[col] = float(values.isna().mean())
            finite = values.dropna().to_numpy(dtype=float)
            if len(finite) == 0:
                self.bin_edges_[col] = np.array([])
                self.expected_dist_[col] = np.array([1.0])
            else:
                edges = np.unique(np.quantile(finite, np.linspace(0, 1, self.bins + 1)))
                self.bin_edges_[col] = edges
                dist, _ = self._column_dist(col, s)
                self.expected_dist_[col] = dist
        return self

    def _column_psi(self, col, s: dataframe_type):
        """单列 PSI：值分布 + 空值率（作为额外类别并入比较）。"""
        actual_dist, actual_null = self._column_dist(col, s)
        expected_dist = self.expected_dist_[col]
        if len(actual_dist) == 0 or len(expected_dist) == 0:
            return 0.0
        if len(actual_dist) != len(expected_dist):
            # 依据基准箱数补齐（理论上一致，防御性处理）
            n = max(len(actual_dist), len(expected_dist))
            expected_dist = np.pad(expected_dist, (0, n - len(expected_dist)))
            actual_dist = np.pad(actual_dist, (0, n - len(actual_dist)))
        psi_value = calc_psi_by_dist(expected_dist, actual_dist)
        e_null = max(self.expected_null_frac_[col], 1e-6)
        a_null = max(actual_null, 1e-6)
        psi_value += (a_null - e_null) * np.log(a_null / e_null)
        return float(psi_value)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        report = {}
        for col in self.cols:
            if col in s.columns:
                report[col] = self._column_psi(col, s)
        self.last_psi_report_ = report
        self.last_drift_cols_ = sorted([col for col, psi in report.items() if psi > self.psi_threshold])
        self.psi_history_.append(dict(report))
        if len(self.last_drift_cols_) > 0:
            print("({}) drift detected, psi > {} in cols: {}".format(
                self.name, self.psi_threshold, self.last_drift_cols_))
        return s

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "bins": self.bins, "psi_threshold": self.psi_threshold,
                "bin_edges_": self.bin_edges_, "expected_dist_": self.expected_dist_,
                "expected_null_frac_": self.expected_null_frac_}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.bins = params["bins"]
        self.psi_threshold = params["psi_threshold"]
        self.bin_edges_ = params["bin_edges_"]
        self.expected_dist_ = params["expected_dist_"]
        self.expected_null_frac_ = params["expected_null_frac_"]
