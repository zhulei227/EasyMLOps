# -*- coding: utf-8 -*-
"""
TS 基础操作: 时间索引、日期特征、时序填充、可逆变换(差分/标准化)。

约定:
    - 时间列经 time_col 指定; group_cols 提供分组(面板数据)能力, None=全局单序列
    - 批量 transform 为纯函数(不推进状态), 缓存仅在 transform_single 演进
    - 除 FixTimeInput 的去重/排序外, 所有操作保持行数与行序(默认), 兼容 auto_test
"""
from collections import deque

import numpy as np
import pandas as pd

from easymlops.ts.core.group_pipe import (GroupTSBase, PipeBase, normalize_cols,
                                          numeric_cols_of)

dataframe_type = pd.DataFrame
dict_type = dict


class FixTimeInput(GroupTSBase):
    """
    时间列规范化: 解析为 datetime(无法解析置 NaT), 可选组内去重与物理排序。

    注意:
        - 去重/物理排序会改变行数或行序, 默认关闭; 开启后该 pipe 不再满足
          auto_test 的行数/index 一致性检查(属离线整理管道, 请单独使用)
    """

    def __init__(self, time_col=None, group_cols=None, sort_rows=False,
                 drop_duplicate_time=False, **kwargs):
        super().__init__(group_cols=group_cols, time_col=time_col, sort_rows=sort_rows, **kwargs)
        self.drop_duplicate_time = drop_duplicate_time

    def _parse(self, s: dataframe_type) -> dataframe_type:
        if self.time_col is None or self.time_col not in s.columns:
            return s
        s = s.copy()
        s[self.time_col] = pd.to_datetime(s[self.time_col], errors="coerce")
        if self.drop_duplicate_time:
            subset = list(self.group_cols) + [self.time_col]
            s = s.drop_duplicates(subset=subset, keep="first")
        return self._finalize(s)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        return self._parse(s)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.time_col in s:
            s[self.time_col] = pd.to_datetime(s[self.time_col], errors="coerce")
        return s

    def udf_get_params(self) -> dict_type:
        return {"drop_duplicate_time": self.drop_duplicate_time}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.drop_duplicate_time = params["drop_duplicate_time"]


class DateFeatureExtract(PipeBase):
    """
    日期特征提取(逐行变换, 与分组无关, 签名保持干净)。

    Args:
        time_col: 时间列名; None 时自动取第一个 datetime64 列
        features: 特征子集, 可选 year/quarter/month/day/hour/minute/weekday/
                  weekofyear/dayofyear/is_weekend/is_month_start/is_month_end
        cyclical: 是否为 month/weekday/hour/minute 追加 sin/cos 周期编码列
        na_fill: 时间不可解析时特征填充值
    """

    SUPPORTED_FEATURES = ("year", "quarter", "month", "day", "hour", "minute", "weekday",
                          "weekofyear", "dayofyear", "is_weekend", "is_month_start",
                          "is_month_end")
    CYCLICAL_PERIODS = {"month": 12, "weekday": 7, "hour": 24, "minute": 60}

    def __init__(self, time_col=None, features=None, cyclical=True, na_fill=0.0, **kwargs):
        super().__init__(**kwargs)
        self.time_col = time_col
        self.features = list(features) if features else ["month", "day", "weekday", "hour", "is_weekend"]
        for f in self.features:
            if f not in self.SUPPORTED_FEATURES:
                raise Exception("DateFeatureExtract unsupported feature: {}, supported: {}".format(
                    f, self.SUPPORTED_FEATURES))
        self.cyclical = cyclical
        self.na_fill = na_fill
        self.feature_names = []

    def _resolve_time_col(self, s: dataframe_type):
        if self.time_col is not None:
            return self.time_col
        for c in s.columns:
            if "datetime" in str(s[c].dtype).lower():
                return c
        return None

    @staticmethod
    def _extract_features(ts: pd.Series, features) -> dict_type:
        dt = ts.dt
        getters = {
            "year": dt.year, "quarter": dt.quarter, "month": dt.month, "day": dt.day,
            "hour": dt.hour, "minute": dt.minute, "weekday": dt.weekday,
            "weekofyear": dt.isocalendar().week.astype(float), "dayofyear": dt.dayofyear,
            "is_weekend": (dt.weekday >= 5).astype(float),
            "is_month_start": dt.is_month_start.astype(float),
            "is_month_end": dt.is_month_end.astype(float),
        }
        return {f: getters[f].astype(float) for f in features}

    def udf_fit(self, s: dataframe_type, **kwargs):
        time_col = self._resolve_time_col(s)
        if time_col is None:
            return self
        names = list(self.features)
        if self.cyclical:
            for f in self.features:
                if f in self.CYCLICAL_PERIODS:
                    names.extend(["{}_sin".format(f), "{}_cos".format(f)])
        self.feature_names = names
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        time_col = self._resolve_time_col(s)
        if time_col is None or not self.feature_names:
            return s
        result = s.copy()
        ts = pd.to_datetime(s[time_col], errors="coerce")
        feats = self._extract_features(ts, self.features)
        for f in self.features:
            result[f] = feats[f].fillna(self.na_fill).to_numpy()
        if self.cyclical:
            for f, period in self.CYCLICAL_PERIODS.items():
                if f not in self.features:
                    continue
                v = feats[f].fillna(self.na_fill).to_numpy()
                result["{}_sin".format(f)] = np.sin(2 * np.pi * v / period)
                result["{}_cos".format(f)] = np.cos(2 * np.pi * v / period)
        return result

    def _features_from_timestamp(self, ts):
        if ts is None or pd.isna(ts):
            return {name: self.na_fill for name in self.feature_names}
        ts_series = pd.to_datetime(pd.Series([ts]), errors="coerce")
        feats = self._extract_features(ts_series, self.features)
        values = {}
        for f in self.features:
            v = feats[f].iloc[0]
            values[f] = self.na_fill if pd.isna(v) else float(v)
        if self.cyclical:
            for f, period in self.CYCLICAL_PERIODS.items():
                if f not in self.features:
                    continue
                v = values[f]
                values["{}_sin".format(f)] = float(np.sin(2 * np.pi * v / period))
                values["{}_cos".format(f)] = float(np.cos(2 * np.pi * v / period))
        return values

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        ts = None
        if self.time_col is not None and self.time_col in s:
            parsed = pd.to_datetime(s[self.time_col], errors="coerce")
            if not pd.isna(parsed):
                ts = parsed
        values = self._features_from_timestamp(ts)
        for name, v in values.items():
            s[name] = v
        return s

    def udf_get_params(self) -> dict_type:
        return {"time_col": self.time_col, "features": self.features,
                "cyclical": self.cyclical, "na_fill": self.na_fill,
                "feature_names": self.feature_names}

    def udf_set_params(self, params: dict_type):
        self.time_col = params["time_col"]
        self.features = params["features"]
        self.cyclical = params["cyclical"]
        self.na_fill = params["na_fill"]
        self.feature_names = params["feature_names"]


class TSFillNa(GroupTSBase):
    """
    时序感知缺失填充(组内计算, 不跨组)。

    Args:
        cols: 待填充数值列; None 表示全部数值列
        method: ffill(前向填充, 单条推理一致性最好) / linear(线性插值, 仅批量,
                单条推理退化为前向值) / value(固定值)
        fill_value: method="value" 的填充值, 也是其余方法的兜底值
        limit: 连续填充的最大条数
    """

    METHODS = ("ffill", "linear", "value")

    def __init__(self, cols=None, method="ffill", fill_value=0.0, limit=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        if method not in self.METHODS:
            raise Exception("TSFillNa method must be one of {}".format(self.METHODS))
        self.method = method
        self.fill_value = fill_value
        self.limit = limit
        self._fill_cache = {}  # (组键, 列) -> 最近一个非缺失值, 仅单条路径使用

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        cols = self.cols or numeric_cols_of(s)

        def fill_group(key, view):
            out = {}
            for col in cols:
                if col not in view.columns:
                    continue
                series = view[col].astype(float)
                if self.method == "ffill":
                    series = series.ffill(limit=self.limit)
                elif self.method == "linear":
                    series = series.interpolate(method="linear", limit=self.limit)
                series = series.fillna(self.fill_value)
                out[col] = series
            return pd.DataFrame(out, index=view.index)

        return self._apply_group_func(s, fill_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        for col in self.cols:
            if col not in s:
                continue
            try:
                value = float(s[col])
            except (TypeError, ValueError):
                value = np.nan
            if np.isnan(value):
                if self.method == "value":
                    value = self.fill_value
                else:
                    # ffill/linear 的单条推理: 使用组内最近一个非缺失值(因果近似)
                    value = self._fill_cache.get((key, col), self.fill_value)
            else:
                self._fill_cache[(key, col)] = value
            s[col] = value
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "method": self.method,
                "fill_value": self.fill_value, "limit": self.limit}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.cols = params["cols"]
        self.method = params["method"]
        self.fill_value = params["fill_value"]
        self.limit = params["limit"]

    def reset_cache_data(self):
        self._fill_cache = {}


class _TSChangeBase(GroupTSBase):
    """
    可逆相对变化基类: forward 计算相对 x_{t-periods} 的变化量, inverse 依据
    fit 时保存的组内末段原始值(level seed)把变化量流式还原为水平值。

    子类实现 _pair_change(prev, cur) 与 _pair_restore(base, change)。

    语义约定(与 zero 填充配套):
        - 组首/组边界变化量被 fillna_strategy 填充(zero 默认), 该行信息固有丢失,
          逆还原时该行输出 seed 水平, 后续行在此基础上链式还原
    """

    def __init__(self, cols=None, periods=1, inverse=False, fillna_strategy="zero", **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.periods = int(periods)
        self.inverse = inverse
        if fillna_strategy not in ("zero", "nan"):
            raise Exception("fillna_strategy must be zero/nan")
        self.fillna_strategy = fillna_strategy
        self.seed_levels_ = {}   # 组键 -> {col: [末段periods个原始值]}, 随 save/load 持久化
        self._raw_cache = {}     # forward 单条路径: (组键, col) -> deque(原始值)
        self._level_cache = {}   # inverse 单条路径: (组键, col) -> deque(水平值)

    def _pair_change(self, prev: float, cur: float) -> float:
        raise Exception("need to implement!")

    def _pair_restore(self, base: float, change: float) -> float:
        raise Exception("need to implement!")

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.cols = self.cols or numeric_cols_of(s)
        self.seed_levels_ = {}
        for key, view in self._iter_group_views(s):
            self.seed_levels_[key] = {}
            for col in self.cols:
                if col in view.columns:
                    values = view[col].astype(float).to_numpy()
                    self.seed_levels_[key][col] = [float(v) for v in values[-self.periods:]]
        self._reset_level_cache()
        return self

    def _reset_level_cache(self):
        self._level_cache = {}
        for key, col_seeds in self.seed_levels_.items():
            for col, seed in col_seeds.items():
                self._level_cache[(key, col)] = deque(seed, maxlen=self.periods)

    def _guard_change(self, change):
        """变化量兜底: inf -> 0; nan -> 按 fillna_strategy 填充"""
        if change is None or (isinstance(change, float) and np.isnan(change)):
            return 0.0 if self.fillna_strategy == "zero" else np.nan
        if isinstance(change, float) and np.isinf(change):
            return 0.0
        return change

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        cols = self.cols or numeric_cols_of(s)

        def change_group(key, view):
            out = {}
            for col in cols:
                if col not in view.columns:
                    continue
                values = view[col].astype(float).to_numpy()
                if not self.inverse:
                    changes = np.full(len(values), np.nan)
                    for i in range(self.periods, len(values)):
                        changes[i] = self._pair_change(values[i - self.periods], values[i])
                    changes = np.where(np.isinf(changes), 0.0, changes)
                    if self.fillna_strategy == "zero":
                        changes = np.where(np.isnan(changes), 0.0, changes)
                    out[col] = pd.Series(changes, index=view.index)
                else:
                    # 纯函数: 从 seed 重建(不推进缓存), 保证与单条路径逐值一致
                    seed = list(self.seed_levels_.get(key, {}).get(col, [0.0] * self.periods))
                    levels = deque(seed, maxlen=self.periods)
                    restored = []
                    for v in values:
                        base = levels[0] if len(levels) == self.periods else 0.0
                        new_level = self._pair_restore(base, 0.0 if pd.isna(v) else float(v))
                        levels.append(new_level)
                        restored.append(new_level)
                    out[col] = pd.Series(restored, index=view.index)
            return pd.DataFrame(out, index=view.index)

        return self._apply_group_func(s, change_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        for col in self.cols:
            if col not in s:
                continue
            try:
                value = float(s[col])
            except (TypeError, ValueError):
                value = np.nan
            if not self.inverse:
                dq = self._raw_cache.setdefault((key, col), deque(maxlen=self.periods))
                # 先取基值再追加: dq 中保存"当前值之前的 periods 个原始值"
                base = dq[0] if len(dq) == self.periods else None
                if base is None or np.isnan(base) or np.isnan(value):
                    change = 0.0 if self.fillna_strategy == "zero" else np.nan
                else:
                    change = self._guard_change(self._pair_change(base, value))
                dq.append(value)
                s[col] = change
            else:
                levels = self._level_cache.setdefault(
                    (key, col), deque(self.seed_levels_.get(key, {}).get(col, [0.0] * self.periods),
                                      maxlen=self.periods))
                base = levels[0] if len(levels) == self.periods else 0.0
                new_level = self._pair_restore(base, 0.0 if np.isnan(value) else value)
                levels.append(new_level)
                s[col] = new_level
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "periods": self.periods, "inverse": self.inverse,
                "fillna_strategy": self.fillna_strategy, "seed_levels_": self.seed_levels_}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.cols = params["cols"]
        self.periods = params["periods"]
        self.inverse = params["inverse"]
        self.fillna_strategy = params["fillna_strategy"]
        self.seed_levels_ = params["seed_levels_"]
        self._reset_level_cache()

    def reset_cache_data(self):
        self._raw_cache = {}
        self._reset_level_cache()


class TSDiff(_TSChangeBase):
    """
    一阶差分(可逆): diff_t = x_t - x_{t-periods}。

    Args:
        cols: 差分列; None 表示全部数值列
        periods: 差分滞后阶
        inverse: False=正向差分, True=反向还原(需 fit 过原始序列)
        fillna_strategy: 组首/组边界变化量的填充, zero(默认)/nan
    """

    def _pair_change(self, prev: float, cur: float) -> float:
        return cur - prev

    def _pair_restore(self, base: float, change: float) -> float:
        return base + change


class TSPctChange(_TSChangeBase):
    """
    百分比变化(可逆): change_t = x_t / x_{t-periods} - 1。
    prev=0 时变化量无定义(inf/nan -> 按 fillna_strategy 填充)。

    Args: 同 TSDiff
    """

    def udf_get_params(self) -> dict_type:
        return {}

    def _pair_change(self, prev: float, cur: float) -> float:
        if prev == 0:
            return np.inf if cur > 0 else (-np.inf if cur < 0 else np.nan)
        return cur / prev - 1.0

    def _pair_restore(self, base: float, change: float) -> float:
        return base * (1.0 + change)


class TSLogReturn(_TSChangeBase):
    """
    对数收益(可逆): change_t = ln(x_t) - ln(x_{t-periods})。要求序列为正。

    Args: 同 TSDiff
    """

    def udf_get_params(self) -> dict_type:
        return {}

    def _pair_change(self, prev: float, cur: float) -> float:
        if prev <= 0 or cur <= 0:
            return np.nan
        return float(np.log(cur) - np.log(prev))

    def _pair_restore(self, base: float, change: float) -> float:
        return base * float(np.exp(change))


class _TSScalerBase(GroupTSBase):
    """
    可逆标准化基类。per_group=False(默认)时使用全局统计;
    group_cols=None 时 per_group 天然失效, 走全局。
    """

    def __init__(self, cols=None, per_group=False, inverse=False,
                 unseen_group="global", **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.per_group = per_group
        self.inverse = inverse
        if unseen_group not in ("global", "error"):
            raise Exception("unseen_group must be global/error")
        self.unseen_group = unseen_group
        self.global_stats_ = {}   # col -> (a, b)
        self.group_stats_ = {}    # 组键 -> {col: (a, b)}

    def _fit_stats(self, values: np.ndarray):
        raise Exception("need to implement!")

    def _map_value(self, value: float, stats) -> float:
        raise Exception("need to implement!")

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.cols = self.cols or numeric_cols_of(s)
        self.global_stats_ = {}
        self.group_stats_ = {}
        for col in self.cols:
            if col in s.columns:
                self.global_stats_[col] = self._fit_stats(s[col].astype(float).to_numpy())
        if self.per_group and self.grouped:
            for key, view in self._iter_group_views(s):
                self.group_stats_[key] = {}
                for col in self.cols:
                    if col in view.columns:
                        self.group_stats_[key][col] = self._fit_stats(view[col].astype(float).to_numpy())
            self._warn_if_too_many_groups(len(self.group_stats_))
        return self

    def _stats_for(self, key, col):
        if self.per_group and self.grouped:
            stats = self.group_stats_.get(key, {}).get(col)
            if stats is not None:
                return stats
            if self.unseen_group == "error":
                raise Exception("unseen group {} for {}, set unseen_group='global' to fallback".format(key, col))
        return self.global_stats_[col]

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        cols = [c for c in self.cols if c in s.columns]

        def map_group(key, view):
            out = {}
            for col in cols:
                stats = self._stats_for(key, col)
                values = view[col].astype(float).to_numpy()
                mapped = np.array([self._map_value(v, stats) for v in values], dtype=float)
                out[col] = pd.Series(mapped, index=view.index)
            return pd.DataFrame(out, index=view.index)

        return self._apply_group_func(s, map_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        for col in self.cols:
            if col not in s:
                continue
            try:
                value = float(s[col])
            except (TypeError, ValueError):
                value = np.nan
            s[col] = self._map_value(value, self._stats_for(key, col))
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "per_group": self.per_group, "inverse": self.inverse,
                "unseen_group": self.unseen_group, "global_stats_": self.global_stats_,
                "group_stats_": self.group_stats_}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.cols = params["cols"]
        self.per_group = params["per_group"]
        self.inverse = params["inverse"]
        self.unseen_group = params["unseen_group"]
        self.global_stats_ = params["global_stats_"]
        self.group_stats_ = params["group_stats_"]


class TSZScoreScaler(_TSScalerBase):
    """Z-Score 标准化(可逆): z = (x - mean) / std"""

    def _fit_stats(self, values: np.ndarray):
        mean = float(np.nanmean(values)) if len(values) else 0.0
        std = float(np.nanstd(values, ddof=0)) if len(values) else 1.0
        return mean, (std if std > 1e-12 else 1.0)

    def _map_value(self, value: float, stats) -> float:
        mean, std = stats
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if self.inverse:
            return value * std + mean
        return (value - mean) / std


class TSMinMaxScaler(_TSScalerBase):
    """Min-Max 标准化(可逆): x' = (x - min) / (max - min)"""

    def _fit_stats(self, values: np.ndarray):
        vmin = float(np.nanmin(values)) if len(values) else 0.0
        vmax = float(np.nanmax(values)) if len(values) else 1.0
        return vmin, (vmax - vmin if vmax - vmin > 1e-12 else 1.0)

    def _map_value(self, value: float, stats) -> float:
        vmin, span = stats
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return np.nan
        if self.inverse:
            return value * span + vmin
        return (value - vmin) / span


class TimeResample(GroupTSBase):
    """
    时间重采样(离线整理管道): 组内按 time_col 重采样到目标频率并聚合。
    会改变行数, 勿纳入 auto_test 验证的 pipeline; transform_single 仅透传。

    Args:
        time_col: 时间列名(必填)
        rule: 重采样频率, 如 "D"/"12H"/"W"/"M"
        agg: 聚合方式, mean/sum/last/first/max/min
        cols: 参与聚合的数值列; None 表示全部数值列
    """

    AGGS = ("mean", "sum", "last", "first", "max", "min")

    def __init__(self, time_col=None, rule="D", agg="mean", cols=None, **kwargs):
        super().__init__(**kwargs)
        self.time_col = time_col
        self.rule = rule
        if agg not in self.AGGS:
            raise Exception("TimeResample agg must be one of {}".format(self.AGGS))
        self.agg = agg
        self.cols = normalize_cols(cols)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        assert self.time_col is not None, "TimeResample requires time_col"
        cols = self.cols or numeric_cols_of(s)
        cols = [c for c in cols if c in s.columns]
        parts = []
        for key, view in self._iter_group_views(s):
            grid = view.set_index(self.time_col)[cols].resample(self.rule).agg(self.agg)
            grid = grid.reset_index()
            for c in self.group_cols:
                if c in view.columns:
                    grid[c] = view[c].iloc[0]
            parts.append(grid)
        return pd.concat(parts, ignore_index=True) if parts else s.copy()

    def udf_get_params(self) -> dict_type:
        return {"time_col": self.time_col, "rule": self.rule, "agg": self.agg, "cols": self.cols}

    def udf_set_params(self, params: dict_type):
        self.time_col = params["time_col"]
        self.rule = params["rule"]
        self.agg = params["agg"]
        self.cols = params["cols"]


class TimeGapsFill(GroupTSBase):
    """
    时间网格补齐(离线整理管道): 组内重索引到 [最早, 最晚] 的完整规则网格,
    缺失时间点补行。会改变行数, 勿纳入 auto_test 验证的 pipeline;
    transform_single 仅透传。

    Args:
        time_col: 时间列名(必填)
        freq: 网格频率(须与数据规律一致, 如 "D"/"H")
        cols: 需补值的数值列; None 表示全部数值列
        fill_value: 缺失行数值填充; None=NaN, 数字=固定值, "ffill"=前向填充
        mark_col: 标记列名(1=补齐行, 0=原始行); None 表示不添加
    """

    def __init__(self, time_col=None, freq="D", cols=None, fill_value=None,
                 mark_col="__is_gap_fill__", **kwargs):
        super().__init__(**kwargs)
        self.time_col = time_col
        self.freq = freq
        self.cols = normalize_cols(cols)
        self.fill_value = fill_value
        self.mark_col = mark_col

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        assert self.time_col is not None, "TimeGapsFill requires time_col"
        cols = self.cols or numeric_cols_of(s)
        cols = [c for c in cols if c in s.columns]
        parts = []
        for key, view in self._iter_group_views(s):
            ts_index = pd.DatetimeIndex(view[self.time_col])
            full_idx = pd.date_range(ts_index.min(), ts_index.max(), freq=self.freq)
            grid = view.set_index(ts_index)[cols].reindex(full_idx)
            is_original = grid.index.isin(ts_index)
            if self.fill_value == "ffill":
                grid = grid.ffill()
            elif self.fill_value is not None:
                grid = grid.fillna(self.fill_value)
            grid = grid.reset_index().rename(columns={"index": self.time_col})
            for c in self.group_cols:
                if c in view.columns:
                    grid[c] = view[c].iloc[0]
            if self.mark_col:
                grid[self.mark_col] = (~is_original).astype(int)
            parts.append(grid)
        return pd.concat(parts, ignore_index=True) if parts else s.copy()

    def udf_get_params(self) -> dict_type:
        return {"time_col": self.time_col, "freq": self.freq, "cols": self.cols,
                "fill_value": self.fill_value, "mark_col": self.mark_col}

    def udf_set_params(self, params: dict_type):
        self.time_col = params["time_col"]
        self.freq = params["freq"]
        self.cols = params["cols"]
        self.fill_value = params["fill_value"]
        self.mark_col = params["mark_col"]


class TSBoxCox(GroupTSBase):
    """
    Box-Cox / Yeo-Johnson 幂变换(可逆, 懒加载 scipy)。fit 时按列估计最优 lambda,
    transform 正向变换, inverse 还原。

    Args:
        cols: 变换列; None 表示全部数值列
        method: boxcox(要求正值) / yeojohnson(支持负值) / auto(有非正值自动用 yeojohnson)
        inverse: False=正向变换, True=还原
    """

    METHODS = ("boxcox", "yeojohnson", "auto")
    LAMBDA_BOUNDS = (-2.0, 2.0)   # 与 sklearn PowerTransformer 一致, 防止极端 lambda 导致逆变换数值失真

    def __init__(self, cols=None, method="auto", inverse=False, **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        if method not in self.METHODS:
            raise Exception("TSBoxCox method must be one of {}".format(self.METHODS))
        self.method = method
        self.inverse = inverse
        self.lambdas_ = {}   # col -> (method, lambda)

    def udf_fit(self, s: dataframe_type, **kwargs):
        from scipy import stats as sps
        self.cols = self.cols or numeric_cols_of(s)
        self.lambdas_ = {}
        for col in self.cols:
            if col not in s.columns:
                continue
            values = s[col].astype(float).dropna().to_numpy()
            method = self.method
            if method == "auto":
                method = "yeojohnson" if (len(values) and values.min() <= 0) else "boxcox"
            if method == "boxcox":
                lam = float(sps.boxcox_normmax(values)) if len(values) else 0.5
            else:
                lam = float(sps.yeojohnson_normmax(values)) if len(values) else 0.5
            lam = float(np.clip(lam, *self.LAMBDA_BOUNDS))
            self.lambdas_[col] = (method, lam)
        return self

    def _forward_one(self, v, method, lam):
        if np.isnan(v):
            return np.nan
        if method == "boxcox":
            return float(sps_boxcox(v, lam))
        return float(_yeojohnson_one(v, lam))

    def _inverse_one(self, v, method, lam):
        if np.isnan(v):
            return np.nan
        if method == "boxcox":
            return float(sps_inv_boxcox(v, lam))
        return float(_yeojohnson_inv_one(v, lam))

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        cols = [c for c in self.cols if c in s.columns]

        def map_group(key, view):
            out = {}
            for col in cols:
                if col not in self.lambdas_:
                    continue
                method, lam = self.lambdas_[col]
                values = view[col].astype(float).to_numpy()
                mapped = np.array([self._inverse_one(v, method, lam) if self.inverse
                                   else self._forward_one(v, method, lam) for v in values],
                                  dtype=float)
                out[col] = pd.Series(mapped, index=view.index)
            return pd.DataFrame(out, index=view.index)

        return self._apply_group_func(s, map_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self.cols:
            if col not in s or col not in self.lambdas_:
                continue
            try:
                value = float(s[col])
            except (TypeError, ValueError):
                value = np.nan
            method, lam = self.lambdas_[col]
            s[col] = self._inverse_one(value, method, lam) if self.inverse \
                else self._forward_one(value, method, lam)
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "method": self.method,
                "inverse": self.inverse, "lambdas_": self.lambdas_}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.method = params["method"]
        self.inverse = params["inverse"]
        self.lambdas_ = params["lambdas_"]


def sps_boxcox(v, lam):
    from scipy.special import boxcox1p  # noqa: F401
    # 手写 boxcox 单点(正值), 避免逐点调用 scipy.stats.boxcox
    if lam == 0:
        return np.log(v)
    return (np.power(v, lam) - 1.0) / lam


def sps_inv_boxcox(v, lam):
    if lam == 0:
        return np.exp(v)
    return np.power(v * lam + 1.0, 1.0 / lam)


def _yeojohnson_one(v, lam):
    if v >= 0:
        if lam == 0:
            return np.log(v + 1.0)
        return (np.power(v + 1.0, lam) - 1.0) / lam
    if lam == 2:
        return -np.log(-v + 1.0)
    return -(np.power(-v + 1.0, 2.0 - lam) - 1.0) / (2.0 - lam)


def _yeojohnson_inv_one(v, lam):
    if v >= 0:
        if lam == 0:
            return np.exp(v) - 1.0
        return np.power(v * lam + 1.0, 1.0 / lam) - 1.0
    if lam == 2:
        return 1.0 - np.exp(-v)
    return 1.0 - np.power(1.0 - v * (2.0 - lam), 1.0 / (2.0 - lam))
