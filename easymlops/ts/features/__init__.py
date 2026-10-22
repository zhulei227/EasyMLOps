# -*- coding: utf-8 -*-
"""
TS 滑动特征 pipes: 滞后 / 滚动统计 / 指数加权。

均继承 GroupM2MRollingPipeBase(分组滚动基类):
    - 批量: 组内时间排序后 pandas 原生计算(无 numpy_ext 依赖)
    - 单条: (组键, 输出列) 分桶缓存因果演进, 新组自动 warmup
    - 窗口不足位置按 fillna_strategy 填充, 默认 "zero"(兼容), 可选 "nan"
"""
from collections import deque

import numpy as np
import pandas as pd

from easymlops.ts.core.group_pipe import GroupM2MRollingPipeBase, GroupTSBase, normalize_cols

dataframe_type = pd.DataFrame
dict_type = dict


class LagFeature(GroupM2MRollingPipeBase):
    """
    滞后特征: output = x_{t-lag}。组边界(组内前 lag 条)输出 fillna_strategy 值。

    Args:
        cols: 输入列; None 表示全部数值列
        lags: 滞后阶列表, 如 [1, 7]; 输出列名 f"{col}_lag{k}"
    """

    def __init__(self, cols=None, lags=(1,), **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.lags = [int(k) for k in lags]
        for k in self.lags:
            if k <= 0:
                raise Exception("LagFeature lags must be positive")

    def _build_specs(self, s: dataframe_type):
        self._specs = []
        for col in self.input_cols:
            for lag in self.lags:
                self._specs.append({"input_col": col, "output_col": "{}_lag{}".format(col, lag),
                                    "window": lag + 1, "min_periods": lag + 1, "lag": lag})

    def _calc_batch(self, spec, series: pd.Series) -> pd.Series:
        return series.shift(spec["lag"])

    def _feature_from_deque(self, spec, dq: deque) -> float:
        return dq[0]

    def udf_get_params(self) -> dict_type:
        params = super().udf_get_params()
        params["lags"] = self.lags
        return params

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.lags = params["lags"]


class RollingFeature(GroupM2MRollingPipeBase):
    """
    滚动统计特征: output = rolling(window, min_periods).{stat}, 组内计算。
    输出列名 f"{col}_roll{window}_{stat}"。

    Args:
        cols: 输入列; None 表示全部数值列
        window: 滚动窗口长度
        stats: 统计量列表, 可选 mean/std/min/max/median (std 为样本标准差 ddof=1)
        min_periods: 触发计算的最小窗口样本数, None 表示 window(全窗才计算)
    """

    SUPPORTED_STATS = ("mean", "std", "min", "max", "median", "skew", "kurt")

    def __init__(self, cols=None, window=7, stats=("mean", "std"), **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.window = int(window)
        if self.window <= 0:
            raise Exception("RollingFeature window must be positive")
        self.stats = list(stats)
        for st in self.stats:
            if st not in self.SUPPORTED_STATS:
                raise Exception("RollingFeature unsupported stat: {}, supported: {}".format(
                    st, self.SUPPORTED_STATS))

    def _build_specs(self, s: dataframe_type):
        mp = self.min_periods if self.min_periods is not None else self.window
        self._specs = []
        for col in self.input_cols:
            for st in self.stats:
                self._specs.append({"input_col": col,
                                    "output_col": "{}_roll{}_{}".format(col, self.window, st),
                                    "window": self.window, "min_periods": mp, "stat": st})

    def _calc_batch(self, spec, series: pd.Series) -> pd.Series:
        roller = series.rolling(self.window, min_periods=spec["min_periods"])
        return getattr(roller, spec["stat"])()

    def _feature_from_deque(self, spec, dq: deque) -> float:
        values = np.asarray(dq, dtype=float)
        values = values[~np.isnan(values)]
        stat = spec["stat"]
        if stat == "mean":
            return float(np.mean(values)) if len(values) else np.nan
        if stat == "std":
            return float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        if stat == "min":
            return float(np.min(values)) if len(values) else np.nan
        if stat == "max":
            return float(np.max(values)) if len(values) else np.nan
        if stat == "median":
            return float(np.median(values)) if len(values) else np.nan
        if stat in ("skew", "kurt"):
            n = len(values)
            if n < 3 or (stat == "kurt" and n < 4):
                return np.nan
            std = float(np.std(values, ddof=1))
            if std == 0:
                return np.nan
            z = (values - np.mean(values)) / std
            if stat == "skew":
                return float(n / ((n - 1) * (n - 2)) * np.sum(z ** 3))
            return float(n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(z ** 4)
                         - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        raise Exception("unsupported stat {}".format(stat))

    def udf_get_params(self) -> dict_type:
        params = super().udf_get_params()
        params.update({"window": self.window, "stats": self.stats})
        return params

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.window = params["window"]
        self.stats = params["stats"]


class EWMAFeature(GroupM2MRollingPipeBase):
    """
    指数加权移动平均(adjust=False 递归口径, 批量与单条严格一致):
    y_t = alpha * x_t + (1 - alpha) * y_{t-1}, y_0 = x_0。输出列名 f"{col}_ewma"。

    Args:
        cols: 输入列; None 表示全部数值列
        span: 与 halflife 二选一(优先 span), alpha = 2 / (span + 1)
        halflife: alpha = 1 - exp(-ln2 / halflife)
    """

    def __init__(self, cols=None, span=None, halflife=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.span = span
        self.halflife = halflife
        if self.span is None and self.halflife is None:
            self.span = 7
        self._ewma_state = {}  # (组键, 输出列) -> 上一期 ewma 值, 仅单条路径使用

    def _alpha(self) -> float:
        if self.span is not None:
            return 2.0 / (self.span + 1.0)
        return 1.0 - np.exp(-np.log(2.0) / self.halflife)

    def _build_specs(self, s: dataframe_type):
        self._specs = [{"input_col": col, "output_col": "{}_ewma".format(col),
                        "window": 1, "min_periods": 1} for col in self.input_cols]

    def _calc_batch(self, spec, series: pd.Series) -> pd.Series:
        if self.span is not None:
            return series.ewm(span=self.span, adjust=False).mean()
        return series.ewm(halflife=self.halflife, adjust=False).mean()

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        alpha = self._alpha()
        for spec in self._specs:
            col_in = spec["input_col"]
            try:
                raw = float(s[col_in])
            except (TypeError, ValueError, KeyError):
                raw = np.nan
            cache_key = (key, spec["output_col"])
            prev = self._ewma_state.get(cache_key)
            if np.isnan(raw):
                value = prev if prev is not None else (0.0 if self.fillna_strategy == "zero" else np.nan)
            elif prev is None or np.isnan(prev):
                value = raw  # pandas adjust=False 首值即 x_0
            else:
                value = alpha * raw + (1.0 - alpha) * prev
            self._ewma_state[cache_key] = value
            s[spec["output_col"]] = self._fill_scalar(value)
        return s

    def udf_get_params(self) -> dict_type:
        params = super().udf_get_params()
        params.update({"span": self.span, "halflife": self.halflife})
        return params

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.span = params["span"]
        self.halflife = params["halflife"]

    def reset_cache_data(self):
        self.history_cache = {}
        self._ewma_state = {}


class ExpandingFeature(GroupM2MRollingPipeBase):
    """
    累积扩展统计特征: expanding(min_periods).{stat}, 组内因果计算(含当前行)。
    输出列名 f"{col}_expanding_{stat}"。单条路径用 Welford 增量算法, 与批量严格一致。

    Args:
        cols: 输入列; None 表示全部数值列
        stats: mean/std(ddof=1)/min/max
        min_periods: 触发计算的最小样本数, 默认 1
    """

    SUPPORTED_STATS = ("mean", "std", "min", "max")

    def __init__(self, cols=None, stats=("mean",), min_periods=1, **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.stats = list(stats)
        for st in self.stats:
            if st not in self.SUPPORTED_STATS:
                raise Exception("ExpandingFeature unsupported stat: {}, supported: {}".format(
                    st, self.SUPPORTED_STATS))
        self.exp_min_periods = int(min_periods)
        self._state = {}  # (组键, 输出列) -> {"n","sum","min","max","m2"}, 仅单条路径

    def _build_specs(self, s: dataframe_type):
        self._specs = [{"input_col": col, "output_col": "{}_expanding_{}".format(col, st),
                        "window": 1, "min_periods": self.exp_min_periods, "stat": st}
                       for col in self.input_cols for st in self.stats]

    def _calc_batch(self, spec, series: pd.Series) -> pd.Series:
        exp = series.expanding(min_periods=spec["min_periods"])
        return getattr(exp, spec["stat"])()

    def _feature_from_deque(self, spec, dq: deque) -> float:
        return np.nan  # 单条路径被完全重载, 此钩子不使用

    def _stat_from_state(self, st, stat):
        n = st["n"]
        if n < self.exp_min_periods or n == 0:
            return np.nan
        if stat == "mean":
            return st["mean"]
        if stat == "std":
            if n < 2:
                return np.nan
            return float(np.sqrt(max(st["m2"], 0.0) / (n - 1)))
        if stat == "min":
            return st["min"]
        if stat == "max":
            return st["max"]
        raise Exception("unsupported stat {}".format(stat))

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        for spec in self._specs:
            col_in = spec["input_col"]
            try:
                raw = float(s[col_in])
            except (TypeError, ValueError, KeyError):
                raw = np.nan
            st = self._state.setdefault((key, spec["output_col"]),
                                        {"n": 0, "mean": 0.0, "m2": 0.0,
                                         "min": np.inf, "max": -np.inf})
            if not np.isnan(raw):
                # Welford 增量更新(先更新再输出, 与 pandas expanding 含当前行一致)
                st["n"] += 1
                delta = raw - st["mean"]
                st["mean"] += delta / st["n"]
                st["m2"] += delta * (raw - st["mean"])
                st["min"] = min(st["min"], raw)
                st["max"] = max(st["max"], raw)
            s[spec["output_col"]] = self._fill_scalar(self._stat_from_state(st, spec["stat"]))
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "stats": self.stats, "exp_min_periods": self.exp_min_periods}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.stats = params["stats"]
        self.exp_min_periods = params["exp_min_periods"]

    def reset_cache_data(self):
        self.history_cache = {}
        self._state = {}


class MovingAverageTrend(GroupM2MRollingPipeBase):
    """
    移动平均趋势项。center=False(默认)为因果滑动平均, 批量与单条严格一致,
    可通过 auto_test; center=True 为中心化移动平均(用到未来值), 仅限离线,
    单条推理退化为 trailing 平均。

    Args:
        cols: 输入列; None 表示全部数值列
        window: 窗口长度
        center: 是否中心化窗口(离线专用)
    """

    def __init__(self, cols=None, window=5, center=False, **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.window = int(window)
        if self.window <= 0:
            raise Exception("MovingAverageTrend window must be positive")
        self.center = center

    def _build_specs(self, s: dataframe_type):
        mp = self.min_periods if self.min_periods is not None else self.window
        self._specs = [{"input_col": col, "output_col": "{}_ma{}".format(col, self.window),
                        "window": self.window, "min_periods": mp} for col in self.input_cols]

    def _calc_batch(self, spec, series: pd.Series) -> pd.Series:
        return series.rolling(self.window, min_periods=spec["min_periods"],
                              center=self.center).mean()

    def _feature_from_deque(self, spec, dq: deque) -> float:
        values = np.asarray(dq, dtype=float)
        values = values[~np.isnan(values)]
        return float(np.mean(values)) if len(values) else np.nan

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "window": self.window, "center": self.center}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.window = params["window"]
        self.center = params["center"]


class SeasonalMean(GroupTSBase):
    """
    季节均值 profile(可逆成分提取): fit 时按组内位置 pos % period 池化均值,
    transform 输出季节项/去季节列。

    相位约定: 位置 = 当前 transform 批次内组内行序(每批次从 0 开始);
    跨批次连续相位请使用单条流式(计数器缓存)或自行对齐批次起点。

    Args:
        cols: 目标数值列; None 表示全部数值列
        period: 季节周期长度
        per_group: 是否按组学 profile; group_cols=None 时天然走全局
        mode: seasonal(仅季节列) / deseasonal(仅去季节列) / both(默认)
    """

    MODES = ("seasonal", "deseasonal", "both")

    def __init__(self, cols=None, period=7, per_group=False, mode="both", **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.period = int(period)
        if self.period <= 0:
            raise Exception("SeasonalMean period must be positive")
        self.per_group = per_group
        if mode not in self.MODES:
            raise Exception("SeasonalMean mode must be one of {}".format(self.MODES))
        self.mode = mode
        self.global_profiles_ = {}  # col -> np.array(period)
        self.group_profiles_ = {}   # 组键 -> {col: np.array(period)}
        self._pos_counter = {}      # 组键 -> 已流式消费的行数, 仅单条路径

    def _empty_profile(self):
        return np.zeros(self.period)

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.cols = self.cols or numeric_cols_of(s)
        self.global_profiles_ = {col: self._empty_profile() for col in self.cols if col in s.columns}
        self.group_profiles_ = {}
        counts = {col: np.zeros(self.period) for col in self.global_profiles_}
        group_counts = {}
        for key, view in self._iter_group_views(s):
            gprofiles = {col: {} for col in self.cols if col in view.columns}
            for col in self.global_profiles_:
                if col not in view.columns:
                    continue
                values = view[col].astype(float).to_numpy()
                for pos, v in enumerate(values):
                    bucket = pos % self.period
                    if not np.isnan(v):
                        counts[col][bucket] += 1
                        self.global_profiles_[col][bucket] += v
                    if self.per_group and self.grouped:
                        gb = gprofiles[col].setdefault("buckets", np.zeros(self.period))
                        gb_c = group_counts.setdefault((key, col), np.zeros(self.period))
                        if not np.isnan(v):
                            gb[bucket] += v
                            gb_c[bucket] += 1
            if self.per_group and self.grouped:
                self.group_profiles_[key] = {}
                for col, gd in gprofiles.items():
                    buckets = gd.get("buckets", np.zeros(self.period))
                    cnt = group_counts.get((key, col), np.zeros(self.period))
                    avg = np.divide(buckets, cnt, out=np.zeros(self.period), where=cnt > 0)
                    self.group_profiles_[key][col] = avg
        for col in self.global_profiles_:
            cnt = counts[col]
            self.global_profiles_[col] = np.divide(self.global_profiles_[col], cnt,
                                                   out=np.zeros(self.period), where=cnt > 0)
        self._warn_if_too_many_groups(len(self.group_profiles_))
        return self

    def _profile_for(self, key, col):
        if self.per_group and self.grouped:
            profile = self.group_profiles_.get(key, {}).get(col)
            if profile is not None:
                return profile
        return self.global_profiles_[col]

    def _output_cols(self, col):
        if self.mode == "seasonal":
            return ["{}_seasonal".format(col)]
        if self.mode == "deseasonal":
            return ["{}_deseasonal".format(col)]
        return ["{}_seasonal".format(col), "{}_deseasonal".format(col)]

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        cols = [c for c in self.cols if c in s.columns]

        def seasonal_group(key, view):
            out = {}
            positions = np.arange(len(view)) % self.period
            for col in cols:
                profile = self._profile_for(key, col)
                seasonal = profile[positions]
                if self.mode in ("seasonal", "both"):
                    out["{}_seasonal".format(col)] = pd.Series(seasonal, index=view.index)
                if self.mode in ("deseasonal", "both"):
                    out["{}_deseasonal".format(col)] = pd.Series(
                        view[col].astype(float).to_numpy() - seasonal, index=view.index)
            return pd.DataFrame(out, index=view.index)

        return self._apply_group_func(s, seasonal_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        pos = self._pos_counter.get(key, 0)
        self._pos_counter[key] = pos + 1
        for col in self.cols:
            if col not in s:
                continue
            try:
                value = float(s[col])
            except (TypeError, ValueError):
                value = np.nan
            seasonal = self._profile_for(key, col)[pos % self.period]
            if self.mode in ("seasonal", "both"):
                s["{}_seasonal".format(col)] = seasonal
            if self.mode in ("deseasonal", "both"):
                s["{}_deseasonal".format(col)] = value - seasonal
        return s

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "period": self.period, "per_group": self.per_group,
                "mode": self.mode, "global_profiles_": self.global_profiles_,
                "group_profiles_": self.group_profiles_}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.period = params["period"]
        self.per_group = params["per_group"]
        self.mode = params["mode"]
        self.global_profiles_ = params["global_profiles_"]
        self.group_profiles_ = params["group_profiles_"]

    def reset_cache_data(self):
        self._pos_counter = {}


class FourierFeature(GroupTSBase):
    """
    傅里叶周期特征: 对组内位置 pos 输出 sin/cos(2*pi*k*pos/period), k=1..order。
    逐点无状态变换(相位约定与 SeasonalMean 一致: 每个批量从 0 起, 单条流式推进)。

    Args:
        time_col: 时间列名(仅用于组内排序, 不参与计算); None 保持原行序
        period: 周期长度(如 7=周, 12=月)
        order: 谐波阶数, 输出 2*order 列
    """

    def __init__(self, time_col=None, period=7, order=2, group_cols=None, **kwargs):
        super().__init__(group_cols=group_cols, time_col=time_col, **kwargs)
        self.period = int(period)
        self.order = int(order)
        if self.period <= 0 or self.order <= 0:
            raise Exception("FourierFeature period/order must be positive")
        self._pos_counter = {}   # 组键 -> 已流式消费的行数, 仅单条路径

    def udf_fit(self, s: dataframe_type, **kwargs):
        return self

    def _fourier_at(self, pos: int) -> dict_type:
        values = {}
        for k in range(1, self.order + 1):
            values["fourier_{}_sin".format(k)] = float(np.sin(2 * np.pi * k * pos / self.period))
            values["fourier_{}_cos".format(k)] = float(np.cos(2 * np.pi * k * pos / self.period))
        return values

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        def fourier_group(key, view):
            return pd.DataFrame({name: [self._fourier_at(pos)[name] for pos in range(len(view))]
                                 for name in self._fourier_at(0)}, index=view.index)

        return self._apply_group_func(s, fourier_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        pos = self._pos_counter.get(key, 0)
        self._pos_counter[key] = pos + 1
        for name, value in self._fourier_at(pos).items():
            s[name] = value
        return s

    def udf_get_params(self) -> dict_type:
        return {"time_col": self.time_col, "period": self.period, "order": self.order}

    def udf_set_params(self, params: dict_type):
        self.time_col = params["time_col"]
        self.period = params["period"]
        self.order = params["order"]

    def reset_cache_data(self):
        self._pos_counter = {}


class RollingCorrCov(GroupTSBase):
    """
    双列滚动相关/协方差(组内计算)。输出列名
    f"{left_col}_{right_col}_roll{window}_{stat}"。

    Args:
        left_col: 左列
        right_col: 右列
        window: 滚动窗口
        stat: corr(皮尔逊相关) / cov(样本协方差 ddof=1)
        min_periods: 触发计算的最小成对样本数, 默认 window
    """

    STATS = ("corr", "cov")

    def __init__(self, left_col, right_col, window=10, stat="corr",
                 min_periods=None, fillna_strategy="zero", **kwargs):
        super().__init__(**kwargs)
        self.left_col = left_col
        self.right_col = right_col
        self.window = int(window)
        if self.window <= 1:
            raise Exception("RollingCorrCov window must be > 1")
        if stat not in self.STATS:
            raise Exception("RollingCorrCov stat must be one of {}".format(self.STATS))
        self.stat = stat
        self.min_periods = max(2, int(min_periods if min_periods is not None else window))
        if fillna_strategy not in ("zero", "nan"):
            raise Exception("fillna_strategy must be zero/nan")
        self.fillna_strategy = fillna_strategy
        self.output_col = "{}_{}_roll{}_{}".format(left_col, right_col, window, stat)
        self._pair_cache = {}  # 组键 -> deque of (l, r)

    def _calc_batch(self, view: dataframe_type) -> pd.Series:
        roller = view[self.left_col].astype(float).rolling(self.window,
                                                           min_periods=self.min_periods)
        if self.stat == "corr":
            return roller.corr(view[self.right_col].astype(float))
        return roller.cov(view[self.right_col].astype(float))

    def _fill(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0 if self.fillna_strategy == "zero" else np.nan
        if isinstance(value, float) and np.isinf(value):
            return 0.0
        return value

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for key, view in self._iter_group_views(s):
            values = self._calc_batch(view).to_numpy(dtype=float)
            values = np.where(np.isinf(values), 0.0, values)
            if self.fillna_strategy == "zero":
                values = np.where(np.isnan(values), 0.0, values)
            result.loc[view.index, self.output_col] = values
        return self._finalize(result)

    def _pair_stat(self, pairs) -> float:
        if len(pairs) < self.min_periods:
            return np.nan
        arr = np.asarray(pairs, dtype=float)
        mask = ~(np.isnan(arr[:, 0]) | np.isnan(arr[:, 1]))
        arr = arr[mask]
        n = len(arr)
        if n < self.min_periods or n < 2:
            return np.nan
        cov = float(np.cov(arr[:, 0], arr[:, 1], ddof=1)[0, 1])
        if self.stat == "cov":
            return cov
        std_l = float(np.std(arr[:, 0], ddof=1))
        std_r = float(np.std(arr[:, 1], ddof=1))
        if std_l == 0 or std_r == 0:
            return np.nan
        return cov / (std_l * std_r)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        try:
            l_value = float(s[self.left_col])
        except (TypeError, ValueError, KeyError):
            l_value = np.nan
        try:
            r_value = float(s[self.right_col])
        except (TypeError, ValueError, KeyError):
            r_value = np.nan
        dq = self._pair_cache.setdefault(key, deque(maxlen=self.window))
        dq.append((l_value, r_value))
        s[self.output_col] = self._fill(self._pair_stat(dq))
        return s

    def udf_get_params(self) -> dict_type:
        return {"left_col": self.left_col, "right_col": self.right_col,
                "window": self.window, "stat": self.stat,
                "min_periods": self.min_periods, "fillna_strategy": self.fillna_strategy,
                "output_col": self.output_col}

    def udf_set_params(self, params: dict_type):
        self.left_col = params["left_col"]
        self.right_col = params["right_col"]
        self.window = params["window"]
        self.stat = params["stat"]
        self.min_periods = params["min_periods"]
        self.fillna_strategy = params["fillna_strategy"]
        self.output_col = params["output_col"]

    def reset_cache_data(self):
        self._pair_cache = {}


class STLDecompose(GroupTSBase):
    """
    STL 季节-趋势分解(离线专用, 懒加载 statsmodels)。
    组内对每列分解并输出 f"{col}_trend" / f"{col}_seasonal" / f"{col}_resid"。
    会新增列且依赖完整序列, 勿纳入 auto_test 验证的 pipeline; transform_single 仅透传。

    Args:
        cols: 分解列; None 表示全部数值列
        period: 季节周期
        robust: 是否鲁棒 STL
    """

    def __init__(self, cols=None, period=7, robust=False, **kwargs):
        super().__init__(**kwargs)
        self.cols = normalize_cols(cols)
        self.period = int(period)
        self.robust = robust

    def udf_fit(self, s: dataframe_type, **kwargs):
        self.cols = self.cols or numeric_cols_of(s)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        from statsmodels.tsa.seasonal import STL
        cols = [c for c in self.cols if c in s.columns]
        parts = []
        for key, view in self._iter_group_views(s):
            result = view.copy()
            for col in cols:
                values = view[col].astype(float).to_numpy()
                if len(values) < 2 * self.period:
                    raise Exception("STLDecompose requires at least 2*period({}) rows per group,"
                                    " got {}".format(2 * self.period, len(values)))
                res = STL(values, period=self.period, robust=self.robust).fit()
                result["{}_trend".format(col)] = res.trend
                result["{}_seasonal".format(col)] = res.seasonal
                result["{}_resid".format(col)] = res.resid
            parts.append(result)
        return pd.concat(parts, ignore_index=True) if parts else s.copy()

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "period": self.period, "robust": self.robust}

    def udf_set_params(self, params: dict_type):
        self.cols = params["cols"]
        self.period = params["period"]
        self.robust = params["robust"]
