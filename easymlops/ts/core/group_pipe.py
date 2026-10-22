# -*- coding: utf-8 -*-
"""
分组时序基类。

GroupTSBase:
    面向"面板数据"(一张DataFrame含多个实体的时序)的通用分组基类。
    所有时序操作在组内按 time_col 排序后计算, 结果按原始行位置写回(index对齐)。
    group_cols=None 时整张表视为单一序列(全局模式)。

GroupM2MRollingPipeBase:
    继承 M2MRollingPipeBase(原基类保留不动)。在其范式上扩展:
    - 单条推理缓存按 (组键, 列) 分桶, 线上出现 fit 未见过的新组时自动warmup
    - 批量路径使用 pandas 原生 rolling/shift(组内计算, 不引入 numpy_ext 依赖)
    - 窗口不足位置按 fillna_strategy 填充("zero"默认兼容, 可选"nan")

约定与限制(务必阅读):
    - 批量 transform 是纯函数(不推进缓存); 缓存只在 transform_single 中演进
      —— 与框架 auto_test 的检测顺序(batch先跑, single逐条喂入)一致
    - 在线单条数据必须按组内时间顺序到达, 否则单条(因果)与批量(内部排序)结果不一致
    - 组键必须低基数(分段/渠道级别), 不要用主键: 每组一份缓存/统计量
    - groupby 全程 dropna=False, 组键为 NaN 的行归入独立伪组, 行数守恒
"""
from collections import deque

import numpy as np
import pandas as pd

from easymlops.ts.core.pipe import PipeBase, M2MRollingPipeBase

dataframe_type = pd.DataFrame
series_type = pd.Series
dict_type = dict

NAN_GROUP_KEY = "__nan__"
ALL_GROUP_KEY = "__all__"
FILL_STRATEGIES = ("zero", "nan")


def normalize_group_value(v):
    """组键值规范化: NaN/None -> 哨兵字符串, 其余转 str(保证 save/load 与 int/str 组键一致)"""
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return NAN_GROUP_KEY
    except Exception:
        return NAN_GROUP_KEY
    return str(v)


def normalize_cols(cols):
    """列参数归一: None->[], str->[str]"""
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def numeric_cols_of(s: dataframe_type):
    return [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]


class GroupTSBase(PipeBase):
    """
    分组时序操作基类(非滚动类)。

    Args:
        group_cols: 分组列名, None/[] = 全局单序列; 可传 1 个或多个列名
        time_col: 时间列名, 提供后组内按其排序后计算(结果仍写回原行位置)
        sort_rows: 是否物理重排输出为 (组键, 时间) 序; 默认 False(保持原行序,
                   与 auto_test 的 index 一致性检查兼容); True 仅建议离线整理时使用
    """

    def __init__(self, group_cols=None, time_col=None, sort_rows=False, **kwargs):
        super().__init__(**kwargs)
        self.group_cols = normalize_cols(group_cols)
        self.time_col = time_col
        self.sort_rows = sort_rows

    def udf_get_params(self) -> dict_type:
        return {"group_cols": self.group_cols, "time_col": self.time_col, "sort_rows": self.sort_rows}

    def udf_set_params(self, params: dict_type):
        self.group_cols = params["group_cols"]
        self.time_col = params["time_col"]
        self.sort_rows = params["sort_rows"]

    @property
    def grouped(self) -> bool:
        return len(self.group_cols) > 0

    def _key_from_record(self, record: dict_type) -> tuple:
        if not self.grouped:
            return (ALL_GROUP_KEY,)
        return tuple(normalize_group_value(record.get(c)) for c in self.group_cols)

    def _sort_view(self, view: dataframe_type) -> dataframe_type:
        if self.time_col is not None and self.time_col in view.columns:
            return view.sort_values(self.time_col, kind="stable")
        return view

    def _iter_group_views(self, s: dataframe_type):
        """yield (组键tuple, 组内时间排序视图); 视图保留原始index"""
        if not self.grouped:
            yield (ALL_GROUP_KEY,), self._sort_view(s)
            return
        for raw_key, view in s.groupby(self.group_cols, dropna=False, sort=False):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            yield tuple(normalize_group_value(k) for k in key), self._sort_view(view)

    def _finalize(self, s: dataframe_type) -> dataframe_type:
        if self.sort_rows and self.time_col is not None:
            by = list(self.group_cols) + [self.time_col]
            return s.sort_values(by, kind="stable")
        return s

    def _apply_group_func(self, s: dataframe_type, func):
        """
        组级计算框架: func(组键, 组内时间排序视图) -> 与视图同index的DataFrame(可含新增列),
        结果按原始行位置写回。返回 _finalize 后的副本。
        """
        result = s.copy()
        for key, view in self._iter_group_views(s):
            out = func(key, view)
            for col in out.columns:
                result.loc[view.index, col] = out[col].to_numpy()
        return self._finalize(result)

    @staticmethod
    def _warn_if_too_many_groups(n_groups, threshold=1000):
        if n_groups > threshold:
            print("(GroupTS) warning: {} groups detected, group-wise state may cost"
                  " too much memory, group_cols should be low-cardinality".format(n_groups))


class GroupM2MRollingPipeBase(M2MRollingPipeBase):
    """
    分组滚动特征基类: 继承 M2MRollingPipeBase(原基类与行为保留), 扩展分组能力。

    子类职责:
        - udf_fit 中解析 input_cols 并构建 self._specs:
          [{"input_col", "output_col", "window", "min_periods"}, ...]
        - 实现 _calc_batch(spec, series) -> pd.Series (组内时间排序后的批量计算)
        - 实现 _feature_from_deque(spec, dq) -> float (单条: dq 已追加当前值)

    缓存: self.history_cache[(组键, 输出列)] = deque(maxlen=window), 仅在
    transform_single 中演进; 部署重启后缓存清空, 前 window-min_periods 条
    输出 fillna_strategy 值(warmup 约定)。
    """

    def __init__(self, group_cols=None, time_col=None, sort_rows=False,
                 fillna_strategy="zero", min_periods=None, **kwargs):
        super().__init__(**kwargs)
        self.group_cols = normalize_cols(group_cols)
        self.time_col = time_col
        self.sort_rows = sort_rows
        if fillna_strategy not in FILL_STRATEGIES:
            raise Exception("fillna_strategy must be one of {}".format(FILL_STRATEGIES))
        self.fillna_strategy = fillna_strategy
        self.min_periods = min_periods
        self._specs = []
        self.history_cache = {}

    # ---------- 组辅助(与 GroupTSBase 同语义, 因 MRO 不便多继承而独立实现) ----------
    @property
    def grouped(self) -> bool:
        return len(self.group_cols) > 0

    def _key_from_record(self, record: dict_type) -> tuple:
        if not self.grouped:
            return (ALL_GROUP_KEY,)
        return tuple(normalize_group_value(record.get(c)) for c in self.group_cols)

    def _sort_view(self, view: dataframe_type) -> dataframe_type:
        if self.time_col is not None and self.time_col in view.columns:
            return view.sort_values(self.time_col, kind="stable")
        return view

    def _iter_group_views(self, s: dataframe_type):
        if not self.grouped:
            yield (ALL_GROUP_KEY,), self._sort_view(s)
            return
        for raw_key, view in s.groupby(self.group_cols, dropna=False, sort=False):
            key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
            yield tuple(normalize_group_value(k) for k in key), self._sort_view(view)

    def _finalize(self, s: dataframe_type) -> dataframe_type:
        if self.sort_rows and self.time_col is not None:
            by = list(self.group_cols) + [self.time_col]
            return s.sort_values(by, kind="stable")
        return s

    @staticmethod
    def _warn_if_too_many_groups(n_groups, threshold=1000):
        if n_groups > threshold:
            print("(GroupM2M) warning: {} groups detected, group-wise cache may cost"
                  " too much memory, group_cols should be low-cardinality".format(n_groups))

    # ---------- 填充策略 ----------
    def _fill_values(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        values = np.where(np.isinf(values), 0.0, values)
        if self.fillna_strategy == "zero":
            values = np.where(np.isnan(values), 0.0, values)
        return values

    def _fill_scalar(self, value):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return 0.0 if self.fillna_strategy == "zero" else np.nan
        if isinstance(value, float) and np.isinf(value):
            return 0.0
        return value

    # ---------- 子类钩子 ----------
    def _build_specs(self, s: dataframe_type):
        """子类在 udf_fit 中调用: 构建 self._specs 并设置 self.output_cols"""
        raise Exception("need to implement!")

    def _calc_batch(self, spec: dict_type, series: series_type) -> series_type:
        """组内时间排序后的批量计算, 返回与 series 同 index 的特征列"""
        raise Exception("need to implement!")

    def _feature_from_deque(self, spec: dict_type, dq: deque) -> float:
        """单条计算: dq(maxlen=window) 已追加当前值"""
        raise Exception("need to implement!")

    # ---------- 框架方法 ----------
    def udf_fit(self, s: dataframe_type, **kwargs):
        if not self.input_cols:
            self.input_cols = numeric_cols_of(s) or list(s.columns)
        self._build_specs(s)
        self.output_cols = [spec["output_col"] for spec in self._specs]
        # 兼容基类字段
        self.history_data = {c: [] for c in self.input_cols}
        if self.grouped:
            self._warn_if_too_many_groups(s.groupby(self.group_cols, dropna=False).ngroups)
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for key, view in self._iter_group_views(s):
            for spec in self._specs:
                values = self._calc_batch(spec, view[spec["input_col"]]).to_numpy(dtype=float)
                result.loc[view.index, spec["output_col"]] = self._fill_values(values)
        return self._finalize(result)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        for spec in self._specs:
            col_in = spec["input_col"]
            try:
                raw = float(s[col_in])
            except (TypeError, ValueError, KeyError):
                raw = np.nan
            dq = self.history_cache.setdefault((key, spec["output_col"]),
                                               deque(maxlen=spec["window"]))
            dq.append(raw)
            if len(dq) >= spec["min_periods"]:
                value = self._feature_from_deque(spec, dq)
            else:
                value = np.nan
            s[spec["output_col"]] = self._fill_scalar(value)
        return s

    def udf_get_params(self) -> dict_type:
        # 仅返回自有键: get_params 会沿继承链逐层聚合, 合并父类键会触发参数冲突告警
        return {"group_cols": self.group_cols, "time_col": self.time_col,
                "sort_rows": self.sort_rows, "fillna_strategy": self.fillna_strategy,
                "min_periods": self.min_periods, "_specs": self._specs}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.group_cols = params["group_cols"]
        self.time_col = params["time_col"]
        self.sort_rows = params["sort_rows"]
        self.fillna_strategy = params["fillna_strategy"]
        self.min_periods = params["min_periods"]
        # specs 属派生状态: fit 后保存/加载, 使 load 后无需重新 fit 即可 transform
        self._specs = params["_specs"]

    def reset_cache_data(self):
        self.history_cache = {}
