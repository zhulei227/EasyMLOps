# -*- coding: utf-8 -*-
import pandas as pd

from easymlops.table.core import *

dataframe_type = pd.DataFrame
dict_type = dict


class PredictionPostProcess(TablePipeObjectBase):
    """
    预测后处理 pipe（v0.2.5 新增）：挂在 Pipeline 尾部（或独立使用），对模型
    输出的预测列做 clip/round/二值化/自定义映射。Kaggle 优胜方案中标准的
    post-processing 环节（如把预测裁剪到业务合法区间、计数类比赛取整、概率
    二值化）。

    处理顺序（对每个目标列依次执行）：threshold 二值化 → clip 截断 →
    round 保留位数 → mapping 自定义映射。

    Args:
        cols: 要处理的列；None 表示全部输出列
        clip: (min, max) 元组，将列裁剪到该区间；None 不裁剪
        threshold: 二值化阈值，列值 > threshold 置 1 否则 0（如概率→标签）；None 不二值化
        round_digits: 保留小数位；None 不处理
        mapping: callable(pd.DataFrame) -> pd.DataFrame，最后的自定义映射；None 跳过
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.postprocess import PredictionPostProcess
        >>> pipeline.pipe(LGBMRegression(y=y_train)) \\
        ...     .pipe(PredictionPostProcess(clip=(0, None), round_digits=0))
    """

    def __init__(self, cols=None, clip=None, threshold=None, round_digits=None,
                 mapping=None, **kwargs):
        super().__init__(**kwargs)
        self.cols = cols
        self.clip = tuple(clip) if clip is not None else None
        self.threshold = threshold
        self.round_digits = round_digits
        self.mapping = mapping
        if self.clip is not None and len(self.clip) != 2:
            raise Exception("clip must be a (min, max) tuple")

    def _target_cols(self, columns):
        if self.cols is None:
            return list(columns)
        if isinstance(self.cols, str):
            return [self.cols]
        return [c for c in self.cols if c in columns]

    def _apply_column(self, series):
        if self.threshold is not None:
            series = (series > self.threshold).astype(float)
        if self.clip is not None:
            lo = self.clip[0] if self.clip[0] is not None else -float("inf")
            hi = self.clip[1] if self.clip[1] is not None else float("inf")
            series = series.clip(lo, hi)
        if self.round_digits is not None:
            series = series.round(self.round_digits)
        return series

    def udf_fit(self, s, **kwargs):
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = s.copy()
        for col in self._target_cols(result.columns):
            result[col] = self._apply_column(result[col])
        if self.mapping is not None:
            result = self.mapping(result)
        return result

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        for col in self._target_cols(s.keys()):
            if col in s:
                value = pd.Series([s[col]])
                s[col] = self._apply_column(value).iloc[0]
        if self.mapping is not None:
            s = self.mapping(s)
        return s
