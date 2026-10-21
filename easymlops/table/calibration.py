# -*- coding: utf-8 -*-
"""
概率校准 Pipe：将模型输出的概率校准为真实发生概率。

说明：
- y 与构造模型时一致（原始标签值）；对每个待校准列 col，
  校准目标为 (y == col) 的0/1指示（one-vs-rest），支持多分类逐类校准。
- cols 不传时默认校准 fit 时输入的所有数值列。

两种用法：

1. 推荐（无偏）：主 pipeline 先拟合，校准 pipe 在验证集预测上单独 fit 后 append：

    table = TablePipeLine()
    table.pipe(FixInput()).pipe(...).pipe(LGBMClassification(y=y_train))
    table.fit(x_train)

    calib = IsotonicCalibrationPipe(y=y_val, cols=[1])
    calib.fit(table.transform(x_val))   # 用验证集的模型输出配对 y_val
    table.pipe(calib)                   # append 后 save/load/transform 均正常

2. 简便（有偏，在训练预测上校准）：直接挂入 pipeline，y 与 x_train 行对齐：

    table.pipe(LGBMClassification(y=y_train)).pipe(IsotonicCalibrationPipe(y=y_train, cols=[1]))
    table.fit(x_train)
"""
import numpy as np
import pandas as pd

from easymlops.table.core import TablePipeObjectBase, dataframe_type, dict_type


class IsotonicCalibrationPipe(TablePipeObjectBase):
    """
    基于 sklearn IsotonicRegression 的概率校准（保序回归，非参数、单调）。

    Example:
        >>> model = LGBMClassification(y=y_train, use_faster_predictor=False)
        >>> pipeline.pipe(model).pipe(IsotonicCalibrationPipe(y=y_val, cols=[1]))
    """

    def __init__(self, y=None, cols=None, out_of_bounds="clip", null_fill=0.5,
                 skip_check_transform_type=True, **kwargs):
        """
        Args:
            y: 校准用标签（与训练同口径的原始标签值），与拟合校准映射的输入数据行对齐
            cols: 待校准的输出概率列（通常为正类列名）；None 表示全部数值列
            out_of_bounds: 超出拟合范围的处理，默认"clip"
            null_fill: 输入无法转成数值时的填充值，默认0.5
            skip_check_transform_type: 概率列名可能为非字符串（如数字1），跳过类型检测
        """
        super().__init__(skip_check_transform_type=skip_check_transform_type, **kwargs)
        self.y = None if y is None else pd.Series(y).reset_index(drop=True)
        self.cols = cols
        self.out_of_bounds = out_of_bounds
        self.null_fill = null_fill
        self.calibrators = {}

    @staticmethod
    def _to_numeric_value(v, null_fill):
        """单值转float：无法转换(字符串/None等)时填 null_fill。与批量路径保持一致。"""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return null_fill
        if np.isnan(v):
            return null_fill
        return v

    def _fit_target(self, col):
        """列 col 的 one-vs-rest 0/1 目标（按字符串口径对齐标签与列名）。"""
        return (self.y.apply(lambda v: str(v) == str(col))).astype(float).to_numpy()

    def udf_fit(self, s: dataframe_type, **kwargs):
        from sklearn.isotonic import IsotonicRegression
        assert self.y is not None, "IsotonicCalibrationPipe need y for calibration"
        assert len(self.y) == len(s), "y length must match calibration input rows"
        if self.cols is None:
            self.cols = [col for col in s.columns
                         if "int" in str(s[col].dtype).lower() or "float" in str(s[col].dtype).lower()]
        if type(self.cols) not in (list, tuple):
            self.cols = [self.cols]
        for col in self.cols:
            x = s[col].apply(lambda v: self._to_numeric_value(v, self.null_fill)).to_numpy(dtype=float)
            iso = IsotonicRegression(out_of_bounds=self.out_of_bounds, y_min=0.0, y_max=1.0)
            iso.fit(x, self._fit_target(col))
            self.calibrators[col] = iso
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        result = dict()
        for col in s.columns:
            if col in self.calibrators:
                x = s[col].apply(lambda v: self._to_numeric_value(v, self.null_fill)).to_numpy(dtype=float)
                result[col] = self.calibrators[col].predict(x)
            else:
                result[col] = s[col].to_numpy()
        return pd.DataFrame(result, index=s.index)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        result = dict()
        for col, value in s.items():
            if col in self.calibrators:
                v = np.array([self._to_numeric_value(value, self.null_fill)], dtype=float)
                result[col] = float(self.calibrators[col].predict(v)[0])
            else:
                result[col] = value
        return result

    def udf_get_params(self) -> dict_type:
        return {"calibrators": self.calibrators, "cols": self.cols,
                "out_of_bounds": self.out_of_bounds, "null_fill": self.null_fill}

    def udf_set_params(self, params: dict_type):
        self.calibrators = params["calibrators"]
        self.cols = params["cols"]
        self.out_of_bounds = params["out_of_bounds"]
        self.null_fill = params["null_fill"]
