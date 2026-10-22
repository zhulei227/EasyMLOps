# -*- coding: utf-8 -*-
"""
统计时序模型(ARIMA 家族, 分组版)。

约定:
    - 继承 GroupTSBase: group_cols=None 时全局单序列; 提供分组列时每组独立拟合
      一个模型, 面板数据各组序列互不干扰
    - 批量 transform: 每组从该组训练序列末端 forecast steps=组内行数, 结果按
      原始行位置写回(行数守恒)
    - 单条 transform_single: 该组 forecast 1 步(与批量第 1 步一致)
    - 拟合好的模型对象随 get_params 持久化(statsmodels/arch results 可 pickle),
      save/load 后无需重新 fit 即可 transform
    - 预测类 pipe 的批量(多步 forecast)与逐条单步推进语义不同, 不纳入 auto_test
"""
import warnings

import numpy as np
import pandas as pd

from easymlops.ts.core.group_pipe import GroupTSBase, normalize_cols

dataframe_type = pd.DataFrame
dict_type = dict


class ArimaRegression(GroupTSBase):
    """
    ARIMA 预测模型(可分组): 组内独立拟合, forecast 延续该组训练序列。

    Args:
        output_col: 预测输出列名
        input_cols: 目标序列列; None 自动取第一个数值列
        p, d, q: AR/差分/MA 阶数
        seasonal_p, seasonal_d, seasonal_q: 季节性阶数
        seasonal_period: 季节周期(>0 时使用 SARIMAX)
        exog_cols: 外生变量列名列表(forecast 步数 = transform 输入行数,
                   外生变量须在输入中提供)
        group_cols, time_col, sort_rows: 分组框架通用参数
    """

    def __init__(self, output_col="arima_predict", input_cols=None, p=1, d=1, q=1,
                 seasonal_p=0, seasonal_d=0, seasonal_q=0, seasonal_period=None,
                 exog_cols=None, **kwargs):
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = normalize_cols(input_cols)
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_p = seasonal_p
        self.seasonal_d = seasonal_d
        self.seasonal_q = seasonal_q
        self.seasonal_period = seasonal_period
        self.exog_cols = normalize_cols(exog_cols)
        self.group_models_ = {}   # 组键 -> 拟合好的 statsmodels results, 随 save/load 持久化

    # ---------- 内部工具 ----------
    def _resolve_input_cols(self, s: dataframe_type):
        if not self.input_cols:
            numeric_cols = [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
            self.input_cols = numeric_cols[:1] if numeric_cols else [s.columns[0]]

    def _exog_of(self, view: dataframe_type):
        if not self.exog_cols:
            return None
        return view[self.exog_cols].to_numpy(dtype=float)

    def _fit_one(self, target: np.ndarray, exog):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if self.seasonal_period is not None and self.seasonal_period > 0:
            model = SARIMAX(
                target,
                exog=exog,
                order=(self.p, self.d, self.q),
                seasonal_order=(self.seasonal_p, self.seasonal_d,
                                self.seasonal_q, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            model = ARIMA(
                target,
                exog=exog,
                order=(self.p, self.d, self.q),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        return model.fit()

    def _forecast_for(self, key, steps: int, exog) -> np.ndarray:
        if key not in self.group_models_:
            raise Exception("group {} not fitted, call fit before transform".format(key))
        results = self.group_models_[key]
        return np.asarray(results.forecast(steps=steps, exog=exog), dtype=float)

    # ---------- 框架钩子 ----------
    def udf_fit(self, s: dataframe_type, **kwargs):
        self._resolve_input_cols(s)
        target_col = self.input_cols[0]
        self.group_models_ = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # statsmodels 收敛告警噪声
            for key, view in self._iter_group_views(s):
                target = view[target_col].astype(float).to_numpy()
                self.group_models_[key] = self._fit_one(target, self._exog_of(view))
        return self

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        def forecast_group(key, view):
            preds = self._forecast_for(key, len(view), self._exog_of(view))
            return pd.DataFrame({self.output_col: pd.Series(preds, index=view.index)})

        return self._apply_group_func(s, forecast_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        exog = None
        if self.exog_cols:
            exog = np.array([[float(s[c]) for c in self.exog_cols]], dtype=float)
        s[self.output_col] = float(self._forecast_for(key, 1, exog)[0])
        return s

    def udf_get_params(self) -> dict_type:
        return {"output_col": self.output_col, "input_cols": self.input_cols,
                "p": self.p, "d": self.d, "q": self.q,
                "seasonal_p": self.seasonal_p, "seasonal_d": self.seasonal_d,
                "seasonal_q": self.seasonal_q, "seasonal_period": self.seasonal_period,
                "exog_cols": self.exog_cols, "group_models_": self.group_models_}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.output_col = params["output_col"]
        self.input_cols = params["input_cols"]
        self.p = params["p"]
        self.d = params["d"]
        self.q = params["q"]
        self.seasonal_p = params["seasonal_p"]
        self.seasonal_d = params["seasonal_d"]
        self.seasonal_q = params["seasonal_q"]
        self.seasonal_period = params["seasonal_period"]
        self.exog_cols = params["exog_cols"]
        self.group_models_ = params["group_models_"]


class SarimaRegression(ArimaRegression):
    """
    SARIMA 季节性预测模型(可分组): seasonal_order 默认 (1,1,1)x12。

    Args: 同 ArimaRegression(seasonal_* 默认开启)
    """

    def __init__(self, output_col="sarima_predict", input_cols=None,
                 p=1, d=1, q=1, seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12,
                 exog_cols=None, **kwargs):
        super().__init__(
            output_col=output_col, input_cols=input_cols,
            p=p, d=d, q=q,
            seasonal_p=seasonal_p, seasonal_d=seasonal_d, seasonal_q=seasonal_q,
            seasonal_period=seasonal_period, exog_cols=exog_cols, **kwargs
        )


class ArimaxRegression(ArimaRegression):
    """
    ARIMAX 带外生变量预测模型(可分组)。

    Args: 同 ArimaRegression(须提供 exog_cols)
    """

    def __init__(self, output_col="arimax_predict", input_cols=None, p=1, d=1, q=1,
                 exog_cols=None, **kwargs):
        super().__init__(
            output_col=output_col, input_cols=input_cols,
            p=p, d=d, q=q,
            seasonal_p=0, seasonal_d=0, seasonal_q=0, seasonal_period=None,
            exog_cols=exog_cols, **kwargs
        )


class SarimaxRegression(ArimaRegression):
    """
    SARIMAX 带外生变量的季节性预测模型(可分组)。

    Args: 同 ArimaRegression(seasonal_* 默认开启)
    """

    def __init__(self, output_col="sarimax_predict", input_cols=None,
                 p=1, d=1, q=1, seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12,
                 exog_cols=None, **kwargs):
        super().__init__(
            output_col=output_col, input_cols=input_cols,
            p=p, d=d, q=q,
            seasonal_p=seasonal_p, seasonal_d=seasonal_d, seasonal_q=seasonal_q,
            seasonal_period=seasonal_period, exog_cols=exog_cols, **kwargs
        )


class GarchRegression(GroupTSBase):
    """
    GARCH 波动率预测模型(可分组, 懒加载 arch)。transform 输出该组条件均值
    多步预测(arch 的 mean 通道, 缩放还原), 列名 output_col。

    Args:
        output_col: 输出列名
        input_cols: 目标序列列; None 自动取第一个数值列
        p, q: GARCH/ARCH 阶数
        vol: 波动率模型 (Garch, EGARCH, GJR_GARCH)
        mean: 均值模型 (Constant, ARX, HARX)
        exog_cols: 外生变量列名列表
        group_cols, time_col, sort_rows: 分组框架通用参数
    """

    def __init__(self, output_col="garch_predict", input_cols=None,
                 p=1, q=1, vol="Garch", mean="Constant",
                 exog_cols=None, **kwargs):
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = normalize_cols(input_cols)
        self.p = p
        self.q = q
        self.vol = vol
        self.mean = mean
        self.exog_cols = normalize_cols(exog_cols)
        self.group_models_ = {}   # 组键 -> 拟合好的 arch results, 随 save/load 持久化

    def _resolve_input_cols(self, s: dataframe_type):
        if not self.input_cols:
            numeric_cols = [c for c in s.columns if np.issubdtype(s[c].dtype, np.number)]
            self.input_cols = numeric_cols[:1] if numeric_cols else [s.columns[0]]

    def _exog_of(self, view: dataframe_type):
        if not self.exog_cols:
            return None
        return view[self.exog_cols].to_numpy(dtype=float)

    def udf_fit(self, s: dataframe_type, **kwargs):
        from arch import arch_model

        self._resolve_input_cols(s)
        target_col = self.input_cols[0]
        self.group_models_ = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # arch/numba 拟合告警噪声
            for key, view in self._iter_group_views(s):
                target = view[target_col].astype(float).to_numpy() * 100
                model = arch_model(
                    target,
                    vol=self.vol,
                    mean=self.mean,
                    p=self.p,
                    q=self.q,
                    x=self._exog_of(view),
                    dist="normal",
                    rescale=False
                )
                self.group_models_[key] = model.fit(disp="off")
        return self

    def _forecast_for(self, key, steps: int, exog) -> np.ndarray:
        if key not in self.group_models_:
            raise Exception("group {} not fitted, call fit before transform".format(key))
        predictions = self.group_models_[key].forecast(horizon=steps, x=exog)
        return np.asarray(predictions.mean.values[-1], dtype=float) / 100

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        def forecast_group(key, view):
            preds = self._forecast_for(key, len(view), self._exog_of(view))
            return pd.DataFrame({self.output_col: pd.Series(preds, index=view.index)})

        return self._apply_group_func(s, forecast_group)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        key = self._key_from_record(s)
        exog = None
        if self.exog_cols:
            exog = np.array([[float(s[c]) for c in self.exog_cols]], dtype=float)
        s[self.output_col] = float(self._forecast_for(key, 1, exog)[0])
        return s

    def udf_get_params(self) -> dict_type:
        return {"output_col": self.output_col, "input_cols": self.input_cols,
                "p": self.p, "q": self.q, "vol": self.vol, "mean": self.mean,
                "exog_cols": self.exog_cols, "group_models_": self.group_models_}

    def udf_set_params(self, params: dict_type):
        super().udf_set_params(params)
        self.output_col = params["output_col"]
        self.input_cols = params["input_cols"]
        self.p = params["p"]
        self.q = params["q"]
        self.vol = params["vol"]
        self.mean = params["mean"]
        self.exog_cols = params["exog_cols"]
        self.group_models_ = params["group_models_"]
