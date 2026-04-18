import pandas as pd
import numpy as np
from easymlops.ts.core.pipe import PipeBase
import warnings

warnings.filterwarnings("ignore")


class ArimaRegression(PipeBase):
    """
    ARIMA时间序列预测模型
    """

    def __init__(self, output_col="arima_predict", input_cols=None, p=1, d=1, q=1,
                 seasonal_p=0, seasonal_d=0, seasonal_q=0, seasonal_period=None,
                 exog_cols=None, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param p: AR阶数
        :param d: 差分阶数
        :param q: MA阶数
        :param seasonal_p: 季节性AR阶数
        :param seasonal_d: 季节性差分阶数
        :param seasonal_q: 季节性MA阶数
        :param seasonal_period: 季节性周期
        :param exog_cols: 外生变量列名列表
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.p = p
        self.d = d
        self.q = q
        self.seasonal_p = seasonal_p
        self.seasonal_d = seasonal_d
        self.seasonal_q = seasonal_q
        self.seasonal_period = seasonal_period
        self.exog_cols = exog_cols
        self.model = None
        self.results = None

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        if self.input_cols is None or len(self.input_cols) == 0:
            numeric_cols = s.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.input_cols = [numeric_cols[0]]
            else:
                self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(float)

        if self.seasonal_period is not None and self.seasonal_period > 0:
            self.model = SARIMAX(
                target,
                exog=s[self.exog_cols].values if self.exog_cols else None,
                order=(self.p, self.d, self.q),
                seasonal_order=(self.seasonal_p, self.seasonal_d, self.seasonal_q, self.seasonal_period),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
        else:
            self.model = ARIMA(
                target,
                exog=s[self.exog_cols].values if self.exog_cols else None,
                order=(self.p, self.d, self.q),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

        self.results = self.model.fit()
        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
        exog = s[self.exog_cols].values if self.exog_cols else None
        predictions = self.results.forecast(steps=len(s), exog=exog)
        s = s.copy()
        s[self.output_col] = predictions
        return s


class SarimaRegression(ArimaRegression):
    """
    SARIMA时间序列预测模型 (季节性ARIMA)
    """

    def __init__(self, output_col="sarima_predict", input_cols=None,
                 p=1, d=1, q=1, seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12,
                 exog_cols=None, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param p: AR阶数
        :param d: 差分阶数
        :param q: MA阶数
        :param seasonal_p: 季节性AR阶数
        :param seasonal_d: 季节性差分阶数
        :param seasonal_q: 季节性MA阶数
        :param seasonal_period: 季节性周期 (如 12 for monthly, 7 for daily)
        :param exog_cols: 外生变量列名列表
        :param kwargs:
        """
        super().__init__(
            output_col=output_col, input_cols=input_cols,
            p=p, d=d, q=q,
            seasonal_p=seasonal_p, seasonal_d=seasonal_d, seasonal_q=seasonal_q,
            seasonal_period=seasonal_period, exog_cols=exog_cols, **kwargs
        )


class ArimaxRegression(ArimaRegression):
    """
    ARIMAX时间序列预测模型 (带外生变量的ARIMA)
    """

    def __init__(self, output_col="arimax_predict", input_cols=None, p=1, d=1, q=1,
                 exog_cols=None, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param p: AR阶数
        :param d: 差分阶数
        :param q: MA阶数
        :param exog_cols: 外生变量列名列表
        :param kwargs:
        """
        super().__init__(
            output_col=output_col, input_cols=input_cols,
            p=p, d=d, q=q,
            seasonal_p=0, seasonal_d=0, seasonal_q=0, seasonal_period=None,
            exog_cols=exog_cols, **kwargs
        )


class SarimaxRegression(ArimaRegression):
    """
    SARIMAX时间序列预测模型 (带外生变量的SARIMA)
    """

    def __init__(self, output_col="sarimax_predict", input_cols=None,
                 p=1, d=1, q=1, seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12,
                 exog_cols=None, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param p: AR阶数
        :param d: 差分阶数
        :param q: MA阶数
        :param seasonal_p: 季节性AR阶数
        :param seasonal_d: 季节性差分阶数
        :param seasonal_q: 季节性MA阶数
        :param seasonal_period: 季节性周期
        :param exog_cols: 外生变量列名列表
        :param kwargs:
        """
        super().__init__(
            output_col=output_col, input_cols=input_cols,
            p=p, d=d, q=q,
            seasonal_p=seasonal_p, seasonal_d=seasonal_d, seasonal_q=seasonal_q,
            seasonal_period=seasonal_period, exog_cols=exog_cols, **kwargs
        )


class GarchRegression(PipeBase):
    """
    GARCH时间序列波动率预测模型
    """

    def __init__(self, output_col="garch_predict", input_cols=None,
                 p=1, q=1, vol="Garch", mean="Constant",
                 exog_cols=None, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param p: GARCH阶数
        :param q: ARCH阶数
        :param vol: 波动率模型类型 (Garch, EGARCH, GJR_GARCH)
        :param mean: 均值模型类型 (Constant, ARX, HARX)
        :param exog_cols: 外生变量列名列表
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.p = p
        self.q = q
        self.vol = vol
        self.mean = mean
        self.exog_cols = exog_cols
        self.model = None
        self.results = None

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        from arch import arch_model

        if self.input_cols is None or len(self.input_cols) == 0:
            numeric_cols = s.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.input_cols = [numeric_cols[0]]
            else:
                self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(float) * 100

        self.model = arch_model(
            target,
            vol=self.vol,
            mean=self.mean,
            p=self.p,
            q=self.q,
            x=s[self.exog_cols].values if self.exog_cols else None,
            dist="normal",
            rescale=False
        )

        self.results = self.model.fit(disp="off")
        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
        predictions = self.results.forecast(horizon=len(s))
        s = s.copy()
        s[self.output_col] = predictions.mean.values[-1] / 100
        return s
