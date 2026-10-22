# -*- coding: utf-8 -*-
# 统计模型测试: ARIMA/SARIMA/ARIMAX/SARIMAX/GARCH(分组框架版)
import os

import numpy as np

from _ts_suite_utils import TSSuite, generate_test_data

from easymlops.ts.statistical.arima import (ArimaRegression, SarimaRegression,
                                            ArimaxRegression, SarimaxRegression,
                                            GarchRegression)

suite = TSSuite("statistical")
df = generate_test_data(200)

# ArimaRegression
suite.verify_forecast_model(
    lambda: ArimaRegression(input_cols=["value"], p=1, d=1, q=1),
    df, "arima_predict", "arima", "tests/arima_model.pkl")

# SarimaRegression
suite.verify_forecast_model(
    lambda: SarimaRegression(input_cols=["value"], p=1, d=1, q=1,
                             seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12),
    df, "sarima_predict", "sarima", "tests/sarima_model.pkl")

# ArimaxRegression(外生变量)
suite.verify_forecast_model(
    lambda: ArimaxRegression(input_cols=["value"], exog_cols=["feature1"], p=1, d=1, q=1),
    df, "arimax_predict", "arimax", "tests/arimax_model.pkl")

# SarimaxRegression(外生变量 + 季节)
suite.verify_forecast_model(
    lambda: SarimaxRegression(input_cols=["value"], exog_cols=["feature1"],
                              p=1, d=1, q=1,
                              seasonal_p=1, seasonal_d=1, seasonal_q=1, seasonal_period=12),
    df, "sarimax_predict", "sarimax", "tests/sarimax_model.pkl")

# GarchRegression(波动率)
suite.verify_forecast_model(
    lambda: GarchRegression(input_cols=["value"], p=1, q=1),
    df, "garch_predict", "garch", "tests/garch_model.pkl", stochastic=True)

# 分组: 每组独立拟合, 输出有限, 各组预测互不相同
group_df = generate_test_data(120, seed=7)
group_df["group"] = ["A"] * 60 + ["B"] * 60
g = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                    group_cols=["group"], time_col="date")
g.fit(group_df)
gout = g.transform(group_df)
firsts = gout.groupby("group")["arima_predict"].first()
suite.check("arima_grouped_finite",
            len(gout) == 120 and
            np.isfinite(gout["arima_predict"].astype(float)).all() and
            firsts.nunique() == 2)

suite.finish()
