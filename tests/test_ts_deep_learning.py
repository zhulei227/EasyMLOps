# -*- coding: utf-8 -*-
# 深度学习模型测试: NBeats/NHiTS/DeepAR/GP(权重持久化 + 单条推理)
import sys

from _ts_suite_utils import TSSuite, generate_test_data

try:
    import torch  # noqa: F401
    from easymlops.ts.deep_learning.nbeats import (NBeatsRegression, NHiTSRegression,
                                                   DeepARRegression, GPRegression)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

suite = TSSuite("deep_learning")
df = generate_test_data(200)

if not HAS_TORCH:
    print("PyTorch not available, all deep learning tests skipped")
    suite.finish()

suite.verify_forecast_model(
    lambda: NBeatsRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5),
    df, "nbeats_predict", "nbeats", "tests/nbeats_model.pkl")

suite.verify_forecast_model(
    lambda: NHiTSRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5),
    df, "nhits_predict", "nhits", "tests/nhits_model.pkl")

suite.verify_forecast_model(
    lambda: DeepARRegression(input_cols=["value"], time_period=30, forecast_horizon=7, epochs=5),
    df, "deepar_predict", "deepar", "tests/deepar_model.pkl")

suite.verify_forecast_model(
    lambda: GPRegression(input_cols=["value"], time_period=30, forecast_horizon=7),
    df, "gp_predict", "gp", "tests/gp_model.pkl")

suite.finish()
