# -*- coding: utf-8 -*-
# 状态空间模型测试: DeepState/Mamba/LiquidS4
import sys

from _ts_suite_utils import TSSuite, generate_test_data

try:
    import torch  # noqa: F401
    from easymlops.ts.state_space.state_space import (DeepStateRegression,
                                                      MambaRegression,
                                                      LiquidS4Regression)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

suite = TSSuite("state_space")
df = generate_test_data(200)

if not HAS_TORCH:
    print("PyTorch not available, all state space tests skipped")
    suite.finish()

suite.verify_forecast_model(
    lambda: DeepStateRegression(input_cols=["value"], time_period=30,
                                forecast_horizon=7, epochs=5),
    df, "deepstate_predict", "deepstate", "tests/deepstate_model.pkl")

suite.verify_forecast_model(
    lambda: MambaRegression(input_cols=["value"], time_period=30,
                            forecast_horizon=7, epochs=5),
    df, "mamba_predict", "mamba", "tests/mamba_model.pkl")

suite.verify_forecast_model(
    lambda: LiquidS4Regression(input_cols=["value"], time_period=30,
                               forecast_horizon=7, epochs=5),
    df, "liquids4_predict", "liquids4", "tests/liquids4_model.pkl")

suite.finish()
