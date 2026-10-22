# -*- coding: utf-8 -*-
# 生成模型测试: VAE/NormalizingFlow/Diffusion
import sys

from _ts_suite_utils import TSSuite, generate_test_data

try:
    import torch  # noqa: F401
    from easymlops.ts.generative.generative import (VAERegression,
                                                    NormalizingFlowRegression,
                                                    DiffusionRegression)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

suite = TSSuite("generative")
df = generate_test_data(200)

if not HAS_TORCH:
    print("PyTorch not available, all generative tests skipped")
    suite.finish()

# VAE/Diffusion 含随机采样, 固定种子保证 save/load 一致性可比
suite.verify_forecast_model(
    lambda: VAERegression(input_cols=["value"], time_period=30,
                          forecast_horizon=7, epochs=5),
    df, "vae_predict", "vae", "tests/vae_model.pkl", stochastic=True)

suite.verify_forecast_model(
    lambda: NormalizingFlowRegression(input_cols=["value"], time_period=30,
                                      forecast_horizon=7, epochs=5),
    df, "normalizingflow_predict", "normalizingflow", "tests/normalizingflow_model.pkl")

suite.verify_forecast_model(
    lambda: DiffusionRegression(input_cols=["value"], time_period=30,
                                forecast_horizon=7, epochs=5),
    df, "diffusion_predict", "diffusion", "tests/diffusion_model.pkl", stochastic=True)

suite.finish()
