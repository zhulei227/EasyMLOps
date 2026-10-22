# -*- coding: utf-8 -*-
# Transformer 模型测试: TFT/Informer/Autoformer/FEDformer/PatchTST/TimesNet/iTransformer
import sys

from _ts_suite_utils import TSSuite, generate_test_data

try:
    import torch  # noqa: F401
    from easymlops.ts.transformer.transformer import (TFTRegression, InformerRegression,
                                                      AutoformerRegression, FEDformerRegression,
                                                      PatchTSTRegression, TimesNetRegression,
                                                      iTransformerRegression)
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

suite = TSSuite("transformer")
df = generate_test_data(200)

if not HAS_TORCH:
    print("PyTorch not available, all transformer tests skipped")
    suite.finish()

CASES = [
    ("tft", lambda: TFTRegression(input_cols=["value"], time_period=30,
                                  forecast_horizon=7, epochs=5), "tft_predict"),
    ("informer", lambda: InformerRegression(input_cols=["value"], time_period=30,
                                            forecast_horizon=7, epochs=5), "informer_predict"),
    ("autoformer", lambda: AutoformerRegression(input_cols=["value"], time_period=30,
                                                forecast_horizon=7, epochs=5), "autoformer_predict"),
    ("fedformer", lambda: FEDformerRegression(input_cols=["value"], time_period=30,
                                              forecast_horizon=7, epochs=5), "fedformer_predict"),
    ("patchtst", lambda: PatchTSTRegression(input_cols=["value"], time_period=30,
                                            forecast_horizon=7, epochs=5), "patchtst_predict"),
    ("timesnet", lambda: TimesNetRegression(input_cols=["value"], time_period=30,
                                            forecast_horizon=7, epochs=5), "timesnet_predict"),
    ("itransformer", lambda: iTransformerRegression(input_cols=["value"], time_period=30,
                                                    forecast_horizon=7, epochs=5), "itransformer_predict"),
]

for name, factory, output_col in CASES:
    suite.verify_forecast_model(factory, df, output_col, name,
                                "tests/{}_model.pkl".format(name))

suite.finish()
