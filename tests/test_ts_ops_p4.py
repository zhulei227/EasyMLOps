# -*- coding: utf-8 -*-
# TS P4 测试: 深度模型统一修复验收(权重持久化 / transform_single / input_cols 自动解析)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd

from easymlops.ts.deep_learning.nbeats import (NBeatsRegression, NHiTSRegression,
                                               DeepARRegression, GPRegression)
from easymlops.ts.transformer.transformer import (TFTRegression, InformerRegression,
                                                  AutoformerRegression, FEDformerRegression,
                                                  PatchTSTRegression, TimesNetRegression,
                                                  iTransformerRegression)
from easymlops.ts.state_space.state_space import (DeepStateRegression, MambaRegression,
                                                  LiquidS4Regression)
from easymlops.ts.generative.generative import (VAERegression, NormalizingFlowRegression,
                                                DiffusionRegression)

PASS, FAIL = [], []
P4_DIR = "d:/datas/share/traes/easymlops-trae/tests"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


# 测试数据: date 在首列(验证 input_cols=None 时自动跳过非数值列)
rng = np.random.RandomState(11)
n = 90
df = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=n, freq="D"),
    "value": np.cumsum(rng.normal(0, 1, n)) + 50.0,
    "feature1": rng.normal(0, 1, n),
})

# 全部 17 个预测模型(最小配置); stochastic=True 的模型 transform 含随机采样, 需固定种子
MODELS = [
    ("NBeats", lambda: NBeatsRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("NHiTS", lambda: NHiTSRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("DeepAR", lambda: DeepARRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("GP", lambda: GPRegression(time_period=16, forecast_horizon=3), False),
    ("TFT", lambda: TFTRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("Informer", lambda: InformerRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("Autoformer", lambda: AutoformerRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("FEDformer", lambda: FEDformerRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("PatchTST", lambda: PatchTSTRegression(time_period=16, forecast_horizon=3, epochs=1,
                                            patch_len=8, stride=4), False),
    ("TimesNet", lambda: TimesNetRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("iTransformer", lambda: iTransformerRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("DeepState", lambda: DeepStateRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("Mamba", lambda: MambaRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("LiquidS4", lambda: LiquidS4Regression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("VAE", lambda: VAERegression(time_period=16, forecast_horizon=3, epochs=1), True),
    ("NormalizingFlow", lambda: NormalizingFlowRegression(time_period=16, forecast_horizon=3, epochs=1), False),
    ("Diffusion", lambda: DiffusionRegression(time_period=16, forecast_horizon=3, epochs=1), True),
]

import torch

for name, factory, stochastic in MODELS:
    print("=== {} ===".format(name))
    try:
        m = factory()
        m.fit(df)

        # 1. 批量 transform: 行数守恒且有限
        if stochastic:
            torch.manual_seed(42)
        out = m.transform(df)
        output_col = [c for c in out.columns if c.endswith("_predict")]
        check("{}_fit_transform_rows".format(name),
              len(out) == n and len(output_col) == 1 and
              np.isfinite(out[output_col[0]].astype(float)).all())

        # 2. input_cols 自动解析为第一个数值列
        check("{}_input_cols_auto".format(name), m.input_cols == ["value"],
              str(m.input_cols))

        # 3. save/load 后预测一致(权重持久化)
        path = os.path.join(P4_DIR, "ts_p4_{}.pkl".format(name.lower()))
        m.save(path)
        m2 = factory()
        m2.load(path)
        if stochastic:
            torch.manual_seed(42)
        out2 = m2.transform(df)
        check("{}_save_load_consistent".format(name),
              np.allclose(out[output_col[0]].astype(float).to_numpy(),
                          out2[output_col[0]].astype(float).to_numpy(), atol=1e-6))
        os.remove(path)

        # 4. transform_single 可用且有限
        single = m.transform_single(dict(df.iloc[-1]))
        check("{}_single_finite".format(name),
              output_col[0] in single and np.isfinite(float(single[output_col[0]])))
    except Exception as e:
        check("{}_fit_transform_rows".format(name), False, "{}: {}".format(type(e).__name__, str(e)[:100]))

# ============ 超参恢复顺序: 非默认超参模型 load 后可正确重建 ============
print("=== hyperparam restore ===")
m = NBeatsRegression(time_period=16, forecast_horizon=3, hidden_units=32, epochs=1)
m.fit(df)
path = os.path.join(P4_DIR, "ts_p4_hu.pkl")
m.save(path)
m2 = NBeatsRegression(time_period=16, forecast_horizon=3, epochs=1)  # 构造时 hidden_units=64(默认)
m2.load(path)
before = m.transform(df)["nbeats_predict"].to_numpy()
after = m2.transform(df)["nbeats_predict"].to_numpy()
check("non_default_hyperparam_restored",
      np.allclose(before, after, atol=1e-6) and abs(before[0]) > 0)
os.remove(path)

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
