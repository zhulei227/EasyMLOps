# -*- coding: utf-8 -*-
# TS P5 测试: GroupForecast 分组预测包装器(统计模型 + torch 深度模型)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from easymlops.ts.core.group_forecast import GroupForecast
from easymlops.ts.statistical import ArimaRegression
from easymlops.ts.utils import split_time_series

PASS, FAIL = [], []
P5_DIR = "d:/datas/share/traes/easymlops-trae/tests"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(11)
n = 90
dates = pd.date_range("2026-01-01", periods=n, freq="D")
rows = []
for g, base in [("A", 50.0), ("B", 120.0)]:
    vals = base + np.cumsum(rng.normal(0, 1, n))
    for t, v in zip(dates, vals):
        rows.append({"group": g, "date": t, "value": v})
data = pd.DataFrame(rows)
train, test = split_time_series(data, time_col="date", group_cols=["group"], test_size=0.3)
test_shuffled = test.sample(frac=1.0, random_state=3)   # 打乱顺序, 验证按原位置写回
test_shuffled = test_shuffled.reset_index(drop=True)

# ============ 1. GroupForecast(ARIMA) 对拍每组的 statsmodels ============
print("=== 1. arima grouped wrapper ===")
gf = GroupForecast(model=ArimaRegression(input_cols=["value"], p=1, d=1, q=1),
                   group_cols=["group"], time_col="date")
gf.fit(train)
out = gf.transform(test_shuffled)

from statsmodels.tsa.arima.model import ARIMA
ok, per_group_first = True, {}
for g in ["A", "B"]:
    gtrain = train[train["group"] == g].sort_values("date")
    gtest = test_shuffled[test_shuffled["group"] == g].sort_values("date")
    expect = np.asarray(ARIMA(gtrain["value"].to_numpy(), order=(1, 1, 1),
                              enforce_stationarity=False,
                              enforce_invertibility=False).fit()
                        .forecast(steps=len(gtest)), dtype=float)
    got = out[out["group"] == g].sort_values("date")["arima_predict"].to_numpy()
    per_group_first[g] = got[0]
    if not np.allclose(got, expect, atol=1e-6):
        ok = False
check("grouped_arima_matches_statsmodels", ok)
check("grouped_arima_rows_kept",
      len(out) == len(test_shuffled) and list(out.index) == list(test_shuffled.index))
check("grouped_arima_independent", per_group_first["A"] != per_group_first["B"])

# ============ 2. GroupForecast(ARIMA) 与原生分组 ArimaRegression 一致 ============
print("=== 2. wrapper vs native grouped ===")
native = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                         group_cols=["group"], time_col="date")
native.fit(train)
native_out = native.transform(test_shuffled)
check("wrapper_equals_native_grouped",
      np.allclose(out.sort_values(["group", "date"])["arima_predict"].to_numpy(),
                  native_out.sort_values(["group", "date"])["arima_predict"].to_numpy(),
                  atol=1e-8))

# ============ 3. transform_single 与 unseen group ============
print("=== 3. single & unseen ===")
ok = True
for g in ["A", "B"]:
    gtrain = train[train["group"] == g].sort_values("date")
    expect = float(np.asarray(ARIMA(gtrain["value"].to_numpy(), order=(1, 1, 1),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False).fit()
                              .forecast(steps=1), dtype=float)[0])
    rec = dict(test_shuffled[test_shuffled["group"] == g].iloc[0])
    if not np.isclose(gf.transform_single(rec)["arima_predict"], expect, atol=1e-6):
        ok = False
check("grouped_arima_single_matches_forecast1", ok)
try:
    gf.transform(pd.DataFrame([{"group": "C", "date": dates[0], "value": 1.0}]))
    check("unseen_group_raises", False, "no exception raised")
except Exception as e:
    check("unseen_group_raises", "not fitted" in str(e), str(e)[:60])

# ============ 4. save/load(含各组模型权重) ============
print("=== 4. save/load ===")
pkl = os.path.join(P5_DIR, "ts_p5_group_forecast.pkl")
gf.save(pkl)
gf_loaded = GroupForecast(group_cols=["group"], time_col="date")
gf_loaded.load(pkl)
out2 = gf_loaded.transform(test_shuffled)
check("grouped_arima_save_load_consistent",
      np.allclose(out.sort_values(["group", "date"])["arima_predict"].to_numpy(),
                  out2.sort_values(["group", "date"])["arima_predict"].to_numpy(),
                  atol=1e-6))
os.remove(pkl)

# ============ 5. torch 深度模型(每家族一个代表) ============
print("=== 5. torch grouped ===")
try:
    import torch  # noqa: F401
    from easymlops.ts.deep_learning.nbeats import NBeatsRegression, GPRegression
    from easymlops.ts.transformer.transformer import TFTRegression
    from easymlops.ts.state_space.state_space import MambaRegression
    from easymlops.ts.generative.generative import DiffusionRegression

    torch.manual_seed(42)
    cases = [
        ("nbeats", lambda: NBeatsRegression(input_cols=["value"], time_period=16,
                                            forecast_horizon=3, epochs=1),
         "nbeats_predict", False),
        ("gp", lambda: GPRegression(input_cols=["value"], time_period=16,
                                    forecast_horizon=3), "gp_predict", False),
        ("tft", lambda: TFTRegression(input_cols=["value"], time_period=16,
                                      forecast_horizon=3, epochs=1), "tft_predict", False),
        ("mamba", lambda: MambaRegression(input_cols=["value"], time_period=16,
                                          forecast_horizon=3, epochs=1), "mamba_predict", False),
        ("diffusion", lambda: DiffusionRegression(input_cols=["value"], time_period=16,
                                                  forecast_horizon=3, epochs=1),
         "diffusion_predict", True),
    ]
    for name, factory, output_col, stochastic in cases:
        torch.manual_seed(42)
        g = GroupForecast(model=factory(), group_cols=["group"], time_col="date")
        g.fit(train)
        if stochastic:
            torch.manual_seed(42)
        gout = g.transform(test_shuffled)
        ok = (len(gout) == len(test_shuffled) and
              np.isfinite(gout[output_col].astype(float)).all())
        # 各组独立: 组首预测不同
        firsts = gout.groupby("group")[output_col].first()
        ok = ok and firsts.nunique() == 2
        check("grouped_{}".format(name), ok)

        # save/load 一致
        pkl = os.path.join(P5_DIR, "ts_p5_{}.pkl".format(name))
        g.save(pkl)
        g2 = GroupForecast(group_cols=["group"], time_col="date")
        g2.load(pkl)
        if stochastic:
            torch.manual_seed(42)
        gout2 = g2.transform(test_shuffled)
        check("grouped_{}_save_load".format(name),
              np.allclose(gout.sort_values(["group", "date"])[output_col].astype(float).to_numpy(),
                          gout2.sort_values(["group", "date"])[output_col].astype(float).to_numpy(),
                          atol=1e-6))
        os.remove(pkl)
        # 单条
        rec = dict(test_shuffled.iloc[0])
        check("grouped_{}_single".format(name),
              output_col in g.transform_single(rec) and
              np.isfinite(float(g.transform_single(dict(rec))[output_col])))
except ImportError:
    print("PyTorch not available, torch grouped tests skipped")

# ============ 6. 模板缺失报错 ============
print("=== 6. template missing ===")
try:
    GroupForecast(group_cols=["group"]).fit(data)
    check("template_required", False, "no exception raised")
except Exception as e:
    check("template_required", "template model" in str(e), str(e)[:60])

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
