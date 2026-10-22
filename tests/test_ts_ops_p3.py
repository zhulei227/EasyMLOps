# -*- coding: utf-8 -*-
# TS P3 操作测试: 统计模型(ARIMA/SARIMA/ARIMAX/SARIMAX 分组迁移 + GARCH)
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from easymlops import TSPipeLine
from easymlops.ts.statistical import (ArimaRegression, SarimaRegression,
                                      ArimaxRegression, SarimaxRegression,
                                      GarchRegression)
from easymlops.ts.features import FourierFeature
from easymlops.ts.utils import split_time_series

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


def manual_fit_forecast(target, exog, p, d, q, sp, sd, sq, period, steps, fc_exog=None):
    """手动 statsmodels 拟合+预测(与框架内实现同参数), 作为对拍基准"""
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if period:
        model = SARIMAX(target, exog=exog, order=(p, d, q),
                        seasonal_order=(sp, sd, sq, period),
                        enforce_stationarity=False, enforce_invertibility=False)
    else:
        model = ARIMA(target, exog=exog, order=(p, d, q),
                      enforce_stationarity=False, enforce_invertibility=False)
    return np.asarray(model.fit().forecast(steps=steps, exog=fc_exog), dtype=float)


rng = np.random.RandomState(11)
n_per_group = 80
dates = pd.date_range("2026-01-01", periods=n_per_group, freq="D")
rows = []
for g, base in [("A", 50.0), ("B", 120.0)]:
    vals = base + np.cumsum(rng.normal(0, 1, n_per_group))
    for t, v in zip(dates, vals):
        rows.append({"group": g, "date": t, "value": v, "ex1": rng.normal()})
data = pd.DataFrame(rows)
train, test = split_time_series(data, time_col="date", group_cols=["group"], test_size=0.3)
test_sorted = test.sort_values(["group", "date"]).reset_index(drop=True)

# ============ 1. ARIMA 全局单序列(对拍 statsmodels) ============
print("=== 1. ARIMA global ===")
single_train = train[train["group"] == "A"].sort_values("date").reset_index(drop=True)
single_test = test[test["group"] == "A"].sort_values("date").reset_index(drop=True)
pipe = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                       group_cols=None, time_col="date")
pipe.fit(single_train)
out = pipe.transform(single_test)
expect = manual_fit_forecast(single_train["value"].to_numpy(), None, 1, 1, 1,
                             0, 0, 0, None, len(single_test))
check("arima_global_matches_statsmodels",
      len(out) == len(single_test) and
      np.allclose(out["arima_predict"].to_numpy(), expect, atol=1e-6))

# ============ 2. ARIMA 分组(每组独立对拍) ============
print("=== 2. ARIMA grouped ===")
gpipe = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                        group_cols=["group"], time_col="date")
gpipe.fit(train)
out = gpipe.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gtrain = train[train["group"] == g].sort_values("date")
    gtest = test_sorted[test_sorted["group"] == g].sort_values("date")
    expect = manual_fit_forecast(gtrain["value"].to_numpy(), None, 1, 1, 1,
                                 0, 0, 0, None, len(gtest))
    got = out[out["group"] == g].sort_values("date")["arima_predict"].to_numpy()
    if not np.allclose(got, expect, atol=1e-6):
        ok = False
check("arima_grouped_matches_per_group_statsmodels", ok)
check("arima_row_count_preserved", len(out) == len(test_sorted) and
      list(out.index) == list(test_sorted.index))

# ============ 3. SARIMA 季节序列(对拍) ============
print("=== 3. SARIMA ===")
stl_rows = []
for g in ["A", "B"]:
    for i, t in enumerate(pd.date_range("2026-01-01", periods=100, freq="D")):
        stl_rows.append({"group": g, "date": t,
                         "value": 10 + 3.0 * np.sin(2 * np.pi * i / 7) +
                                  0.02 * i + rng.normal(0, 0.2),
                         "ex1": rng.normal()})
stl_data = pd.DataFrame(stl_rows)
strain, stest = split_time_series(stl_data, time_col="date",
                                  group_cols=["group"], test_size=0.3)
stest_sorted = stest.sort_values(["group", "date"]).reset_index(drop=True)
spipe = SarimaRegression(input_cols=["value"], p=1, d=0, q=1,
                         seasonal_p=1, seasonal_d=0, seasonal_q=1, seasonal_period=7,
                         group_cols=["group"], time_col="date")
spipe.fit(strain)
sout = spipe.transform(stest_sorted)
ok = True
for g in ["A", "B"]:
    gtrain = strain[strain["group"] == g].sort_values("date")
    gtest = stest_sorted[stest_sorted["group"] == g].sort_values("date")
    expect = manual_fit_forecast(gtrain["value"].to_numpy(), None, 1, 0, 1,
                                 1, 0, 1, 7, len(gtest))
    got = sout[sout["group"] == g].sort_values("date")["sarima_predict"].to_numpy()
    if not np.allclose(got, expect, atol=1e-6):
        ok = False
check("sarima_grouped_matches_statsmodels", ok)
# 季节性预测应优于均值基线
ok = True
for g in ["A", "B"]:
    gtest = stest_sorted[stest_sorted["group"] == g]
    got = sout[sout["group"] == g].sort_values("date")["sarima_predict"].to_numpy()
    truth = gtest.sort_values("date")["value"].to_numpy()
    rmse_pred = float(np.sqrt(np.mean((got - truth) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((truth - gtrain["value"].mean()) ** 2)))
    if rmse_pred >= rmse_naive:
        ok = False
check("sarima_beats_mean_baseline", ok)

# ============ 4. ARIMAX / SARIMAX 外生变量(对拍) ============
print("=== 4. ARIMAX & SARIMAX ===")
axpipe = ArimaxRegression(input_cols=["value"], exog_cols=["ex1"], p=1, d=1, q=1,
                          group_cols=["group"], time_col="date")
axpipe.fit(train)
axout = axpipe.transform(test_sorted)
sxpipe = SarimaxRegression(input_cols=["value"], exog_cols=["ex1"], p=1, d=0, q=1,
                           seasonal_p=1, seasonal_d=0, seasonal_q=1, seasonal_period=7,
                           group_cols=["group"], time_col="date")
sxpipe.fit(strain)
sxout = sxpipe.transform(stest_sorted)
ok_ax, ok_sx = True, True
for g in ["A", "B"]:
    gtrain = train[train["group"] == g].sort_values("date")
    gtest = test_sorted[test_sorted["group"] == g].sort_values("date")
    expect = manual_fit_forecast(gtrain["value"].to_numpy(),
                                 gtrain["ex1"].to_numpy().reshape(-1, 1),
                                 1, 1, 1, 0, 0, 0, None, len(gtest),
                                 gtest["ex1"].to_numpy().reshape(-1, 1))
    got = axout[axout["group"] == g].sort_values("date")["arimax_predict"].to_numpy()
    if not np.allclose(got, expect, atol=1e-6):
        ok_ax = False
    gstrain = strain[strain["group"] == g].sort_values("date")
    gstest = stest_sorted[stest_sorted["group"] == g].sort_values("date")
    expect_s = manual_fit_forecast(gstrain["value"].to_numpy(),
                                   gstrain["ex1"].to_numpy().reshape(-1, 1),
                                   1, 0, 1, 1, 0, 1, 7, len(gstest),
                                   gstest["ex1"].to_numpy().reshape(-1, 1))
    got_s = sxout[sxout["group"] == g].sort_values("date")["sarimax_predict"].to_numpy()
    if not np.allclose(got_s, expect_s, atol=1e-6):
        ok_sx = False
check("arimax_grouped_matches_statsmodels", ok_ax)
check("sarimax_grouped_matches_statsmodels", ok_sx)

# ============ 5. transform_single = 每组 forecast(1) ============
print("=== 5. transform_single ===")
ok = True
for g in ["A", "B"]:
    gtrain = train[train["group"] == g].sort_values("date")
    gtest = test_sorted[test_sorted["group"] == g].sort_values("date")
    expect = float(manual_fit_forecast(gtrain["value"].to_numpy(), None, 1, 1, 1,
                                       0, 0, 0, None, 1)[0])
    rec = dict(gtest.iloc[0])
    got = gpipe.transform_single(rec)["arima_predict"]
    if not np.isclose(got, expect, atol=1e-6):
        ok = False
check("arima_single_equals_forecast1", ok)

# ============ 6. 未见新组报错 ============
print("=== 6. unseen group ===")
try:
    gpipe.transform(pd.DataFrame([{"group": "C", "date": dates[0], "value": 1.0,
                                   "ex1": 0.0}]))
    check("unseen_group_raises", False, "no exception raised")
except Exception as e:
    check("unseen_group_raises", "not fitted" in str(e), str(e)[:80])

# ============ 7. pipeline 集成 ============
print("=== 7. pipeline ===")
p1 = TSPipeLine()
p1.pipe(FourierFeature(period=7, order=2, group_cols=["group"], time_col="date")) \
  .pipe(ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                        group_cols=["group"], time_col="date"))
p1.fit(train)
pout = p1.transform(test_sorted)
check("pipeline_arima_rows_and_col",
      len(pout) == len(test_sorted) and "arima_predict" in pout.columns and
      np.isfinite(pout["arima_predict"]).all())

# ============ 8. save/load(模型对象持久化) ============
print("=== 8. save/load ===")
before = gpipe.transform(test_sorted)
gpipe.save("d:/datas/share/traes/easymlops-trae/tests/ts_p3_arima.pkl")
loaded = ArimaRegression(input_cols=["value"], p=1, d=1, q=1,
                         group_cols=["group"], time_col="date")
loaded.load("d:/datas/share/traes/easymlops-trae/tests/ts_p3_arima.pkl")
after = loaded.transform(test_sorted)
check("arima_model_persisted",
      np.allclose(before["arima_predict"].to_numpy(), after["arima_predict"].to_numpy()))

# ============ 9. GARCH ============
print("=== 9. GARCH ===")
try:
    vol = 0.02 + 0.02 * np.abs(np.sin(np.arange(n_per_group) / 10.0))
    g_rows = []
    for g in ["A", "B"]:
        rets = rng.normal(0.001, vol)
        for t, r in zip(dates, rets):
            g_rows.append({"group": g, "date": t, "ret": r})
    gdata = pd.DataFrame(g_rows)
    gtrain, gtest = split_time_series(gdata, time_col="date",
                                      group_cols=["group"], test_size=0.3)
    gtest_sorted = gtest.sort_values(["group", "date"]).reset_index(drop=True)
    garch = GarchRegression(input_cols=["ret"], p=1, q=1,
                            group_cols=["group"], time_col="date")
    garch.fit(gtrain)
    gout = garch.transform(gtest_sorted)
    check("garch_fit_transform", len(gout) == len(gtest_sorted) and
          np.isfinite(gout["garch_predict"]).all())
    rec = dict(gtest_sorted.iloc[0])
    check("garch_single_finite",
          np.isfinite(garch.transform_single(rec)["garch_predict"]))
except Exception as e:
    check("garch_fit_transform", False, str(e)[:120])
    check("garch_single_finite", False, "skipped due to error above")

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
