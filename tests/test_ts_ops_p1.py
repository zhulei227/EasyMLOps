# -*- coding: utf-8 -*-
# TS P1 操作测试: 重采样/网格补齐/expanding/季节均值/移动平均/平稳性检验/滚动回测
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from easymlops import TSPipeLine
from easymlops.table.ensemble import Parallel
from easymlops.ts.preprocessing import TimeResample, TimeGapsFill
from easymlops.ts.features import (ExpandingFeature, MovingAverageTrend, SeasonalMean,
                                   RollingFeature, LagFeature, FourierFeature)
from easymlops.ts.utils import calc_adf, calc_kpss, calc_acf_pacf, walk_forward_eval

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(7)

# 面板数据: 每组 60 天
n_per_group = 60
dates = pd.date_range("2026-01-01", periods=n_per_group, freq="D")
rows = []
for g, base in [("A", 10.0), ("B", 30.0)]:
    vals = base + np.cumsum(rng.normal(0, 1, n_per_group))
    for t, v in zip(dates, vals):
        rows.append({"group": g, "date": t, "value": v})
data = pd.DataFrame(rows)

# ============ 1. TimeResample ============
print("=== 1. TimeResample ===")
rs = TimeResample(time_col="date", rule="2D", agg="mean", group_cols=["group"])
rs.fit(data)
out = rs.transform(data)
check("resample_row_count", len(out) == 2 * (n_per_group // 2 + n_per_group % 2),
      "rows={}".format(len(out)))
ok = True
for g in ["A", "B"]:
    gout = out[out["group"] == g].reset_index(drop=True)
    gsrc = data[data["group"] == g].reset_index(drop=True)
    expect = gsrc["value"].groupby(gsrc.index // 2).mean().to_numpy()
    if not np.allclose(gout["value"].to_numpy(), expect):
        ok = False
check("resample_mean_matches_manual", ok)
check("resample_group_kept", set(out["group"].unique()) == {"A", "B"})

# ============ 2. TimeGapsFill ============
print("=== 2. TimeGapsFill ===")
has_gap = data.drop(index=[10, 10 + n_per_group]).reset_index(drop=True)  # 每组删 1 行
gf = TimeGapsFill(time_col="date", freq="D", fill_value=0.0,
                  mark_col="is_filled", group_cols=["group"])
gf.fit(has_gap)
out = gf.transform(has_gap)
check("gaps_row_restored", len(out) == len(data), "rows={}".format(len(out)))
check("gaps_mark_count", int(out["is_filled"].sum()) == 2)
check("gaps_fill_value", (out[out["is_filled"] == 1]["value"] == 0.0).all())
check("gaps_group_kept", (out.groupby("group").size() == n_per_group).all())

# ============ 3. ExpandingFeature ============
print("=== 3. ExpandingFeature ===")
exp = ExpandingFeature(cols=["value"], stats=["mean", "std"], min_periods=2,
                       group_cols=["group"], time_col="date")
exp.fit(data)
out = exp.transform(data)
ok = True
for g in ["A", "B"]:
    gvals = data[data["group"] == g]["value"].to_numpy()
    s = pd.Series(gvals)
    expect_mean = s.expanding(min_periods=2).mean().fillna(0.0).to_numpy()
    expect_std = s.expanding(min_periods=2).std().fillna(0.0).to_numpy()
    got_m = out[out["group"] == g]["value_expanding_mean"].to_numpy()
    got_s = out[out["group"] == g]["value_expanding_std"].to_numpy()
    if not (np.allclose(got_m, expect_mean) and np.allclose(got_s, expect_std)):
        ok = False
check("expanding_matches_pandas", ok)
exp.reset_cache_data()
tb = data.iloc[-20:].reset_index(drop=True)
batch_out = exp.transform(tb)
exp.reset_cache_data()
singles = pd.DataFrame([exp.transform_single(dict(r)) for r in tb.to_dict("records")])
check("expanding_batch_single_consistent",
      np.allclose(batch_out["value_expanding_mean"].to_numpy(),
                  singles["value_expanding_mean"].to_numpy()) and
      np.allclose(batch_out["value_expanding_std"].to_numpy(),
                  singles["value_expanding_std"].to_numpy(), equal_nan=True))

# ============ 4. SeasonalMean ============
print("=== 4. SeasonalMean ===")
effect = np.array([5.0, 1.0, 1.0, 1.0, 1.0, -3.0, -3.0])
seasonal_rows = []
for g, base in [("A", 0.0), ("B", 100.0)]:
    for i, t in enumerate(pd.date_range("2026-01-01", periods=70, freq="D")):
        seasonal_rows.append({"group": g, "date": t,
                              "value": base + effect[i % 7] + rng.normal(0, 0.1)})
seasonal = pd.DataFrame(seasonal_rows)

sm = SeasonalMean(cols=["value"], period=7, per_group=True, mode="both",
                  group_cols=["group"], time_col="date")
sm.fit(seasonal)
out = sm.transform(seasonal)
check("seasonal_profile_close_to_effect",
      all(abs(sm.group_profiles_[("A",)]["value"][i] - effect[i]) < 0.2 for i in range(7)))
resid_var = out["value_deseasonal"].var()
orig_var = seasonal["value"].var()
check("deseasonal_reduces_variance", resid_var < orig_var * 0.2,
      "{:.3f} -> {:.3f}".format(orig_var, resid_var))
check("seasonal_cols_present",
      "value_seasonal" in out.columns and "value_deseasonal" in out.columns)
sm.reset_cache_data()
s0 = sm.transform_single(dict(seasonal.iloc[0]))
s1 = sm.transform_single(dict(seasonal.iloc[1]))
check("seasonal_single_phase_advance",
      abs(s0["value_seasonal"] - effect[0]) < 0.2 and abs(s1["value_seasonal"] - effect[1]) < 0.2)

# ============ 5. MovingAverageTrend ============
print("=== 5. MovingAverageTrend ===")
mat = MovingAverageTrend(cols=["value"], window=5, center=False,
                         group_cols=["group"], time_col="date")
mat.fit(data)
out = mat.transform(data)
ok = True
for g in ["A", "B"]:
    gvals = data[data["group"] == g]["value"].to_numpy()
    expect = pd.Series(gvals).rolling(5, min_periods=5).mean().fillna(0.0).to_numpy()
    if not np.allclose(out[out["group"] == g]["value_ma5"].to_numpy(), expect):
        ok = False
check("ma_trailing_matches_pandas", ok)
mat.reset_cache_data()
tb = data.iloc[-10:].reset_index(drop=True)
batch_out = mat.transform(tb)
mat.reset_cache_data()
singles = pd.DataFrame([mat.transform_single(dict(r)) for r in tb.to_dict("records")])
check("ma_batch_single_consistent",
      np.allclose(batch_out["value_ma5"].to_numpy(), singles["value_ma5"].to_numpy()))
matc = MovingAverageTrend(cols=["value"], window=5, center=True,
                          group_cols=["group"], time_col="date")
matc.fit(data)
outc = matc.transform(data)
ok = True
for g in ["A", "B"]:
    gvals = data[data["group"] == g]["value"].to_numpy()
    expect = pd.Series(gvals).rolling(5, center=True, min_periods=5).mean().fillna(0.0).to_numpy()
    if not np.allclose(outc[out["group"] == g]["value_ma5"].to_numpy(), expect):
        ok = False
check("ma_centered_matches_pandas", ok)

# ============ 6. 平稳性检验 ============
print("=== 6. stationarity tests ===")
white_noise = pd.Series(rng.normal(0, 1, 500))
trending = pd.Series(np.arange(500) * 0.1 + rng.normal(0, 1, 500).cumsum())
adf_st = calc_adf(white_noise)
adf_tr = calc_adf(trending)
check("adf_noise_stationary", adf_st["stationary"], "p={:.3f}".format(adf_st["pvalue"]))
check("adf_trending_nonstationary", not adf_tr["stationary"], "p={:.3f}".format(adf_tr["pvalue"]))
kpss_st = calc_kpss(white_noise)
check("kpss_noise_stationary", kpss_st["stationary"], "p={:.3f}".format(kpss_st["pvalue"]))

# ============ 7. ACF/PACF ============
print("=== 7. acf/pacf ===")
ar = np.zeros(800)
for i in range(1, 800):
    ar[i] = 0.8 * ar[i - 1] + rng.normal(0, 1)
acf_v, pacf_v = calc_acf_pacf(ar, nlags=10)
check("acf_lag1_close_to_phi", abs(acf_v[1] - 0.8) < 0.1, "acf[1]={:.3f}".format(acf_v[1]))
check("acf_decays", acf_v[5] < acf_v[1])

# ============ 8. walk_forward_eval ============
print("=== 8. walk_forward_eval ===")
train_lens = []


def naive_predict(train_df, test_df):
    train_lens.append(len(train_df))
    return np.full(len(test_df), train_df["value"].iloc[-1])


wf = walk_forward_eval(data, naive_predict, target_col="value", time_col="date",
                       group_cols=["group"], min_train_size=40, horizon=5)
check("wf_has_results", len(wf) > 0, "points={}".format(len(wf)))
check("wf_folds_per_group", wf.groupby("group")["fold"].nunique().eq(4).all(),
      "(60-40)/5=4 折/组")
check("wf_no_leakage", all(l == 40 + f * 5 for f, l in
                           zip(sorted(wf[wf["group"] == "A"]["fold"].unique()),
                               train_lens[:4])))
check("wf_y_alignment",
      np.allclose(wf[(wf["group"] == "A") & (wf["fold"] == 0)]["y_true"].to_numpy(),
                  data[data["group"] == "A"]["value"].to_numpy()[40:45]))

# ============ 9. auto_test 验收(行数守恒 pipes) ============
print("=== 9. auto_test ===")
p1 = TSPipeLine()
p1.pipe(ExpandingFeature(cols=["value"], stats=["mean", "std"],
                         group_cols=["group"], time_col="date")) \
  .pipe(MovingAverageTrend(cols=["value"], window=5, group_cols=["group"], time_col="date")) \
  .pipe(SeasonalMean(cols=["value"], period=7, mode="both",
                     group_cols=["group"], time_col="date"))
p1.fit(data)
try:
    p1.auto_test(data, sample=30)
    check("auto_test_p1_pipeline", True)
except Exception as e:
    check("auto_test_p1_pipeline", False, str(e)[:200])

# ============ 10. save/load(SeasonalMean profile 持久化) ============
print("=== 10. save/load ===")
sm.save("d:/datas/share/traes/easymlops-trae/tests/ts_p1_sm.pkl")
sm_loaded = SeasonalMean(cols=["value"], period=7, per_group=True, mode="both",
                         group_cols=["group"], time_col="date")
sm_loaded.load("d:/datas/share/traes/easymlops-trae/tests/ts_p1_sm.pkl")
before = sm.transform(seasonal)
after = sm_loaded.transform(seasonal)
check("seasonal_profile_persisted",
      np.allclose(before["value_seasonal"].to_numpy(), after["value_seasonal"].to_numpy()))

# ============ 11. Parallel 分支(复用表模块, 同输入并行多特征) ============
print("=== 11. parallel ===")
par = TSPipeLine()
par.pipe(Parallel(pipe_objects=[
    RollingFeature(cols=["value"], window=5, stats=["mean"],
                   group_cols=["group"], time_col="date"),
    LagFeature(cols=["value"], lags=[1], group_cols=["group"], time_col="date"),
])) \
  .pipe(FourierFeature(period=7, order=1, group_cols=["group"], time_col="date"))
par.fit(data)
pout = par.transform(data)
expect_cols = {"value_roll5_mean", "value_lag1", "fourier_1_sin", "fourier_1_cos"}
check("parallel_cols_present",
      len(pout) == len(data) and expect_cols.issubset(set(pout.columns)))
check("parallel_rows_kept", list(pout.index) == list(data.index))
psingle = par.transform_single(dict(data.iloc[0]))
check("parallel_single_keys", expect_cols.issubset(set(psingle.keys())))
par_path = "d:/datas/share/traes/easymlops-trae/tests/ts_p1_parallel.pkl"
par.save(par_path)
par_loaded = TSPipeLine()
par_loaded.load(par_path)
pout2 = par_loaded.transform(data)
same = all(np.allclose(pout[c].astype(float), pout2[c].astype(float), atol=1e-8)
           for c in pout.columns if c not in ("group", "date"))
check("parallel_save_load_consistent", same)
os.remove(par_path)

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
