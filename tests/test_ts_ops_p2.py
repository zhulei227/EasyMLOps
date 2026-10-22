# -*- coding: utf-8 -*-
# TS P2 操作测试: 百分比变化/对数收益/BoxCox/skew-kurt/滚动相关/傅里叶/STL/熵/Hurst
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from easymlops import TSPipeLine
from easymlops.ts.preprocessing import TSDiff, TSPctChange, TSLogReturn, TSBoxCox
from easymlops.ts.features import RollingFeature, RollingCorrCov, FourierFeature, STLDecompose
from easymlops.ts.utils import calc_sample_entropy, calc_hurst

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(11)
n_per_group = 80
dates = pd.date_range("2026-01-01", periods=n_per_group, freq="D")
rows = []
for g, base in [("A", 50.0), ("B", 120.0)]:
    vals = base + np.cumsum(rng.normal(0, 1, n_per_group))  # 保持正值
    for t, v in zip(dates, vals):
        rows.append({"group": g, "date": t, "value": v, "value2": v + rng.normal(0, 1)})
data = pd.DataFrame(rows)
from easymlops.ts.utils import split_time_series
train, test = split_time_series(data, time_col="date", group_cols=["group"], test_size=0.3)
test_sorted = test.sort_values(["group", "date"]).reset_index(drop=True)

# ============ 1. TSPctChange ============
print("=== 1. TSPctChange ===")
pc = TSPctChange(cols=["value"], periods=1, group_cols=["group"], time_col="date")
pc.fit(train)
out = pc.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gvals = test_sorted[test_sorted["group"] == g]["value"].to_numpy()
    expect = pd.Series(gvals).pct_change(1).fillna(0.0).to_numpy()
    if not np.allclose(out[out["group"] == g]["value"].to_numpy(), expect):
        ok = False
check("pct_change_matches_pandas", ok)

inv_pc = TSPctChange(cols=["value"], periods=1, inverse=True,
                     group_cols=["group"], time_col="date")
inv_pc.fit(train)
real = test_sorted.copy()
for g in ["A", "B"]:
    gidx = test_sorted[test_sorted["group"] == g].index
    seed = inv_pc.seed_levels_[(g,)]["value"][0]
    real.loc[gidx[0], "value"] = test_sorted.loc[gidx[0], "value"] / seed - 1.0
    real.loc[gidx[1:], "value"] = (test_sorted.loc[gidx[1:], "value"].to_numpy() /
                                   test_sorted.loc[gidx[:-1], "value"].to_numpy() - 1.0)
restored = inv_pc.transform(real)
check("pct_change_inverse_reconstruct",
      np.allclose(restored["value"].to_numpy(), test_sorted["value"].to_numpy(), atol=1e-9))

# ============ 2. TSLogReturn ============
print("=== 2. TSLogReturn ===")
lr = TSLogReturn(cols=["value"], periods=1, group_cols=["group"], time_col="date")
lr.fit(train)
out = lr.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gvals = test_sorted[test_sorted["group"] == g]["value"].to_numpy()
    expect = np.concatenate([[0.0], np.log(gvals[1:] / gvals[:-1])])
    if not np.allclose(out[out["group"] == g]["value"].to_numpy(), expect):
        ok = False
check("log_return_matches_manual", ok)

inv_lr = TSLogReturn(cols=["value"], periods=1, inverse=True,
                     group_cols=["group"], time_col="date")
inv_lr.fit(train)
real = test_sorted.copy()
for g in ["A", "B"]:
    gidx = test_sorted[test_sorted["group"] == g].index
    seed = inv_lr.seed_levels_[(g,)]["value"][0]
    real.loc[gidx[0], "value"] = np.log(test_sorted.loc[gidx[0], "value"] / seed)
    real.loc[gidx[1:], "value"] = np.log(test_sorted.loc[gidx[1:], "value"].to_numpy() /
                                         test_sorted.loc[gidx[:-1], "value"].to_numpy())
restored = inv_lr.transform(real)
check("log_return_inverse_reconstruct",
      np.allclose(restored["value"].to_numpy(), test_sorted["value"].to_numpy(), atol=1e-9))

# TSDiff 回归(重构后行为不变)
diff = TSDiff(cols=["value"], periods=1, group_cols=["group"], time_col="date")
diff.fit(train)
dout = diff.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gvals = test_sorted[test_sorted["group"] == g]["value"].to_numpy()
    expect = np.concatenate([[0.0], np.diff(gvals)])
    if not np.allclose(dout[dout["group"] == g]["value"].to_numpy(), expect):
        ok = False
check("diff_regression_after_refactor", ok)

# ============ 3. TSBoxCox ============
print("=== 3. TSBoxCox ===")
bc = TSBoxCox(cols=["value"], method="auto", group_cols=["group"], time_col="date")
bc.fit(train)
check("boxcox_auto_positive_uses_boxcox", bc.lambdas_["value"][0] == "boxcox")
tr = bc.transform(test)
back = TSBoxCox(cols=["value"], method="auto", inverse=True,
                group_cols=["group"], time_col="date")
back.fit(train)  # inverse 模式实例直接 fit(set_params 会携带 inverse=False 覆盖模式)
restored = back.transform(tr)
check("boxcox_roundtrip", np.allclose(restored["value"].to_numpy(),
                                      test["value"].to_numpy(), atol=1e-7))
neg_data = train.copy()
neg_data["value"] = neg_data["value"] - neg_data["value"].mean()
bneg = TSBoxCox(cols=["value"], method="auto", group_cols=["group"], time_col="date")
bneg.fit(neg_data)
check("boxcox_auto_negative_uses_yeojohnson", bneg.lambdas_["value"][0] == "yeojohnson")
tneg = bneg.transform(neg_data)
rbneg = TSBoxCox(cols=["value"], method="auto", inverse=True,
                 group_cols=["group"], time_col="date")
rbneg.fit(neg_data)
restored_neg = rbneg.transform(tneg)
check("yeojohnson_roundtrip", np.allclose(restored_neg["value"].to_numpy(),
                                          neg_data["value"].to_numpy(), atol=1e-7))

# ============ 4. RollingFeature skew/kurt ============
print("=== 4. RollingFeature skew/kurt ===")
rf = RollingFeature(cols=["value"], window=10, stats=["skew", "kurt"],
                    group_cols=["group"], time_col="date")
rf.fit(train)
out = rf.transform(test_sorted)
ok_s, ok_k = True, True
for g in ["A", "B"]:
    gvals = test_sorted[test_sorted["group"] == g]["value"].to_numpy()
    s = pd.Series(gvals).rolling(10, min_periods=10)
    expect_skew = s.skew().fillna(0.0).to_numpy()
    expect_kurt = s.kurt().fillna(0.0).to_numpy()
    got_s = out[out["group"] == g]["value_roll10_skew"].to_numpy()
    got_k = out[out["group"] == g]["value_roll10_kurt"].to_numpy()
    if not np.allclose(got_s, expect_skew, atol=1e-8):
        ok_s = False
    if not np.allclose(got_k, expect_kurt, atol=1e-8):
        ok_k = False
check("rolling_skew_matches_pandas", ok_s)
check("rolling_kurt_matches_pandas", ok_k)

# ============ 5. RollingCorrCov ============
print("=== 5. RollingCorrCov ===")
rc = RollingCorrCov("value", "value2", window=10, stat="corr",
                    group_cols=["group"], time_col="date")
rc.fit(train)
out = rc.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gv = test_sorted[test_sorted["group"] == g]
    expect = gv["value"].rolling(10, min_periods=10).corr(gv["value2"]).fillna(0.0).to_numpy()
    if not np.allclose(out[out["group"] == g][rc.output_col].to_numpy(), expect, atol=1e-8):
        ok = False
check("rolling_corr_matches_pandas", ok)
rc.reset_cache_data()
tb = test_sorted.iloc[-20:].reset_index(drop=True)
batch_out = rc.transform(tb)
rc.reset_cache_data()
singles = pd.DataFrame([rc.transform_single(dict(r)) for r in tb.to_dict("records")])
check("corr_batch_single_consistent",
      np.allclose(batch_out[rc.output_col].to_numpy(), singles[rc.output_col].to_numpy()))
rcc = RollingCorrCov("value", "value2", window=10, stat="cov",
                     group_cols=["group"], time_col="date")
rcc.fit(train)
outc = rcc.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gv = test_sorted[test_sorted["group"] == g]
    expect = gv["value"].rolling(10, min_periods=10).cov(gv["value2"]).fillna(0.0).to_numpy()
    if not np.allclose(outc[outc["group"] == g][rcc.output_col].to_numpy(), expect, atol=1e-8):
        ok = False
check("rolling_cov_matches_pandas", ok)

# ============ 6. FourierFeature ============
print("=== 6. FourierFeature ===")
ff = FourierFeature(period=7, order=2, group_cols=["group"], time_col="date")
ff.fit(test_sorted)
out = ff.transform(test_sorted)
ok = True
for g in ["A", "B"]:
    gout = out[out["group"] == g].reset_index(drop=True)
    for k in range(1, 3):
        expect_sin = np.sin(2 * np.pi * k * np.arange(len(gout)) / 7)
        if not np.allclose(gout["fourier_{}_sin".format(k)].to_numpy(), expect_sin):
            ok = False
check("fourier_matches_formula", ok)
check("fourier_cols_count", sum(1 for c in out.columns if c.startswith("fourier_")) == 4)
ff.reset_cache_data()
s0 = ff.transform_single(dict(test_sorted.iloc[0]))
s1 = ff.transform_single(dict(test_sorted.iloc[1]))
check("fourier_single_phase", abs(s0["fourier_1_sin"]) < 1e-9 and
      abs(s1["fourier_1_sin"] - np.sin(2 * np.pi / 7)) < 1e-9)

# ============ 7. STLDecompose ============
print("=== 7. STLDecompose ===")
stl_rows = []
for g in ["A", "B"]:
    for i, t in enumerate(pd.date_range("2026-01-01", periods=70, freq="D")):
        stl_rows.append({"group": g, "date": t,
                         "value": 10 + 3.0 * np.sin(2 * np.pi * i / 7) + rng.normal(0, 0.05)})
stl_data = pd.DataFrame(stl_rows)
stl = STLDecompose(cols=["value"], period=7, group_cols=["group"], time_col="date")
stl.fit(stl_data)
out = stl.transform(stl_data)
check("stl_cols_present",
      all(c in out.columns for c in ["value_trend", "value_seasonal", "value_resid"]))
recon = out["value_trend"] + out["value_seasonal"] + out["value_resid"]
check("stl_reconstruct_original", np.allclose(recon.to_numpy(), stl_data["value"].to_numpy()))
resid_std = out.groupby("group")["value_resid"].std()
check("stl_resid_small", (resid_std < 0.2).all(), "resid std={:.3f}".format(resid_std.max()))

# ============ 8. 熵与 Hurst ============
print("=== 8. entropy & hurst ===")
sinusoid = pd.Series(np.sin(2 * np.pi * np.arange(300) / 7))
noise = pd.Series(rng.normal(0, 1, 300))
check("sample_entropy_noise_gt_sinusoid",
      calc_sample_entropy(noise) > calc_sample_entropy(sinusoid),
      "noise={:.3f}, sinusoid={:.3f}".format(calc_sample_entropy(noise),
                                             calc_sample_entropy(sinusoid)))
h_noise = calc_hurst(noise)
h_trend = calc_hurst(pd.Series(np.cumsum(rng.normal(0, 1, 500)) + np.arange(500) * 0.05))
check("hurst_noise_near_0_5", 0.3 < h_noise < 0.7, "{:.3f}".format(h_noise))
check("hurst_trending_gt_noise", h_trend > h_noise, "{:.3f}".format(h_trend))

# ============ 9. auto_test 验收 ============
print("=== 9. auto_test ===")
p1 = TSPipeLine()
p1.pipe(TSPctChange(cols=["value"], group_cols=["group"], time_col="date")) \
  .pipe(RollingFeature(cols=["value"], window=10, stats=["skew", "kurt"],
                       group_cols=["group"], time_col="date")) \
  .pipe(RollingCorrCov("value", "value2", window=10, stat="corr",
                       group_cols=["group"], time_col="date")) \
  .pipe(FourierFeature(period=7, order=2,
                       group_cols=["group"], time_col="date")) \
  .pipe(TSBoxCox(cols=["value2"], method="auto", group_cols=["group"], time_col="date"))
p1.fit(train)
try:
    p1.auto_test(train, sample=30)
    check("auto_test_p2_pipeline", True)
except Exception as e:
    check("auto_test_p2_pipeline", False, str(e)[:200])

# ============ 10. save/load(TSBoxCox lambda 持久化) ============
print("=== 10. save/load ===")
bc.save("d:/datas/share/traes/easymlops-trae/tests/ts_p2_bc.pkl")
bc_loaded = TSBoxCox(cols=["value"], method="auto", group_cols=["group"], time_col="date")
bc_loaded.load("d:/datas/share/traes/easymlops-trae/tests/ts_p2_bc.pkl")
before = bc.transform(test)
after = bc_loaded.transform(test)
check("boxcox_lambda_persisted",
      np.allclose(before["value"].to_numpy(), after["value"].to_numpy()))

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
