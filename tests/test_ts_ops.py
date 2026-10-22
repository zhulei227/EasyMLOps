# -*- coding: utf-8 -*-
# TS 基础操作测试: 分组语义/批单一致性/auto_test/save-load/防泄漏
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from easymlops import TSPipeLine
from easymlops.ts.preprocessing import (FixTimeInput, DateFeatureExtract, TSFillNa,
                                        TSDiff, TSZScoreScaler, TSMinMaxScaler)
from easymlops.ts.features import LagFeature, RollingFeature, EWMAFeature
from easymlops.ts.utils import calc_mape, calc_smape, calc_mase, split_time_series

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(42)

# 面板数据: 两组交错(组内时间有序), 每组 60 条
n_per_group = 60
dates = pd.date_range("2026-01-01", periods=n_per_group, freq="D")
rows = []
for g in ["A", "B"]:
    vals = 10 + np.cumsum(rng.normal(0, 1, n_per_group))
    for t, v in zip(dates, vals):
        rows.append({"group": g, "date": t, "value": v, "value2": v * 2 + rng.normal(0, 0.1)})
data = pd.DataFrame(rows)
# 交错序: 按时间逐日交替 A,B,A,B... 组内时间有序
alt_rows = []
for i in range(n_per_group):
    for g in ["A", "B"]:
        alt_rows.append(data[(data["group"] == g) & (data["date"] == dates[i])].iloc[0].to_dict())
ordered = pd.DataFrame(alt_rows)

train, test = split_time_series(ordered, time_col="date", group_cols=["group"], test_size=0.25)

# ============ 1. split_time_series 防泄漏 ============
print("=== 1. split_time_series ===")
check("split_row_count", len(train) + len(test) == len(ordered),
      "{}+{}={}".format(len(train), len(test), len(ordered)))
ok_no_leak = True
for g in ["A", "B"]:
    if train[train["group"] == g]["date"].max() >= test[test["group"] == g]["date"].min():
        ok_no_leak = False
check("split_no_leak_per_group", ok_no_leak)

# ============ 2. LagFeature 分组边界与对拍 ============
print("=== 2. LagFeature ===")
lag = LagFeature(cols=["value"], lags=[1, 3], group_cols=["group"], time_col="date")
lag.fit(train)
out = lag.transform(ordered)
check("lag_row_count", len(out) == len(ordered))
check("lag_index_preserved", (out.index == ordered.index).all())
for g in ["A", "B"]:
    gvals = ordered[ordered["group"] == g]["value"].to_numpy()
    expect_lag1 = np.concatenate([[0.0], gvals[:-1]])
    got = out[out["group"] == g]["value_lag1"].to_numpy()
    check("lag1_group_{}_boundary_zero".format(g), abs(got[0]) == 0.0)
    check("lag1_group_{}_values".format(g), np.allclose(got, expect_lag1))

# ============ 3. RollingFeature 对拍 pandas ============
print("=== 3. RollingFeature ===")
roll = RollingFeature(cols=["value"], window=5, stats=["mean", "std"], min_periods=3,
                      group_cols=["group"], time_col="date")
roll.fit(train)
out = roll.transform(ordered)
ok = True
for g in ["A", "B"]:
    gvals = ordered[ordered["group"] == g]["value"].to_numpy()
    expect = pd.Series(gvals).rolling(5, min_periods=3).mean().fillna(0.0).to_numpy()
    got = out[out["group"] == g]["value_roll5_mean"].to_numpy()
    if not np.allclose(got, expect):
        ok = False
check("rolling_mean_matches_pandas", ok)

# ============ 4. EWMAFeature 批量 vs 单条(严格一致) ============
print("=== 4. EWMAFeature ===")
ewma = EWMAFeature(cols=["value"], span=5, group_cols=["group"], time_col="date")
ewma.fit(train)
tb = test.reset_index(drop=True)
batch_out = ewma.transform(tb)
ewma.reset_cache_data()
single_vals = [ewma.transform_single(dict(r))["value_ewma"] for r in tb.to_dict("records")]
check("ewma_batch_single_consistent",
      np.allclose(batch_out["value_ewma"].to_numpy(), np.array(single_vals)))

# ============ 5. 分组滚动批/单一致性 ============
print("=== 5. batch vs single (grouped rolling) ===")
roll2 = RollingFeature(cols=["value", "value2"], window=4, stats=["mean"],
                       group_cols=["group"], time_col="date")
roll2.fit(train)
batch_out = roll2.transform(tb)
roll2.reset_cache_data()
single_df = pd.DataFrame([roll2.transform_single(dict(r)) for r in tb.to_dict("records")])
check("rolling_batch_single_consistent",
      np.allclose(batch_out["value_roll4_mean"].to_numpy(),
                  single_df["value_roll4_mean"].to_numpy()))

# ============ 6. 未见新组 warmup ============
print("=== 6. unseen group warmup ===")
new_group_rows = tb[tb["group"] == "A"].copy()
new_group_rows["group"] = "NEW"
single_new = roll2.transform_single(dict(new_group_rows.iloc[0]))
check("unseen_group_no_error", abs(single_new["value_roll4_mean"]) == 0.0, "warmup head -> 0")

# ============ 7. TSDiff 可逆 ============
print("=== 7. TSDiff ===")
diff = TSDiff(cols=["value"], periods=1, group_cols=["group"], time_col="date")
diff.fit(train)
test_sorted = test.sort_values(["group", "date"]).reset_index(drop=True)
diff_out = diff.transform(test_sorted)
expected_diff = test_sorted.copy()
for g in ["A", "B"]:
    mask = expected_diff["group"] == g
    expected_diff.loc[mask, "value"] = expected_diff.loc[mask, "value"].diff().fillna(0.0)
check("diff_matches_pandas",
      np.allclose(diff_out["value"].to_numpy(), expected_diff["value"].to_numpy()))

inv = TSDiff(cols=["value"], periods=1, inverse=True, group_cols=["group"], time_col="date")
inv.fit(train)
restored = inv.transform(diff_out)
# 语义1: zero 填充的组首差分信息固有丢失 -> 该行还原为 seed 水平
for g in ["A", "B"]:
    gidx = test_sorted[test_sorted["group"] == g].index
    seed = inv.seed_levels_[(g,)]["value"][0]
    check("diff_inverse_head_is_seed_{}".format(g),
          abs(restored.loc[gidx[0], "value"] - seed) < 1e-9)

# 语义2(部署口径): 差分流从 fit 边界后开始且首条差分真实(模型预测的转移量)
# -> 全量精确还原
real_diff = test_sorted.copy()
for g in ["A", "B"]:
    gidx = test_sorted[test_sorted["group"] == g].index
    seed = inv.seed_levels_[(g,)]["value"][0]
    real_diff.loc[gidx[0], "value"] = test_sorted.loc[gidx[0], "value"] - seed
    real_diff.loc[gidx[1:], "value"] = (test_sorted.loc[gidx[1:], "value"].to_numpy() -
                                        test_sorted.loc[gidx[:-1], "value"].to_numpy())
restored2 = inv.transform(real_diff)
check("diff_inverse_reconstruct",
      np.allclose(restored2["value"].to_numpy(), test_sorted["value"].to_numpy(), atol=1e-9))

inv.reset_cache_data()
single_levels = [inv.transform_single(dict(r))["value"] for r in diff_out.to_dict("records")]
check("diff_inverse_batch_single_consistent",
      np.allclose(restored["value"].to_numpy(), np.array(single_levels)))

# ============ 8. Scaler 全局/分组/可逆/新组回退 ============
print("=== 8. Scalers ===")
sc = TSZScoreScaler(cols=["value"], per_group=False, group_cols=["group"], time_col="date")
sc.fit(train)
z = sc.transform(test)
mu, sd = train["value"].mean(), train["value"].std(ddof=0)
check("zscore_global", np.allclose(z["value"].to_numpy(), (test["value"].to_numpy() - mu) / sd))
zinv = TSZScoreScaler(cols=["value"], per_group=False, inverse=True,
                      group_cols=["group"], time_col="date")
zinv.fit(train)
back = zinv.transform(z)
check("zscore_inverse_roundtrip", np.allclose(back["value"].to_numpy(), test["value"].to_numpy()))

scg = TSZScoreScaler(cols=["value"], per_group=True, group_cols=["group"], time_col="date")
scg.fit(train)
zg = scg.transform(test)
ok = True
for g in ["A", "B"]:
    mask_tr, mask_te = train["group"] == g, test["group"] == g
    m, s = train[mask_tr]["value"].mean(), train[mask_tr]["value"].std(ddof=0)
    expect = (test[mask_te]["value"].to_numpy() - m) / (s if s > 1e-12 else 1.0)
    if not np.allclose(zg[mask_te]["value"].to_numpy(), expect):
        ok = False
check("zscore_per_group_stats", ok)

newg = test[test["group"] == "A"].copy()
newg["group"] = "NEW"
zg_new = scg.transform(newg)
check("unseen_group_fallback_global",
      np.allclose(zg_new["value"].to_numpy(), (newg["value"].to_numpy() - mu) / sd))

mm = TSMinMaxScaler(cols=["value"], group_cols=["group"], time_col="date")
mm.fit(train)
mmt = mm.transform(test)
vmin, vmax = train["value"].min(), train["value"].max()
check("minmax_global", np.allclose(mmt["value"].to_numpy(),
                                   (test["value"].to_numpy() - vmin) / (vmax - vmin)))

# ============ 9. TSFillNa 分组不跨组 ============
print("=== 9. TSFillNa ===")
has_na = ordered.copy()
head_index = has_na.groupby("group").head(1).index
has_na.loc[head_index, "value"] = np.nan  # 每组首行缺失
fn = TSFillNa(cols=["value"], method="ffill", fill_value=-999.0,
              group_cols=["group"], time_col="date")
fn.fit(ordered)
filled = fn.transform(has_na)
check("fillna_no_cross_group", (filled.loc[head_index, "value"] == -999.0).all(),
      "组首无前值, 不得借用其他组数据")
check("fillna_other_rows_intact",
      np.allclose(filled.drop(index=head_index)["value"].to_numpy(),
                  ordered.drop(index=head_index)["value"].to_numpy()))

# ============ 10. DateFeatureExtract ============
print("=== 10. DateFeatureExtract ===")
de = DateFeatureExtract(time_col="date", features=["month", "weekday", "hour", "is_weekend"])
de.fit(ordered)
de_out = de.transform(ordered)
check("date_features_added",
      all(c in de_out.columns for c in ["month", "weekday", "hour", "is_weekend",
                                        "month_sin", "month_cos", "weekday_sin", "weekday_cos"]))
check("date_values_match_pandas",
      (de_out["month"].to_numpy() == ordered["date"].dt.month.to_numpy()).all() and
      (de_out["weekday"].to_numpy() == ordered["date"].dt.weekday.to_numpy()).all())
check("cyclical_in_range",
      de_out["month_sin"].abs().max() <= 1.0 + 1e-9 and de_out["month_cos"].abs().max() <= 1.0 + 1e-9)

# ============ 11. FixTimeInput 乱序解析 ============
print("=== 11. FixTimeInput ===")
shuffled = ordered.sample(frac=1, random_state=1).reset_index(drop=True)
fti = FixTimeInput(time_col="date", group_cols=["group"], sort_rows=False)
fti.fit(shuffled)
parsed = fti.transform(shuffled)
check("fixtime_parse_dtype", str(parsed["date"].dtype).startswith("datetime64"))
check("fixtime_row_order_kept", (parsed.index == shuffled.index).all())

# ============ 12. save/load 一致性(状态性 pipe) ============
print("=== 12. save/load ===")
pipe = TSPipeLine()
pipe.pipe(LagFeature(cols=["value"], lags=[1], group_cols=["group"], time_col="date")) \
    .pipe(RollingFeature(cols=["value"], window=3, stats=["mean"], group_cols=["group"], time_col="date"))
pipe.fit(train)
before = pipe.transform(test)
pipe.save("d:/datas/share/traes/easymlops-trae/tests/ts_ops_model.pkl")
loaded = TSPipeLine()
loaded.load("d:/datas/share/traes/easymlops-trae/tests/ts_ops_model.pkl")
after = loaded.transform(test)
check("ts_save_load_consistent",
      np.allclose(before["value_roll3_mean"].to_numpy(), after["value_roll3_mean"].to_numpy()) and
      np.allclose(before["value_lag1"].to_numpy(), after["value_lag1"].to_numpy()))

inv.save("d:/datas/share/traes/easymlops-trae/tests/ts_ops_inv.pkl")
inv_loaded = TSDiff(cols=["value"], periods=1, inverse=True, group_cols=["group"], time_col="date")
inv_loaded.load("d:/datas/share/traes/easymlops-trae/tests/ts_ops_inv.pkl")
r2 = inv_loaded.transform(diff_out)
check("diff_seed_persisted", np.allclose(r2["value"].to_numpy(), restored["value"].to_numpy()))

# ============ 13. auto_test 验收(核心) ============
print("=== 13. auto_test ===")
p1 = TSPipeLine()
p1.pipe(LagFeature(cols=["value"], lags=[1, 2], group_cols=["group"], time_col="date")) \
  .pipe(RollingFeature(cols=["value"], window=5, stats=["mean", "std"],
                       group_cols=["group"], time_col="date")) \
  .pipe(EWMAFeature(cols=["value"], span=5, group_cols=["group"], time_col="date")) \
  .pipe(DateFeatureExtract(time_col="date")) \
  .pipe(TSFillNa(cols=["value"], method="ffill", group_cols=["group"], time_col="date"))
p1.fit(train)
try:
    p1.auto_test(train, sample=30)
    check("auto_test_feature_pipeline", True)
except Exception as e:
    check("auto_test_feature_pipeline", False, str(e)[:200])

p2 = TSPipeLine()
p2.pipe(TSDiff(cols=["value"], group_cols=["group"], time_col="date")) \
  .pipe(TSZScoreScaler(cols=["value"], group_cols=["group"], time_col="date")) \
  .pipe(TSMinMaxScaler(cols=["value2"], group_cols=["group"], time_col="date")) \
  .pipe(FixTimeInput(time_col="date", group_cols=["group"]))
p2.fit(train)
try:
    p2.auto_test(train, sample=30)
    check("auto_test_transform_pipeline", True)
except Exception as e:
    check("auto_test_transform_pipeline", False, str(e)[:200])

# ============ 14. 评估指标 ============
print("=== 14. metrics ===")
y_true = np.array([1.0, 2.0, 3.0, 4.0])
y_pred = np.array([1.1, 2.0, 2.7, 4.4])
y_hist = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5])
mape_v = calc_mape(y_true, y_pred)
smape_v = calc_smape(y_true, y_pred)
mase_v = calc_mase(y_true, y_pred, y_hist, m=1)
manual_mape = np.mean(np.abs((y_true - y_pred) / y_true))
check("mape_matches_manual", abs(mape_v - manual_mape) < 1e-12, "{:.4f}".format(mape_v))
check("smape_in_range", 0 <= smape_v <= 2, "{:.4f}".format(smape_v))
naive_mae = np.mean(np.abs(np.diff(y_hist)))
check("mase_matches_manual", abs(mase_v - np.mean(np.abs(y_true - y_pred)) / naive_mae) < 1e-12,
      "{:.4f}".format(mase_v))
check("mase_better_than_naive", mase_v < 1.0)

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
