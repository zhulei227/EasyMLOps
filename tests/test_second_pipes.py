# -*- coding: utf-8 -*-
# 第二梯队 pipe 验收: GroupAggregateFeatures / ClusterFeatures / QuantileClip /
# LogTransform / SqrtTransform / CountEncoding / CategoryCross / CorrFilter (v0.2.5)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd

from easymlops.table.features import (GroupAggregateFeatures, ClusterFeatures,
                                      QuantileClip, LogTransform, SqrtTransform)
from easymlops.table.encoding import CountEncoding, CategoryCross, TargetEncoding
from easymlops.table.feature_selection import CorrFilter
from easymlops.table.core import TablePipeLine

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_second_pipes.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(7)
n = 300
df = pd.DataFrame({
    "city": rng.choice(["bj", "sh", "sz"], n),
    "product": rng.choice(["a", "b", "c", "d"], n),
    "income": np.abs(rng.normal(8000, 2000, n)) + 1,
    "value": rng.normal(100, 20, n),
    "count_feat": rng.poisson(5, n),
    "y": rng.choice([0, 1], n),
})
train, test = df[:240].copy(), df[240:].copy()

# ============ 1. GroupAggregateFeatures ============
print("=== 1. GroupAggregateFeatures ===")
gaf = GroupAggregateFeatures(group_cols=["city"], stats=["mean", "count"])
gaf.fit(train)
out = gaf.transform(test.copy())
check("gaf_cols_added", all(c in out.columns for c in
                            ["income_mean", "income_count", "value_mean", "value_count"]))
check("gaf_rows_kept", len(out) == len(test) and list(out.index) == list(test.index))
expect = train.groupby("city")["income"].mean()
check("gaf_matches_pandas",
      np.allclose(out[out["city"] == "bj"]["income_mean"], expect["bj"], atol=1e-8))
unseen = test.copy()
unseen["city"] = "zz"
out_u = gaf.transform(unseen)
check("gaf_unseen_group_fallback",
      np.allclose(out_u["income_mean"], train["income"].mean(), atol=1e-8) and
      (out_u["income_count"] == len(train)).all())
s = gaf.transform_single(dict(test.iloc[0]))
check("gaf_single_consistent",
      abs(s["income_mean"] - out.iloc[0]["income_mean"]) < 1e-9)
gaf.save(PKL)
gaf2 = GroupAggregateFeatures(group_cols=["city"], stats=["mean", "count"])
gaf2.load(PKL)
check("gaf_save_load_consistent",
      np.allclose(gaf2.transform(test.copy())["income_mean"], out["income_mean"], atol=1e-12))
gaf_m = GroupAggregateFeatures(group_cols=["city", "product"], stats=["mean"])
gaf_m.fit(train)
check("gaf_multi_key", "income_mean" in gaf_m.transform(test.copy()).columns)

# ============ 2. ClusterFeatures ============
print("=== 2. ClusterFeatures ===")
cf = ClusterFeatures(cols=["value", "count_feat"], n_clusters=3, random_state=7)
cf.fit(train)
outc = cf.transform(test.copy())
check("cluster_cols_added",
      "kmeans_cluster_id" in outc.columns and "kmeans_dist_0" in outc.columns and
      len(outc) == len(test))
check("cluster_id_valid", outc["kmeans_cluster_id"].isin(range(3)).all())
cf.save(PKL)
cf2 = ClusterFeatures(n_clusters=3, random_state=7)
cf2.load(PKL)
check("cluster_save_load_consistent",
      np.allclose(outc["kmeans_cluster_id"].to_numpy(dtype=float),
                  cf2.transform(test.copy())["kmeans_cluster_id"].to_numpy(dtype=float)))
s = cf.transform_single(dict(test.iloc[0]))
check("cluster_single", "kmeans_cluster_id" in s and "kmeans_dist_0" in s)

# ============ 3. QuantileClip ============
print("=== 3. QuantileClip ===")
qc = QuantileClip(cols=["income"], lower=0.05, upper=0.95)
qc.fit(train)
outq = qc.transform(test.copy())
lo = train["income"].quantile(0.05)
hi = train["income"].quantile(0.95)
check("quantile_clip_bounds", (outq["income"] >= lo - 1e-9).all() and
      (outq["income"] <= hi + 1e-9).all())
check("quantile_clip_matches_manual",
      np.allclose(train["income"].clip(lo, hi), qc.transform(train.copy())["income"], atol=1e-9))
s = qc.transform_single(dict(test.iloc[0]))
check("quantile_clip_single", lo - 1e-9 <= s["income"] <= hi + 1e-9)

# ============ 4. LogTransform / SqrtTransform ============
print("=== 4. Log/Sqrt ===")
lt = LogTransform(cols=["income"])
outl = lt.fit(train).transform(test.copy())
check("log_transform", np.allclose(outl["income"], np.log1p(test["income"]), atol=1e-12))
st = SqrtTransform(cols=["count_feat"])
outs = st.fit(train).transform(test.copy())
check("sqrt_transform", np.allclose(outs["count_feat"], np.sqrt(test["count_feat"]), atol=1e-12))
s = lt.transform_single(dict(test.iloc[0]))
check("log_single", abs(s["income"] - np.log1p(test.iloc[0]["income"])) < 1e-12)

# ============ 5. CountEncoding ============
print("=== 5. CountEncoding ===")
ce = CountEncoding(cols=["city"], normalize=True)
ce.fit(train)
oute = ce.transform(test.copy())
freq = train["city"].value_counts(normalize=True)
check("count_encoding_matches_pandas",
      np.allclose(oute[oute["city"] == "sh"]["city"], freq["sh"], atol=1e-12))
unseen_row = ce.transform(test.iloc[[0]].assign(city="zz"))
check("count_encoding_unseen", unseen_row["city"].iloc[0] == 0)
s = ce.transform_single(dict(test.iloc[0]))
check("count_encoding_single", abs(s["city"] - freq[test.iloc[0]["city"]]) < 1e-12)
ce.save(PKL)
ce2 = CountEncoding(cols=["city"], normalize=True)
ce2.load(PKL)
check("count_encoding_save_load",
      np.allclose(ce2.transform(test.copy())["city"], oute["city"], atol=1e-12))

# ============ 6. CategoryCross（含与 TargetEncoding 链接） ============
print("=== 6. CategoryCross ===")
cc = CategoryCross(cols=[["city", "product"]])
cc.fit(train)
outx = cc.transform(test.copy())
check("cross_col_created", "city_product_cross" in outx.columns and
      len(outx) == len(test))
expected_combo = "{}_{}".format(test.iloc[0]["city"], test.iloc[0]["product"])
check("cross_value_correct",
      outx.iloc[0]["city_product_cross"] == expected_combo)
unseen_row = cc.transform(test.copy().iloc[[0]].assign(city="zz", product="x"))
check("cross_unseen_fallback", unseen_row["city_product_cross"].iloc[0] == "__other__")
# 单条
s = cc.transform_single(dict(test.copy().iloc[0].to_dict(), city="bj", product="a"))
check("cross_single", s["city_product_cross"] == "bj_a")
# 与 TargetEncoding 组链(交叉列做目标编码)
te = TargetEncoding(cols=[("city_product_cross", "city_product_cross")], y=train["y"])
te.fit(train.assign(city_product_cross=cc.transform(train.copy())["city_product_cross"]))
te_out = te.transform(outx)
check("cross_target_encoding_chain", "city_product_cross" in te_out.columns and
      np.isfinite(te_out["city_product_cross"].astype(float)).all())
# drop_origin_cols
ccd = CategoryCross(cols=[["city", "product"]], drop_origin_cols=True)
ccd.fit(train)
check("cross_drop_origin", "city" not in ccd.transform(test.copy()).columns and
      "city_product_cross" in ccd.transform(test.copy()).columns)

# ============ 7. CorrFilter ============
print("=== 7. CorrFilter ===")
corr_df = train.copy()
corr_df["value_copy"] = corr_df["value"]  # 与 value 完全重复（冗余特征）
cfilt = CorrFilter(threshold=0.95, y=corr_df["y"])
cfilt.fit(corr_df)
check("corr_filter_dropped_redundant",
      "value_copy" in cfilt.dropped_cols_ and "value" not in cfilt.dropped_cols_)
outf = cfilt.transform(corr_df)
check("corr_filter_rows_kept", len(outf) == len(corr_df) and
      "value" in outf.columns and "value_copy" not in outf.columns)
cfilt.save(PKL)
cfilt2 = CorrFilter(threshold=0.95)
cfilt2.load(PKL)
check("corr_filter_save_load",
      list(cfilt2.transform(corr_df).columns) == list(outf.columns))

# ============ 8. Pipeline 集成 ============
print("=== 8. pipeline integration ===")
pipeline = TablePipeLine()
pipeline.pipe(GroupAggregateFeatures(group_cols=["city"], stats=["mean"])) \
        .pipe(CountEncoding(cols=["city"], normalize=True)) \
        .pipe(QuantileClip(cols=["income"]))
pipeline.fit(train)
pout = pipeline.transform(test.copy())
check("pipeline_all_pipes", len(pout) == len(test) and
      "income_mean" in pout.columns and "city" in pout.columns and
      (pout["income"] >= train["income"].quantile(0.01) - 1e-9).all())
psingle = pipeline.transform_single(dict(test.iloc[0]))
check("pipeline_single", "income_mean" in psingle and "income" in psingle)

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
