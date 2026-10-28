# -*- coding: utf-8 -*-
# 第三梯队 pipe 验收: TargetEncoding 增强(smoothing_k/OOF) / NullImportanceFilter /
# RFEFilter / TextTfidfSvd / OptunaTuner (v0.2.5)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from easymlops.table.encoding import TargetEncoding
from easymlops.table.feature_selection import NullImportanceFilter, RFEFilter
from easymlops.table.text_features import TextTfidfSvd

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_third_pipes.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


# ============ 1. TargetEncoding: smoothing_k ============
print("=== 1. TargetEncoding smoothing_k ===")
rng = np.random.RandomState(7)
n = 300
df = pd.DataFrame({
    "city": ["rare"] * 5 + rng.choice(["bj", "sh"], n - 5).tolist(),
    "y": [1] * 5 + rng.choice([0, 1], n - 5).tolist(),
})
te_plain = TargetEncoding(cols=["city"], y=df["y"], min_sample=0)
te_plain.fit(df)
te_smooth = TargetEncoding(cols=["city"], y=df["y"], smoothing_k=50, min_sample=0)
te_smooth.fit(df)
plain_rare = te_plain.target_map_detail["city"]["rare"]
smooth_rare = te_smooth.target_map_detail["city"]["rare"]
global_mean = float(df["y"].mean())
check("te_smoothing_k_pulls_to_global",
      abs(smooth_rare - global_mean) < abs(plain_rare - global_mean) and
      min(plain_rare, global_mean) <= smooth_rare <= max(plain_rare, global_mean),
      "plain={:.3f} smooth={:.3f} global={:.3f}".format(plain_rare, smooth_rare, global_mean))

# ============ 2. TargetEncoding: OOF 防泄漏 ============
print("=== 2. TargetEncoding OOF ===")
df_te = df.copy()
te = TargetEncoding(cols=["city"], y=df["y"], cv=5, random_state=42, error_value=-1)
encoded = te.fit_transform_oof(df_te.copy())
check("te_oof_rows", len(encoded) == len(df))
check("te_oof_finite", np.isfinite(encoded["city"].astype(float)).all())
# 防泄漏: 'rare' 类别只出现在 1 个折中, 其所在行的 OOF 编码用 error_value(-1),
# 而全量映射编码为该类别均值(1.0) -> 两者必然不同
full_on_train = te.transform(df_te.copy())
diff_rows = int((encoded["city"].astype(float).to_numpy() !=
                 full_on_train["city"].astype(float).to_numpy()).sum())
check("te_oof_prevents_leakage", diff_rows > 0, "diff rows={}".format(diff_rows))
# 测试行 transform 仍使用全量映射
check("te_transform_uses_full_map",
      np.allclose(te.transform(df_te.iloc[[-1]].copy())["city"].astype(float).iloc[0],
                  te.target_map_detail["city"][df.iloc[-1]["city"]], atol=1e-12))
te.save(PKL)
te2 = TargetEncoding(cols=["city"], cv=5)
te2.load(PKL)
check("te_save_load", np.allclose(te2.transform(df_te.copy())["city"].astype(float),
                                  te.transform(df_te.copy())["city"].astype(float), atol=1e-12))

# ============ 3. NullImportanceFilter ============
print("=== 3. NullImportanceFilter ===")
x_clf, y_clf = make_classification(n_samples=400, n_features=10, n_informative=5,
                                   n_redundant=0, class_sep=2.0, random_state=7)
clf_df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(10)])
y_clf = pd.Series(y_clf)
nif = NullImportanceFilter(y=y_clf, n_runs=15, percentile=75,
                           native_init_params={"n_estimators": 60})
nif.fit(clf_df)
check("nif_selects_informative",
      len(set(["f0", "f1", "f2", "f3", "f4"]).intersection(nif.selected_cols)) >= 3,
      "selected={}".format(nif.selected_cols))
check("nif_drops_noise", len(nif.selected_cols) < 10,
      "selected={} of 10".format(len(nif.selected_cols)))
outf = nif.transform(clf_df)
check("nif_rows_kept", len(outf) == len(clf_df) and
      list(outf.columns) == nif.selected_cols)
nif.save(PKL)
nif2 = NullImportanceFilter()
nif2.load(PKL)
check("nif_save_load", nif2.selected_cols == nif.selected_cols)

# ============ 4. RFEFilter ============
print("=== 4. RFEFilter ===")
x_reg, y_reg = make_regression(n_samples=300, n_features=10, n_informative=4,
                               random_state=7)
reg_df = pd.DataFrame(x_reg, columns=[f"f{i}" for i in range(10)])
y_reg = pd.Series(y_reg)
rfe = RFEFilter(y=y_reg, n_features_to_select=5, task="regression",
                native_init_params={"n_estimators": 40})
rfe.fit(reg_df)
check("rfe_select_count", len(rfe.selected_cols) == 5,
      "selected={}".format(rfe.selected_cols))
check("rfe_keeps_informative",
      len(set(["f0", "f1", "f2", "f3"]).intersection(rfe.selected_cols)) >= 3)
outr = rfe.transform(reg_df)
check("rfe_rows_kept", len(outr) == len(reg_df) and
      list(outr.columns) == rfe.selected_cols)

# ============ 5. TextTfidfSvd ============
print("=== 5. TextTfidfSvd ===")
words = ["good service fast delivery", "bad quality slow", "excellent support",
         "terrible experience", "average product okay"]
text_df = pd.DataFrame({
    "remark": [words[i % len(words)] for i in range(100)],
    "num": rng.normal(0, 1, 100),
})
ttrain, ttest = text_df[:80].copy(), text_df[80:].copy()
tfs = TextTfidfSvd(cols=["remark"], n_components=3)
tfs.fit(ttrain)
outs = tfs.transform(ttest)
check("tfidf_svd_cols", all("remark_svd_{}".format(i) in outs.columns for i in range(3)) and
      len(outs) == len(ttest))
check("tfidf_svd_finite", np.isfinite(outs[["remark_svd_0", "remark_svd_1"]].values).all())
tfs.save(PKL)
tfs2 = TextTfidfSvd(cols=["remark"], n_components=3)
tfs2.load(PKL)
check("tfidf_svd_save_load",
      np.allclose(outs[["remark_svd_0"]].values,
                  tfs2.transform(ttest.copy())[["remark_svd_0"]].values, atol=1e-10))
s = tfs.transform_single(dict(ttest.iloc[0]))
check("tfidf_svd_single", "remark_svd_0" in s and
      abs(s["remark_svd_0"] - outs.iloc[0]["remark_svd_0"]) < 1e-10)

# ============ 6. OptunaTuner（未安装 optuna 时跳过） ============
print("=== 6. OptunaTuner ===")
try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if HAS_OPTUNA:
    from easymlops.table.tuner import OptunaTuner
    from easymlops.table.regression import LGBMRegression
    x_reg, y_reg = make_regression(n_samples=300, n_features=8, n_informative=5,
                                   noise=10.0, random_state=7)
    reg_df = pd.DataFrame(x_reg, columns=[f"f{i}" for i in range(8)])
    y_reg = pd.Series(y_reg)
    tuner = OptunaTuner(
        model_class=LGBMRegression,
        fixed_params={"y": y_reg, "use_faster_predictor": False},
        search_space={
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": lambda t: t.suggest_int("num_leaves", 8, 32),
        },
        n_trials=3, cv=2, task="regression", seed=7)
    tuner.fit(reg_df)
    check("tuner_best_params", tuner.best_params_ is not None and
          "learning_rate" in tuner.best_params_)
    outr = tuner.transform(reg_df)
    pred_col = outr.columns[0]
    check("tuner_transform_finite", len(outr) == len(reg_df) and
          np.isfinite(outr[pred_col]).all())
    tuner.save(PKL)
    tuner2 = OptunaTuner()
    tuner2.load(PKL)
    check("tuner_save_load",
          np.allclose(outr[pred_col].to_numpy(),
                      tuner2.transform(reg_df)[pred_col].to_numpy(), atol=1e-10))
else:
    print("[SKIP] OptunaTuner tests (optuna not installed)")

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
