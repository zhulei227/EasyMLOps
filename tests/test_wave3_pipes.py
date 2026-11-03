# -*- coding: utf-8 -*-
# 第三波 pipe 验收 (v0.2.10): AugmentationPipe / HillClimbingStackingClassification /
# FFTSpectralFeature / TabularMLP label_smoothing
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from easymlops.table.ensemble import AugmentationPipe, HillClimbingStackingClassification
from easymlops.table.classification import LGBMClassification, LogisticRegressionClassification
from easymlops.table.nn import TabularMLPClassification
from easymlops.ts.features import FFTSpectralFeature

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_wave3.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(7)
n = 500
x_clf, y_clf = make_classification(n_samples=n, n_features=6, n_informative=4,
                                   n_classes=2, weights=[0.85, 0.15], random_state=7)
clf_df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(6)])
clf_df["city"] = rng.choice(["bj", "sh"], n)
y_clf = pd.Series(y_clf)
train_c, test_c = clf_df[:350].copy(), clf_df[350:].copy()
y_train_c = y_clf[:350]

x_reg, y_reg = make_regression(n_samples=n, n_features=5, n_informative=4,
                               noise=10.0, random_state=7)
reg_df = pd.DataFrame(x_reg, columns=[f"f{i}" for i in range(5)])
y_reg = pd.Series(y_reg)
train_r, test_r = reg_df[:350].copy(), reg_df[350:].copy()
y_train_r = y_reg[:350]

# ============ 1. AugmentationPipe: smote ============
print("=== 1. AugmentationPipe smote ===")
from easymlops.table.regression import LGBMRegression
aug = AugmentationPipe(
    base_pipe=TabularMLPClassification(y=y_train_c, epochs=5, hidden_dims=(16,),
                                       categorical_cols=["city"], random_state=42),
    method="smote", sampling_ratio=1.0, k_neighbors=4, random_state=42)
aug.fit(train_c)
outa = aug.transform(test_c.copy())
check("smote_transform", np.isfinite(outa.astype(float).values).all() and
      len(outa) == len(test_c))
aug.save(PKL)
aug2 = AugmentationPipe(method="smote")
aug2.load(PKL)
check("smote_save_load", np.allclose(outa.astype(float).values,
                                     aug2.transform(test_c.copy()).astype(float).values,
                                     atol=1e-9))


# ============ 2. AugmentationPipe: oversample / mixup / gaussian ============
print("=== 2. AugmentationPipe other methods ===")
aug_o = AugmentationPipe(
    base_pipe=TabularMLPClassification(y=y_train_c, epochs=3, hidden_dims=(16,),
                                       categorical_cols=["city"], random_state=42),
    method="oversample", sampling_ratio=1.0, random_state=42)
aug_o.fit(train_c)
check("oversample_transform", np.isfinite(
    aug_o.transform(test_c.copy()).astype(float).values).all())
aug_m = AugmentationPipe(
    base_pipe=LGBMRegression(y=y_train_r, use_faster_predictor=False),
    method="mixup", n_augment=100, alpha=0.2, random_state=42)
aug_m.fit(train_r)
outr = aug_m.transform(test_r.copy())
pred_col_m = outr.columns[0]
check("mixup_transform", len(outr) == len(test_r) and
      np.isfinite(outr[pred_col_m].astype(float)).all())
aug_g = AugmentationPipe(
    base_pipe=LGBMRegression(y=y_train_r, use_faster_predictor=False),
    method="gaussian", noise_std=0.1, random_state=42)
aug_g.fit(train_r)
gout = aug_g.transform(test_r.copy())
gcol = [c for c in gout.columns if c not in test_r.columns][0] if \
    len([c for c in gout.columns if c not in test_r.columns]) else gout.columns[0]
check("gaussian_transform", np.isfinite(gout[gcol].astype(float)).all())
aug_m.save(PKL)
aug_m2 = AugmentationPipe(method="mixup")
aug_m2.load(PKL)
check("mixup_save_load", np.allclose(
    outr[pred_col_m].astype(float).to_numpy(),
    aug_m2.transform(test_r.copy())[pred_col_m].astype(float).to_numpy(), atol=1e-9))

# ============ 3. HillClimbingStackingClassification ============
print("=== 3. HillClimbingStackingClassification ===")
from easymlops.table.ensemble import Parallel
from easymlops import TablePipeLine

pipeline = TablePipeLine()
pipeline.pipe(Parallel(pipe_objects=[
    LGBMClassification(y=y_train_c, prefix="lgbm", use_faster_predictor=False,
                       native_init_params={"bagging_fraction": 0.8, "bagging_freq": 1}),
    LogisticRegressionClassification(y=y_train_c, prefix="lr"),
]))
pipeline.pipe(HillClimbingStackingClassification(y=y_train_c, eval_function="auc",
                                                 n_splits=3, random_state=42))
stack_df = clf_df.drop(columns=["city"])   # 纯数值子集(LGBM 不吃字符串列)
stack_train, stack_test = stack_df[:350].copy(), stack_df[350:].copy()
pipeline.fit(stack_train)
outp = pipeline.transform(stack_test.copy())
prob_cols = [c for c in outp.columns if c in (0, 1, "0", "1")]
check("hc_prob_cols", len(prob_cols) == 2)
pc = outp[prob_cols].astype(float)
check("hc_prob_valid", np.allclose(pc.sum(axis=1), 1.0, atol=1e-5) and
      np.isfinite(pc.values).all())
pipeline.save(PKL)
pipeline2 = TablePipeLine()
pipeline2.load(PKL)
pc2 = pipeline2.transform(stack_test.copy())[prob_cols].astype(float)
check("hc_save_load", np.allclose(pc.values, pc2.values, atol=1e-9))


# ============ 4. TabularMLP label_smoothing ============
print("=== 4. label smoothing ===")
mls = TabularMLPClassification(y=y_train_c, epochs=5, hidden_dims=(16,),
                               categorical_cols=["city"], random_state=42,
                               label_smoothing=0.1)
mls.fit(train_c)
outls = mls.transform(test_c.copy())
check("label_smoothing_trains", np.isfinite(outls.values).all() and
      np.allclose(outls.sum(axis=1), 1.0, atol=1e-5))
mls.save(PKL)
mls2 = TabularMLPClassification()
mls2.load(PKL)
check("label_smoothing_persist", np.allclose(outls.values, mls2.transform(test_c.copy()).values,
                                             atol=1e-9))

# ============ 5. FFTSpectralFeature (TS) ============
print("=== 5. FFTSpectralFeature ===")
n_ts = 120
ts_df = pd.DataFrame({
    "group": ["A"] * n_ts + ["B"] * n_ts,
    "date": list(pd.date_range("2026-01-01", periods=n_ts, freq="D")) * 2,
    "value": np.concatenate([
        np.sin(2 * np.pi * np.arange(n_ts) / 10) + rng.normal(0, 0.1, n_ts),
        rng.normal(0, 1, n_ts),
    ]),
})
fft_pipe = FFTSpectralFeature(cols=["value"], window=32, top_k=3)
fft_pipe.fit(ts_df)
ts_out = fft_pipe.transform(ts_df.copy())
fft_cols = [c for c in ts_out.columns if "_fft" in c]
check("fft_cols_present", len(fft_cols) == 5 and
      all(np.isfinite(ts_out[c].iloc[45:115]).all() and
          np.isfinite(ts_out[c].iloc[165:235]).all() for c in fft_cols),
      "cols={}".format(fft_cols[:3]))
# 周期组主频应接近 10 采样周期 -> domfreq ≈ 1/10 = 0.1
dom_a = ts_out[ts_out["group"] == "A"]["value_fft32_domfreq"].iloc[60]
dom_b = ts_out[ts_out["group"] == "B"]["value_fft32_domfreq"].iloc[60]
check("fft_domfreq_meaningful", abs(dom_a - 0.1) < 0.05,
      "A={:.3f} B={:.3f}".format(dom_a, dom_b))
fft_pipe.save(PKL)
fft_pipe2 = FFTSpectralFeature(cols=["value"], window=32, top_k=3)
fft_pipe2.load(PKL)
ts_out2 = fft_pipe2.transform(ts_df.copy())
check("fft_save_load", np.allclose(ts_out[fft_cols].iloc[45:115].to_numpy(),
                                   ts_out2[fft_cols].iloc[45:115].to_numpy(), atol=1e-9))

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("")
print("================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
