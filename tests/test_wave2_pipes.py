# -*- coding: utf-8 -*-
# 第二波 pipe 验收: TextStatsFeatures / AnomalyScoreFeatures / GMM 聚类 / focal loss /
# StackingClassification / DistillationPipe / group_kfold_eval (v0.2.7)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from easymlops.table.text_features import TextStatsFeatures
from easymlops.table.features import AnomalyScoreFeatures, ClusterFeatures
from easymlops.table.nn import TabularMLPClassification
from easymlops.table.ensemble import StackingClassification
from easymlops.table.distillation import DistillationPipe
from easymlops.table.validation import group_kfold_eval
from easymlops.table.classification import LGBMClassification, LogisticRegressionClassification
from easymlops.table.regression import LGBMRegression

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_wave2.pkl"


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
                                   n_classes=2, random_state=7)
base_df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(6)])
base_df["city"] = rng.choice(["bj", "sh", "sz"], n)
base_df["group_id"] = rng.choice(np.arange(50), n)     # 分组 CV 用
y_clf = pd.Series(y_clf)
x_reg, y_reg = make_regression(n_samples=n, n_features=6, n_informative=4,
                               noise=10.0, random_state=7)
reg_df = pd.DataFrame(x_reg, columns=[f"f{i}" for i in range(6)])
y_reg = pd.Series(y_reg)
stack_df = base_df.drop(columns=["city"])   # 纯数值子集(LGBM 不吃字符串列)

# ============ 1. TextStatsFeatures ============
print("=== 1. TextStatsFeatures ===")
ts_df = pd.DataFrame({"remark": ["Good service 123", "bad", "", "EXCELLENT Support", None]})
ts = TextStatsFeatures(cols=["remark"])
ts.fit(ts_df)
outs = ts.transform(ts_df.copy())
check("tstats_cols", all(c in outs.columns for c in
                         ["remark_len", "remark_words", "remark_digit_ratio", "remark_upper_ratio"]))
row0 = outs.iloc[0]
check("tstats_values", row0["remark_len"] == 16 and row0["remark_words"] == 3 and
      abs(row0["remark_digit_ratio"] - 3 / 16) < 1e-9)
s = ts.transform_single({"remark": "AB 12"})
check("tstats_single", s["remark_len"] == 5 and s["remark_words"] == 2)

# ============ 2. AnomalyScoreFeatures ============
print("=== 2. AnomalyScoreFeatures ===")
an_df = pd.DataFrame({"v": np.concatenate([rng.normal(0, 1, 280), rng.normal(0, 1, 20) + 20])})
an = AnomalyScoreFeatures(cols=["v"], n_estimators=50, random_state=7)
an.fit(an_df)
outa = an.transform(an_df.copy())
check("anomaly_col", "anomaly_score" in outa.columns and
      np.isfinite(outa["anomaly_score"]).all())
outliers = outa["anomaly_score"].to_numpy()[280:]
normals = outa["anomaly_score"].to_numpy()[:280]
check("anomaly_score_meaningful", outliers.mean() > normals.mean(),
      "outlier={:.3f} normal={:.3f}".format(outliers.mean(), normals.mean()))
an.save(PKL)
an2 = AnomalyScoreFeatures()
an2.load(PKL)
check("anomaly_save_load", np.allclose(outa["anomaly_score"],
                                       an2.transform(an_df.copy())["anomaly_score"]))
s = an.transform_single({"v": 25.0})
check("anomaly_single", "anomaly_score" in s and np.isfinite(s["anomaly_score"]))

# ============ 3. ClusterFeatures GMM ============
print("=== 3. ClusterFeatures GMM ===")
gmm = ClusterFeatures(cols=["value"], n_clusters=2, method="gmm", random_state=7)
gmmdf = pd.DataFrame({"value": np.concatenate([rng.normal(0, 1, 150),
                                               rng.normal(10, 1, 150)])})
gmm.fit(gmmdf)
outg = gmm.transform(gmmdf.copy())
check("gmm_cols", "kmeans_cluster_id" in outg.columns and "kmeans_dist_0" in outg.columns and
      np.isfinite(outg["kmeans_dist_0"]).all())
check("gmm_two_clusters", outg["kmeans_cluster_id"].nunique() == 2)
gmm.save(PKL)
gmm2 = ClusterFeatures(n_clusters=2, method="gmm")
gmm2.load(PKL)
check("gmm_save_load", np.allclose(outg["kmeans_cluster_id"],
                                   gmm2.transform(gmmdf.copy())["kmeans_cluster_id"]))

# ============ 4. focal / weighted loss (TabularMLP) ============
print("=== 4. focal/weighted loss ===")
x_c, y_c = make_classification(n_samples=500, n_features=6, n_informative=4,
                               weights=[0.9, 0.1], random_state=7)
clf_df = pd.DataFrame(x_c, columns=[f"f{i}" for i in range(6)])
y_c = pd.Series(y_c)
mf = TabularMLPClassification(y=y_c, epochs=6, hidden_dims=(32,), random_state=42,
                              loss="focal", focal_gamma=2.0,
                              class_weight={0: 1.0, 1: 5.0})
mf.fit(clf_df)
outf = mf.transform(clf_df.copy())
check("focal_loss_trains", np.isfinite(outf.values).all() and
      np.allclose(outf.sum(axis=1), 1.0, atol=1e-5))
mf.save(PKL)
mf2 = TabularMLPClassification()
mf2.load(PKL)
check("focal_save_load", np.allclose(outf.values, mf2.transform(clf_df.copy()).values, atol=1e-6))

# ============ 5. StackingClassification ============
print("=== 5. StackingClassification ===")
stack = StackingClassification(
    base_pipes=[LGBMClassification(y=y_clf, use_faster_predictor=False),
                LogisticRegressionClassification(y=y_clf)],
    meta_pipe=LogisticRegressionClassification(y=y_clf), cv=4, random_state=42)
stack.fit(stack_df)
outs = stack.transform(stack_df.copy())
check("stack_prob_cols", set(outs.columns) == {0, 1} and len(outs) == len(stack_df))
check("stack_prob_valid", np.allclose(outs.sum(axis=1).to_numpy(), 1.0, atol=1e-5) and
      np.isfinite(outs.values).all())
stack.save(PKL)
stack2 = StackingClassification()
stack2.load(PKL)
check("stack_save_load", np.allclose(outs.values, stack2.transform(base_df.copy()).values,
                                     atol=1e-6))
s = stack.transform_single(dict(stack_df.iloc[0]))
check("stack_single", set(s.keys()) == {0, 1} and abs(sum(s.values()) - 1.0) < 1e-5)

# ============ 6. DistillationPipe ============
print("=== 6. DistillationPipe ===")
distill = DistillationPipe(
    teacher_pipe=LGBMClassification(y=y_clf, use_faster_predictor=False),
    student_params={"hidden_dims": (32,), "epochs": 5},
    alpha=0.7, temperature=2.0, task="classification")
distill.fit(stack_df)
outd = distill.transform(stack_df.copy())
check("distill_clf_prob", list(outd.columns) == [0, 1] and
      np.allclose(outd.sum(axis=1).to_numpy(), 1.0, atol=1e-5))
distill.save(PKL)
distill2 = DistillationPipe(task="classification")
distill2.load(PKL)
check("distill_save_load", np.allclose(outd.values, distill2.transform(base_df.copy()).values,
                                       atol=1e-6))
# 回归蒸馏
distill_r = DistillationPipe(
    teacher_pipe=LGBMRegression(y=y_reg, use_faster_predictor=False),
    student_params={"hidden_dims": (32,), "epochs": 5}, task="regression")
distill_r.fit(reg_df)
outr = distill_r.transform(reg_df.copy())
check("distill_reg_finite", np.isfinite(outr["pred"]).all() and len(outr) == len(reg_df))

# ============ 7. group_kfold_eval ============
print("=== 7. group_kfold_eval ===")
result = group_kfold_eval(
    build_model=lambda fold_y: LGBMRegression(y=fold_y, use_faster_predictor=False),
    x=reg_df, y=y_reg, groups=base_df["group_id"], cv=3, task="regression")
check("gkf_scores", len(result["scores"]) == 3 and
      all(np.isfinite(v) and v > 0 for v in result["scores"]) and
      "mean" in result and "std" in result)
result_c = group_kfold_eval(
    build_model=lambda fold_y: LGBMClassification(y=fold_y, use_faster_predictor=False),
    x=stack_df, y=y_clf, groups=base_df["group_id"], cv=3, task="classification")
check("gkf_classification", len(result_c["scores"]) == 3 and
      all(np.isfinite(v) for v in result_c["scores"]))

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
