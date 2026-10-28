# -*- coding: utf-8 -*-
# 新增 pipe 验收: SeedAveraging / PseudoLabelPipe / PredictionPostProcess (v0.2.5)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from easymlops.table.ensemble import SeedAveraging, PseudoLabelPipe
from easymlops.table.postprocess import PredictionPostProcess
from easymlops.table.classification import LGBMClassification
from easymlops.table.regression import LGBMRegression
from easymlops import TablePipeLine

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_new_pipes.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


def make_clf_data(n=500):
    x, y = make_classification(n_samples=n, n_features=8, n_informative=5,
                               n_classes=2, random_state=7)
    cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    y = pd.Series(y, name="label")
    return df, y


def make_reg_data(n=500):
    x, y = make_regression(n_samples=n, n_features=8, n_informative=5,
                           noise=10.0, random_state=7)
    cols = [f"f{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    y = pd.Series(y, name="target")
    return df, y


df_clf, y_clf = make_clf_data()
train_clf, test_clf = df_clf[:400], df_clf[400:]
y_train_clf, y_test_clf = y_clf[:400], y_clf[400:]

# ============ 1. SeedAveraging 分类 ============
print("=== 1. SeedAveraging classification ===")
sa = SeedAveraging(base_pipe=LGBMClassification(y=y_train_clf, use_faster_predictor=False,
                             native_init_params={"bagging_fraction": 0.8, "bagging_freq": 1,
                                                 "feature_fraction": 0.8}),
                   n_seeds=3, seed_start=42)
sa.fit(train_clf)
out = sa.transform(test_clf)
check("sa_clf_rows_and_cols", len(out) == len(test_clf) and
      list(out.columns) == [0, 1] and
      np.isfinite(out.values).all())
check("sa_clf_prob_valid",
      ((out.values >= 0) & (out.values <= 1)).all() and
      np.allclose(out.sum(axis=1).to_numpy(), 1.0, atol=1e-6))
# 确定性: 相同种子配置 -> 相同预测
sa2 = SeedAveraging(base_pipe=LGBMClassification(y=y_train_clf, use_faster_predictor=False,
                             native_init_params={"bagging_fraction": 0.8, "bagging_freq": 1,
                                                 "feature_fraction": 0.8}),
                    n_seeds=3, seed_start=42)
sa2.fit(train_clf)
check("sa_clf_deterministic",
      np.allclose(out.values, sa2.transform(test_clf).values, atol=1e-10))
# 不同种子 -> 不同模型(预测有差异)
sa3 = SeedAveraging(base_pipe=LGBMClassification(y=y_train_clf, use_faster_predictor=False,
                             native_init_params={"bagging_fraction": 0.8, "bagging_freq": 1,
                                                 "feature_fraction": 0.8}),
                    n_seeds=3, seed_start=100)
sa3.fit(train_clf)
check("sa_clf_seeds_change_model",
      not np.allclose(out.values, sa3.transform(test_clf).values, atol=1e-10))
# save/load
sa.save(PKL)
sa_loaded = SeedAveraging(n_seeds=3, seed_start=42)
sa_loaded.load(PKL)
check("sa_clf_save_load_consistent",
      np.allclose(out.values, sa_loaded.transform(test_clf).values, atol=1e-10))
# 单条推理
single = sa.transform_single(dict(test_clf.iloc[0]))
check("sa_clf_single", set(single.keys()) == {0, 1} and
      abs(sum(single.values()) - 1.0) < 1e-6)

# ============ 2. SeedAveraging 回归 ============
print("=== 2. SeedAveraging regression ===")
df_reg, y_reg = make_reg_data()
train_reg, test_reg = df_reg[:400], df_reg[400:]
y_train_reg, y_test_reg = y_reg[:400], y_reg[400:]
sar = SeedAveraging(base_pipe=LGBMRegression(y=y_train_reg, use_faster_predictor=False),
                    n_seeds=3, seed_start=42)
sar.fit(train_reg)
outr = sar.transform(test_reg)
pred_col = outr.columns[0]
check("sa_reg_rows_and_finite", len(outr) == len(test_reg) and
      np.isfinite(outr[pred_col]).all())
sar.save(PKL)
sar_loaded = SeedAveraging(n_seeds=3, seed_start=42)
sar_loaded.load(PKL)
check("sa_reg_save_load_consistent",
      np.allclose(outr[pred_col].to_numpy(),
                  sar_loaded.transform(test_reg)[pred_col].to_numpy(), atol=1e-10))

# ============ 3. PseudoLabelPipe 分类 ============
print("=== 3. PseudoLabelPipe classification ===")
pl = PseudoLabelPipe(base_pipe=LGBMClassification(y=y_train_clf, use_faster_predictor=False,
                             native_init_params={"bagging_fraction": 0.8, "bagging_freq": 1,
                                                 "feature_fraction": 0.8}),
                     confidence=0.6, n_rounds=1)
pl.fit(train_clf, unlabeled=test_clf)   # test_clf 特征作为无标签数据
outp = pl.transform(test_clf)
check("pl_clf_rows_and_prob", len(outp) == len(test_clf) and
      np.isfinite(outp.values).all() and
      np.allclose(outp.sum(axis=1).to_numpy(), 1.0, atol=1e-6))
pl.save(PKL)
pl_loaded = PseudoLabelPipe()
pl_loaded.load(PKL)
check("pl_clf_save_load_consistent",
      np.allclose(outp.values, pl_loaded.transform(test_clf).values, atol=1e-10))
single = pl.transform_single(dict(test_clf.iloc[0]))
check("pl_clf_single", set(single.keys()) == {0, 1} and
      abs(sum(single.values()) - 1.0) < 1e-6)

# ============ 4. PseudoLabelPipe 回归 ============
print("=== 4. PseudoLabelPipe regression ===")
plr = PseudoLabelPipe(base_pipe=LGBMRegression(y=y_train_reg, use_faster_predictor=False),
                      n_rounds=1)
plr.fit(train_reg, unlabeled=test_reg)
outr2 = plr.transform(test_reg)
check("pl_reg_rows_and_finite", len(outr2) == len(test_reg) and
      np.isfinite(outr2[pred_col]).all())

# ============ 5. PredictionPostProcess ============
print("=== 5. PredictionPostProcess ===")
df_src = pd.DataFrame({"pred": [0.1, 5.0, 2.34567, -1.0], "other": [1, 2, 3, 4]})
pp = PredictionPostProcess(cols=["pred"], clip=(0.0, 3.0), round_digits=2)
pp.fit(df_src)
outpp = pp.transform(df_src)
check("pp_clip_round",
      np.allclose(outpp["pred"].to_numpy(), [0.1, 3.0, 2.35, 0.0]) and
      (outpp["other"].to_numpy() == [1, 2, 3, 4]).all())
# threshold 二值化
ppb = PredictionPostProcess(cols=["pred"], threshold=0.5)
ppb.fit(pd.DataFrame({"pred": [0.0]}))
outb = ppb.transform(pd.DataFrame({"pred": [0.3, 0.7, 0.5]}))
check("pp_threshold", (outb["pred"].to_numpy() == [0.0, 1.0, 0.0]).all())
# 单条一致性
s = pp.transform_single(dict(df_src.iloc[2]))
check("pp_single_consistent", abs(s["pred"] - 2.35) < 1e-9)
# mapping 自定义映射
ppm = PredictionPostProcess(mapping=lambda d: d.assign(pred=d["pred"] * 10))
ppm.fit(pd.DataFrame({"pred": [0.0]}))
outm = ppm.transform(pd.DataFrame({"pred": [1.0, 2.0]}))
check("pp_mapping", np.allclose(outm["pred"].to_numpy(), [10.0, 20.0]))

# ============ 6. Pipeline 集成 ============
print("=== 6. pipeline integration ===")
pipeline = TablePipeLine()
pipeline.pipe(LGBMRegression(y=y_train_reg, use_faster_predictor=False)) \
        .pipe(PredictionPostProcess(cols=[pred_col], clip=(None, None), round_digits=3))
pipeline.fit(train_reg)
pout = pipeline.transform(test_reg)
check("pp_pipeline_integration", len(pout) == len(test_reg) and
      np.isfinite(pout[pred_col]).all() and
      (pout[pred_col] == pout[pred_col].round(3)).all())
psingle = pipeline.transform_single(dict(test_reg.iloc[0]))
check("pp_pipeline_single", pred_col in psingle and
      abs(psingle[pred_col] - round(psingle[pred_col], 3)) < 1e-9)

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
