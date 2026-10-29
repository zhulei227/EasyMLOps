# -*- coding: utf-8 -*-
# TabNet 验收: 分类/回归/save-load/单条推理/确定性/类别Embedding/pipeline (v0.2.7)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from easymlops.table.nn import TabNetClassification, TabNetRegression

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_tabnet.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if not HAS_TORCH:
    print("[SKIP] torch not installed")
    sys.exit(0)

rng = np.random.RandomState(7)
n = 500
x_clf, y_clf = make_classification(n_samples=n, n_features=6, n_informative=4,
                                   n_classes=2, random_state=7)
clf_df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(6)])
clf_df["city"] = rng.choice(["bj", "sh", "sz"], n)
y_clf = pd.Series(y_clf)
train_c, test_c = clf_df[:400].copy(), clf_df[400:].copy()
y_train_c = y_clf[:400]

# ============ 1. 分类 ============
print("=== 1. TabNetClassification ===")
m = TabNetClassification(y=y_train_c, n_d=16, n_a=16, n_steps=3, epochs=8,
                         random_state=42)
m.fit(train_c)
out = m.transform(test_c.copy())
check("clf_prob_cols", set(out.columns) == {0, 1} and len(out) == len(test_c))
check("clf_prob_valid", np.isfinite(out.values).all() and
      np.allclose(out.sum(axis=1).to_numpy(), 1.0, atol=1e-5))
m.save(PKL)
m2 = TabNetClassification(n_d=16, n_a=16, n_steps=3)
m2.load(PKL)
out2 = m2.transform(test_c.copy())
check("clf_save_load_consistent", np.allclose(out.values, out2.values, atol=1e-6))
s = m.transform_single(dict(test_c.iloc[0]))
check("clf_single", set(s.keys()) == {0, 1} and abs(sum(s.values()) - 1.0) < 1e-5)
m3 = TabNetClassification(y=y_train_c, n_d=16, n_a=16, n_steps=3, epochs=8,
                          random_state=42)
m3.fit(train_c)
check("clf_deterministic", np.allclose(out.values, m3.transform(test_c.copy()).values,
                                       atol=1e-6))

# ============ 2. 回归(含类别列 Embedding) ============
print("=== 2. TabNetRegression ===")
x_reg, y_reg = make_regression(n_samples=n, n_features=6, n_informative=4,
                               noise=10.0, random_state=7)
reg_df = pd.DataFrame(x_reg, columns=[f"f{i}" for i in range(6)])
reg_df["region"] = rng.choice(["east", "west"], n)
y_reg = pd.Series(y_reg)
train_r, test_r = reg_df[:400].copy(), reg_df[400:].copy()
y_train_r = y_reg[:400]
mr = TabNetRegression(y=y_train_r, n_d=16, n_a=16, n_steps=3, epochs=8,
                      random_state=42)
mr.fit(train_r)
outr = mr.transform(test_r.copy())
check("reg_pred_finite", mr.pred_name in outr.columns and
      np.isfinite(outr[mr.pred_name]).all() and len(outr) == len(test_r))
mr.save(PKL)
mr2 = TabNetRegression(n_d=16, n_a=16, n_steps=3)
mr2.load(PKL)
check("reg_save_load_consistent",
      np.allclose(outr[mr.pred_name].to_numpy(),
                  mr2.transform(test_r.copy())[mr.pred_name].to_numpy(), atol=1e-5))

# ============ 3. pipeline 集成 ============
print("=== 3. pipeline integration ===")
from easymlops import TablePipeLine
pipeline = TablePipeLine()
pipeline.pipe(TabNetClassification(y=y_train_c, n_steps=2, epochs=5, random_state=42))
pipeline.fit(train_c)
pout = pipeline.transform(test_c.copy())
check("pipeline_integration", len(pout) == len(test_c) and
      set(pout.columns) >= {0, 1} and np.isfinite(pout.values).all())
psingle = pipeline.transform_single(dict(test_c.iloc[0]))
check("pipeline_single", set(psingle.keys()) >= {0, 1} and
      abs(sum(psingle[k] for k in (0, 1)) - 1.0) < 1e-5)

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
