# -*- coding: utf-8 -*-
# TabularMLP 验收: 分类/回归/Embedding/未见类别/save-load/单条推理/确定性 (v0.2.6)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

from easymlops.table.nn import TabularMLPClassification, TabularMLPRegression
from easymlops import TablePipeLine

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_nn.pkl"


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

# ============ 数据: 数值 + 类别混合 ============
rng = np.random.RandomState(7)
n = 600
x_clf, y_clf = make_classification(n_samples=n, n_features=6, n_informative=4,
                                   n_classes=2, random_state=7)
clf_df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(6)])
clf_df["city"] = rng.choice(["bj", "sh", "sz", "gz"], n)          # 类别列
clf_df.loc[rng.choice(n, 30), "f0"] = np.nan                      # 数值缺失
clf_df.loc[rng.choice(n, 20), "city"] = None                      # 类别缺失
y_clf = pd.Series(y_clf)
train_c, test_c = clf_df[:480].copy(), clf_df[480:].copy()
y_train_c, y_test_c = y_clf[:480], y_clf[480:]

# ============ 1. 分类 ============
print("=== 1. TabularMLPClassification ===")
m = TabularMLPClassification(y=y_train_c, epochs=8, hidden_dims=(64, 32),
                             categorical_cols=["city"], random_state=42)
m.fit(train_c)
out = m.transform(test_c.copy())
check("clf_prob_cols", set(out.columns) == {0, 1} and len(out) == len(test_c))
check("clf_prob_valid", ((out.values >= 0) & (out.values <= 1)).all() and
      np.allclose(out.sum(axis=1).to_numpy(), 1.0, atol=1e-5))
check("clf_col_split", m.num_cols_ == [f"f{i}" for i in range(6)] and
      m.cat_cols_ == ["city"])
# 未见类别/缺失自动映射 id 0, 不报错
weird = test_c.copy().iloc[[0]].assign(city="__never_seen__")
outw = m.transform(weird)
check("clf_unseen_category", np.isfinite(outw.values).all())
# 确定性
m2 = TabularMLPClassification(y=y_train_c, epochs=8, hidden_dims=(64, 32),
                              categorical_cols=["city"], random_state=42)
m2.fit(train_c)
check("clf_deterministic", np.allclose(out.values, m2.transform(test_c.copy()).values,
                                       atol=1e-6))
# save/load(权重持久化)
m.save(PKL)
m_loaded = TabularMLPClassification()
m_loaded.load(PKL)
out2 = m_loaded.transform(test_c.copy())
check("clf_save_load_consistent", np.allclose(out.values, out2.values, atol=1e-6))
# 单条
s = m.transform_single(dict(test_c.iloc[0]))
check("clf_single", set(s.keys()) == {0, 1} and abs(sum(s.values()) - 1.0) < 1e-5)

# ============ 2. 回归 ============
print("=== 2. TabularMLPRegression ===")
x_reg, y_reg = make_regression(n_samples=n, n_features=6, n_informative=4,
                               noise=10.0, random_state=7)
reg_df = pd.DataFrame(x_reg, columns=[f"f{i}" for i in range(6)])
reg_df["region"] = rng.choice(["east", "west"], n)
y_reg = pd.Series(y_reg)
train_r, test_r = reg_df[:480].copy(), reg_df[480:].copy()
y_train_r, y_test_r = y_reg[:480], y_reg[480:]
mr = TabularMLPRegression(y=y_train_r, epochs=8, hidden_dims=(64, 32),
                          categorical_cols=["region"], random_state=42)
mr.fit(train_r)
outr = mr.transform(test_r.copy())
pred_col = mr.pred_name
check("reg_pred_col", pred_col in outr.columns and len(outr) == len(test_r) and
      np.isfinite(outr[pred_col]).all())
mr.save(PKL)
mr_loaded = TabularMLPRegression()
mr_loaded.load(PKL)
check("reg_save_load_consistent",
      np.allclose(outr[pred_col].to_numpy(),
                  mr_loaded.transform(test_r.copy())[pred_col].to_numpy(), atol=1e-5))
s = mr.transform_single(dict(test_r.iloc[0]))
check("reg_single", pred_col in s and np.isfinite(s[pred_col]))

# ============ 3. Pipeline 集成 ============
print("=== 3. pipeline integration ===")
pipeline = TablePipeLine()
pipeline.pipe(TabularMLPRegression(y=y_train_r, epochs=5, hidden_dims=(32,),
                                   categorical_cols=["region"], random_state=42))
pipeline.fit(train_r)
pout = pipeline.transform(test_r.copy())
check("pipeline_integration", len(pout) == len(test_r) and
      np.isfinite(pout["pred"]).all())
psingle = pipeline.transform_single(dict(test_r.iloc[0]))
check("pipeline_single", "pred" in psingle and np.isfinite(psingle["pred"]))

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("\n================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
