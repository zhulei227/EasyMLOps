# -*- coding: utf-8 -*-
"""示例3：回归任务训练与预测（已验证跑通，easymlops 0.2.4）
运行: python scripts/example_regression.py
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error, r2_score
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import OneHotEncoding
from easymlops.table.regression import LGBMRegression, LinearRegression
from easymlops.table.ensemble import Parallel

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)
X, y = make_regression(n_samples=600, n_features=6, n_informative=4, noise=10.0, random_state=42)
df = pd.DataFrame(X, columns=[f"x{i}" for i in range(6)])
df["region"] = np.random.choice(["east", "west", "north"], 600)
df["target"] = y
# 保存样例数据便于单独查看
df.to_csv(os.path.join(OUT, "sample_regression.csv"), index=False)

train, test = df[:450], df[450:]
x_train = train.drop(columns=["target"])
x_test = test.drop(columns=["target"])
y_test = test["target"]

pipe = TablePipeLine()
pipe.pipe(FixInput()) \
    .pipe(FillNa(cols=[f"x{i}" for i in range(6)], fill_mode="mean")) \
    .pipe(OneHotEncoding(cols=["region"])) \
    .pipe(Parallel([
        LGBMRegression(y=train["target"], use_faster_predictor=False, prefix="lgbm"),
        LinearRegression(y=train["target"], prefix="lr"),
    ]))

pipe.fit(x_train)
pred = pipe.transform(x_test)
print("[REG] 输出列:", list(pred.columns))
lgbm_pred = pred[[c for c in pred.columns if "lgbm" in c][0]].values
print("[REG] LGBM RMSE: %.4f | R2: %.4f" % (
    mean_squared_error(y_test, lgbm_pred) ** 0.5, r2_score(y_test, lgbm_pred)))
lr_pred = pred[[c for c in pred.columns if "lr" in c][0]].values
print("[REG] Linear RMSE: %.4f | R2: %.4f" % (
    mean_squared_error(y_test, lr_pred) ** 0.5, r2_score(y_test, lr_pred)))
print("REGRESSION PASSED")
