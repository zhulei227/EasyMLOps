# -*- coding: utf-8 -*-
"""示例2：分类任务训练、预测、评估、持久化（已验证跑通，easymlops 0.2.4）
运行: python scripts/example_classification.py
产物: scripts/output/cls_model.pkl（供 deploy_fastapi.py 使用）
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, roc_auc_score
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import OneHotEncoding, LabelEncoding
from easymlops.table.classification import LGBMClassification
from easymlops.table.ensemble import Parallel

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

np.random.seed(42)
X, y = make_classification(n_samples=800, n_features=8, n_informative=5, random_state=42)
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
df["gender"] = np.random.choice(["M", "F"], 800)      # 类别特征
df["label"] = y
# 保存样例数据便于单独查看
df.to_csv(os.path.join(OUT, "sample_classification.csv"), index=False)

train, test = df[:600], df[600:]
x_train, y_train = train.drop(columns=["label"]), train["label"]
x_test, y_test = test.drop(columns=["label"]), test["label"]

pipe = TablePipeLine()
pipe.pipe(FixInput()) \
    .pipe(FillNa(cols=[f"f{i}" for i in range(8)], fill_mode="mean")) \
    .pipe(Parallel([
        OneHotEncoding(cols=["gender"]),
        LabelEncoding(cols=["gender"]),
    ])) \
    .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))

pipe.fit(x_train)
pred = pipe.transform(x_test)
print("[CLS] 输出列:", list(pred.columns))
# 输出列为各类别概率（列名为类别id），取类别1的概率
prob_col = "1" if "1" in pred.columns else pred.columns[-1]
y_prob = pred[prob_col].values
if y_prob.ndim > 1:
    y_prob = y_prob[:, 1]
y_pred = (y_prob > 0.5).astype(int)
print("[CLS] Accuracy: %.4f | AUC: %.4f" % (accuracy_score(y_test, y_pred), roc_auc_score(y_test, y_prob)))

# 单条预测
single = pipe.transform_single(x_test.to_dict("records")[0])
print("[CLS] 单条预测可用:", single is not None)

# 持久化（save 存参数字典，load 需先构造实例）
model_path = os.path.join(OUT, "cls_model.pkl")
pipe.save(model_path)
pipe2 = TablePipeLine()
pipe2.load(model_path)
prob2 = pipe2.transform(x_test)[prob_col].values
print("[CLS] 加载后预测一致:", bool(np.allclose(y_prob, prob2)))
print("CLASSIFICATION PASSED")
