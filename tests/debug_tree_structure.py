# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
import numpy as np
import lightgbm as lgb

np.random.seed(42)
n_samples = 100

df = pd.DataFrame({
    "feature1": np.random.randn(n_samples) * 10,
    "feature2": np.random.randint(0, 20, n_samples),
    "feature3": np.random.choice([1, 2, 3, 4, 5], n_samples),
    "feature4": np.random.randn(n_samples) * 5
})

y = (df["feature1"] > 0).astype(int) + (df["feature2"] > 10).astype(int)
y = (y >= 1).astype(int)

X = df.values

params = {
    "objective": "regression",
    "boosting_type": "gbdt",
    "num_leaves": 8,
    "max_depth": 3,
    "learning_rate": 0.1,
    "n_estimators": 5,
    "min_child_samples": 10,
    "verbosity": -1,
    "feature_name": list(df.columns)
}

model = lgb.LGBMRegressor(**params)
model.fit(X, y)

model_dict = model.booster_.dump_model()

print("=== Model structure ===")
for i, tree_data in enumerate(model_dict["tree_info"]):
    print(f"\nTree {i}:")
    print(f"  num_leaves: {tree_data.get('num_leaves')}")
    print(f"  tree_structure: {tree_data['tree_structure']}")
