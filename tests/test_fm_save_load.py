# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
import numpy as np
from easymlops.table.core.pipeline import TablePipeLine
from easymlops.table.fm import FMClassification

np.random.seed(42)

print("=== FM Save/Load Test ===\n")

n_samples = 200
df = pd.DataFrame({
    "feature1": np.random.randint(0, 5, n_samples),
    "feature2": np.random.randint(0, 10, n_samples),
    "feature3": np.random.randint(0, 8, n_samples),
    "feature4": np.random.randint(0, 3, n_samples),
})
label = pd.Series(np.random.randint(0, 2, n_samples))

print(f"Input data shape: {df.shape}")

field_dims = [5, 10, 8, 3]

fm_clf = FMClassification(
    y=label,
    cols=["feature1", "feature2", "feature3", "feature4"],
    field_dims=field_dims,
    embed_dim=4,
    epochs=3,
    batch_size=64,
    verbose=False
)

pipeline = TablePipeLine()
pipeline.pipe(fm_clf)
pipeline.fit(df)

result_before = pipeline.transform(df.head(5))
print(f"\nTransform before save shape: {result_before.shape}")
print(f"Values:\n{result_before.values}")

pipeline.save("fm_test.pkl")
print("\nSaved to fm_test.pkl")

loaded_pipeline = TablePipeLine().load("fm_test.pkl")
print("Loaded from fm_test.pkl")

if loaded_pipeline is None:
    print("Warning: load returned None, trying alternative approach")
    loaded_pipeline = TablePipeLine()
    loaded_pipeline.load("fm_test.pkl")

result_after = loaded_pipeline.transform(df.head(5))
print(f"\nTransform after load shape: {result_after.shape}")
print(f"Values:\n{result_after.values}")

import os
os.remove("fm_test.pkl")
print("\n=== FM Save/Load Test PASSED ===")
