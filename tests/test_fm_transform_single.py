# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
import numpy as np
from easymlops.table.core.pipeline import TablePipeLine
from easymlops.table.fm import FMClassification

np.random.seed(42)

print("=== FM transform_single Test ===\n")

n_samples = 200
df = pd.DataFrame({
    "feature1": np.random.randint(0, 5, n_samples),
    "feature2": np.random.randint(0, 10, n_samples),
    "feature3": np.random.randint(0, 8, n_samples),
    "feature4": np.random.randint(0, 3, n_samples),
})
label = pd.Series(np.random.randint(0, 2, n_samples))

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

single_data = df.iloc[0].to_dict()
print(f"Input single data: {single_data}")

result = pipeline.transform_single(single_data)
print(f"\nTransform single result: {result}")

print("\n=== FM transform_single Test PASSED ===")
