# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
import numpy as np
from easymlops.table.core.pipeline import TablePipeLine
from easymlops.table.encoding import LGBMLeafEncoder

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

print("=== LGBMLeafEncoder Test ===")
print(f"Input data shape: {df.shape}")
print(f"Label distribution: {y.value_counts().to_dict()}")

print("\n1. Creating pipeline with encoder...")
pipeline = TablePipeLine()
encoder = LGBMLeafEncoder(
    y=y,
    n_estimators=5,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.1,
    min_child_samples=10
)
pipeline.pipe(encoder)

print("2. Fitting...")
pipeline.fit(df)
print("   Fit passed")

print("\n3. Testing describe()...")
descriptions = encoder.describe()
print(f"   Total leaves: {len(descriptions)}")
for idx, desc in list(descriptions.items())[:5]:
    print(f"   Leaf {idx}: {desc}")

print("\n4. Transforming...")
result = pipeline.transform(df)
print(f"   Transform result shape: {result.shape}")
print(f"   Sample result (first 5 rows, first 10 cols):\n{result.iloc[:5, :10]}")

print("\n5. Checking leaf命中情况...")
leaf_sums = result.sum(axis=0)
print(f"   Leaves with hits: {(leaf_sums > 0).sum()}")
print(f"   Leaves without hits: {(leaf_sums == 0).sum()}")

print("\n6. Testing transform_single...")
sample_dict = df.iloc[0].to_dict()
single_result = encoder.transform_single(sample_dict)
print(f"   Single result (first 10): {list(single_result.items())[:10]}")

print("\n7. Testing save/load via pipeline...")
pipeline.save("d:/datas/share/traes/easymlops-trae/test_lgbm_leaf_encoder.pkl")
print("   Save passed")

loaded_pipeline = TablePipeLine()
loaded_pipeline.load("d:/datas/share/traes/easymlops-trae/test_lgbm_leaf_encoder.pkl")
print("   Load passed")

print("\n8. Testing transform after load (no re-fit needed)...")
result2 = loaded_pipeline.transform(df.head(5))
print(f"   Transform after load shape: {result2.shape}")

print("\n9. Verifying describe after load...")
loaded_encoder = loaded_pipeline.models[0]
descriptions2 = loaded_encoder.describe()
print(f"   Describe after load: {len(descriptions2)} leaves")

print("\n=== All LGBMLeafEncoder tests PASSED ===")
