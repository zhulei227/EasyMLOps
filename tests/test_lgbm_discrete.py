# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
import numpy as np
from easymlops.table.core.pipeline import TablePipeLine
from easymlops.table.encoding import LGBMLeafEncoder

np.random.seed(42)
n_samples = 200

df = pd.DataFrame({
    "feature1": np.random.randn(n_samples) * 10,
    "feature2": np.random.randint(0, 20, n_samples),
    "feature3": np.random.choice([1, 2, 3, 4, 5], n_samples),
    "feature4": np.random.choice([10, 20, 30], n_samples),
})

y = (df["feature1"] > 0).astype(int) + (df["feature3"] > 3).astype(int)
y = (y >= 1).astype(int)

print("=== LGBMLeafEncoder Discrete Features Test ===")
print(f"Input data shape: {df.shape}")
print(f"Feature types:")
for col in df.columns:
    print(f"  {col}: {df[col].nunique()} unique values")

print("\n1. Creating pipeline with encoder...")
pipeline = TablePipeLine()
encoder = LGBMLeafEncoder(
    y=y,
    n_estimators=3,
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
print("   Sample descriptions:")
for idx, desc in list(descriptions.items())[:10]:
    print(f"   Leaf {idx}: {desc}")

print("\n4. Transforming...")
result = pipeline.transform(df)
print(f"   Transform result shape: {result.shape}")

print("\n5. Testing save/load...")
pipeline.save("d:/datas/share/traes/easymlops-trae/test_lgbm_discrete.pkl")
loaded_pipeline = TablePipeLine()
loaded_pipeline.load("d:/datas/share/traes/easymlops-trae/test_lgbm_discrete.pkl")
print("   Save/Load passed")

print("\n6. Verify describe after load...")
loaded_encoder = loaded_pipeline.models[0]
descriptions2 = loaded_encoder.describe()
print(f"   Loaded describe: {len(descriptions2)} leaves")
print("   Sample:")
for idx, desc in list(descriptions2.items())[:5]:
    print(f"   Leaf {idx}: {desc}")

print("\n=== Discrete Features Test PASSED ===")
