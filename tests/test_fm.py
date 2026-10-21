# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import pandas as pd
import numpy as np
from easymlops.table.core.pipeline import TablePipeLine
from easymlops.table.fm import FMClassification, FMRegression, FFMClassification, FFMRegression, DeepFMClassification, DeepFMRegression

np.random.seed(42)

print("=== FM Module Test ===\n")

n_samples = 500
df = pd.DataFrame({
    "feature1": np.random.randint(0, 5, n_samples),
    "feature2": np.random.randint(0, 10, n_samples),
    "feature3": np.random.randint(0, 8, n_samples),
    "feature4": np.random.randint(0, 3, n_samples),
})
label = pd.Series(np.random.randint(0, 2, n_samples))

print(f"Input data shape: {df.shape}")
print(f"Label distribution: {label.value_counts().to_dict()}\n")

field_dims = [5, 10, 8, 3]

print("=" * 50)
print("1. Testing FMClassification...")
print("=" * 50)
try:
    fm_clf = FMClassification(
        y=label,
        cols=["feature1", "feature2", "feature3", "feature4"],
        field_dims=field_dims,
        embed_dim=4,
        epochs=5,
        batch_size=64,
        verbose=False
    )
    pipeline = TablePipeLine()
    pipeline.pipe(fm_clf)
    pipeline.fit(df)
    result = pipeline.transform(df.head(10))
    print(f"Transform result shape: {result.shape}")
    print(f"Result columns: {result.columns.tolist()}")
    print("FMClassification: PASSED\n")
except Exception as e:
    print(f"FMClassification: FAILED - {e}\n")

print("=" * 50)
print("2. Testing FMRegression...")
print("=" * 50)
try:
    y_reg = pd.Series(np.random.randn(n_samples) * 10)
    fm_reg = FMRegression(
        y=y_reg,
        cols=["feature1", "feature2", "feature3", "feature4"],
        field_dims=field_dims,
        embed_dim=4,
        epochs=5,
        batch_size=64,
        verbose=False
    )
    pipeline2 = TablePipeLine()
    pipeline2.pipe(fm_reg)
    pipeline2.fit(df)
    result2 = pipeline2.transform(df.head(10))
    print(f"Transform result shape: {result2.shape}")
    print(f"Result columns: {result2.columns.tolist()}")
    print("FMRegression: PASSED\n")
except Exception as e:
    print(f"FMRegression: FAILED - {e}\n")

print("=" * 50)
print("3. Testing FFMClassification...")
print("=" * 50)
try:
    ffm_clf = FFMClassification(
        y=label,
        cols=["feature1", "feature2", "feature3", "feature4"],
        field_dims=field_dims,
        embed_dim=4,
        epochs=5,
        batch_size=64,
        verbose=False
    )
    pipeline3 = TablePipeLine()
    pipeline3.pipe(ffm_clf)
    pipeline3.fit(df)
    result3 = pipeline3.transform(df.head(10))
    print(f"Transform result shape: {result3.shape}")
    print(f"Result columns: {result3.columns.tolist()}")
    print("FFMClassification: PASSED\n")
except Exception as e:
    print(f"FFMClassification: FAILED - {e}\n")

print("=" * 50)
print("4. Testing FFMRegression...")
print("=" * 50)
try:
    ffm_reg = FFMRegression(
        y=y_reg,
        cols=["feature1", "feature2", "feature3", "feature4"],
        field_dims=field_dims,
        embed_dim=4,
        epochs=5,
        batch_size=64,
        verbose=False
    )
    pipeline4 = TablePipeLine()
    pipeline4.pipe(ffm_reg)
    pipeline4.fit(df)
    result4 = pipeline4.transform(df.head(10))
    print(f"Transform result shape: {result4.shape}")
    print(f"Result columns: {result4.columns.tolist()}")
    print("FFMRegression: PASSED\n")
except Exception as e:
    print(f"FFMRegression: FAILED - {e}\n")

print("=" * 50)
print("5. Testing DeepFMClassification...")
print("=" * 50)
try:
    deepfm_clf = DeepFMClassification(
        y=label,
        cols=["feature1", "feature2", "feature3", "feature4"],
        field_dims=field_dims,
        embed_dim=4,
        hidden_layers=[16, 8],
        epochs=3,
        batch_size=64,
        verbose=False
    )
    pipeline5 = TablePipeLine()
    pipeline5.pipe(deepfm_clf)
    pipeline5.fit(df)
    result5 = pipeline5.transform(df.head(10))
    print(f"Transform result shape: {result5.shape}")
    print(f"Result columns: {result5.columns.tolist()}")
    print("DeepFMClassification: PASSED\n")
except Exception as e:
    print(f"DeepFMClassification: FAILED - {e}\n")

print("=" * 50)
print("6. Testing DeepFMRegression...")
print("=" * 50)
try:
    deepfm_reg = DeepFMRegression(
        y=y_reg,
        cols=["feature1", "feature2", "feature3", "feature4"],
        field_dims=field_dims,
        embed_dim=4,
        hidden_layers=[16, 8],
        epochs=3,
        batch_size=64,
        verbose=False
    )
    pipeline6 = TablePipeLine()
    pipeline6.pipe(deepfm_reg)
    pipeline6.fit(df)
    result6 = pipeline6.transform(df.head(10))
    print(f"Transform result shape: {result6.shape}")
    print(f"Result columns: {result6.columns.tolist()}")
    print("DeepFMRegression: PASSED\n")
except Exception as e:
    print(f"DeepFMRegression: FAILED - {e}\n")

print("=== All FM Module Tests Complete ===")
