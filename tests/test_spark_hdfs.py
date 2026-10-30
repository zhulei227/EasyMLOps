# -*- coding: utf-8 -*-
# Spark/HDFS 增强验收: 存储层/分块等价性/类型cast/契约守卫/端到端(Spark local, 可选) (v0.2.8+)
import os
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from easymlops.core.storage import save_model_bytes, load_model_bytes, is_remote_path
from easymlops.table.core.pipeline import TablePipeLine
from easymlops.table.core.spark_compat import (build_pipe_spec, apply_pipe_spec_to_pandas,
                                               _load_pipe_from_spec, clear_executor_cache)
from easymlops.table.encoding import CountEncoding, TargetEncoding
from easymlops.table.nn import TabularMLPClassification
from easymlops.table.preprocessing import FillNa

PASS, FAIL = [], []
PKL = "d:/datas/share/traes/easymlops-trae/tests/_tmp_spark.pkl"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


rng = np.random.RandomState(7)
n = 400
x_clf, y_clf = make_classification(n_samples=n, n_features=5, n_informative=3, random_state=7)
df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(5)])
df["city"] = rng.choice(["bj", "sh", "sz"], n)
df.loc[rng.choice(n, 20), "f0"] = np.nan
y_clf = pd.Series(y_clf)
train, test = df[:300].copy(), df[300:].copy()
y_train = y_clf[:300]

# ============ 1. 存储层: memory:// 远程语义 ============
print("=== 1. storage layer ===")
check("storage_is_remote", is_remote_path("webhdfs://nn:9870/a.pkl") and
      is_remote_path("hdfs://nn:9000/a.pkl") and is_remote_path("memory://a.pkl") and
      not is_remote_path(r"C:\data\a.pkl") and not is_remote_path("local/a.pkl"))
save_model_bytes(b"abc", "memory://dir/m.pkl")
check("storage_roundtrip", load_model_bytes("memory://dir/m.pkl") == b"abc")

# pipeline 走存储层持久化(memory://)
pl = TablePipeLine()
pl.pipe(FillNa(cols=["f0"], fill_detail={"f0": 0.0})).pipe(
    TargetEncoding(cols=[("city", "city")], y=y_train)).pipe(CountEncoding(cols=["city"]))
pl.fit(train)
pl.save("memory://models/pl.pkl")
pl2 = TablePipeLine()
pl2.load("memory://models/pl.pkl")
check("storage_pipeline_save_load",
      np.allclose(pl.transform(test.copy())["city"].astype(float),
                  pl2.transform(test.copy())["city"].astype(float), atol=1e-9))

# ============ 2. spec 序列化: pipeline 与单 pipe(含模型) ============
print("=== 2. pipe spec ===")
spec_pipeline = build_pipe_spec(pl)
restored = _load_pipe_from_spec(spec_pipeline)
check("spec_pipeline_roundtrip",
      np.allclose(restored.transform(test.copy())["city"].astype(float),
                  pl.transform(test.copy())["city"].astype(float), atol=1e-9))

mlp = TabularMLPClassification(y=y_train, epochs=3, hidden_dims=(16,), random_state=42)
mlp.fit(train)
spec_pipe = build_pipe_spec(mlp)
clear_executor_cache()
restored_mlp = _load_pipe_from_spec(spec_pipe)
check("spec_pipe_roundtrip_model",
      np.allclose(mlp.transform(test.copy()).values,
                  restored_mlp.transform(test.copy()).values, atol=1e-9))

# ============ 3. 分块等价性(模拟 mapInPandas 多分区) ============
print("=== 3. chunk equivalence ===")
clear_executor_cache()
whole = pl.transform(test.copy())
chunks = np.array_split(test.reset_index(drop=True), 3)
parts = [apply_pipe_spec_to_pandas(spec_pipeline,
                                   {c: str(train[c].dtype) for c in train.columns}, ch.copy())
         for ch in chunks]
merged = pd.concat(parts).reset_index(drop=True)
check("equivalence_pipeline",
      len(merged) == len(whole) and
      np.allclose(merged["city"].astype(float).to_numpy(),
                  whole["city"].astype(float).to_numpy(), atol=1e-9))
clear_executor_cache()
whole_mlp = mlp.transform(test.copy()).reset_index(drop=True)
chunks_m = np.array_split(test.reset_index(drop=True), 4)
parts_m = [apply_pipe_spec_to_pandas(spec_pipe,
                                     {c: str(train[c].dtype) for c in train.columns}, ch.copy())
           for ch in chunks_m]
merged_mlp = pd.concat(parts_m).reset_index(drop=True)
check("equivalence_model_pipe",
      np.allclose(merged_mlp.sort_index(axis=1).values,
                  whole_mlp.sort_index(axis=1).values, atol=1e-9))

# ============ 4. 类型 cast: Arrow 类型漂移模拟 ============
print("=== 4. dtype casting ===")
arrow_df = test.copy()
arrow_df["f0"] = arrow_df["f0"].astype(object)   # 模拟 Spark Arrow: 含 NaN 数值列变 object
clear_executor_cache()
casted = apply_pipe_spec_to_pandas(spec_pipeline,
                                   {c: str(train[c].dtype) for c in train.columns},
                                   arrow_df.copy())
check("cast_object_numeric", np.isfinite(casted["f0"].astype(float)).all())

# ============ 5. 契约守卫: TS pipe 拒绝分布式 ============
print("=== 5. contract guard ===")
from easymlops.ts.core.pipe import PipeBase as TSPipeBase
check("ts_guard_flag", getattr(TSPipeBase, "distributed_safe", True) is False)
check("table_default_safe", getattr(TablePipeLine, "distributed_safe", True) is True)

# ============ 6. 端到端: Spark local(可选, 需 pyspark+java) ============
print("=== 6. spark end-to-end (optional) ===")
try:
    import pyspark  # noqa: F401
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False

if HAS_PYSPARK:
    try:
        if not os.environ.get("JAVA_HOME") and os.path.isdir(r"D:\apps\java"):
            jdir = None
            for cand in os.listdir(r"D:\apps\java"):
                if cand.startswith("jdk") or cand.startswith("jre"):
                    jdir = os.path.join(r"D:\apps\java", cand)
            if jdir:
                os.environ["JAVA_HOME"] = jdir
                os.environ["PATH"] = os.path.join(jdir, "bin") + os.pathsep + os.environ.get("PATH", "")
        from pyspark.sql import SparkSession
        spark = (SparkSession.builder.master("local[2]")
                 .appName("easymlops-transform-spark-test")
                 .config("spark.ui.enabled", "false")
                 .config("spark.sql.shuffle.partitions", "2")
                 .config("spark.driver.host", "127.0.0.1")
                 .getOrCreate())
        spark_df = spark.createDataFrame(
            [(float(r.f0), float(r.f1), float(r.f2), float(r.f3), float(r.f4), str(r.city))
             for r in test.itertuples()],
            schema=["f0", "f1", "f2", "f3", "f4", "city"])
        out_sdf = pl.transform_spark(spark_df)
        out_spark = out_sdf.toPandas().sort_values("f1").reset_index(drop=True)
        whole_df = pl.transform(test.copy()).reset_index(drop=True)
        check("spark_end_to_end", len(out_spark) == len(whole_df) and
              np.allclose(out_spark["city"].astype(float).to_numpy(),
                          whole_df["city"].astype(float).to_numpy(), atol=1e-9))
        single_pipe_spark = mlp.transform_spark(spark_df)
        out_single = single_pipe_spark.toPandas().sort_values("f1").reset_index(drop=True)
        check("spark_single_pipe", np.isfinite(
            out_single[[c for c in out_single.columns]].astype(float).values).all())
        spark.stop()
    except Exception as exc:
        check("spark_end_to_end", False, "exception: {}: {}".format(type(exc).__name__, str(exc)[:120]))
else:
    print("[SKIP] pyspark not installed; core equivalence already verified without Spark")

# ============ 汇总 ============
if os.path.exists(PKL):
    os.remove(PKL)
print("")
print("================ SUMMARY ================")
print("PASS: {}, FAIL: {}".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
