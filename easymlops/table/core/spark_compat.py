# -*- coding: utf-8 -*-
"""
transform_spark 核心层（v0.2.8+ 新增）。

设计:
    - build_pipe_spec(pipe): 在 driver 侧把 pipe(pipeline 或单 pipe)序列化为
      可广播的字节说明（含类路径 + 参数快照, 模型类含拟合结果）
    - apply_pipe_spec_to_pandas(spec, input_dtypes, pdf): executor/partition 侧
      按哈希惰性单例恢复 pipe 并做行级 transform——**不依赖 pyspark**, 可独立单测
    - infer_output_schema(result_pdf): 用 driver 侧样本变换结果推导 Spark schema

一致性保证: table pipe 的 transform 均为行级确定变换(逐行结果与批量/单条一致),
mapInPandas 分区计算与本地 transform 结果等价(仅全局行序不保证, 属 Spark 语义)。

限制:
    - 仅行级变换 pipe 支持(默认 distributed_safe=True); TS 窗口类 pipe 显式拒绝
    - 含用户 lambda 的参数(如 PredictionPostProcess(mapping=lambda...))不可 pickle,
      分布式场景需传模块级函数
"""
import hashlib
import importlib
import pickle

import numpy as np
import pandas as pd

__all__ = ["build_pipe_spec", "apply_pipe_spec_to_pandas", "infer_output_schema",
           "clear_executor_cache"]

_executor_cache = {}   # spec 哈希 -> 已恢复 pipe（executor 生命周期内复用）



def build_pipe_spec(pipe) -> bytes:
    """
    driver 侧: 把 pipe 序列化为可广播字节。

    Args:
        pipe: 已 fit 的 TablePipeLine 或任意 TablePipeObjectBase

    Returns:
        bytes: {"kind": "pipeline"/"pipe", "class_path"/"state"} 的 pickle
    """
    from easymlops.table.core.pipeline import TablePipeLine
    if isinstance(pipe, TablePipeLine):
        structs = [pipe.extract_structs(m) for m in pipe.models]
        params = [m.get_params() for m in pipe.models]
        state = pickle.dumps({"structs": structs, "params": params})
        return pickle.dumps({"kind": "pipeline", "state": state})
    class_path = "{}.{}".format(pipe.__class__.__module__, pipe.__class__.__name__)
    state = pickle.dumps(pipe.get_params())
    return pickle.dumps({"kind": "pipe", "class_path": class_path, "state": state})


def _load_pipe_from_spec(spec_bytes: bytes):
    """由 spec 字节恢复已 fit 的 pipe。"""
    spec = pickle.loads(spec_bytes)
    if spec["kind"] == "pipeline":
        from easymlops.table.core.pipeline import TablePipeLine
        p = TablePipeLine()
        sp = pickle.loads(spec["state"])
        for struct in sp["structs"]:
            p.pipe(p.build_structs(struct))
        for i, param in enumerate(sp["params"]):
            p.models[i].set_params(param)
        return p
    mod_name, cls_name = spec["class_path"].rsplit(".", 1)
    cls = getattr(importlib.import_module(mod_name), cls_name)
    instance = cls()
    instance.set_params(pickle.loads(spec["state"]))
    return instance


def _get_cached_pipe(spec_bytes: bytes):
    """executor 侧惰性单例: 同一份 spec 在进程内只恢复一次。"""
    key = hashlib.md5(spec_bytes).hexdigest()
    if key not in _executor_cache:
        _executor_cache[key] = _load_pipe_from_spec(spec_bytes)
    return _executor_cache[key]


def clear_executor_cache():
    """清空 executor 侧 pipe 缓存（测试用）。"""
    _executor_cache.clear()



def cast_input_columns(pdf: pd.DataFrame, input_dtypes: dict) -> pd.DataFrame:
    """
    按 driver 侧记录的 dtype 对齐 Arrow 转换造成的类型漂移:
        Spark INT -> pandas nullable Int64 / object; DATE -> datetime64; DECIMAL -> object
    只 cast fit 时用到的列（input_col_names），多余列不动。
    """
    for col, dtype in input_dtypes.items():
        if col not in pdf.columns:
            continue
        try:
            np_dtype = np.dtype(dtype)
        except TypeError:
            continue
        if np.issubdtype(np_dtype, np.floating):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype("float64")
        elif np.issubdtype(np_dtype, np.integer) or np_dtype == np.bool_:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").fillna(0).astype("int64")
        elif np_dtype == object:
            pdf[col] = pdf[col].astype(str)
        # datetime64 保持原样（Spark DATE/TIMESTAMP -> datetime64 天然对齐）
    return pdf


def apply_pipe_spec_to_pandas(spec_bytes: bytes, input_dtypes: dict,
                              pdf: pd.DataFrame) -> pd.DataFrame:
    """
    executor/partition 侧入口：恢复 pipe（惰性单例）并对分区 pandas 批做行级 transform。
    该函数不依赖 pyspark，可独立单测（分块等价性）。
    """
    pipe = _get_cached_pipe(spec_bytes)
    pdf = cast_input_columns(pdf.copy(), input_dtypes)
    return pipe.transform(pdf)


def infer_output_schema(result_pdf: pd.DataFrame):
    """
    由 driver 侧样本变换结果推导 Spark StructType（惰性导入 pyspark）。
    """
    from pyspark.sql.types import (StructType, StructField, DoubleType, LongType,
                                   StringType, BooleanType, FloatType, IntegerType)
    mapping = {"float64": DoubleType(), "float32": FloatType(), "int64": LongType(),
               "int32": IntegerType(), "bool": BooleanType(), "object": StringType(),
               "string": StringType()}
    fields = []
    for col in result_pdf.columns:
        t = str(result_pdf[col].dtype)
        fields.append(StructField(col, mapping.get(t, StringType()), True))
    return StructType(fields)


def transform_spark_dataframe(pipe, spark_df):
    """
    transform_spark 统一入口（driver 侧编排，供 TablePipeLine/TablePipeObjectBase 委托）。

    Args:
        pipe: 已 fit 的 TablePipeLine 或单 pipe（须满足 distributed_safe=True）
        spark_df: pyspark DataFrame

    Returns:
        pyspark DataFrame: 变换结果（列结构 = 本地 transform 的输出结构）
    """
    if not getattr(pipe, "distributed_safe", True):
        raise Exception(
            "{} does not support distributed transform (row-dependent windows); "
            "use local transform instead".format(pipe.__class__.__name__))
    try:
        import pyspark  # noqa: F401
    except ImportError:
        raise ImportError("transform_spark requires pyspark: pip install pyspark")

    spec = build_pipe_spec(pipe)
    # driver 侧取样本: 记录输入 dtype(类型对齐) + 推导输出 schema
    sample = spark_df.limit(2).toPandas()
    if len(sample) == 0:
        raise Exception("spark_df is empty, cannot infer schema")
    input_dtypes = {c: str(sample[c].dtype) for c in sample.columns}
    result_sample = _load_pipe_from_spec(spec).transform(sample.copy())
    schema = infer_output_schema(result_sample)

    def runner(iterator):
        for pdf in iterator:
            yield apply_pipe_spec_to_pandas(spec, input_dtypes, pdf)

    return spark_df.mapInPandas(runner, schema)



