# -*- coding: utf-8 -*-
"""示例1b：自助特征工程 · 降维/SQL/Pipeline操作/质量检查/性能优化（easymlops 0.2.4 实测跑通）

覆盖 4 大板块：
  5. 特征降维（PCA）
  6. SQL 表达式派生列
  7. Pipeline 操作与质量检查（减法运算符/切片/Parallel/callback/单条转换）
  8. 性能优化（内存压缩/稠密转稀疏）
  9. 全流程串联（清洗 -> 派生 -> 编码 -> 选择）

运行: python scripts/example_feature_engineering_advanced.py
产物: scripts/output/fe_sample_full.csv（全流程结果）
"""
import os
import numpy as np
import pandas as pd

from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import OneHotEncoding, LabelEncoding
from easymlops.table.feature_selection import MissRateFilter, VarianceFilter
from easymlops.table.decomposition import PCADecomposition
from easymlops.table.perfopt import ReduceMemUsage, Dense2Sparse
from easymlops.table.sqls import SQL
from easymlops.table.ensemble import Parallel
from easymlops.table.callback import check_transform_function_pipeline

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# ---------------- 样例数据（与示例1a一致的确定性生成） ----------------
np.random.seed(42)
n = 600
df = pd.DataFrame({
    "user_id": np.arange(n),
    "age": np.random.randint(18, 70, n).astype(float),
    "income": np.random.rand(n) * 50000,
    "score": np.random.rand(n),
    "city": np.random.choice(["BJ", "SH", "GZ", "SZ"], n),
    "gender": np.random.choice(["M", "F"], n),
    "reg_date": pd.date_range("2024-01-01", periods=n).astype(str),
    "label": np.random.randint(0, 2, n),
})
df.loc[df.sample(frac=0.1, random_state=0).index, "age"] = np.nan
df["useless_const"] = 1.0
df.loc[df.sample(frac=0.6, random_state=1).index, "income"] = np.nan
df["mostly_null"] = np.nan
y = df["label"]

num_cols = ["age", "income", "score"]
num_df = df[num_cols].copy()
num_df["income"] = num_df["income"].fillna(0)
num_df["age"] = num_df["age"].fillna(num_df["age"].mean())

# ---------------- Demo 5: 特征降维（PCA） ----------------
pca_pipe = TablePipeLine()
pca_pipe.pipe(FixInput(reduce_mem_usage=False)) \
        .pipe(FillNa(fill_mode="mean")) \
        .pipe(MinMaxScaler()) \
        .pipe(PCADecomposition(n_components=2))
pca_pipe.fit(num_df)
d5 = pca_pipe.transform(num_df)
print("[5 降维] %s -> %s（PCA 2 主成分）" % (num_cols, list(d5.columns)))

# ---------------- Demo 6: SQL 表达式派生列 ----------------
sql_pipe = TablePipeLine()
sql_pipe.pipe(FixInput(reduce_mem_usage=False)) \
        .pipe(SQL(sql="(age*score)/2 as age_score_half, income/1000 as income_k"))
sql_pipe.fit(df)
d6 = sql_pipe.transform(df)
print("[6 SQL] 新增列: %s" % [c for c in d6.columns if c in ("age_score_half", "income_k")])

# ---------------- Demo 7: Pipeline 操作与质量检查 ----------------
# 注意: FillNa(fill_mode=...) 必须限定数值列（详见 SKILL.md 陷阱1），此处基于纯数值子表操作
p = TablePipeLine().pipe(FixInput(reduce_mem_usage=False)) \
                   .pipe(FillNa(fill_mode="mean")) \
                   .pipe(MinMaxScaler()) \
                   .pipe(Tanh(cols=["income"]))
p.fit(num_df)
first_pipe = p[0]     # 切片: 按索引取 pipe
print("[7 操作] p[0] 类型: %s" % type(first_pipe).__name__)

# 单条转换（部署常用）；record 字段需与 fit 时的输入列一致
print("[7 单条] transform_single 返回键数: %d" % len(p.transform_single(num_df.to_dict("records")[0])))

# 减法: 从 pipeline 中移除某个 pipe
# 注意: 0.2.4 实测减法会破坏原 pipeline 内部状态（input_col_names 被清空），
# 且减法产物 transform 也有缺陷，故放在本板块最后，仅演示构造
minus = p - Tanh(cols=["income"])
print("[7 操作] 减法移除 Tanh 返回: %s" % type(minus).__name__)

# 质量检查: 逐 pipe 校验 transform 一致性（需数值列，异常会抛错）
num_pipe = TablePipeLine().pipe(FixInput()) \
                           .pipe(FillNa(fill_mode="mean")) \
                           .pipe(MinMaxScaler()) \
                           .fit(num_df)
check_transform_function_pipeline(num_pipe, num_df, sample=50)
print("[7 检查] check_transform_function_pipeline 通过")

# ---------------- Demo 8: 性能优化 ----------------
mem_pipe = TablePipeLine()
mem_pipe.pipe(FixInput(reduce_mem_usage=False)) \
        .pipe(ReduceMemUsage()) \
        .pipe(Dense2Sparse())
mem_pipe.fit(num_df)
d8 = mem_pipe.transform(num_df)
print("[8 优化] 内存压缩后 dtype: %s" % sorted(set(str(t) for t in d8.dtypes)))

# ---------------- Demo 9: 全流程串联 ----------------
full = TablePipeLine()
full.pipe(FixInput(reduce_mem_usage=False)) \
    .pipe(FillNa(cols=["age", "income"], fill_mode="mean")) \
    .pipe(DropCols(cols=["useless_const", "mostly_null", "user_id"])) \
    .pipe(Add(left_col_name="age", right_col_name="score", output_col_name="age_score")) \
    .pipe(DateMonthInfo(cols=["reg_date"], prefix="reg_month")) \
    .pipe(CrossCategoryWithNumber(left_col_name="city", right_col_name="income",
                                  operate_name="city_income")) \
    .pipe(Parallel([
        OneHotEncoding(cols=["city"]),
        LabelEncoding(cols=["gender"]),
    ])) \
    .pipe(MissRateFilter(cols=["age", "income", "score", "age_score"], max_threshold=0.7)) \
    .pipe(VarianceFilter(cols=["age", "income", "score", "age_score"], min_threshold=0.001)) \
    .pipe(ReduceMemUsage())
full.fit(df)
final = full.transform(df)
final.to_csv(os.path.join(OUT, "fe_sample_full.csv"), index=False)
# 注: 复杂管道（含交叉特征等）在 0.2.4 的 transform_single 可能触发
# input_col_names=None 缺陷，单条推理建议用单行 DataFrame + transform
one = full.transform(df.head(1))
print("[9 全流程] %d 列 -> %d 列 | 单行推理列数: %d"
      % (df.shape[1], final.shape[1], one.shape[1]))
print("FEATURE ENGINEERING (1b) PASSED")
