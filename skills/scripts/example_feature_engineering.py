# -*- coding: utf-8 -*-
"""示例1a：自助特征工程 · 清洗/派生/编码/选择（easymlops 0.2.4 实测跑通）

覆盖 4 大板块：
  1. 数据清洗与类型规范化
  2. 特征派生（二元运算/多列汇总/日期特征/交叉特征/分箱/裁剪/激活）
  3. 特征编码（Label/OneHot/Target/WOE/LGBM叶子编码）
  4. 特征选择（过滤式 + 嵌入式）

运行: python scripts/example_feature_engineering.py
产物: scripts/output/fe_sample_raw.csv（原始样例，同步于 assets/data）
"""
import os
import numpy as np
import pandas as pd

from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import (OneHotEncoding, LabelEncoding, TargetEncoding,
                                      WOEEncoding, LGBMLeafEncoder)
from easymlops.table.feature_selection import (MissRateFilter, VarianceFilter, PersonCorrFilter,
                                               Chi2Filter, IVFilter, LREmbed, LGBMEmbed)
from easymlops.table.ensemble import Parallel

OUT = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT, exist_ok=True)

# ---------------- 样例数据（确定性生成） ----------------
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
df.loc[df.sample(frac=0.1, random_state=0).index, "age"] = np.nan     # 缺失值
df["useless_const"] = 1.0                                             # 零方差列
df.loc[df.sample(frac=0.6, random_state=1).index, "income"] = np.nan  # 高缺失列(60%)
df["mostly_null"] = np.nan                                            # 全缺失列
df.to_csv(os.path.join(OUT, "fe_sample_raw.csv"), index=False)

y = df["label"]
print("=" * 60)
print("样例数据: %d 行 x %d 列 | 缺失: age(10%%), income(60%%), mostly_null(100%%)"
      % (df.shape[0], df.shape[1]))

# ---------------- Demo 1: 数据清洗与类型规范化 ----------------
clean = TablePipeLine()
clean.pipe(FixInput(reduce_mem_usage=False)) \
     .pipe(TransToFloat(cols=["age", "income", "score"])) \
     .pipe(TransToCategory(cols=["city", "gender"])) \
     .pipe(TransToInt(cols=["user_id"])) \
     .pipe(FillNa(cols=["age"], fill_mode="mean")) \
     .pipe(FillNa(cols=["income", "mostly_null"],
                  fill_detail={"income": 0, "mostly_null": 0})) \
     .pipe(DropCols(cols=["useless_const", "mostly_null"])) \
     .pipe(TransToLower(cols=["city"]))
clean.fit(df)
d1 = clean.transform(df)
print("-" * 60)
print("[1 清洗] 列: %s" % list(d1.columns))
print("[1 清洗] age缺失已填充: %s | city已转小写: %s"
      % (int(d1["age"].isna().sum()) == 0, set(d1["city"]) <= {"bj", "sh", "gz", "sz"}))

# ---------------- Demo 2: 特征派生 ----------------
derived = TablePipeLine()
derived.pipe(FixInput(reduce_mem_usage=False)) \
       .pipe(FillNa(cols=["age", "income"], fill_mode="mean")) \
       .pipe(Add(left_col_name="age", right_col_name="score", output_col_name="age_plus_score")) \
       .pipe(Divide(left_col_name="income", right_col_name="age", output_col_name="income_per_age")) \
       .pipe(Sum(cols=["age", "score"], output_col_name="sum_age_score", drop_input=False)) \
       .pipe(Mean(cols=["age", "income", "score"], output_col_name="mean_3", drop_input=False)) \
       .pipe(DateMonthInfo(cols=["reg_date"], prefix="reg_month")) \
       .pipe(DateHourInfo(cols=["reg_date"], prefix="reg_hour")) \
       .pipe(CrossCategoryWithNumber(left_col_name="city", right_col_name="income",
                                     operate_name="city_income")) \
       .pipe(CrossNumberWithNumber(left_col_name="age", right_col_name="score",
                                   operate_name="age_score")) \
       .pipe(Bins(cols=["age", "income", "score"], n_bins=5)) \
       .pipe(Clip(clip_detail={"score": (0.0, 1.0)})) \
       .pipe(Tanh(cols=["income"], prefix="tanh_income")) \
       .pipe(IsNull(cols=["reg_date"], prefix="isnull_regdate"))
derived.fit(df)
d2 = derived.transform(df)
print("-" * 60)
print("[2 派生] 输出列数: %d" % d2.shape[1])
print("[2 派生] 二元/汇总列: %s" % [c for c in d2.columns
                                    if c in ("age_plus_score", "income_per_age",
                                             "sum_age_score", "mean_3")])
print("[2 派生] 日期列: %s | 交叉列: %s"
      % (d2.filter(like="reg_").columns.tolist()[:4],
         d2.filter(like="city_income").columns.tolist()[:3]))
print("[2 派生] 分箱唯一值数(age): %d" % d2["age"].nunique())

# ---------------- Demo 3: 特征编码 ----------------
# 注意: LGBMLeafEncoder 会将输出整体替换为叶子列，故单独成管道演示
encoded = TablePipeLine()
encoded.pipe(FixInput(reduce_mem_usage=False)) \
       .pipe(FillNa(cols=["age", "income"], fill_mode="mean")) \
       .pipe(Parallel([
           OneHotEncoding(cols=["city"]),
           LabelEncoding(cols=["gender"]),   # 原地替换
       ])) \
       .pipe(TargetEncoding(cols=["city"], y=y)) \
       .pipe(WOEEncoding(cols=["gender"], y=y))
encoded.fit(df)
d3 = encoded.transform(df)
print("-" * 60)
print("[3 编码] 输出列数: %d" % d3.shape[1])
print("[3 编码] OneHot列: %s" % [c for c in d3.columns if c.startswith("city_")])

leaf_pipe = TablePipeLine()
leaf_pipe.pipe(FixInput(reduce_mem_usage=False)) \
         .pipe(FillNa(cols=["age", "income"], fill_mode="mean")) \
         .pipe(LGBMLeafEncoder(y=y, n_estimators=5, max_depth=3, cols=["age", "income"]))
leaf_pipe.fit(df)
d3b = leaf_pipe.transform(df)
print("[3 编码] LGBM叶子编码列数: %d（输出被整体替换为叶子列）" % d3b.shape[1])

# ---------------- Demo 4: 特征选择 ----------------
num_cols = ["age", "income", "score"]
num_df = df[num_cols].copy()
num_df["income"] = num_df["income"].fillna(0)
num_df["age"] = num_df["age"].fillna(num_df["age"].mean())

selector = TablePipeLine()
selector.pipe(FixInput(reduce_mem_usage=False)) \
        .pipe(MissRateFilter(max_threshold=0.7)) \
        .pipe(VarianceFilter(min_threshold=0.001)) \
        .pipe(PersonCorrFilter(y=y, max_threshold=0.95)) \
        .pipe(MinMaxScaler()) \
        .pipe(Chi2Filter(y=y)) \
        .pipe(IVFilter(y=y, min_threshold=0.0001)) \
        .pipe(LREmbed(y=y)) \
        .pipe(LGBMEmbed(y=y))
selector.fit(num_df)
d4 = selector.transform(num_df)
print("-" * 60)
print("[4 选择] %s -> %s" % (num_cols, list(d4.columns)))
print("FEATURE ENGINEERING (1a) PASSED")
