# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, r"d:/datas/share/traes/easymlops-trae")
import numpy as np
import pandas as pd
import easymlops

rng = np.random.RandomState(7)
n = 600
df = pd.DataFrame({
    "city": rng.choice(["bj", "sh", "sz"], n),
    "product": rng.choice(["a", "b", "c", "d"], n),
    "income": np.abs(rng.normal(8000, 2000, n)) + 1,
    "value": rng.normal(100, 20, n),
})
train, test = df[:480].copy(), df[480:].copy()

from easymlops.table.features import GroupAggregateFeatures
from easymlops.table.encoding import CategoryCross, CountEncoding

p = easymlops.TablePipeLine()
p.pipe(GroupAggregateFeatures(group_cols=["city"], stats=["mean", "count"])) \
 .pipe(CategoryCross(cols=[["city", "product"]])) \
 .pipe(CountEncoding(cols=["city"], normalize=True))
p.fit(train)
out = p.transform(test.copy())
print("final cols:", list(out.columns))
g = p.models[0]
print("gaf input_col_names:", g.input_col_names)
print("gaf output_col_names:", g.output_col_names)
