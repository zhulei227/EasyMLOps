# 时序任务 (TSPipeLine)

```python
from easymlops import TSPipeLine
from easymlops.ts.dnn import TSCNNRegression

ts = TSPipeLine()

# 时序CNN回归
ts.pipe(TSCNNRegression(
    label="target",
    seq_len=30,
    features=[...],
    ...
))

# 训练
ts.fit(x_train, y=y_train)

# 预测
predictions = ts.transform(x_test)
```

## 时序数据处理

TSPipeLine 继承自 TablePipeLine，因此也可以使用表格数据的预处理方法：

```python
from easymlops.table.preprocessing import *

# 使用表格预处理
ts.pipe(FixInput())
ts.pipe(FillNa(strategy="mean"))
ts.pipe(MinMaxScaler(cols=[...]))
```

## 回调函数

```python
from easymlops.ts.core.callback import *

# 形状检查
check_shape(ts_pipe, batch_transform, single_transform)

# 列检查
check_columns(ts_pipe, batch_transform, single_transform)

# 数据类型检查
check_data_type(ts_pipe, batch_transform, single_transform)

# 数据一致性检查
check_data_same(ts_pipe, batch_transform, single_transform)
```
