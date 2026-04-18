# 完整示例

## 示例1：表格分类任务 (Titanic)

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *

# 加载数据
data = pd.read_csv("titanic.csv")
x_train = data[:800]
x_test = data[800:]
y_train = x_train["Survived"]
y_test = x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]

# 构建Pipeline
table = TablePipeLine()
table.pipe(FixInput())\
     .pipe(TransToCategory(cols=["Sex", "Embarked", "Cabin"]))\
     .pipe(FillNa(strategy="mean"))\
     .pipe(LabelEncoding())\
     .pipe(LGBMClassification(label="Survived", n_estimators=100))

# 训练
table.fit(x_train, y=y_train)

# 预测
predictions = table.transform(x_test)

# 评估
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

# 保存
table.save("model.pkl")
```

---

## 示例2：文本分类任务

```python
import pandas as pd
from easymlops import NLPPipeline
from easymlops.nlp.preprocessing import *
from easymlops.nlp.representation import *
from easymlops.nlp.text_classification import *

# 加载数据
data = pd.read_csv("sentiment.csv")
x_train = data[:6000]
x_test = data[6000:]
y_train = x_train["label"]
y_test = x_test["label"]
del x_train["label"]
del x_test["label"]

# 构建Pipeline
nlp = NLPPipeline()
nlp.pipe(Lower())\
   .pipe(RemovePunctuation())\
   .pipe(ExtractJieBaWords())\
   .pipe(RemoveStopWords())\
   .pipe(TFIDF(max_features=5000))\
   .pipe(TextCNNClassification(
       label="label",
       vocab_size=5000,
       embedding_dim=128,
       num_filters=100,
       kernel_sizes=[2, 3, 4],
       epochs=10
   ))

# 训练
nlp.fit(x_train, y=y_train)

# 预测
predictions = nlp.transform(x_test)

# 评估
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))

# 保存
nlp.save("nlp_model.pkl")
```

---

## 示例3：回归任务

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.regression import *
from sklearn.metrics import mean_squared_error
import numpy as np

# 加载数据
data = pd.read_csv("house_prices.csv")
x_train = data[:1000]
x_test = data[1000:]
y_train = x_train["SalePrice"]
y_test = x_test["SalePrice"]
del x_train["SalePrice"]
del x_test["SalePrice"]

# 构建Pipeline
table = TablePipeLine()
table.pipe(FixInput())\
     .pipe(FillNa(strategy="mean"))\
     .pipe(LabelEncoding())\
     .pipe(MinMaxScaler())\
     .pipe(LGBMRegression(
         label="SalePrice",
         n_estimators=200,
         learning_rate=0.05
     ))

# 训练
table.fit(x_train, y=y_train)

# 预测
predictions = table.transform(x_test)

# 评估
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse}")

# 保存
table.save("regression_model.pkl")
```

---

## 示例4：特征工程完整流程

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.decomposition import *
from easymlops.table.feature_selection import *

# 加载数据
data = pd.read_csv("data.csv")

# 构建完整特征工程Pipeline
table = TablePipeLine()

# 1. 数据清洗
table.pipe(FixInput())\
     .pipe(FillNa(strategy="mean"))\
     .pipe(TransToCategory(cols=["cat1", "cat2"]))

# 2. 特征编码
table.pipe(LabelEncoding())

# 3. 特征选择 - 过滤式
table.pipe(MissRateFilter(threshold=0.5))\
     .pipe(VarianceFilter(threshold=0.0))\
     .pipe(PersonCorrFilter(threshold=0.8))

# 4. 特征降维
table.pipe(PCADecomposition(n_components=20))

# 训练
table.fit(x_train)

# 转换
x_train_processed = table.transform(x_train)
x_test_processed = table.transform(x_test)

print(f"原始特征数: {x_train.shape[1]}")
print(f"处理后特征数: {x_train_processed.shape[1]}")
```

---

## 示例5：Stacking 集成

```python
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *
from easymlops.table.strategy import HillClimbingStackingRegression
from easymlops.table.regression import LGBMRegression

# 加载数据
data = pd.read_csv("data.csv")
x_train = data[:800]
x_test = data[800:]
y_train = x_train["label"]
y_test = x_test["label"]

# 定义基模型
base_models = [
    ("lgbm", LGBMClassification(label="label", n_estimators=50)),
    ("rf", RandomForestClassification(label="label", n_estimators=50)),
    ("dt", DecisionTreeClassification(label="label"))
]

# 定义元模型
meta_model = LGBMRegression(label="label", n_estimators=20)

# 构建Stacking Pipeline
table = TablePipeLine()
table.pipe(FixInput())\
     .pipe(FillNa(strategy="mean"))\
     .pipe(LabelEncoding())\
     .pipe(HillClimbingStackingRegression(
         label="label",
         base_models=base_models,
         meta_model=meta_model,
         n_folds=5
     ))

# 训练
table.fit(x_train, y=y_train)

# 预测
predictions = table.transform(x_test)

# 评估
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, predictions))
```

---

## 示例6：生产部署

```python
# 保存为production.py

from flask import Flask, request, jsonify
import pickle
import pandas as pd
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 加载模型
with open("models/table_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        logger.info(f"Received request: {data}")
        
        df = pd.DataFrame([data])
        result = pipeline.transform(df)
        
        logger.info(f"Prediction result: {result.tolist()}")
        return jsonify({
            "status": "success",
            "prediction": result.tolist()
        })
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
```

启动服务：
```bash
python production.py
```

测试：
```bash
curl -X POST http://localhost:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"feature1": 1.0, "feature2": 2.0, "feature3": "A"}'
```

---

## 示例7：使用 Faiss 进行相似文本检索

```python
import pandas as pd
from easymlops import NLPPipeline
from easymlops.nlp.preprocessing import *
from easymlops.nlp.representation import *
from easymlops.nlp.similarity import FaissSimilarity

# 准备数据
texts = pd.DataFrame({
    "text": [
        "机器学习是人工智能的一个分支",
        "深度学习是机器学习的子领域",
        "自然语言处理用于处理文本数据",
        "计算机视觉用于处理图像",
        "推荐系统用于个性化推荐"
    ]
})

# 构建Pipeline
nlp = NLPPipeline()
nlp.pipe(Lower())\
   .pipe(ExtractJieBaWords())\
   .pipe(TFIDF(max_features=100))\
   .pipe(FaissSimilarity(index_type="IVF", metric="L2"))

# 训练 (构建索引)
nlp.fit(texts)

# 查询
query = "人工智能和机器学习"
results = nlp.transform(pd.DataFrame({"text": [query]}))

print("查询:", query)
print("相似文本:", results)
```

---

## 示例8：使用 AutoML

```python
from easymlops.automl import AutoMLTab

# 创建AutoML实例
automl = AutoMLTab(
    trn_path="train.csv",
    val_path="val.csv",
    label="target",
    task_type="regression",
    eval_metric="mae",
    eval_metric_direct="minimize",
    max_iter=3,
    max_time=3600,
    early_stop_steps=5
)

# 运行AutoML
automl.run()

# 获取最佳pipeline
best_pipeline = automl.get_best_pipeline()

# 预测
predictions = automl.predict(x_test)

# 保存AutoML结果
automl.save("automl_result.pkl")
```
