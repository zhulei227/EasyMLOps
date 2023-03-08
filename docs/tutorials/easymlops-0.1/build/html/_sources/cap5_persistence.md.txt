# 持久化
这里持久化操作不保存结构，只保存参数


```python
#数据准备
import os
os.chdir("../../")#与easymlops同级目录
import pandas as pd
data=pd.read_csv("./data/demo.csv")
print(data.head(5).to_markdown())
```

    |    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
    |---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
    |  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
    |  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
    |  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
    |  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          |
    |  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          |
    


```python
x_train=data[:500]
x_test=data[500:]
y_train=x_train["Survived"]
y_test=x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]
```


```python
from easymlops import NLPPipeline
from easymlops.table.preprocessing import *
from easymlops.table.ensemble import *
from easymlops.nlp.preprocessing import *
from easymlops.nlp.representation import *
from easymlops.table.perfopt import *
from easymlops.table.classification import *
```


```python
nlp=NLPPipeline()
nlp.pipe(FixInput())\
   .pipe(TablePipeLine().pipe(FillNa(cols=["Name","Sex"]))
                        .pipe(SelectCols(cols=["Name","Sex"]))
                        .pipe(Lower())
                        .pipe(RemovePunctuation()))\
   .pipe(Parallel([LsiTopicModel(num_topics=4),Word2VecModel(embedding_size=4),TFIDF()]))\
   .pipe(DropCols(cols=["Name","Sex"]))\
   .pipe(LGBMClassification(y=y_train,support_sparse_input=True,native_init_params={"max_depth": 2}, native_fit_params={"num_boost_round": 128}))

x_test_new=nlp.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |         0 |        1 |
    |----:|----------:|---------:|
    | 500 | 0.782645  | 0.217355 |
    | 501 | 0.0778462 | 0.922154 |
    | 502 | 0.0778462 | 0.922154 |
    | 503 | 0.155608  | 0.844392 |
    | 504 | 0.254     | 0.746    |
    

## 保存模型
保存操作很简单，直接save


```python
nlp.save("nlp.pkl")
```

## 加载模型  

由于只保存了模型参数，所以需要将训练阶段的结构再次申明一次（**结构必须完全一致**，比如上面pipeline嵌套了pipeline的情况，里面的pipeline也不能展开），另外，**参数可以不需设置**


```python
nlp=NLPPipeline()
nlp.pipe(FixInput())\
   .pipe(TablePipeLine().pipe(FillNa())
                        .pipe(SelectCols())
                        .pipe(Lower())
                        .pipe(RemovePunctuation()))\
   .pipe(Parallel([LsiTopicModel(),Word2VecModel(),TFIDF()]))\
   .pipe(DropCols())\
   .pipe(LGBMClassification())

nlp.load("nlp.pkl")
```


```python
print(nlp.transform(x_test).head(5).to_markdown())
```

    |     |         0 |        1 |
    |----:|----------:|---------:|
    | 500 | 0.782645  | 0.217355 |
    | 501 | 0.0778462 | 0.922154 |
    | 502 | 0.0778462 | 0.922154 |
    | 503 | 0.155608  | 0.844392 |
    | 504 | 0.254     | 0.746    |
    


```python

```
