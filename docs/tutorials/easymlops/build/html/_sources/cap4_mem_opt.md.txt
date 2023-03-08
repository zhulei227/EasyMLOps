# 内存优化 

1）pandas再读取数据后，通常会将数据设置为64位(float64,int64)等，而我们的实际数据通常不需要这么大的存储访问，所以第一个思路就是缩小数据类型；
2）对于ont-hot以及bow/tfidf这类特征工程之后得到的是稀疏矩阵，所以第二个思路就是将稠密矩阵转稀疏矩阵


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
    

## ReduceMemUsage


注意：ReduceMemUsage会更加训练数据设置最小的数据类型，对于预测数据，如果不在该范围内，会进行截断


```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.perfopt import *
```


```python
table=TablePipeLine()
table.pipe(FixInput(reduce_mem_usage=False))\
  .pipe(FillNa())\
  .pipe(ReduceMemUsage())

table.fit(data)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x1940db09cc8>




```python
#优化前(K)
table.transform(data,run_to_layer=-2).memory_usage().sum()//1024
```




    83




```python
#优化后(K)
table.transform(data,run_to_layer=-1).memory_usage().sum()//1024
```




    43



## Dense2Sparse  

注意：该模块的潜在问题是，后续pipe模块需要提供对稀疏矩阵的支持（不过，目前内置的pipe模块基本都支持~）


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(OneHotEncoding(cols=["Pclass","Sex","Name","Ticket","Embarked","Cabin"],drop_col=True))\
  .pipe(Dense2Sparse())

table.fit(data)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x1940db2fec8>




```python
#优化前(K)
table.transform(data,run_to_layer=-2).memory_usage().sum()//1024
```




    1512




```python
#优化后(K)
table.transform(data,run_to_layer=-1).memory_usage().sum()//1024
```




    45



基于Sklearn实现的BOW/TFIDF模型，输出已经是稀疏矩阵了，不过Dense2Sparse还能进一步减少内存，原理是内部会调用一次ReduceMemUsage，减小数据类型


```python
from easymlops import NLPPipeline
from easymlops.nlp.representation import *
from easymlops.nlp.preprocessing import *
```


```python
text=pd.read_csv("./data/demo2.csv",encoding="gbk").sample(frac=1)[["review"]]
```


```python
nlp=NLPPipeline()
nlp.pipe(ExtractChineseWords())\
   .pipe(ExtractJieBaWords())\
   .pipe(BagOfWords())\
   .pipe(Dense2Sparse())

nlp.fit(text)
```




    <easymlops.nlp.core.pipeline_object.NLPPipeline at 0x1caf144b3c8>




```python
#优化前(K)
nlp.transform(text,run_to_layer=-2).memory_usage().sum()//1024
```




    4568




```python
#优化后(K)
nlp.transform(text,run_to_layer=-1).memory_usage().sum()//1024
```




    1939




```python

```
