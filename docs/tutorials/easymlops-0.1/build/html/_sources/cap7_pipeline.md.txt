# pipeline操作  

pipeline是承载pipe的流程工具，它具有**聚合pipe**的功能，也具有**拆散pipe**的功能，由于建模流程的灵活性，经常要在某些pipe模块中流转调试优化，所以pipeline设计的要求是：**聚**是基础，**拆**是核心，下面依次介绍相应功能  


```python
#准备数据
import os
os.chdir("../../")#与easymlops同级目录
import pandas as pd
data=pd.read_csv("./data/demo.csv")
x_train=data[:500]
x_test=data[500:]
y_train=x_train["Survived"]
y_test=x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]
```

## pipeline也是pipe  

pipeline继承了pipe的基础功能，所以这意味这pipeline也可以当着pipe用，这样pipeline就可以再嵌入pipeline



```python
from easymlops import NLPPipeline,TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.ensemble import *
from easymlops.table.encoding import *
from easymlops.nlp.preprocessing import *
from easymlops.nlp.representation import *
from easymlops.table.perfopt import *
from easymlops.table.classification import *
from easymlops.table.decomposition import *
from easymlops.table.extend import Normalization
```


```python
nlp=NLPPipeline()
nlp.pipe(FixInput())\
   .pipe(TablePipeLine().pipe(FillNa(cols=["Name","Sex"]))
                        .pipe(SelectCols(cols=["Name","Sex"]))
                        .pipe(Lower()))\
   .pipe(Parallel([LsiTopicModel(num_topics=4),
                   Word2VecModel(embedding_size=4),
                   TablePipeLine().pipe(RemovePunctuation())\
                                  .pipe(TFIDF())]))\
   .pipe(DropCols(cols=["Name","Sex"]))\
   .pipe(LGBMClassification(y=y_train,support_sparse_input=True,native_init_params={"max_depth": 2}, native_fit_params={"num_boost_round": 128}))

x_test_new=nlp.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |        0 |        1 |
    |----:|---------:|---------:|
    | 500 | 0.815788 | 0.184212 |
    | 501 | 0.236939 | 0.763061 |
    | 502 | 0.198483 | 0.801517 |
    | 503 | 0.331877 | 0.668123 |
    | 504 | 0.331877 | 0.668123 |
    

## 聚合pipe为pipeline  
上面构建pipeline的方式与我们通常建模流程相反，实际的建模时分别训练构建好各个pipe模块（还会反复在某些模块迭代优化多轮）后再进行组合，如下


```python
fillna_pipe=TablePipeLine().pipe(FixInput()).pipe(FillNa())
x_train_new=fillna_pipe.fit(x_train).transform(x_train)
print(x_train_new.head(5).to_markdown())
```

    |    |   PassengerId |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |     Fare | Cabin   | Embarked   |
    |---:|--------------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|---------:|:--------|:-----------|
    |  0 |             1 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25    | nan     | S          |
    |  1 |             2 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.3125  | C85     | C          |
    |  2 |             3 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.92578 | nan     | S          |
    |  3 |             4 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.0938  | C123    | S          |
    |  4 |             5 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.04688 | nan     | S          |
    


```python
pipe_onehot_encoding=OneHotEncoding(cols=["Pclass","Sex"],drop_col=False)
x_train_new=pipe_onehot_encoding.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |   PassengerId |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |     Fare | Cabin   | Embarked   |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|---------:|:--------|:-----------|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |             1 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25    | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    |  1 |             2 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.3125  | C85     | C          |          0 |          1 |          0 |          0 |            1 |
    |  2 |             3 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.92578 | nan     | S          |          1 |          0 |          0 |          0 |            1 |
    |  3 |             4 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.0938  | C123    | S          |          0 |          1 |          0 |          0 |            1 |
    |  4 |             5 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.04688 | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    


```python
pipe_label_encoding=LabelEncoding(cols=["Sex","Pclass"])
x_train_new=pipe_label_encoding.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |   PassengerId |   Pclass | Name                                                |   Sex |   Age |   SibSp |   Parch | Ticket           |     Fare | Cabin   | Embarked   |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|---------:|:----------------------------------------------------|------:|------:|--------:|--------:|:-----------------|---------:|:--------|:-----------|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |             1 |        1 | Braund, Mr. Owen Harris                             |     1 |    22 |       1 |       0 | A/5 21171        |  7.25    | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    |  1 |             2 |        2 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) |     2 |    38 |       1 |       0 | PC 17599         | 71.3125  | C85     | C          |          0 |          1 |          0 |          0 |            1 |
    |  2 |             3 |        1 | Heikkinen, Miss. Laina                              |     2 |    26 |       0 |       0 | STON/O2. 3101282 |  7.92578 | nan     | S          |          1 |          0 |          0 |          0 |            1 |
    |  3 |             4 |        2 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        |     2 |    35 |       1 |       0 | 113803           | 53.0938  | C123    | S          |          0 |          1 |          0 |          0 |            1 |
    |  4 |             5 |        1 | Allen, Mr. William Henry                            |     1 |    35 |       0 |       0 | 373450           |  8.04688 | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    


```python
pipe_target_encoding=TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train)
x_train_new=pipe_target_encoding.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |   PassengerId |   Pclass |   Name |   Sex |   Age |   SibSp |   Parch |   Ticket |     Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|---------:|-------:|------:|------:|--------:|--------:|---------:|---------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |             1 |        1 |      0 |     1 |    22 |       1 |       0 |      0   |  7.25    | 0.317829 |   0.334254 |          1 |          0 |          0 |          1 |            0 |
    |  1 |             2 |        2 |      1 |     2 |    38 |       1 |       0 |      1   | 71.3125  | 1        |   0.521739 |          0 |          1 |          0 |          0 |            1 |
    |  2 |             3 |        1 |      1 |     2 |    26 |       0 |       0 |      1   |  7.92578 | 0.317829 |   0.334254 |          1 |          0 |          0 |          0 |            1 |
    |  3 |             4 |        2 |      1 |     2 |    35 |       1 |       0 |      0.5 | 53.0938  | 0.5      |   0.334254 |          0 |          1 |          0 |          0 |            1 |
    |  4 |             5 |        1 |      0 |     1 |    35 |       0 |       0 |      0   |  8.04688 | 0.317829 |   0.334254 |          1 |          0 |          0 |          1 |            0 |
    


```python
pipe_normal=Normalizer()
x_train_new=pipe_normal.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |   PassengerId |   Pclass |   Name |   Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |   Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|---------:|-------:|------:|------:|--------:|--------:|---------:|-------:|--------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |         -1.73 |    -0.81 |  -0.79 | -0.77 | -0.07 |    0.37 |   -0.49 |    -0.84 |  -0.52 |   -0.29 |      -0.6  |       0.89 |      -0.55 |      -0.52 |       0.77 |        -0.77 |
    |  1 |         -1.72 |     0.43 |   1.26 |  1.3  |  0.84 |    0.37 |   -0.49 |     1.33 |   0.83 |    2.59 |       1.57 |      -1.12 |       1.82 |      -0.52 |      -1.3  |         1.3  |
    |  2 |         -1.71 |    -0.81 |   1.26 |  1.3  |  0.16 |   -0.5  |   -0.49 |     1.33 |  -0.5  |   -0.29 |      -0.6  |       0.89 |      -0.55 |      -0.52 |      -1.3  |         1.3  |
    |  3 |         -1.71 |     0.43 |   1.26 |  1.3  |  0.67 |    0.37 |   -0.49 |     0.25 |   0.45 |    0.48 |      -0.6  |      -1.12 |       1.82 |      -0.52 |      -1.3  |         1.3  |
    |  4 |         -1.7  |    -0.81 |  -0.79 | -0.77 |  0.67 |   -0.5  |   -0.49 |    -0.84 |  -0.5  |   -0.29 |      -0.6  |       0.89 |      -0.55 |      -0.52 |       0.77 |        -0.77 |
    


```python
pipe_pca=PCADecomposition(n_components=8)
x_train_new=pipe_pca.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |        0 |         1 |          2 |          3 |          4 |        5 |        6 |         7 |
    |---:|---------:|----------:|-----------:|-----------:|-----------:|---------:|---------:|----------:|
    |  0 | -2.33683 |  0.709077 |  0.0760812 |  0.0745287 |  0.0564233 | 1.24346  | 1.08436  | -0.621034 |
    |  1 |  4.13163 | -0.143481 |  1.78519   | -0.909605  |  0.0755622 | 1.83577  | 0.771872 | -0.538038 |
    |  2 |  1.64632 |  2.17324  | -1.10978   | -0.9792    | -0.447786  | 0.802416 | 1.83454  | -0.185458 |
    |  3 |  2.83715 | -0.17054  |  0.641344  |  0.0803461 | -1.11253   | 1.18825  | 1.3536   | -1.2982   |
    |  4 | -2.313   |  0.324529 |  0.0531388 | -0.487603  | -0.459747  | 1.07981  | 1.53396  | -0.207536 |
    


```python
pipeline_lr=TablePipeLine()
pipeline_lr.pipe(LogisticRegressionClassification(y=y_train))\
           .pipe(Normalization())

x_train_new=pipeline_lr.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |     0 |     1 |
    |---:|------:|------:|
    |  0 | 50.03 | 49.97 |
    |  1 | 41.8  | 58.2  |
    |  2 | 41.81 | 58.19 |
    |  3 | 41.9  | 58.1  |
    |  4 | 50.03 | 49.97 |
    

最后再新建一个pipeline取组合前面的pipe模块，**注意，由于各pipe已经fit好了，组合后的pipeline不要再次fit，只需运行transform/transform_single等其他函数就好**


```python
pipeline_combine=TablePipeLine()
pipeline_combine.pipe(fillna_pipe)\
                .pipe(pipe_onehot_encoding)\
                .pipe(pipe_label_encoding)\
                .pipe(pipe_target_encoding)\
                .pipe(pipe_normal)\
                .pipe(pipe_pca)\
                .pipe(pipeline_lr)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x14e68e704c8>




```python
print(pipeline_combine.transform(x_test).head(5).to_markdown())
```

    |     |     0 |     1 |
    |----:|------:|------:|
    | 500 | 50.02 | 49.98 |
    | 501 | 49.91 | 50.09 |
    | 502 | 49.84 | 50.16 |
    | 503 | 49.95 | 50.05 |
    | 504 | 48.38 | 51.62 |
    

## pipeline运行到指定位置

有时候需要定位数据从最初运行到指定pipe模块的输出结果，这里提供了三种方式：  
- index是int，运行到最外层指定的那一层  
- index是str, 运行到指定`name`的那个pipe
- index是list或tuple，运行到指定嵌入层的pipe


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(Parallel([OneHotEncoding(cols=["Pclass","Sex"],drop_col=False),
                 TablePipeLine().pipe(TargetEncoding(cols=["Embarked","Cabin"],y=y_train,name="target_encoding")).pipe(FillNa()),
                 WOEEncoding(cols=["Sex","Pclass"],y=y_train),
                 TablePipeLine().pipe(LabelEncoding(cols=["Name","Ticket"])).pipe(DropCols(cols=["Name","Sex","Ticket","Cabin","Emarked"]))]))\
  .pipe(TargetEncoding(cols=["Name","Cabin","Embarked","Ticket"],y=y_train))\
  .pipe(FillNa())\
  .pipe(Normalizer())\
  .pipe(PCADecomposition(n_components=8))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |         0 |          1 |          2 |          3 |         4 |        5 |         6 |          7 |
    |----:|----------:|-----------:|-----------:|-----------:|----------:|---------:|----------:|-----------:|
    | 500 | -2.31073  | -0.0907158 | -0.192874  |  0.510718  |  0.203634 | -1.86856 | -0.259263 | -0.254359  |
    | 501 |  0.178031 | -1.92063   | -0.317246  |  1.0355    | -1.90838  | -1.60718 | -1.48618  |  0.609996  |
    | 502 |  0.113005 | -2.26533   | -0.104388  |  1.14062   | -1.46752  | -1.49979 | -2.15435  | -0.0432104 |
    | 503 | -0.023234 | -1.64508   | -0.733197  | -0.0263899 | -2.07429  | -2.24935 |  0.102076 | -0.128     |
    | 504 |  2.47388  |  0.87968   |  0.0426266 | -0.233503  | -1.78844  | -1.92329 | -1.0166   | -1.35483   |
    

### index为int的情况


```python
#查看第1层的数据
print(table.transform(x_test,run_to_layer=1).head(5).to_markdown())
```

    |     |   PassengerId |   Pclass | Name                           | Sex    |   Age |   SibSp |   Parch |   Ticket |     Fare | Cabin   | Embarked   |
    |----:|--------------:|---------:|:-------------------------------|:-------|------:|--------:|--------:|---------:|---------:|:--------|:-----------|
    | 500 |           501 |        3 | Calic, Mr. Petar               | male   |    17 |       0 |       0 |   315086 |  8.66406 | nan     | S          |
    | 501 |           502 |        3 | Canavan, Miss. Mary            | female |    21 |       0 |       0 |   364846 |  7.75    | nan     | Q          |
    | 502 |           503 |        3 | O'Sullivan, Miss. Bridget Mary | female |     0 |       0 |       0 |   330909 |  7.62891 | nan     | Q          |
    | 503 |           504 |        3 | Laitinen, Miss. Kristina Sofia | female |    37 |       0 |       0 |     4135 |  9.58594 | nan     | S          |
    | 504 |           505 |        1 | Maioni, Miss. Roberta          | female |    16 |       0 |       0 |   110152 | 86.5     | B79     | S          |
    


```python
#查看第2层的数据
print(table.transform(x_test,run_to_layer=2).head(5).to_markdown())
```

    |     |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female | Name                           |      Sex |   Ticket | Cabin   |   PassengerId |   Pclass |   Age |   SibSp |   Parch |     Fare | Embarked   |
    |----:|-----------:|-----------:|-----------:|-----------:|-------------:|:-------------------------------|---------:|---------:|:--------|--------------:|---------:|------:|--------:|--------:|---------:|:-----------|
    | 500 |          1 |          0 |          0 |          1 |            0 | Calic, Mr. Petar               |  1.11138 |   315086 | nan     |           501 |        3 |    17 |       0 |       0 |  8.66406 | S          |
    | 501 |          1 |          0 |          0 |          0 |            1 | Canavan, Miss. Mary            | -1.56999 |   364846 | nan     |           502 |        3 |    21 |       0 |       0 |  7.75    | Q          |
    | 502 |          1 |          0 |          0 |          0 |            1 | O'Sullivan, Miss. Bridget Mary | -1.56999 |   330909 | nan     |           503 |        3 |     0 |       0 |       0 |  7.62891 | Q          |
    | 503 |          1 |          0 |          0 |          0 |            1 | Laitinen, Miss. Kristina Sofia | -1.56999 |     4135 | nan     |           504 |        3 |    37 |       0 |       0 |  9.58594 | S          |
    | 504 |          0 |          1 |          0 |          0 |            1 | Maioni, Miss. Roberta          | -1.56999 |   110152 | B79     |           505 |        1 |    16 |       0 |       0 | 86.5     | S          |
    


```python
#查看倒数第3层的数据
print(table.transform(x_test,run_to_layer=-3).head(5).to_markdown())
```

    |     |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |   Name |      Sex |   Ticket |    Cabin |   PassengerId |   Pclass |   Age |   SibSp |   Parch |     Fare |   Embarked |
    |----:|-----------:|-----------:|-----------:|-----------:|-------------:|-------:|---------:|---------:|---------:|--------------:|---------:|------:|--------:|--------:|---------:|-----------:|
    | 500 |          1 |          0 |          0 |          1 |            0 |      0 |  1.11138 |        0 | 0.317829 |           501 |        3 |    17 |       0 |       0 |  8.66406 |   0.334254 |
    | 501 |          1 |          0 |          0 |          0 |            1 |      0 | -1.56999 |        0 | 0.317829 |           502 |        3 |    21 |       0 |       0 |  7.75    |   0.511111 |
    | 502 |          1 |          0 |          0 |          0 |            1 |      0 | -1.56999 |        0 | 0.317829 |           503 |        3 |     0 |       0 |       0 |  7.62891 |   0.511111 |
    | 503 |          1 |          0 |          0 |          0 |            1 |      0 | -1.56999 |        0 | 0.317829 |           504 |        3 |    37 |       0 |       0 |  9.58594 |   0.334254 |
    | 504 |          0 |          1 |          0 |          0 |            1 |      0 | -1.56999 |        1 | 0        |           505 |        1 |    16 |       0 |       0 | 86.5     |   0.334254 |
    

### index为list或tuple
嵌套层的范围:将run_to_layer设置为list结构，分别指定逐层的index


```python
#查看第2层(Parallel)的第0层(OneHotEncoding)
print(table.transform(x_test,run_to_layer=[2,0]).head(5).to_markdown())
```

    |     |   PassengerId |   Pclass | Name                           | Sex    |   Age |   SibSp |   Parch |   Ticket |     Fare | Cabin   | Embarked   |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |----:|--------------:|---------:|:-------------------------------|:-------|------:|--------:|--------:|---------:|---------:|:--------|:-----------|-----------:|-----------:|-----------:|-----------:|-------------:|
    | 500 |           501 |        3 | Calic, Mr. Petar               | male   |    17 |       0 |       0 |   315086 |  8.66406 | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    | 501 |           502 |        3 | Canavan, Miss. Mary            | female |    21 |       0 |       0 |   364846 |  7.75    | nan     | Q          |          1 |          0 |          0 |          0 |            1 |
    | 502 |           503 |        3 | O'Sullivan, Miss. Bridget Mary | female |     0 |       0 |       0 |   330909 |  7.62891 | nan     | Q          |          1 |          0 |          0 |          0 |            1 |
    | 503 |           504 |        3 | Laitinen, Miss. Kristina Sofia | female |    37 |       0 |       0 |     4135 |  9.58594 | nan     | S          |          1 |          0 |          0 |          0 |            1 |
    | 504 |           505 |        1 | Maioni, Miss. Roberta          | female |    16 |       0 |       0 |   110152 | 86.5     | B79     | S          |          0 |          1 |          0 |          0 |            1 |
    


```python
#运行到第2层(Parallel)的第1层(TablePipeLine)的第0层(TargetEncoding)
print(table.transform(x_test,run_to_layer=[2,1,0]).head(5).to_markdown())
```

    |     |   PassengerId |   Pclass | Name                           | Sex    |   Age |   SibSp |   Parch |   Ticket |     Fare |    Cabin |   Embarked |
    |----:|--------------:|---------:|:-------------------------------|:-------|------:|--------:|--------:|---------:|---------:|---------:|-----------:|
    | 500 |           501 |        3 | Calic, Mr. Petar               | male   |    17 |       0 |       0 |   315086 |  8.66406 | 0.317829 |   0.334254 |
    | 501 |           502 |        3 | Canavan, Miss. Mary            | female |    21 |       0 |       0 |   364846 |  7.75    | 0.317829 |   0.511111 |
    | 502 |           503 |        3 | O'Sullivan, Miss. Bridget Mary | female |     0 |       0 |       0 |   330909 |  7.62891 | 0.317829 |   0.511111 |
    | 503 |           504 |        3 | Laitinen, Miss. Kristina Sofia | female |    37 |       0 |       0 |     4135 |  9.58594 | 0.317829 |   0.334254 |
    | 504 |           505 |        1 | Maioni, Miss. Roberta          | female |    16 |       0 |       0 |   110152 | 86.5     | 0        |   0.334254 |
    

### index为str
命名访问:如果有对模块命名，可以通过指定名称的方式访问


```python
#查看到模块名为target_encoding的数据
print(table.transform(x_test,run_to_layer="target_encoding").head(5).to_markdown())
```

    |     |   PassengerId |   Pclass | Name                           | Sex    |   Age |   SibSp |   Parch |   Ticket |     Fare |    Cabin |   Embarked |
    |----:|--------------:|---------:|:-------------------------------|:-------|------:|--------:|--------:|---------:|---------:|---------:|-----------:|
    | 500 |           501 |        3 | Calic, Mr. Petar               | male   |    17 |       0 |       0 |   315086 |  8.66406 | 0.317829 |   0.334254 |
    | 501 |           502 |        3 | Canavan, Miss. Mary            | female |    21 |       0 |       0 |   364846 |  7.75    | 0.317829 |   0.511111 |
    | 502 |           503 |        3 | O'Sullivan, Miss. Bridget Mary | female |     0 |       0 |       0 |   330909 |  7.62891 | 0.317829 |   0.511111 |
    | 503 |           504 |        3 | Laitinen, Miss. Kristina Sofia | female |    37 |       0 |       0 |     4135 |  9.58594 | 0.317829 |   0.334254 |
    | 504 |           505 |        1 | Maioni, Miss. Roberta          | female |    16 |       0 |       0 |   110152 | 86.5     | 0        |   0.334254 |
    

## 获取指定的pipe模块
有时候我们想获取指定pipe模块，并调用其函数方法,这里可以通过下标索引(从0开始),也可以通过name进行索引，与上面类似
### index为int


```python
#获取-4层的target encoding
print(table[-4].show_detail().head(5).to_markdown())
```

    |    | col   | bin_value                                      |   target_value |
    |---:|:------|:-----------------------------------------------|---------------:|
    |  0 | Name  | Abbott, Mrs. Stanton (Rosa Hunt)               |              1 |
    |  1 | Name  | Abelson, Mr. Samuel                            |              0 |
    |  2 | Name  | Adahl, Mr. Mauritz Nils Martin                 |              0 |
    |  3 | Name  | Adams, Mr. John                                |              0 |
    |  4 | Name  | Ahlin, Mrs. Johan (Johanna Persdotter Larsson) |              0 |
    

### index为str


```python
#这里获取的是2->1层的target_encoding，因为它取名name="target_encoding"
print(table["target_encoding"].show_detail().head(5).to_markdown())
```

    |    | col      | bin_value   |   target_value |
    |---:|:---------|:------------|---------------:|
    |  0 | Embarked | C           |       0.521739 |
    |  1 | Embarked | Q           |       0.511111 |
    |  2 | Embarked | S           |       0.334254 |
    |  3 | Embarked | nan         |       1        |
    |  4 | Cabin    | A14         |       0        |
    

### index为tuple或list


```python
#获取2->2的woe encoding
print(table[2,2].show_detail().head(5).to_markdown())
```

    |    | col    | bin_value   |   bad_num |   bad_rate |   good_num |   good_rate |       woe |        iv |
    |---:|:-------|:------------|----------:|-----------:|-----------:|------------:|----------:|----------:|
    |  0 | Sex    | male        |        54 |   0.279793 |        261 |    0.850163 |  1.11138  | 0.633897  |
    |  1 | Sex    | female      |       139 |   0.720207 |         46 |    0.149837 | -1.56999  | 0.895475  |
    |  2 | Pclass | 3           |        78 |   0.404145 |        201 |    0.654723 |  0.482439 | 0.120889  |
    |  3 | Pclass | 1           |        66 |   0.341969 |         50 |    0.162866 | -0.741789 | 0.132856  |
    |  4 | Pclass | 2           |        49 |   0.253886 |         56 |    0.18241  | -0.330626 | 0.0236317 |
    

## pipeline的切片式调用
可以通过切分的方式获取部分连续的pipe模块，并组装为Pipeline，这样可以更加方便灵活的获取中间结果


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(WOEEncoding(cols=["Sex","Pclass"],y=y_train))\
  .pipe(LabelEncoding(cols=["Name","Ticket"]))\
  .pipe(TargetEncoding(cols=["Embarked","Cabin"],y=y_train,name="target_encoding"))\
  .pipe(FillNa())\
  .pipe(Normalizer())\
  .pipe(PCADecomposition(n_components=8))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x14e68f33e08>




```python
#运行前3层
step1=table[:3].transform(x_test[:5])
print(step1.to_markdown())
```

    |     |   PassengerId |   Pclass | Name                           | Sex    |   Age |   SibSp |   Parch |   Ticket |     Fare | Cabin   | Embarked   |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |----:|--------------:|---------:|:-------------------------------|:-------|------:|--------:|--------:|---------:|---------:|:--------|:-----------|-----------:|-----------:|-----------:|-----------:|-------------:|
    | 500 |           501 |        3 | Calic, Mr. Petar               | male   |    17 |       0 |       0 |   315086 |  8.66406 | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    | 501 |           502 |        3 | Canavan, Miss. Mary            | female |    21 |       0 |       0 |   364846 |  7.75    | nan     | Q          |          1 |          0 |          0 |          0 |            1 |
    | 502 |           503 |        3 | O'Sullivan, Miss. Bridget Mary | female |     0 |       0 |       0 |   330909 |  7.62891 | nan     | Q          |          1 |          0 |          0 |          0 |            1 |
    | 503 |           504 |        3 | Laitinen, Miss. Kristina Sofia | female |    37 |       0 |       0 |     4135 |  9.58594 | nan     | S          |          1 |          0 |          0 |          0 |            1 |
    | 504 |           505 |        1 | Maioni, Miss. Roberta          | female |    16 |       0 |       0 |   110152 | 86.5     | B79     | S          |          0 |          1 |          0 |          0 |            1 |
    


```python
#运行中间3,4层
step2=table[3:5].transform(step1)
print(step2.to_markdown())
```

    |     |   PassengerId |    Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |     Fare | Cabin   | Embarked   |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |----:|--------------:|----------:|-------:|---------:|------:|--------:|--------:|---------:|---------:|:--------|:-----------|-----------:|-----------:|-----------:|-----------:|-------------:|
    | 500 |           501 |  0.482439 |      0 |  1.11138 |    17 |       0 |       0 |        0 |  8.66406 | nan     | S          |          1 |          0 |          0 |          1 |            0 |
    | 501 |           502 |  0.482439 |      0 | -1.56999 |    21 |       0 |       0 |        0 |  7.75    | nan     | Q          |          1 |          0 |          0 |          0 |            1 |
    | 502 |           503 |  0.482439 |      0 | -1.56999 |     0 |       0 |       0 |        0 |  7.62891 | nan     | Q          |          1 |          0 |          0 |          0 |            1 |
    | 503 |           504 |  0.482439 |      0 | -1.56999 |    37 |       0 |       0 |        0 |  9.58594 | nan     | S          |          1 |          0 |          0 |          0 |            1 |
    | 504 |           505 | -0.741789 |      0 | -1.56999 |    16 |       0 |       0 |      231 | 86.5     | B79     | S          |          0 |          1 |          0 |          0 |            1 |
    


```python
#运行5层及之后
step3=table[5:].transform(step2)
print(step3.to_markdown())
```

    |     |          0 |        1 |          2 |          3 |        4 |          5 |         6 |         7 |
    |----:|-----------:|---------:|-----------:|-----------:|---------:|-----------:|----------:|----------:|
    | 500 | -2.14936   | 0.307524 |  0.522612  |  0.0884669 | 0.324718 | -0.239891  | -0.435528 | -0.423045 |
    | 501 |  0.212004  | 2.3181   | -1.29001   | -0.468946  | 1.54553  | -0.0966754 |  0.778773 |  0.458949 |
    | 502 |  0.0531926 | 2.53845  | -1.51151   | -0.26218   | 1.53283  |  0.732909  |  0.733354 | -0.243668 |
    | 503 |  0.0885673 | 1.9982   | -1.13547   | -0.944341  | 0.546968 | -1.65149   | -0.116201 | -0.12588  |
    | 504 |  2.66786   | 0.110183 | -0.0890267 | -0.155736  | 0.269069 | -0.958493  |  1.60561  | -2.50238  |
    


```python

```
