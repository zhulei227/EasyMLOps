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

    |     |        0 |         1 |
    |----:|---------:|----------:|
    | 500 | 0.919121 | 0.0808785 |
    | 501 | 0.189551 | 0.810449  |
    | 502 | 0.189551 | 0.810449  |
    | 503 | 0.255328 | 0.744672  |
    | 504 | 0.264882 | 0.735118  |
    

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

    |    |   PassengerId |    Pclass |      Name |       Sex |        Age |     SibSp |   Parch |    Ticket |      Fare |     Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|----------:|----------:|----------:|-----------:|----------:|--------:|----------:|----------:|----------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |      -1.72859 | -0.810644 | -0.792884 | -0.766356 | -0.0710908 |  0.369594 | -0.4875 | -0.83866  | -0.515811 | -0.287586 |  -0.597846 |   0.890008 |  -0.549621 |   -0.51558 |   0.766356 |    -0.766356 |
    |  1 |      -1.72166 |  0.432675 |  1.26122  |  1.30488  |  0.84106   |  0.369594 | -0.4875 |  1.33403  |  0.831116 |  2.59023  |   1.56826  |  -1.12359  |   1.81944  |   -0.51558 |  -1.30488  |     1.30488  |
    |  2 |      -1.71473 | -0.810644 |  1.26122  |  1.30488  |  0.156947  | -0.497998 | -0.4875 |  1.33403  | -0.501603 | -0.287586 |  -0.597846 |   0.890008 |  -0.549621 |   -0.51558 |  -1.30488  |     1.30488  |
    |  3 |      -1.70781 |  0.432675 |  1.26122  |  1.30488  |  0.670032  |  0.369594 | -0.4875 |  0.247687 |  0.448063 |  0.480923 |  -0.597846 |  -1.12359  |   1.81944  |   -0.51558 |  -1.30488  |     1.30488  |
    |  4 |      -1.70088 | -0.810644 | -0.792884 | -0.766356 |  0.670032  | -0.497998 | -0.4875 | -0.83866  | -0.499057 | -0.287586 |  -0.597846 |   0.890008 |  -0.549621 |   -0.51558 |   0.766356 |    -0.766356 |
    


```python
pipe_pca=PCADecomposition(n_components=8)
x_train_new=pipe_pca.fit(x_train_new).transform(x_train_new)
print(x_train_new.head(5).to_markdown())
```

    |    |        0 |         1 |          2 |          3 |          4 |        5 |        6 |         7 |
    |---:|---------:|----------:|-----------:|-----------:|-----------:|---------:|---------:|----------:|
    |  0 | -2.33692 |  0.710067 |  0.0803019 |  0.0746554 |  0.0515367 | 1.25277  | 1.07024  | -0.624323 |
    |  1 |  4.13372 | -0.15474  |  1.78063   | -0.908822  |  0.0571593 | 1.84565  | 0.755661 | -0.561293 |
    |  2 |  1.65117 |  2.18103  | -1.10575   | -0.975744  | -0.441426  | 0.820884 | 1.82729  | -0.175748 |
    |  3 |  2.83835 | -0.176422 |  0.638332  |  0.0833258 | -1.11821   | 1.18948  | 1.34569  | -1.29006  |
    |  4 | -2.31382 |  0.326691 |  0.0522898 | -0.487427  | -0.461197  | 1.09001  | 1.5257   | -0.2117   |
    


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




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x13aadde1bc8>




```python
print(pipeline_combine.transform(x_test).head(5).to_markdown())
```

    |     |     0 |     1 |
    |----:|------:|------:|
    | 500 | 50.02 | 49.98 |
    | 501 | 49.9  | 50.1  |
    | 502 | 49.84 | 50.16 |
    | 503 | 49.95 | 50.05 |
    | 504 | 48.35 | 51.65 |
    

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

    |     |          0 |          1 |         2 |         3 |         4 |        5 |        6 |          7 |
    |----:|-----------:|-----------:|----------:|----------:|----------:|---------:|---------:|-----------:|
    | 500 | -2.31036   | -0.0928624 | -0.19398  |  0.511277 |  0.202237 | -1.86472 | -0.24938 | -0.255635  |
    | 501 |  0.180246  | -1.91993   | -0.315418 |  1.02128  | -1.91574  | -1.62147 | -1.47288 |  0.604476  |
    | 502 |  0.115275  | -2.264     | -0.104098 |  1.12641  | -1.47905  | -1.51371 | -2.14166 | -0.0441479 |
    | 503 | -0.0202802 | -1.64604   | -0.728745 | -0.034924 | -2.07495  | -2.24773 |  0.12019 | -0.132674  |
    | 504 |  2.47437   |  0.881527  |  0.045653 | -0.240732 | -1.78536  | -1.93257 | -1.00486 | -1.33923   |
    

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




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x13ab1ad0648>




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

    |     |         0 |        1 |          2 |          3 |        4 |         5 |         6 |         7 |
    |----:|----------:|---------:|-----------:|-----------:|---------:|----------:|----------:|----------:|
    | 500 | -2.14842  | 0.307872 |  0.526932  |  0.0881118 | 0.324999 | -0.238535 | -0.432583 | -0.424289 |
    | 501 |  0.212949 | 2.32202  | -1.28604   | -0.469809  | 1.53792  | -0.098157 |  0.775934 |  0.459946 |
    | 502 |  0.053812 | 2.543    | -1.50479   | -0.263976  | 1.52535  |  0.729823 |  0.737169 | -0.240717 |
    | 503 |  0.089901 | 2.00564  | -1.13093   | -0.942879  | 0.544493 | -1.64931  | -0.119709 | -0.125324 |
    | 504 |  2.67342  | 0.110068 | -0.0911749 | -0.154845  | 0.265834 | -0.959033 |  1.61703  | -2.49404  |
    


```python

```
