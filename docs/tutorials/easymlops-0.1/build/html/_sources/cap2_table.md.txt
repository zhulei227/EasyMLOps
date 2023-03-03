# Table任务 

这里以titanic生存数据集为例，介绍相关模块的使用方法


```python
#数据准备
import os
os.chdir("../../")#与easymlops同级目录
import pandas as pd
data=pd.read_csv("./data/demo.csv")
data["date1"]="2020-03-06"
data["date2"]="2023-01-04"
print(data.head(5).to_markdown())
```

    |    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   | date1      | date2      |
    |---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|:-----------|:-----------|
    |  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          | 2020-03-06 | 2023-01-04 |
    |  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          | 2020-03-06 | 2023-01-04 |
    |  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          | 2020-03-06 | 2023-01-04 |
    |  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          | 2020-03-06 | 2023-01-04 |
    |  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          | 2020-03-06 | 2023-01-04 |
    


```python
x_train=data[:500]
x_test=data[500:]
y_train=x_train["Survived"]
y_test=x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]
```

## 数据清洗

- FixInput：固定输入column的顺序、数据类型（这个模块十分重要，所有table任务第一个pipe建议是它）
- FillNa：空值填充，包括均值、中位数、众数、指定默认值的方式  
- TransToCategory、TransToFloat、TransToInt：分别转换为类别、浮点、整型数据  
- ClipString：字符串切割；
- Replace：替换指定的字符串；
- TransToLower、TransToUpper：英文转小写、大写
- MapValues：将某些离散值或某些区间内的值映射为某个指定值
- Clip：对连续型变量按绝对范围或百分比切割   
- MinMaxScaler、Normalizer：归一化
- Bins：等距、等频、聚类分箱
- Add、Subtract、Multiply、Divide、DivideExact、Mod：加减乘除、整除、求余等运算，支持两列col计算，以及单列和某个指定值计算
- Equal、GreaterThan、GreaterEqualThan、LessThan、LessEqualThan：=,>,>=,<,<=等比较操作  
- And、Or：与、或操作  
- DateDayDiff：日期相减操作


```python
from easymlops import TablePipeLine
```


```python
from easymlops.table.preprocessing import *
from easymlops.table.ensemble import Parallel
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(Parallel([IsNull(),IsNotNull()]))\
  .pipe(FillNa(cols=["Cabin","Ticket","Parch","Fare","Sex"],fill_mode="mode"))\
  .pipe(FillNa(cols=["Age"],fill_mode="mean"))\
  .pipe(FillNa(fill_detail={"Embarked":"N"}))\
  .pipe(FillNa())\
  .pipe(TransToCategory(cols=["Cabin","Embarked","Name"]))\
  .pipe(TransToFloat(cols=["Age","Fare"]))\
  .pipe(TransToInt(cols=["Pclass","PassengerId","SibSp","Parch"]))\
  .pipe(ClipString(cols=["Name"],default_clip_index=(0,10)))\
  .pipe(Replace(cols=["Cabin"],source_values=["nan","N","B79"],target_value="nan"))\
  .pipe(TransToLower(cols=["Ticket","Cabin","Embarked","Name","Sex"]))\
  .pipe(MapValues(map_detail={"Cabin":[(["nan","NaN"],"n")],"Age":[("(-1,10)",10),("[10,20]",20)]}))\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(Clip(cols=["Fare"],percent_range=(1,99),name="clip_fare"))\
  .pipe(MinMaxScaler(cols=[("Age","Age_minmax")]))\
  .pipe(Normalizer(cols=[("Fare","Fare_normal")]))\
  .pipe(Bins(n_bins=10,strategy="uniform",cols=[("Age","Age_uni")]))\
  .pipe(Bins(n_bins=10,strategy="quantile",cols=[("Age","Age_quan")]))\
  .pipe(Bins(n_bins=10,strategy="kmeans",cols=[("Fare","Fare_km")]))\
  .pipe(Parallel([Add(left_col_name="Pclass",right_col_name="SibSp"),
                  Subtract(left_col_name="Pclass",right_col_name="Fare"),
                  Multiply(left_col_name="Fare",right_col_name="Age"),
                  Divide(left_col_name="Age_minmax",right_col_name="Fare_normal"),
                  DivideExact(left_col_name="Age_minmax",right_col_name="Pclass"),
                  Mod(left_col_name="PassengerId",right_col_name="Pclass")]))\
  .pipe(Parallel([Equal(left_col_name="Pclass",right_col_name="SibSp"),
                  GreaterThan(left_col_name="Pclass",right_col_name="Fare"),
                  GreaterEqualThan(left_col_name="Fare",right_col_name="Age"),
                  LessThan(left_col_name="Age_minmax",right_col_name="Fare_normal"),
                  LessEqualThan(left_col_name="Age_minmax",right_col_name="Pclass"),
                  And(left_col_name="PassengerId",right_col_name="Pclass"),
                  Or(left_col_name="Pclass",right_col_name="SibSp")]))\
  .pipe(DateDayDiff(left_col_name="date2",right_col_name="date1"))


x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Pclass_equal_SibSp |   Pclass_greater_than_Fare |   Fare_greater_equal_than_Age |   Age_minmax_less_than_Fare_normal |   Age_minmax_less_equal_than_Pclass |   PassengerId_and_Pclass |   Pclass_add_SibSp |   Pclass_subtract_Fare |   Fare_multiply_Age |   Age_minmax_divide_Fare_normal |   Age_minmax_divide_exact_Pclass |   PassengerId_is_null |   Pclass_is_null |   Name_is_null |   Sex_is_null |   Age_is_null |   SibSp_is_null |   Parch_is_null |   Ticket_is_null |   Fare_is_null |   Cabin_is_null |   Embarked_is_null |   date1_is_null |   date2_is_null |   PassengerId |   Pclass | Name       | Sex    |     Age |   SibSp |   Parch |   Ticket |     Fare | Cabin   | Embarked   | date1      | date2      |   PassengerId_is_not_null |   Pclass_is_not_null |   Name_is_not_null |   Sex_is_not_null |   Age_is_not_null |   SibSp_is_not_null |   Parch_is_not_null |   Ticket_is_not_null |   Fare_is_not_null |   Cabin_is_not_null |   Embarked_is_not_null |   date1_is_not_null |   date2_is_not_null |   Age_minmax |   Fare_normal |   Age_uni |   Age_quan |   Fare_km |   PassengerId_mod_Pclass |   Pclass_or_SibSp |   date2_day_diff_date1 |
    |----:|---------------------:|---------------------------:|------------------------------:|-----------------------------------:|------------------------------------:|-------------------------:|-------------------:|-----------------------:|--------------------:|--------------------------------:|---------------------------------:|----------------------:|-----------------:|---------------:|--------------:|--------------:|----------------:|----------------:|-----------------:|---------------:|----------------:|-------------------:|----------------:|----------------:|--------------:|---------:|:-----------|:-------|--------:|--------:|--------:|---------:|---------:|:--------|:-----------|:-----------|:-----------|--------------------------:|---------------------:|-------------------:|------------------:|------------------:|--------------------:|--------------------:|---------------------:|-------------------:|--------------------:|-----------------------:|--------------------:|--------------------:|-------------:|--------------:|----------:|-----------:|----------:|-------------------------:|------------------:|-----------------------:|
    | 500 |                    0 |                          0 |                             0 |                                  0 |                                   1 |                        1 |                  3 |               -5.66406 |              173.25 |                       -0.316333 |                                0 |                     0 |                0 |              0 |             0 |             0 |               0 |               0 |                0 |              0 |               1 |                  0 |               0 |               0 |           501 |        3 | calic, mr. | male   | 20      |       0 |       0 |   315086 |  8.66406 | n       | s          | 2020-03-06 | 2023-01-04 |                         1 |                    1 |                  1 |                 1 |                 1 |                   1 |                   1 |                    1 |                  1 |                   0 |                      1 |                   1 |                   1 |     0.163934 |     -0.518233 |         1 |          1 |         1 |                        0 |                 1 |                   1034 |
    | 501 |                    0 |                          0 |                             0 |                                  0 |                                   1 |                        1 |                  3 |               -4.75    |              162.75 |                       -0.334449 |                                0 |                     0 |                0 |              0 |             0 |             0 |               0 |               0 |                0 |              0 |               1 |                  0 |               0 |               0 |           502 |        3 | canavan, m | female | 21      |       0 |       0 |   364846 |  7.75    | n       | q          | 2020-03-06 | 2023-01-04 |                         1 |                    1 |                  1 |                 1 |                 1 |                   1 |                   1 |                    1 |                  1 |                   0 |                      1 |                   1 |                   1 |     0.180328 |     -0.539179 |         1 |          2 |         1 |                        1 |                 1 |                   1034 |
    | 502 |                    0 |                          0 |                             0 |                                  0 |                                   1 |                        1 |                  3 |               -4.62891 |              222.75 |                       -0.580871 |                                0 |                     0 |                0 |              0 |             0 |             1 |               0 |               0 |                0 |              0 |               1 |                  0 |               0 |               0 |           503 |        3 | o'sullivan | female | 29.2031 |       0 |       0 |   330909 |  7.62891 | n       | q          | 2020-03-06 | 2023-01-04 |                         1 |                    1 |                  1 |                 1 |                 0 |                   1 |                   1 |                    1 |                  1 |                   0 |                      1 |                   1 |                   1 |     0.314805 |     -0.541954 |         3 |          5 |         1 |                        2 |                 1 |                   1034 |
    | 503 |                    0 |                          0 |                             0 |                                  0 |                                   1 |                        1 |                  3 |               -6.58594 |              354.75 |                       -0.890396 |                                0 |                     0 |                0 |              0 |             0 |             0 |               0 |               0 |                0 |              0 |               1 |                  0 |               0 |               0 |           504 |        3 | laitinen,  | female | 37      |       0 |       0 |     4135 |  9.58594 | n       | s          | 2020-03-06 | 2023-01-04 |                         1 |                    1 |                  1 |                 1 |                 1 |                   1 |                   1 |                    1 |                  1 |                   0 |                      1 |                   1 |                   1 |     0.442623 |     -0.497108 |         4 |          6 |         1 |                        0 |                 1 |                   1034 |
    | 504 |                    0 |                          0 |                             1 |                                  1 |                                   1 |                        1 |                  1 |              -85.5     |             1730    |                        0.12955  |                                0 |                     0 |                0 |              0 |             0 |             0 |               0 |               0 |                0 |              0 |               0 |                  0 |               0 |               0 |           505 |        1 | maioni, mi | female | 20      |       0 |       0 |   110152 | 86.5     | n       | s          | 2020-03-06 | 2023-01-04 |                         1 |                    1 |                  1 |                 1 |                 1 |                   1 |                   1 |                    1 |                  1 |                   1 |                      1 |                   1 |                   1 |     0.163934 |      1.26542  |         1 |          1 |         4 |                        0 |                 1 |                   1034 |
    


```python
del x_train["date1"]
del x_train["date2"]
del x_test["date1"]
del x_test["date2"]
```

## 特征编码
- OneHotEncoding：新特征以{col名称}_{变量值}的方式表示
- LabelEncoding：将变量值map为1,2,3...
- TargetEncoding：支持smoothing  
- WOEEncoding


```python
from easymlops.table.encoding import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Sex"],drop_col=False))\
  .pipe(LabelEncoding(cols=["Sex"]))\
  .pipe(TargetEncoding(cols=["Name","Ticket"],smoothing=True,y=y_train))\
  .pipe(WOEEncoding(cols=["Pclass","Embarked","Cabin"],y=y_train,name="woe"))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   PassengerId |    Pclass |   Name |   Sex |   Age |   SibSp |   Parch |   Ticket |     Fare |    Cabin |   Embarked |   Sex_male |   Sex_female |
    |----:|--------------:|----------:|-------:|------:|------:|--------:|--------:|---------:|---------:|---------:|-----------:|-----------:|-------------:|
    | 500 |           501 |  0.482439 |      0 |     1 |    17 |       0 |       0 | 0        |  8.66406 | 0.299607 |   0.224849 |          1 |            0 |
    | 501 |           502 |  0.482439 |      0 |     2 |    21 |       0 |       0 | 0        |  7.75    | 0.299607 |  -0.508609 |          0 |            1 |
    | 502 |           503 |  0.482439 |      0 |     2 |     0 |       0 |       0 | 0        |  7.62891 | 0.299607 |  -0.508609 |          0 |            1 |
    | 503 |           504 |  0.482439 |      0 |     2 |    37 |       0 |       0 | 0        |  9.58594 | 0.299607 |   0.224849 |          0 |            1 |
    | 504 |           505 | -0.741789 |      0 |     2 |    16 |       0 |       0 | 0.387228 | 86.5     | 0        |   0.224849 |          0 |            1 |
    

## 特征降维
- PCADecomposition：PCA降维
- NMFDecomposition：注意，对于小于0的输入，会截断为0


```python
from easymlops.table.decomposition import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
  .pipe(Parallel([PCADecomposition(n_components=4,prefix="pca"),NMFDecomposition(n_components=4,prefix="nmf")]))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   pca_0 |    pca_1 |     pca_2 |     pca_3 |   nmf_0 |    nmf_1 |       nmf_2 |      nmf_3 |
    |----:|--------:|---------:|----------:|----------:|--------:|---------:|------------:|-----------:|
    | 500 | 250.112 | -27.0943 |  -5.83468 | -0.180072 | 6.20683 | 0.217971 | 0.502871    | 0          |
    | 501 | 251.118 | -27.7806 |  -1.80052 | -0.126844 | 6.21923 | 0.19184  | 0.6215      | 0.00316003 |
    | 502 | 252.018 | -29.1482 | -22.7548  | -0.559051 | 6.2316  | 0.191844 | 0.000106315 | 0.0141703  |
    | 503 | 253.218 | -25.0384 |  14.0518  |  0.198428 | 6.24398 | 0.234273 | 1.09488     | 0          |
    | 504 | 255.219 |  50.4986 | -11.4019  | -1.09081  | 6.2555  | 2.23452  | 0.473346    | 0.00359195 |
    

## 特征选择

### 过滤式
通过`min_threshold`和`max_threshold`来进行筛选，如果设置为`None`表示对应方向不做约束
- MissRateFilter：缺失率
- VarianceFilter：方差
- PersonCorrFilter：相关系数
- PSIFilter：PSI，主要是模型稳定性
- Chi2Filter：主要针对离散变量 
- MutualInfoFilter：互信息  
- IVFilter：IV值，即WOE的加权和


```python
from easymlops.table.feature_selection import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(MissRateFilter(max_threshold=0.1))\
  .pipe(VarianceFilter(min_threshold=0.1))\
  .pipe(PersonCorrFilter(min_threshold=0.1,y=y_train,name="person"))\
  .pipe(PSIFilter(oot_x=x_test,cols=["Pclass","Sex","Embarked"],name="psi",max_threshold=0.5))\
  .pipe(LabelEncoding(cols=["Sex","Ticket","Embarked","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name"],y=y_train))\
  .pipe(Chi2Filter(y=y_train,name="chi2"))\
  .pipe(MutualInfoFilter(y=y_train))\
  .pipe(IVFilter(y=y_train,name="iv",cols=["Sex","Fare"],min_threshold=0.05))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Pclass |   Name |   Sex |   Ticket |     Fare |   Embarked |
    |----:|---------:|-------:|------:|---------:|---------:|-----------:|
    | 500 |        1 |      0 |     1 |        0 |  8.66406 |          1 |
    | 501 |        1 |      0 |     2 |        0 |  7.75    |          3 |
    | 502 |        1 |      0 |     2 |        0 |  7.62891 |          3 |
    | 503 |        1 |      0 |     2 |        0 |  9.58594 |          1 |
    | 504 |        2 |      0 |     2 |      231 | 86.5     |          1 |
    


```python
#查看PSI分布
print(table[-6].show_detail().head(5).to_markdown())
```

    |    | col    | bin_value   |   ins_num |   ins_rate |   oot_num |   oot_rate |         psi |
    |---:|:-------|:------------|----------:|-----------:|----------:|-----------:|------------:|
    |  0 | Pclass | 3           |       279 |      0.558 |       212 |   0.542199 | 0.000453869 |
    |  1 | Pclass | 1           |       116 |      0.232 |       100 |   0.255754 | 0.0023156   |
    |  2 | Pclass | 2           |       105 |      0.21  |        79 |   0.202046 | 0.000307118 |
    |  3 | Sex    | male        |       315 |      0.63  |       262 |   0.670077 | 0.00247163  |
    |  4 | Sex    | female      |       185 |      0.37  |       129 |   0.329923 | 0.00459451  |
    

### 嵌入式
同样，通过`min_threshold`和`max_threshold`来进行筛选，如果设置为`None`表示对应方向不做约束

- LREmbed：线性模型，对系数取了绝对值做评估
- LGBMEmbed：lgbm决策树，评估标准有`importance_type="split"`（默认），以及`importance_type="gain"`



```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(LabelEncoding(cols=["Sex","Ticket","Embarked","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Cabin"],y=y_train))\
  .pipe(LREmbed(y=y_train,min_threshold=0.01))\
  .pipe(LGBMEmbed(y=y_train,min_threshold=0.01))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Pclass |   Name |   Sex |   SibSp |   Parch |    Cabin |   Embarked |
    |----:|---------:|-------:|------:|--------:|--------:|---------:|-----------:|
    | 500 |        1 |      0 |     1 |       0 |       0 | 0.317829 |          1 |
    | 501 |        1 |      0 |     2 |       0 |       0 | 0.317829 |          3 |
    | 502 |        1 |      0 |     2 |       0 |       0 | 0.317829 |          3 |
    | 503 |        1 |      0 |     2 |       0 |       0 | 0.317829 |          1 |
    | 504 |        2 |      0 |     2 |       0 |       0 | 0        |          1 |
    


```python
#查看LR权重分布
print(table[-2].show_detail().to_markdown())
```

    |    |   PassengerId |   Pclass |    Name |     Sex |        Age |    SibSp |     Parch |      Ticket |       Fare |   Cabin |   Embarked |
    |---:|--------------:|---------:|--------:|--------:|-----------:|---------:|----------:|------------:|-----------:|--------:|-----------:|
    |  0 |   0.000168929 | 0.200302 | 6.51026 | 1.03629 | 0.00765414 | 0.145938 | 0.0530416 | 0.000512668 | 0.00329262 | 0.90755 |   0.154712 |
    


```python
#查看Lgbm中split次数(默认)分布
print(table[-1].show_detail().to_markdown())
```

    |    |   Pclass |   Name |   Sex |   SibSp |   Parch |   Cabin |   Embarked |
    |---:|---------:|-------:|------:|--------:|--------:|--------:|-----------:|
    |  0 |       35 |    100 |    12 |      16 |      16 |       4 |          7 |
    

## 分类 

模型封装了很多，包括lgbm,lr,nb....等，具体可以查看后面的`API`


```python
from easymlops.table.classification import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass", "Sex"], drop_col=False)) \
  .pipe(WOEEncoding(cols=["Ticket", "Embarked", "Cabin", "Sex", "Pclass"], y=y_train)) \
  .pipe(LabelEncoding(cols=["Name"]))\
  .pipe(LGBMClassification(y=y_train,native_init_params={"max_depth":2},native_fit_params={"num_boost_round":128}))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |         0 |        1 |
    |----:|----------:|---------:|
    | 500 | 0.923326  | 0.076674 |
    | 501 | 0.373652  | 0.626348 |
    | 502 | 0.37838   | 0.62162  |
    | 503 | 0.670166  | 0.329834 |
    | 504 | 0.0684703 | 0.93153  |
    


```python
#获取sabass特征重要性
table[-1].get_contrib(table.transform_single(x_test.to_dict("record")[0],run_to_layer=-2))
```




    {'Sex': -0.5461501854144144,
     'Cabin': -0.1270063345547848,
     'Age': 0.05115085845832483,
     'Fare': -0.1265214828241528,
     'SibSp': 0.0441638895513395,
     'Ticket': -0.005658456561668252,
     'PassengerId': -0.2263838236137358,
     'Pclass_2': -0.016025418181148575}



## 回归建模
目前只封装了lgbm，后续将会封装其他模型


```python
from easymlops.table.regression import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass", "Sex"], drop_col=False)) \
  .pipe(WOEEncoding(cols=["Ticket", "Embarked", "Cabin", "Sex", "Pclass"], y=y_train)) \
  .pipe(LabelEncoding(cols=["Name"]))\
  .pipe(LGBMRegression(y=y_train,objective="poisson"))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |      pred |
    |----:|----------:|
    | 500 | 0.0929963 |
    | 501 | 0.716432  |
    | 502 | 0.654662  |
    | 503 | 0.309583  |
    | 504 | 0.916929  |
    


```python
#获取sabass特征重要性
table[-1].get_contrib(table.transform_single(x_test.to_dict("record")[0],run_to_layer=-2))
```




    {'Sex': -0.5899834278029109,
     'Cabin': -0.13150175316983173,
     'Fare': -0.1301026457360956,
     'Age': 0.275645992578784,
     'PassengerId': -0.39285142163510006,
     'Pclass_2': 0.0868239954973954,
     'SibSp': 0.048242598427820915,
     'Parch': -0.01823350433998343,
     'Pclass': -0.01402473665988247,
     'Ticket': -0.0030424253099006976,
     'Embarked': -0.005601108385329127}



## Stacking建模
主要通过`Parallel`的并行化实现  
注意：`Parallel`内部本质是顺序执行，如果后面pipe模块的输出col名称与前面有重复，将会覆盖，所以对于会被覆盖的col建议取别名，比如下面的`LabelEncoding`，通过设置`cols=[(原col1,新col1),(原col2,新col2)]`将新生成的结果赋值到新col中，而不改变原col的值


```python
table = TablePipeLine()
table.pipe(FixInput()) \
  .pipe(FillNa()) \
  .pipe(Parallel([OneHotEncoding(cols=["Pclass", "Sex"]), LabelEncoding(cols=[("Sex","Sex_label"), ("Pclass","Pclass_label")]),
                    TargetEncoding(cols=["Name", "Ticket", "Embarked", "Cabin", "Sex"], y=y_train)])) \
  .pipe(Parallel([PCADecomposition(n_components=2, prefix="pca"), NMFDecomposition(n_components=2, prefix="nmf")]))\
  .pipe(Parallel([LGBMClassification(y=y_train, prefix="lgbm"), LogisticRegressionClassification(y=y_train, prefix="lr")]))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   lgbm_0 |    lgbm_1 |     lr_0 |     lr_1 |
    |----:|---------:|----------:|---------:|---------:|
    | 500 | 0.965218 | 0.0347825 | 0.651417 | 0.348583 |
    | 501 | 0.98153  | 0.0184698 | 0.65506  | 0.34494  |
    | 502 | 0.979139 | 0.0208607 | 0.647266 | 0.352734 |
    | 503 | 0.808796 | 0.191204  | 0.656613 | 0.343387 |
    | 504 | 0.184484 | 0.815516  | 0.449149 | 0.550851 |
    


```python

```
