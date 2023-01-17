
# EasyMLOps  
  
## 介绍   
`EasyMLOps`包以`Pipline`的方式构建建模任务，可直接进行模型训练、预测(离线，在线)，测试(离线在线预测一致性、预测性能)等功能，通过外套一层Flask或FastApi即可直接部署生产，目前主要功能有：

#### 1. 模型的训练&预测&保存

- 数据清洗，数据自动填充、转换、盖帽等：easymlops.ml.preprocessing
- 特征表达，包括Target、Label、Onehot Encoding以及PCA降维等：easymlops.ml.representation
- 分类模型，包括lgbm决策树、logistic回归、svm等传统机器学习模型：easymlops.ml.classification 
- stacking，通过Parallel模块，可以在同一阶段进行多个模型的训练，这样可以很方面的构建stacking模型：easymlops.ensemble.Parallel

#### 2. 自定义pipe模块

- fit,tranform：最少只需实现这两函数即可接入pipeline中
- set_params,get_params:实现这两函数可以对模块持久化
- transform_single:支持生产预测

#### 3. 模型的分拆&组合&中间层特征抽取  

- pipeml的子模块也可以是pipeml，这样方便逐块建模再组合
- pipeml可以提取中间层数据，方便复用别人的模型，继续做自己下一步工作:pipeobj.transform(data,run_to_layer=指定层数或模块名)   

#### 4. 支持生产部署:数据一致性测试&性能测试&日志记录

- pipeobj.transform_single(data)即可对生产数据(通常转换为dict)进行预测
- pipeobj.auto_check_transform(data)可以对数据一致性以及各个pipe模块性能做测试  
- pipeobj.transform_single(data,logger)可以追踪记录pipeline预测中每一步信息  

#### 5. 训练性能优化（主要是减少内存占用）

- easymlops.ml.perfopt.ReduceMemUsage模块:修改数据类型，比如某列特征数据范围在float16内，而目前的数据类型是float64，则将float64修改为float16
- easymlops.ml.perfopt.Dense2Sparse模块:将稠密矩阵转换为稀疏矩阵（含0量很多时使用），注意后续的pipe模块要提供对稀疏矩阵的支持(easymlops.ml.classification下的模块基本都支持)  

#### 6. 文本NLP处理
- 去停用词，去标点符号，去特定字符，抽取中文字符，分词，关键词提取等数据清洗操作：easymlops.nlp.preprocessing
- 文本特征提取，包括bow,tfidf等传统模型；lda,lsi等主题模型；fastext,word2vec,doc2vec等词向量模型：easymlops.nlp.representation

## 0.安装
```bash
pip install easymlops
```  
或

```bash
pip install git+https://github.com/zhulei227/EasyMLOps
```  
或

```bash
git clone https://github.com/zhulei227/EasyMLOps.git
cd EasyMLOps
python setup.py install
```  
或  

将整个easymlops包拷贝到你所运行代码的同级目录，然后安装依赖包  
```bash
pip install -r requirements.txt
```

## 1. 基本使用  

导入`PipeML`主程序


```python
from easymlops import PipeML
```

准备`pandas.DataFrame`格式的数据


```python
import pandas as pd
data=pd.read_csv("./data/demo.csv")
data.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



拆分训练测试


```python
x_train=data[:500]
x_test=data[500:]
y_train=x_train["Survived"]
y_test=x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]
```

### 1.1 数据清洗


```python
from easymlops.ml.preprocessing import *
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa(cols=["Cabin","Ticket","Parch"],fill_mode="mode"))\
  .pipe(FillNa(cols=["Age"],fill_mode="mean"))\
  .pipe(FillNa(fill_detail={"Embarked":"N"}))\
  .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
  .pipe(TransToFloat(cols=["Age","Fare"]))\
  .pipe(TransToInt(cols=["PassengerId","Survived","SibSp","Parch"]))\
  .pipe(TransToLower(cols=["Ticket","Cabin","Embarked","Name","Sex"]))\
  .pipe(CategoryMapValues(map_detail={"Cabin":(["nan","NaN"],"n")}))\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(Clip(cols=["Fare"],percent_range=(10,99),name="clip_fare"))

x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>3</td>
      <td>calic, mr. petar</td>
      <td>male</td>
      <td>17.000000</td>
      <td>0</td>
      <td>0</td>
      <td>315086</td>
      <td>8.6625</td>
      <td>c23 c25 c27</td>
      <td>s</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>3</td>
      <td>canavan, miss. mary</td>
      <td>female</td>
      <td>21.000000</td>
      <td>0</td>
      <td>0</td>
      <td>364846</td>
      <td>7.7500</td>
      <td>c23 c25 c27</td>
      <td>q</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>3</td>
      <td>o'sullivan, miss. bridget mary</td>
      <td>female</td>
      <td>29.204774</td>
      <td>0</td>
      <td>0</td>
      <td>330909</td>
      <td>7.7175</td>
      <td>c23 c25 c27</td>
      <td>q</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>3</td>
      <td>laitinen, miss. kristina sofia</td>
      <td>female</td>
      <td>37.000000</td>
      <td>0</td>
      <td>0</td>
      <td>4135</td>
      <td>9.5875</td>
      <td>c23 c25 c27</td>
      <td>s</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>maioni, miss. roberta</td>
      <td>female</td>
      <td>16.000000</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>b79</td>
      <td>s</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 特征工程


```python
from easymlops.ml.representation import *
```


```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa(cols=["Cabin","Ticket","Parch"],fill_mode="mode"))\
  .pipe(FillNa(cols=["Age"],fill_mode="mean"))\
  .pipe(FillNa(fill_detail={"Embarked":"N"}))\
  .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
  .pipe(TransToFloat(cols=["Age","Fare"]))\
  .pipe(TransToInt(cols=["PassengerId","Survived","SibSp","Parch"]))\
  .pipe(TransToLower(cols=["Ticket","Cabin","Embarked","Name","Sex"]))\
  .pipe(CategoryMapValues(map_detail={"Cabin":(["nan","NaN"],"n")}))\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(Clip(cols=["Fare"],percent_range=(10,99),name="clip_fare"))\
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
  .pipe(FillNa(fill_number_value=0))

x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Pclass_3</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>17.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.319693</td>
      <td>0.334254</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>21.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.319693</td>
      <td>0.511111</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>29.204774</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7175</td>
      <td>0.319693</td>
      <td>0.511111</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>37.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.319693</td>
      <td>0.334254</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>16.000000</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.000000</td>
      <td>0.334254</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 1.3 分类模型


```python
from easymlops.ml.classification import *

ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa(cols=["Cabin","Ticket","Parch"],fill_mode="mode"))\
  .pipe(FillNa(cols=["Age"],fill_mode="mean"))\
  .pipe(FillNa(fill_detail={"Embarked":"N"}))\
  .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
  .pipe(TransToFloat(cols=["Age","Fare"]))\
  .pipe(TransToInt(cols=["PassengerId","Survived","SibSp","Parch"]))\
  .pipe(TransToLower(cols=["Ticket","Cabin","Embarked","Name","Sex"]))\
  .pipe(CategoryMapValues(map_detail={"Cabin":(["nan","NaN"],"n")}))\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(Clip(cols=["Fare"],percent_range=(10,99),name="clip_fare"))\
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
  .pipe(FillNa(fill_number_value=0))\
  .pipe(LGBMClassification(y=y_train))

x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
  </tbody>
</table>
</div>



### 1.4 模型持久化


```python
#保存
ml.save("ml.pkl")
```


```python
#导入
#由于只保留了模型参数，所以需要重新声明模型结构信息(参数无需传入;但导入也没有问题，这样还可以给调用者提供更多的建模信息)
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(FillNa())\
  .pipe(FillNa())\
  .pipe(TransToCategory())\
  .pipe(TransToFloat())\
  .pipe(TransToInt())\
  .pipe(TransToLower())\
  .pipe(CategoryMapValues())\
  .pipe(Clip())\
  .pipe(Clip())\
  .pipe(OneHotEncoding())\
  .pipe(LabelEncoding())\
  .pipe(TargetEncoding())\
  .pipe(FillNa())\
  .pipe(LGBMClassification())
```




    <easymlops.pipeml.PipeML at 0x1f0260c5eb8>




```python
ml.load("ml.pkl")
ml.transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.999998</td>
      <td>0.000002</td>
    </tr>
  </tbody>
</table>
</div>



### 1.5 Stacking建模


```python
from easymlops.ml.ensemble import Parallel
ml = PipeML()
ml.pipe(FixInput()) \
  .pipe(TransToCategory(cols=["Cabin", "Embarked"])) \
  .pipe(TransToFloat(cols=["Age", "Fare"])) \
  .pipe(TransToInt(cols=["PassengerId", "SibSp", "Parch"])) \
  .pipe(Clip(cols=["Age"], default_clip=(1, 99), name="clip_name")) \
  .pipe(Clip(cols=["Fare"], percent_range=(10, 99), name="clip_fare")) \
  .pipe(Parallel([OneHotEncoding(cols=["Pclass", "Sex"]), LabelEncoding(cols=["Sex", "Pclass"]),
                    TargetEncoding(cols=["Name", "Ticket", "Embarked", "Cabin", "Sex"], y=y_train)])) \
  .pipe(FillNa(fill_number_value=0, fill_category_value="missing")) \
  .pipe(Parallel([PCADecomposition(n_components=2, prefix="pca"), NMFDecomposition(n_components=2, prefix="nmf")]))

ml.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pca_0</th>
      <th>pca_1</th>
      <th>nmf_0</th>
      <th>nmf_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.113628</td>
      <td>-26.857317</td>
      <td>6.211156</td>
      <td>0.080100</td>
    </tr>
    <tr>
      <th>1</th>
      <td>251.118945</td>
      <td>-27.505963</td>
      <td>6.226342</td>
      <td>0.071882</td>
    </tr>
    <tr>
      <th>2</th>
      <td>252.020439</td>
      <td>-28.977182</td>
      <td>6.224453</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253.219795</td>
      <td>-24.617566</td>
      <td>6.261636</td>
      <td>0.183279</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255.231018</td>
      <td>50.696607</td>
      <td>6.249646</td>
      <td>2.115887</td>
    </tr>
  </tbody>
</table>
</div>




```python
ml = PipeML()
ml.pipe(FixInput()) \
  .pipe(TransToCategory(cols=["Cabin", "Embarked"])) \
  .pipe(TransToFloat(cols=["Age", "Fare"])) \
  .pipe(TransToInt(cols=["PassengerId", "SibSp", "Parch"])) \
  .pipe(Clip(cols=["Age"], default_clip=(1, 99), name="clip_name")) \
  .pipe(Clip(cols=["Fare"], percent_range=(10, 99), name="clip_fare")) \
  .pipe(Parallel([OneHotEncoding(cols=["Pclass", "Sex"]), LabelEncoding(cols=["Sex", "Pclass"]),
                    TargetEncoding(cols=["Name", "Ticket", "Embarked", "Cabin", "Sex"], y=y_train)])) \
  .pipe(FillNa(fill_number_value=0, fill_category_value="missing")) \
  .pipe(Parallel([PCADecomposition(n_components=4, prefix="pca"), NMFDecomposition(n_components=4, prefix="nmf")]))\
  .pipe(Parallel([LGBMClassification(y=y_train, prefix="lgbm"), LogisticRegressionClassification(y_train, prefix="lr")]))

ml.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lgbm_0</th>
      <th>lgbm_1</th>
      <th>lr_0</th>
      <th>lr_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.997739</td>
      <td>0.002261</td>
      <td>0.632065</td>
      <td>0.367935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.989129</td>
      <td>0.010871</td>
      <td>0.649331</td>
      <td>0.350669</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.993015</td>
      <td>0.006985</td>
      <td>0.563674</td>
      <td>0.436326</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.984683</td>
      <td>0.015317</td>
      <td>0.703988</td>
      <td>0.296012</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.308946</td>
      <td>0.691054</td>
      <td>0.309790</td>
      <td>0.690210</td>
    </tr>
  </tbody>
</table>
</div>



## 2. 自定义pipe模块  
自需要实现一个包含了fit,transform,get_params和set_params这四个函数的类即可：  
- fit:用于拟合模块中参数
- transform:利用fit阶段拟合的参数去转换数据
- get_params/set_params:主要用于模型的持久化   

比如如下实现了一个对逐列归一化的模块


```python
from easymlops.base import PipeObject
import numpy as np
import scipy.stats as ss
class Normalization(PipeObject):
    def __init__(self, normal_range=100, normal_type="cdf", std_range=10):
        PipeObject.__init__(self)
        self.normal_range = normal_range
        self.normal_type = normal_type
        self.std_range = std_range
        self.mean_std = dict()

    def fit(self, s):
        if self.normal_type == "cdf":
            for col in s.columns:
                col_value = s[col]
                mean = np.median(col_value)
                std = np.std(col_value) * self.std_range
                self.mean_std[col] = (mean, std)
        self.input_col_names = s.columns.tolist()
        return self

    def transform(self, s):
        s_ = copy.copy(s)
        if self.normal_type == "cdf":
            for col in s_.columns:
                if col in self.mean_std:
                    s_[col] = np.round(
                        ss.norm.cdf((s_[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in s_.columns:
                if col in self.mean_std:
                    s_[col] = self.normal_range * s_[col]
        self.output_col_names = s_.columns.tolist()
        return s_

    def get_params(self):
        params = PipeObject.get_params(self)
        params.update({"mean_std": self.mean_std, "normal_range": self.normal_range, "normal_type": self.normal_type})
        return params

    def set_params(self, params):
        PipeObject.set_params(self, params)
        self.mean_std = params["mean_std"]
        self.normal_range = params["normal_range"]
        self.normal_type = params["normal_type"]
```


```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(TargetEncoding(cols=["Embarked"],y=y_train))\
  .pipe(FillNa())

x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>17.0</td>
      <td>8.6625</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>501</th>
      <td>21.0</td>
      <td>7.7500</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>502</th>
      <td>0.0</td>
      <td>7.6292</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>503</th>
      <td>37.0</td>
      <td>9.5875</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>504</th>
      <td>16.0</td>
      <td>86.5000</td>
      <td>0.334254</td>
    </tr>
  </tbody>
</table>
</div>




```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(TargetEncoding(cols=["Embarked"],y=y_train))\
  .pipe(FillNa())\
  .pipe(Normalization())
  
x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>48.41</td>
      <td>49.51</td>
      <td>50.00</td>
    </tr>
    <tr>
      <th>501</th>
      <td>49.32</td>
      <td>49.44</td>
      <td>58.35</td>
    </tr>
    <tr>
      <th>502</th>
      <td>44.56</td>
      <td>49.43</td>
      <td>58.35</td>
    </tr>
    <tr>
      <th>503</th>
      <td>52.95</td>
      <td>49.59</td>
      <td>50.00</td>
    </tr>
    <tr>
      <th>504</th>
      <td>48.18</td>
      <td>56.02</td>
      <td>50.00</td>
    </tr>
  </tbody>
</table>
</div>



## 3. 模型的分拆&组合&中间层特征抽取
PipeML的pipe对象也可以是一个PipeML，所以我们以将过长的pipeline拆分为多个pipeline，分别fit后再进行组合，避免后面流程的pipe模块更新又要重新fit前面的pipe模块

### 3.1 分拆训练


```python
ml1=PipeML()
ml1.pipe(FixInput())\
   .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
   .pipe(TransToFloat(cols=["Age","Fare"]))\
   .pipe(TransToInt(cols=["PassengerId","SibSp","Parch"]))\
   .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
   .pipe(Clip(cols=["Fare"],percent_range=(10,99),name="clip_fare"))\
   .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
   .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
   .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
   .pipe(FillNa())\
   .pipe(PCADecomposition(n_components=8))

x_train_new=ml1.fit(x_train).transform(x_train)
x_test_new=ml1.transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.113748</td>
      <td>-26.852814</td>
      <td>-5.607253</td>
      <td>-0.118244</td>
      <td>-1.098243</td>
      <td>-0.120703</td>
      <td>0.040973</td>
      <td>-0.124744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>251.119044</td>
      <td>-27.500182</td>
      <td>-1.567217</td>
      <td>-0.089340</td>
      <td>-0.022752</td>
      <td>-1.143059</td>
      <td>-0.179981</td>
      <td>-1.018557</td>
    </tr>
    <tr>
      <th>2</th>
      <td>252.020539</td>
      <td>-28.971093</td>
      <td>-22.514046</td>
      <td>-0.529986</td>
      <td>-0.020735</td>
      <td>-0.921948</td>
      <td>-0.135084</td>
      <td>-1.051066</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253.219892</td>
      <td>-24.611896</td>
      <td>14.259275</td>
      <td>0.240977</td>
      <td>-0.040018</td>
      <td>-1.309767</td>
      <td>-0.212481</td>
      <td>-1.001431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255.230488</td>
      <td>50.687678</td>
      <td>-11.883178</td>
      <td>-1.155351</td>
      <td>0.263995</td>
      <td>-0.411215</td>
      <td>-0.385816</td>
      <td>-0.610259</td>
    </tr>
  </tbody>
</table>
</div>




```python
ml2=PipeML()
ml2.pipe(LogisticRegressionClassification(y=y_train))\
   .pipe(Normalization())

x_test_new=ml2.fit(x_train_new).transform(x_test_new)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.11</td>
      <td>49.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.98</td>
      <td>50.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.91</td>
      <td>50.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.02</td>
      <td>49.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43.85</td>
      <td>56.15</td>
    </tr>
  </tbody>
</table>
</div>



### 3.2 组合


```python
ml_combine=PipeML()
ml_combine.pipe(ml1).pipe(ml2)

ml_combine.transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.11</td>
      <td>49.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.98</td>
      <td>50.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.91</td>
      <td>50.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.02</td>
      <td>49.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43.85</td>
      <td>56.15</td>
    </tr>
  </tbody>
</table>
</div>



### 3.3 组合模块的持久化  

主要是导入的时候需要按分拆的格式，逐步整合


```python
ml_combine.save("ml_combine.pkl")
```


```python
ml1=PipeML()
ml1.pipe(FixInput())\
   .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
   .pipe(TransToFloat(cols=["Age","Fare"]))\
   .pipe(TransToInt(cols=["PassengerId","SibSp","Parch"]))\
   .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
   .pipe(Clip(cols=["Fare"],percent_range=(10,99),name="clip_fare"))\
   .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
   .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
   .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"]))\
   .pipe(FillNa())\
   .pipe(PCADecomposition(n_components=8))

ml2=PipeML()
ml2.pipe(LogisticRegressionClassification(y=y_train))\
   .pipe(Normalization())

ml_combine=PipeML()
ml_combine.pipe(ml1).pipe(ml2)

ml_combine.load("ml_combine.pkl")
```


```python
ml_combine.transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>50.11</td>
      <td>49.89</td>
    </tr>
    <tr>
      <th>1</th>
      <td>49.98</td>
      <td>50.02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>49.91</td>
      <td>50.09</td>
    </tr>
    <tr>
      <th>3</th>
      <td>50.02</td>
      <td>49.98</td>
    </tr>
    <tr>
      <th>4</th>
      <td>43.85</td>
      <td>56.15</td>
    </tr>
  </tbody>
</table>
</div>



### 3.4 中间层特征抽取

我们有时候可能想看看pipeline过程中特征逐层的变化情况，以及复用别人的特征工程（但又不需要最后几步的变化），transform/transform_single中的run_to_layer就可以排上用场了


```python
x_train
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>496</td>
      <td>3</td>
      <td>Yousseff, Mr. Gerious</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>2627</td>
      <td>14.4583</td>
      <td>NaN</td>
      <td>C</td>
    </tr>
    <tr>
      <th>496</th>
      <td>497</td>
      <td>1</td>
      <td>Eustis, Miss. Elizabeth Mussey</td>
      <td>female</td>
      <td>54.0</td>
      <td>1</td>
      <td>0</td>
      <td>36947</td>
      <td>78.2667</td>
      <td>D20</td>
      <td>C</td>
    </tr>
    <tr>
      <th>497</th>
      <td>498</td>
      <td>3</td>
      <td>Shellard, Mr. Frederick William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>C.A. 6212</td>
      <td>15.1000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>498</th>
      <td>499</td>
      <td>1</td>
      <td>Allison, Mrs. Hudson J C (Bessie Waldo Daniels)</td>
      <td>female</td>
      <td>25.0</td>
      <td>1</td>
      <td>2</td>
      <td>113781</td>
      <td>151.5500</td>
      <td>C22 C26</td>
      <td>S</td>
    </tr>
    <tr>
      <th>499</th>
      <td>500</td>
      <td>3</td>
      <td>Svensson, Mr. Olof</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>0</td>
      <td>350035</td>
      <td>7.7958</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 11 columns</p>
</div>




```python
ml1=PipeML()
ml1.pipe(FixInput())\
   .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
   .pipe(TransToFloat(cols=["Age","Fare"]))\
   .pipe(TransToInt(cols=["PassengerId","SibSp","Parch"]))\
   .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
   .pipe(Clip(cols=["Fare"],percent_range=(10,99),name="clip_fare"))\
   .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
   .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
   .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
   .pipe(FillNa())\
   .pipe(PCADecomposition(n_components=8))

x_test_new=ml1.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.113748</td>
      <td>-26.852814</td>
      <td>-5.607253</td>
      <td>-0.118244</td>
      <td>-1.098243</td>
      <td>-0.120703</td>
      <td>0.040973</td>
      <td>-0.124744</td>
    </tr>
    <tr>
      <th>1</th>
      <td>251.119044</td>
      <td>-27.500182</td>
      <td>-1.567217</td>
      <td>-0.089340</td>
      <td>-0.022752</td>
      <td>-1.143059</td>
      <td>-0.179981</td>
      <td>-1.018557</td>
    </tr>
    <tr>
      <th>2</th>
      <td>252.020539</td>
      <td>-28.971093</td>
      <td>-22.514046</td>
      <td>-0.529986</td>
      <td>-0.020735</td>
      <td>-0.921948</td>
      <td>-0.135084</td>
      <td>-1.051066</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253.219892</td>
      <td>-24.611896</td>
      <td>14.259275</td>
      <td>0.240977</td>
      <td>-0.040018</td>
      <td>-1.309767</td>
      <td>-0.212481</td>
      <td>-1.001431</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255.230488</td>
      <td>50.687678</td>
      <td>-11.883178</td>
      <td>-1.155351</td>
      <td>0.263995</td>
      <td>-0.411215</td>
      <td>-0.385816</td>
      <td>-0.610259</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看第2层的数据
ml1.transform(x_test,run_to_layer=1).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>3</td>
      <td>Calic, Mr. Petar</td>
      <td>male</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>315086</td>
      <td>8.6625</td>
      <td>nan</td>
      <td>S</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>3</td>
      <td>Canavan, Miss. Mary</td>
      <td>female</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>364846</td>
      <td>7.7500</td>
      <td>nan</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>3</td>
      <td>O'Sullivan, Miss. Bridget Mary</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330909</td>
      <td>7.6292</td>
      <td>nan</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>3</td>
      <td>Laitinen, Miss. Kristina Sofia</td>
      <td>female</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>4135</td>
      <td>9.5875</td>
      <td>nan</td>
      <td>S</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>Maioni, Miss. Roberta</td>
      <td>female</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>B79</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看倒数第3层的数据
ml1.transform(x_test,run_to_layer=-3).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Pclass_3</th>
      <th>Pclass_1</th>
      <th>Pclass_2</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.317829</td>
      <td>0.334254</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.317829</td>
      <td>0.511111</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7175</td>
      <td>0.317829</td>
      <td>0.511111</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.317829</td>
      <td>0.334254</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.000000</td>
      <td>0.334254</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看到模块名为clip_fare的数据
ml1.transform(x_test,run_to_layer="clip_fare").head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>3</td>
      <td>Calic, Mr. Petar</td>
      <td>male</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>315086</td>
      <td>8.6625</td>
      <td>nan</td>
      <td>S</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>3</td>
      <td>Canavan, Miss. Mary</td>
      <td>female</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>364846</td>
      <td>7.7500</td>
      <td>nan</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>3</td>
      <td>O'Sullivan, Miss. Bridget Mary</td>
      <td>female</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330909</td>
      <td>7.7175</td>
      <td>nan</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>3</td>
      <td>Laitinen, Miss. Kristina Sofia</td>
      <td>female</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>4135</td>
      <td>9.5875</td>
      <td>nan</td>
      <td>S</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>Maioni, Miss. Roberta</td>
      <td>female</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>86.5000</td>
      <td>B79</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



## 4. 支持生产部署:数据一致性测试&性能测试&日志记录

通常，在生产线上使用pandas效率并不高，且生产的输入格式通常是字典格式(json)，所以如果需要部署生产，我们需要额外添加一个函数：  
- transform_single:实现与transform一致的功能，而input和output需要修改为字典格式  

###  4.1 transform_single



```python
ml1.transform_single({'PassengerId': 1,
 'Survived': 0,
 'Pclass': 3,
 'Name': 'Braund, Mr. Owen Harris',
 'Sex': 'male',
 'Age': 22.0,
 'SibSp': 1,
 'Parch': 0,
 'Ticket': 'A/5 21171',
 'Fare': 7.25,
 'Embarked': 'S'})
```




    {0: -249.81966525008983,
     1: -20.108644461714313,
     2: 1.272686718451494,
     3: 0.5173581256201676,
     4: -0.8759345476698472,
     5: 0.098536572656829,
     6: -0.3357057915922265,
     7: -0.0069406716383854095}




```python
ml1.transform_single({'PassengerId': 1,
 'Survived': 0,
 'Pclass': 3,
 'Name': 'Braund, Mr. Owen Harris',
 'Sex': 'male',
 'Age': 22.0,
 'SibSp': 1,
 'Parch': 0,
 'Ticket': 'A/5 21171',
 'Fare': 7.25,
 'Embarked': 'S'},run_to_layer=-3)
```




    {'PassengerId': 1,
     'Pclass': 1,
     'Name': 0,
     'Sex': 1,
     'Age': 22.0,
     'SibSp': 1,
     'Parch': 0,
     'Ticket': 0.0,
     'Fare': 7.7175,
     'Cabin': 0.3178294573643411,
     'Embarked': 0.3342541436464088,
     'Pclass_3': 1,
     'Pclass_1': 0,
     'Pclass_2': 0,
     'Sex_male': 1,
     'Sex_female': 0}



### 4.2 自定义pipe模块
接着为前面的Normalization类再加一个transform_single函数


```python
class NormalizationExtend(Normalization):
    def transform_single(self, s):
        s_ = copy.copy(s)
        if self.normal_type == "cdf":
            for col in s_.keys():
                if col in self.mean_std:
                    s_[col] = np.round(
                        ss.norm.cdf((s_[col] - self.mean_std[col][0]) / self.mean_std[col][1]) * self.normal_range, 2)
        elif self.normal_type == "range":
            for col in s_.keys():
                if col in self.mean_std:
                    s_[col] = self.normal_range * s_[col]
        return s_
```


```python
#重新训练前面组合模型中的第二个模型
ml2=PipeML()
ml2.pipe(LogisticRegressionClassification(y=y_train))\
   .pipe(NormalizationExtend())

ml2.fit(x_train_new)
```




    <easymlops.pipeml.PipeML at 0x1f015ebf358>




```python
ml_combine=PipeML()
ml_combine.pipe(ml1).pipe(ml2)

ml_combine.transform_single({'PassengerId': 1,
 'Survived': 0,
 'Pclass': 3,
 'Name': 'Braund, Mr. Owen Harris',
 'Sex': 'male',
 'Age': 22.0,
 'SibSp': 1,
 'Parch': 0,
 'Ticket': 'A/5 21171',
 'Fare': 7.25,
 'Embarked': 'S'})
```




    {0: 50.11, 1: 49.89}



### 4.3 数据一致性测试&性能测试
部署生产环境之前，我们通常要关注两点：  
- 离线训练模型和在线预测模型的一致性，即tranform和transform_single的一致性；  
- transform_single对当条数据的预测性能  

这些可以通过调用如下函数，进行自动化测试：  
- auto_check_transform：只要有打印[success]，则表示一致性测试通过，性能测试表示为[*]毫秒/每条数据，如果有异常则会直接抛出，并中断后续pipe模块的测试


```python
ml1.auto_check_transform(x_test)
```

    (<class 'easymlops.ml.preprocessing.FixInput'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.preprocessing.TransToCategory'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.preprocessing.TransToFloat'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.preprocessing.TransToInt'>)  module transform check [success], single transform speed:[0.01]ms/it
    (clip_name)  module transform check [success], single transform speed:[0.01]ms/it
    (clip_fare)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.representation.OneHotEncoding'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.representation.LabelEncoding'>)  module transform check [success], single transform speed:[0.0]ms/it
    (<class 'easymlops.ml.representation.TargetEncoding'>)  module transform check [success], single transform speed:[0.0]ms/it
    (<class 'easymlops.ml.preprocessing.FillNa'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.representation.PCADecomposition'>)  module transform check [success], single transform speed:[2.1]ms/it
    

### 4.4 日志记录 
日志通常只需要在生产中使用，所以只在transform_single可用


```python
import logging
logger=logging.getLogger()#logging的具体使用方法还请另行查资料
```


```python
extend_log_info={"user_id":1,"time":"2023-01-12 15:04:32"}
```


```python
ml1.transform_single({'PassengerId': 1,
 'Survived': 0,
 'Pclass': 3,
 'Name': 'Braund, Mr. Owen Harris',
 'Sex': 'male',
 'Age': 22.0,
 'SibSp': 1,
 'Parch': 0,
 'Ticket': 'A/5 21171',
 'Fare': 7.25,
 'Embarked': 'S'},logger=logger,log_base_dict=extend_log_info)
```




    {0: -249.81966525008983,
     1: -20.108644461714313,
     2: 1.272686718451494,
     3: 0.5173581256201676,
     4: -0.8759345476698472,
     5: 0.098536572656829,
     6: -0.3357057915922265,
     7: -0.0069406716383854095}



## 5. 训练性能优化
主要是优化内存使用情况，下面看一个比较特殊点的(特征OneHot展开)


```python
from easymlops.ml.perfopt import *

ml=PipeML()
ml.pipe(FixInput())\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(OneHotEncoding(cols=["Pclass","Sex","Name","Ticket","Embarked","Cabin"],drop_col=True))\
  .pipe(FillNa())\
  .pipe(ReduceMemUsage())\
  .pipe(Dense2Sparse())

ml.fit(x_train).transform(x_train).shape
```




    (500, 1021)




```python
#不做优化时的内存消耗:3988K
ml.transform(x_train,run_to_layer=-3).memory_usage().sum()//1024
```




    3988




```python
#做了ReduceMemUsage后的内存消耗:500K(整体下降87%)
ml.transform(x_train,run_to_layer=-2).memory_usage().sum()//1024
```




    500




```python
#做了ReduceMemUsage和后的内存消耗:24K(整体下降99%)
ml.transform(x_train,run_to_layer=-1).memory_usage().sum()//1024
```




    24



easymlops.ml.classification中的模块对Dense2Sparse基本都支持，比如LightGBM


```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(OneHotEncoding(cols=["Pclass","Sex","Name","Ticket","Embarked","Cabin"],drop_col=True))\
  .pipe(FillNa())\
  .pipe(ReduceMemUsage())\
  .pipe(Dense2Sparse())\
  .pipe(LGBMClassification(y=y_train))
```




    <easymlops.pipeml.PipeML at 0x1f03b57a898>




```python
ml.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.959436</td>
      <td>0.040564</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.043707</td>
      <td>0.956293</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.054812</td>
      <td>0.945188</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.431609</td>
      <td>0.568391</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.141188</td>
      <td>0.858812</td>
    </tr>
  </tbody>
</table>
</div>



## 6. 文本NLP处理  

### 6.1 文本清洗


```python
from easymlops.nlp.preprocessing import *

#先将Name以外的特征数值化
nlp=PipeML()
nlp.pipe(FixInput())\
   .pipe(TargetEncoding(cols=["Sex","Ticket","Cabin","Embarked"],y=y_train))\
   .pipe(FillNa())

nlp.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>3</td>
      <td>Calic, Mr. Petar</td>
      <td>0.171429</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.0</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>3</td>
      <td>Canavan, Miss. Mary</td>
      <td>0.751351</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.0</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>3</td>
      <td>O'Sullivan, Miss. Bridget Mary</td>
      <td>0.751351</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.6292</td>
      <td>0.0</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>3</td>
      <td>Laitinen, Miss. Kristina Sofia</td>
      <td>0.751351</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.0</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>Maioni, Miss. Roberta</td>
      <td>0.751351</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.0</td>
      <td>0.334254</td>
    </tr>
  </tbody>
</table>
</div>




```python
#接下来把Name中所有字符转小写，然后将所有标点符号用空格代替
nlp=PipeML()
nlp.pipe(FixInput())\
   .pipe(TargetEncoding(cols=["Sex","Ticket","Cabin","Embarked"],y=y_train))\
   .pipe(FillNa())\
   .pipe(Lower(cols=["Name"]))\
   .pipe(ReplacePunctuation(cols=["Name"],symbols=" "))

nlp.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>3</td>
      <td>calic  mr  petar</td>
      <td>0.171429</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.0</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>3</td>
      <td>canavan  miss  mary</td>
      <td>0.751351</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.0</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>3</td>
      <td>o sullivan  miss  bridget mary</td>
      <td>0.751351</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.6292</td>
      <td>0.0</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>3</td>
      <td>laitinen  miss  kristina sofia</td>
      <td>0.751351</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.0</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>maioni  miss  roberta</td>
      <td>0.751351</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.0</td>
      <td>0.334254</td>
    </tr>
  </tbody>
</table>
</div>



### 6.2 文本特征表示  
注意：后续模型处理的最小单位需要在原column中以空格分隔，比如上面第一行"calic mr petar"会分别把"calic","mr","petar"当作独立的词处理


```python
from easymlops.nlp.representation import *
#构建tfidf模型
nlp=PipeML()
nlp.pipe(FixInput())\
   .pipe(TargetEncoding(cols=["Sex","Ticket","Cabin","Embarked"],y=y_train))\
   .pipe(FillNa())\
   .pipe(Lower(cols=["Name"]))\
   .pipe(ReplacePunctuation(cols=["Name"],symbols=" "))\
   .pipe(TFIDF(cols=["Name"]))\
   .pipe(DropCols(cols=["Name"]))

nlp.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>...</th>
      <th>tfidf_Name_yarred</th>
      <th>tfidf_Name_yoto</th>
      <th>tfidf_Name_young</th>
      <th>tfidf_Name_youseff</th>
      <th>tfidf_Name_yousif</th>
      <th>tfidf_Name_youssef</th>
      <th>tfidf_Name_yousseff</th>
      <th>tfidf_Name_yrois</th>
      <th>tfidf_Name_zabour</th>
      <th>tfidf_Name_zimmerman</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501</td>
      <td>3</td>
      <td>0.171429</td>
      <td>17.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.0</td>
      <td>0.334254</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502</td>
      <td>3</td>
      <td>0.751351</td>
      <td>21.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.0</td>
      <td>0.511111</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503</td>
      <td>3</td>
      <td>0.751351</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>7.6292</td>
      <td>0.0</td>
      <td>0.511111</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504</td>
      <td>3</td>
      <td>0.751351</td>
      <td>37.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.0</td>
      <td>0.334254</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505</td>
      <td>1</td>
      <td>0.751351</td>
      <td>16.0</td>
      <td>0</td>
      <td>0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.0</td>
      <td>0.334254</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 994 columns</p>
</div>




```python
#构建lsi主题模型+word2vec词向量模型
nlp=PipeML()
nlp.pipe(FixInput())\
   .pipe(TargetEncoding(cols=["Sex","Ticket","Cabin","Embarked"],y=y_train))\
   .pipe(FillNa())\
   .pipe(Lower(cols=["Name"]))\
   .pipe(ReplacePunctuation(cols=["Name"],symbols=" "))\
   .pipe(SelectCols(cols=["Name"]))\
   .pipe(Parallel([LsiTopicModel(cols=["Name"],num_topics=4),Word2VecModel(embedding_size=4,cols=["Name"])]))\
   .pipe(DropCols(cols=["Name"]))

nlp.fit(x_train).transform(x_test).head(5)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>lsi_Name_0</th>
      <th>lsi_Name_1</th>
      <th>lsi_Name_2</th>
      <th>lsi_Name_3</th>
      <th>w2v_Name_0</th>
      <th>w2v_Name_1</th>
      <th>w2v_Name_2</th>
      <th>w2v_Name_3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>2.132522</td>
      <td>0.648002</td>
      <td>-0.075903</td>
      <td>-0.093426</td>
      <td>-0.183208</td>
      <td>-0.017150</td>
      <td>0.403337</td>
      <td>0.551337</td>
    </tr>
    <tr>
      <th>501</th>
      <td>2.034848</td>
      <td>-0.608232</td>
      <td>-0.733353</td>
      <td>-0.013251</td>
      <td>-0.085437</td>
      <td>-0.033206</td>
      <td>0.293308</td>
      <td>0.416924</td>
    </tr>
    <tr>
      <th>502</th>
      <td>2.040231</td>
      <td>-0.616313</td>
      <td>-0.747628</td>
      <td>-0.022868</td>
      <td>-0.051665</td>
      <td>0.000399</td>
      <td>0.277010</td>
      <td>0.309539</td>
    </tr>
    <tr>
      <th>503</th>
      <td>2.026293</td>
      <td>-0.579113</td>
      <td>-0.735068</td>
      <td>0.012226</td>
      <td>-0.140556</td>
      <td>0.009208</td>
      <td>0.393155</td>
      <td>0.443419</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2.025096</td>
      <td>-0.573417</td>
      <td>-0.720064</td>
      <td>0.010258</td>
      <td>-0.140556</td>
      <td>0.009208</td>
      <td>0.393155</td>
      <td>0.443419</td>
    </tr>
  </tbody>
</table>
</div>




```python
nlp.auto_check_transform(x_test)
```

    (<class 'easymlops.ml.preprocessing.FixInput'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.representation.TargetEncoding'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.preprocessing.FillNa'>)  module transform check [success], single transform speed:[0.02]ms/it
    (<class 'easymlops.nlp.preprocessing.Lower'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.nlp.preprocessing.ReplacePunctuation'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.preprocessing.SelectCols'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.ensemble.Parallel'>)  module transform check [success], single transform speed:[5.4]ms/it
    (<class 'easymlops.ml.preprocessing.DropCols'>)  module transform check [success], single transform speed:[0.0]ms/it
    

## TODO  

- 加入更多通用的数据清洗、特征工程模块，比如WOEEncoding


```python

```
