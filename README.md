# EasyMLOps  
  
## 介绍   
`EasyMLOps`包以`Pipline`的方式构建建模任务，可直接进行模型训练、预测(离线，在线)，测试(离线在线预测一致性、预测性能)等功能，通过外套一层Flask或FastApi即可直接部署生产，目前主要功能有：

### 1. 基础建模模块

- 数据清洗，数据自动填充、转换、盖帽、归一化、分箱等：easymlops.ml.preprocessing
- 特征处理:
  - 特征编码，包括Target、Label、Onehot Encoding、WOEEncoding等：easymlops.ml.encoding
  - 特征降维，包括PCA、NFM等：easymlops.ml.decomposition 
  - 特征选择:easymlops.ml.feature_selection
    - 过滤式：包括饱和度、方差、相关性、卡方、P-value、互信息、IV、PSI等  
    - 嵌入式：包括LR、LightGBM等
- 分类模型，包括lgbm决策树、logistic回归、svm等传统机器学习模型：easymlops.ml.classification 
- stacking，通过Parallel模块，可以在同一阶段进行多个模型的训练，这样可以很方面的构建stacking模型：easymlops.ensemble.Parallel

### 2. 文本NLP处理模块
- 文本清洗，包括去停用词，去标点符号，去特定字符，抽取中文字符，jieba中文分词，关键词提取、ngram特征提取等数据清洗操作：easymlops.nlp.preprocessing
- 特征提取，包括bow,tfidf等传统模型；lda,lsi等主题模型；fastext,word2vec,doc2vec等词向量模型：easymlops.nlp.representation

### 3. 训练性能优化模块（主要是减少内存占用）

- easymlops.ml.perfopt.ReduceMemUsage模块:修改数据类型，比如某列特征数据范围在float16内，而目前的数据类型是float64，则将float64修改为float16
- easymlops.ml.perfopt.Dense2Sparse模块:将稠密矩阵转换为稀疏矩阵（含0量很多时使用），注意后续的pipe模块要提供对稀疏矩阵的支持(easymlops.ml.classification下的模块基本都支持) 

### 4. Pipeline流程的分拆&组合&运行到指定层&中间层pipe模块获取  

- pipeml的子模块也可以是pipeml，这样方便逐块建模再组合
- pipeml可以提取中间层数据，方便复用别人的模型，继续做自己下一步工作:pipeobj.transform(data,run_to_layer=指定层数或模块名) 
- 获取指定pipe模块的两种方式

### 5.pipeline流程的训练&预测&持久化

- 训练接口：fit
- 预测接口：transform/transform_single分别进行批量预测和单条数据预测
- 持久化：save/load

### 6. 自定义pipe模块及其接口扩展

- fit,tranform：最少只需实现这两函数即可接入pipeline中
- set_params,get_params:实现这两函数可以对模块持久化
- transform_single:支持生产预测  
- 扩展自定义函数接口及其调用方式  
- 进阶函数接口:_fit,_transform,_transofrm_single,_set_params,_get_params


### 7. 生产部署:数据一致性测试&性能测试&日志记录

- pipeobj.transform_single(data)即可对生产数据(通常转换为dict)进行预测
- pipeobj.auto_check_transform(data)可以对数据一致性以及各个pipe模块性能做测试  
- pipeobj.transform_single(data,logger)可以追踪记录pipeline预测中每一步信息  



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

## 1. 基础建模模块 

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
  .pipe(FillNa(cols=["Cabin","Ticket","Parch","Fare"],fill_mode="mode"))\
  .pipe(FillNa(cols=["Age"],fill_mode="mean"))\
  .pipe(FillNa(fill_detail={"Embarked":"N"}))\
  .pipe(TransToCategory(cols=["Cabin","Embarked"]))\
  .pipe(TransToFloat(cols=["Age","Fare"]))\
  .pipe(TransToInt(cols=["Pclass","PassengerId","Survived","SibSp","Parch"]))\
  .pipe(TransToLower(cols=["Ticket","Cabin","Embarked","Name","Sex"]))\
  .pipe(CategoryMapValues(map_detail={"Cabin":(["nan","NaN"],"n")}))\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(Clip(cols=["Fare"],percent_range=(1,99),name="clip_fare"))\
  .pipe(MinMaxScaler(cols=[("Age","Age_minmax")]))\
  .pipe(Normalizer(cols=[("Fare","Fare_normal")]))\
  .pipe(Bins(n_bins=10,strategy="uniform",cols=[("Age","Age_uni")]))\
  .pipe(Bins(n_bins=10,strategy="quantile",cols=[("Age","Age_quan")]))\
  .pipe(Bins(n_bins=10,strategy="kmeans",cols=[("Fare","Fare_km")]))

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
      <th>Age_minmax</th>
      <th>Fare_normal</th>
      <th>Age_uni</th>
      <th>Age_quan</th>
      <th>Fare_km</th>
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
      <td>n</td>
      <td>s</td>
      <td>0.228571</td>
      <td>-0.523659</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
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
      <td>n</td>
      <td>q</td>
      <td>0.285714</td>
      <td>-0.544624</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
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
      <td>n</td>
      <td>q</td>
      <td>0.402925</td>
      <td>-0.545371</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
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
      <td>n</td>
      <td>s</td>
      <td>0.514286</td>
      <td>-0.502407</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>0.0</td>
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
      <td>0.214286</td>
      <td>1.264706</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>4.0</td>
    </tr>
  </tbody>
</table>
</div>



### 1.2 特征处理

#### 1.2.1 特征编码


```python
from easymlops.ml.encoding import *
```


```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Sex"],drop_col=False))\
  .pipe(LabelEncoding(cols=["Sex"]))\
  .pipe(TargetEncoding(cols=["Name","Ticket"],y=y_train))\
  .pipe(WOEEncoding(cols=["Pclass","Embarked","Cabin"],y=y_train,name="woe"))\
  .pipe(FillNa())

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
      <th>Sex_male</th>
      <th>Sex_female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>1</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.299607</td>
      <td>0.224849</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>2</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.299607</td>
      <td>-0.508609</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.6292</td>
      <td>0.299607</td>
      <td>-0.508609</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>2</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.299607</td>
      <td>0.224849</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505.0</td>
      <td>-0.741789</td>
      <td>0</td>
      <td>2</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.000000</td>
      <td>0.224849</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看woe encoding层的详情
ml["woe"].show_detail().head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>bin_value</th>
      <th>bad_num</th>
      <th>bad_rate</th>
      <th>good_num</th>
      <th>good_rate</th>
      <th>woe</th>
      <th>iv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pclass</td>
      <td>3.0</td>
      <td>78</td>
      <td>0.404145</td>
      <td>201</td>
      <td>0.654723</td>
      <td>0.482439</td>
      <td>0.120889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pclass</td>
      <td>1.0</td>
      <td>66</td>
      <td>0.341969</td>
      <td>50</td>
      <td>0.162866</td>
      <td>-0.741789</td>
      <td>0.132856</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass</td>
      <td>2.0</td>
      <td>49</td>
      <td>0.253886</td>
      <td>56</td>
      <td>0.182410</td>
      <td>-0.330626</td>
      <td>0.023632</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Embarked</td>
      <td>S</td>
      <td>121</td>
      <td>0.626943</td>
      <td>241</td>
      <td>0.785016</td>
      <td>0.224849</td>
      <td>0.035543</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Embarked</td>
      <td>C</td>
      <td>48</td>
      <td>0.248705</td>
      <td>44</td>
      <td>0.143322</td>
      <td>-0.551169</td>
      <td>0.058083</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.2.2 特征降维


```python
from easymlops.ml.decomposition import *

ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
  .pipe(PCADecomposition(n_components=4))

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
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>250.112231</td>
      <td>-27.096138</td>
      <td>-5.834393</td>
      <td>-0.180067</td>
    </tr>
    <tr>
      <th>1</th>
      <td>251.117645</td>
      <td>-27.780886</td>
      <td>-1.800322</td>
      <td>-0.126839</td>
    </tr>
    <tr>
      <th>2</th>
      <td>252.017857</td>
      <td>-29.148327</td>
      <td>-22.754604</td>
      <td>-0.559045</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253.218297</td>
      <td>-25.037062</td>
      <td>14.051899</td>
      <td>0.198425</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255.219620</td>
      <td>50.498192</td>
      <td>-11.402204</td>
      <td>-1.090810</td>
    </tr>
  </tbody>
</table>
</div>



#### 1.2.3 特征选择
##### 过滤式


```python
from easymlops.ml.feature_selection import *
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(MissRateFilter(max_threshold=0.1))\
  .pipe(VarianceFilter(min_threshold=0.1))\
  .pipe(PersonCorrFilter(min_threshold=0.1,y=y_train,name="person"))\
  .pipe(PSIFilter(oot_x=x_test,cols=["Pclass","Sex","Embarked"],name="psi",max_threshold=0.5))\
  .pipe(LabelEncoding(cols=["Sex","Ticket","Embarked","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Cabin"],y=y_train))\
  .pipe(Chi2Filter(y=y_train,name="chi2"))\
  .pipe(MutualInfoFilter(y=y_train))\
  .pipe(IVFilter(y=y_train,name="iv",cols=["Sex","Fare"],min_threshold=0.05))

x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0.317829</td>
      <td>1</td>
    </tr>
    <tr>
      <th>501</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7.7500</td>
      <td>0.317829</td>
      <td>3</td>
    </tr>
    <tr>
      <th>502</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>7.6292</td>
      <td>0.317829</td>
      <td>3</td>
    </tr>
    <tr>
      <th>503</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>9.5875</td>
      <td>0.317829</td>
      <td>1</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>354</td>
      <td>86.5000</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看psi计算详情
ml["psi"].show_detail().head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>bin_value</th>
      <th>ins_num</th>
      <th>ins_rate</th>
      <th>oot_num</th>
      <th>oot_rate</th>
      <th>psi</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Pclass</td>
      <td>3.0</td>
      <td>279</td>
      <td>0.558</td>
      <td>212</td>
      <td>0.542199</td>
      <td>0.000454</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Pclass</td>
      <td>1.0</td>
      <td>116</td>
      <td>0.232</td>
      <td>100</td>
      <td>0.255754</td>
      <td>0.002316</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass</td>
      <td>2.0</td>
      <td>105</td>
      <td>0.210</td>
      <td>79</td>
      <td>0.202046</td>
      <td>0.000307</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sex</td>
      <td>male</td>
      <td>315</td>
      <td>0.630</td>
      <td>262</td>
      <td>0.670077</td>
      <td>0.002472</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Sex</td>
      <td>female</td>
      <td>185</td>
      <td>0.370</td>
      <td>129</td>
      <td>0.329923</td>
      <td>0.004595</td>
    </tr>
  </tbody>
</table>
</div>



##### 嵌入式


```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(LabelEncoding(cols=["Sex","Ticket","Embarked","Pclass"]))\
  .pipe(TargetEncoding(cols=["Name","Cabin"],y=y_train))\
  .pipe(LREmbed(y=y_train,min_threshold=0.01))\
  .pipe(LGBMEmbed(y=y_train,min_threshold=0.01))
  

x_test_new=ml.fit(x_train).transform(x_test)
x_test_new.head(5)
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.317829</td>
      <td>1</td>
    </tr>
    <tr>
      <th>501</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.317829</td>
      <td>3</td>
    </tr>
    <tr>
      <th>502</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.317829</td>
      <td>3</td>
    </tr>
    <tr>
      <th>503</th>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.317829</td>
      <td>1</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看LR权重分布
ml[-2].show_detail()
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
      <td>0.000114</td>
      <td>0.201071</td>
      <td>6.509879</td>
      <td>1.03353</td>
      <td>0.007445</td>
      <td>0.165752</td>
      <td>0.042254</td>
      <td>0.000275</td>
      <td>0.003045</td>
      <td>0.91086</td>
      <td>0.158658</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看Lgbm中split次数(默认)分布
ml[-1].show_detail()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>81</td>
      <td>256</td>
      <td>38</td>
      <td>32</td>
      <td>29</td>
      <td>9</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



### 1.3 分类模型


```python
from easymlops.ml.classification import *

ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa())\
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



### 1.4 Stacking建模


```python
from easymlops.ml.ensemble import Parallel
ml = PipeML()
ml.pipe(FixInput()) \
  .pipe(FillNa()) \
  .pipe(Parallel([OneHotEncoding(cols=["Pclass", "Sex"]), LabelEncoding(cols=[("Sex","Sex_label"), ("Pclass","Pclass_label")]),
                    TargetEncoding(cols=["Name", "Ticket", "Embarked", "Cabin", "Sex"], y=y_train)])) \
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
      <td>250.111868</td>
      <td>-27.103302</td>
      <td>6.209714</td>
      <td>0.178558</td>
    </tr>
    <tr>
      <th>1</th>
      <td>251.117271</td>
      <td>-27.787295</td>
      <td>6.225051</td>
      <td>0.168702</td>
    </tr>
    <tr>
      <th>2</th>
      <td>252.017480</td>
      <td>-29.155072</td>
      <td>6.222640</td>
      <td>0.096320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253.217925</td>
      <td>-25.043321</td>
      <td>6.260848</td>
      <td>0.267476</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255.220248</td>
      <td>50.507856</td>
      <td>6.249152</td>
      <td>2.144759</td>
    </tr>
  </tbody>
</table>
</div>




```python
ml = PipeML()
ml.pipe(FixInput()) \
  .pipe(FillNa()) \
  .pipe(Parallel([OneHotEncoding(cols=["Pclass", "Sex"]), LabelEncoding(cols=["Sex", "Pclass"]),
                    TargetEncoding(cols=["Name", "Ticket", "Embarked", "Cabin", "Sex"], y=y_train)])) \
  .pipe(Parallel([PCADecomposition(n_components=2, prefix="pca"), NMFDecomposition(n_components=2, prefix="nmf")]))\
  .pipe(Parallel([LGBMClassification(y=y_train, prefix="lgbm"), LogisticRegressionClassification(y=y_train, prefix="lr")]))

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
      <td>0.964262</td>
      <td>0.035738</td>
      <td>0.649694</td>
      <td>0.350306</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.981094</td>
      <td>0.018906</td>
      <td>0.653345</td>
      <td>0.346655</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.987265</td>
      <td>0.012735</td>
      <td>0.645544</td>
      <td>0.354456</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.866328</td>
      <td>0.133672</td>
      <td>0.654837</td>
      <td>0.345163</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.301595</td>
      <td>0.698405</td>
      <td>0.445869</td>
      <td>0.554131</td>
    </tr>
  </tbody>
</table>
</div>



## 2. 文本NLP处理模块
目前主要包含文本清洗和文本特征抽取
### 2.1 文本清洗


```python
#Name中所有字符转小写，然后将所有标点符号用空格代替
from easymlops.nlp.preprocessing import *
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
      <td>501.0</td>
      <td>3.0</td>
      <td>calic  mr  petar</td>
      <td>0.171429</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.6625</td>
      <td>0.317829</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502.0</td>
      <td>3.0</td>
      <td>canavan  miss  mary</td>
      <td>0.751351</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.7500</td>
      <td>0.317829</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503.0</td>
      <td>3.0</td>
      <td>o sullivan  miss  bridget mary</td>
      <td>0.751351</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>7.6292</td>
      <td>0.317829</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504.0</td>
      <td>3.0</td>
      <td>laitinen  miss  kristina sofia</td>
      <td>0.751351</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>9.5875</td>
      <td>0.317829</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505.0</td>
      <td>1.0</td>
      <td>maioni  miss  roberta</td>
      <td>0.751351</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>86.5000</td>
      <td>0.000000</td>
      <td>0.334254</td>
    </tr>
  </tbody>
</table>
</div>



### 2.2 文本特征提取  
注意：后续模型处理的最小单位需要在原column中以空格分隔，比如上面第一行"calic mr petar"会分别把"calic","mr","petar"当作独立的词处理


```python
from easymlops.nlp.representation import *
#构建tfidf模型
nlp=PipeML()
nlp.pipe(FixInput())\
   .pipe(SelectCols(cols=["Name","Age"]))\
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
      <th>Age</th>
      <th>tfidf_Name_</th>
      <th>tfidf_Name_a</th>
      <th>tfidf_Name_abbott</th>
      <th>tfidf_Name_abelson</th>
      <th>tfidf_Name_achem</th>
      <th>tfidf_Name_achille</th>
      <th>tfidf_Name_achilles</th>
      <th>tfidf_Name_ada</th>
      <th>tfidf_Name_adahl</th>
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
      <td>17.0</td>
      <td>0.285612</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>21.0</td>
      <td>0.346207</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>0.0</td>
      <td>0.194849</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>37.0</td>
      <td>0.289940</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
      <td>16.0</td>
      <td>0.627690</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
<p>5 rows × 985 columns</p>
</div>




```python
#构建lsi主题模型+word2vec词向量模型
nlp=PipeML()
nlp.pipe(FixInput())\
   .pipe(SelectCols(cols=["Name"]))\
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
      <td>0.648004</td>
      <td>-0.075895</td>
      <td>-0.093468</td>
      <td>-0.183208</td>
      <td>-0.017150</td>
      <td>0.403337</td>
      <td>0.551337</td>
    </tr>
    <tr>
      <th>501</th>
      <td>2.034848</td>
      <td>-0.608228</td>
      <td>-0.733369</td>
      <td>-0.013220</td>
      <td>-0.085437</td>
      <td>-0.033206</td>
      <td>0.293308</td>
      <td>0.416924</td>
    </tr>
    <tr>
      <th>502</th>
      <td>2.040231</td>
      <td>-0.616304</td>
      <td>-0.747661</td>
      <td>-0.022293</td>
      <td>-0.051665</td>
      <td>0.000399</td>
      <td>0.277010</td>
      <td>0.309539</td>
    </tr>
    <tr>
      <th>503</th>
      <td>2.026293</td>
      <td>-0.579115</td>
      <td>-0.735073</td>
      <td>0.012305</td>
      <td>-0.140556</td>
      <td>0.009208</td>
      <td>0.393155</td>
      <td>0.443419</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2.025096</td>
      <td>-0.573417</td>
      <td>-0.720071</td>
      <td>0.010259</td>
      <td>-0.140556</td>
      <td>0.009208</td>
      <td>0.393155</td>
      <td>0.443419</td>
    </tr>
  </tbody>
</table>
</div>



## 3. 训练性能优化模块
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




    501




```python
#做了ReduceMemUsage和后的内存消耗:24K(整体下降99%)
ml.transform(x_train,run_to_layer=-1).memory_usage().sum()//1024
```




    24




```python
#easymlops.ml.classification中的模块对Dense2Sparse基本都支持，比如LightGBM
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(Clip(cols=["Age"],default_clip=(1,99),name="clip_name"))\
  .pipe(OneHotEncoding(cols=["Pclass","Sex","Name","Ticket","Embarked","Cabin"],drop_col=True))\
  .pipe(FillNa())\
  .pipe(ReduceMemUsage())\
  .pipe(Dense2Sparse())\
  .pipe(LGBMClassification(y=y_train))

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



## 4. Pipeline流程的分拆&组合&运行到指定层&中间层pipe模块获取
- 建模的过程通常式逐步迭代的，在每一层可能都要做多次调整，再进行下一步建模，但按照上面的建模方式，每次调整了最后一层，都要将前面的所有层再次运行一次，这样很费时费力；
- 所以如果能将整个pipeline分拆成为多个子pipeline然后再组合，方面迭代式的建模开发；
- 所以，这里将PipeML对象也设计为了一个pipe模块(即PipeML和上面介绍的FixInput、FillNa、TargetEncoding...等可以视为同级),所以PipeML也可以pipe一个PipeML对象

### 4.1 分拆


```python
#比如先做特征工程
ml1=PipeML()
ml1.pipe(FixInput())\
   .pipe(FillNa())\
   .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
   .pipe(LabelEncoding(cols=["Sex","Pclass"]))\
   .pipe(TargetEncoding(cols=["Name","Ticket","Embarked","Cabin"],y=y_train))\
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
      <td>250.112231</td>
      <td>-27.096138</td>
      <td>-5.834393</td>
      <td>-0.180067</td>
      <td>-1.124374</td>
      <td>-0.103224</td>
      <td>0.040820</td>
      <td>-0.131174</td>
    </tr>
    <tr>
      <th>1</th>
      <td>251.117645</td>
      <td>-27.780886</td>
      <td>-1.800322</td>
      <td>-0.126839</td>
      <td>-0.081313</td>
      <td>-1.153031</td>
      <td>-0.191893</td>
      <td>-1.016990</td>
    </tr>
    <tr>
      <th>2</th>
      <td>252.017857</td>
      <td>-29.148327</td>
      <td>-22.754604</td>
      <td>-0.559045</td>
      <td>-0.052818</td>
      <td>-0.926988</td>
      <td>-0.143329</td>
      <td>-1.040929</td>
    </tr>
    <tr>
      <th>3</th>
      <td>253.218297</td>
      <td>-25.037062</td>
      <td>14.051899</td>
      <td>0.198425</td>
      <td>-0.116402</td>
      <td>-1.322741</td>
      <td>-0.226845</td>
      <td>-1.006329</td>
    </tr>
    <tr>
      <th>4</th>
      <td>255.219620</td>
      <td>50.498192</td>
      <td>-11.402204</td>
      <td>-1.090810</td>
      <td>0.354341</td>
      <td>-0.394043</td>
      <td>-0.365711</td>
      <td>-0.548342</td>
    </tr>
  </tbody>
</table>
</div>




```python
#然后模型训练
ml2=PipeML().pipe(LogisticRegressionClassification(y=y_train))
ml2.fit(x_train_new).transform(x_test_new).head(5)
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
      <td>0.996235</td>
      <td>0.003765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.979202</td>
      <td>0.020798</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.970884</td>
      <td>0.029116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.984536</td>
      <td>0.015464</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.204515</td>
      <td>0.795485</td>
    </tr>
  </tbody>
</table>
</div>



### 4.2 组合


```python
ml_combine=PipeML().pipe(ml1).pipe(ml2)
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
      <td>0.996235</td>
      <td>0.003765</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.979202</td>
      <td>0.020798</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.970884</td>
      <td>0.029116</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.984536</td>
      <td>0.015464</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.204515</td>
      <td>0.795485</td>
    </tr>
  </tbody>
</table>
</div>



### 4.3 运行到指定层
我们有时候可能想看看pipeline过程中特征逐层的变化情况，以及复用别人的特征工程（但又不需要最后几步的变化），transform/transform_single中的run_to_layer就可以排上用场了


```python
ml=PipeML()
ml.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(WOEEncoding(cols=["Sex","Pclass"],y=y_train))\
  .pipe(LabelEncoding(cols=["Name","Ticket"]))\
  .pipe(TargetEncoding(cols=["Embarked","Cabin"],y=y_train,name="target_encoding"))\
  .pipe(FillNa())\
  .pipe(PCADecomposition(n_components=8))

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
      <td>-395.474344</td>
      <td>16.246479</td>
      <td>-12.526273</td>
      <td>-39.863262</td>
      <td>-3.525237</td>
      <td>1.171257</td>
      <td>-0.642503</td>
      <td>0.425204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-396.034758</td>
      <td>17.057039</td>
      <td>-12.883940</td>
      <td>-40.405460</td>
      <td>0.505058</td>
      <td>-1.880799</td>
      <td>-0.632594</td>
      <td>0.421864</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-396.602256</td>
      <td>17.387752</td>
      <td>-13.794505</td>
      <td>-42.144470</td>
      <td>-20.413251</td>
      <td>-1.752624</td>
      <td>-0.926964</td>
      <td>0.291678</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-397.322960</td>
      <td>18.434067</td>
      <td>-13.508230</td>
      <td>-37.263258</td>
      <td>16.299650</td>
      <td>-1.968569</td>
      <td>-0.408951</td>
      <td>0.530000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-238.496005</td>
      <td>274.260464</td>
      <td>150.607307</td>
      <td>75.272168</td>
      <td>-21.145699</td>
      <td>-1.725589</td>
      <td>0.589471</td>
      <td>-0.513614</td>
    </tr>
  </tbody>
</table>
</div>




```python
#查看第2层的数据
ml.transform(x_test,run_to_layer=1).head(5)
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
      <td>501.0</td>
      <td>3.0</td>
      <td>Calic, Mr. Petar</td>
      <td>male</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>315086</td>
      <td>8.6625</td>
      <td>nan</td>
      <td>S</td>
    </tr>
    <tr>
      <th>501</th>
      <td>502.0</td>
      <td>3.0</td>
      <td>Canavan, Miss. Mary</td>
      <td>female</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>364846</td>
      <td>7.7500</td>
      <td>nan</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>502</th>
      <td>503.0</td>
      <td>3.0</td>
      <td>O'Sullivan, Miss. Bridget Mary</td>
      <td>female</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>330909</td>
      <td>7.6292</td>
      <td>nan</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>503</th>
      <td>504.0</td>
      <td>3.0</td>
      <td>Laitinen, Miss. Kristina Sofia</td>
      <td>female</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4135</td>
      <td>9.5875</td>
      <td>nan</td>
      <td>S</td>
    </tr>
    <tr>
      <th>504</th>
      <td>505.0</td>
      <td>1.0</td>
      <td>Maioni, Miss. Roberta</td>
      <td>female</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
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
ml.transform(x_test,run_to_layer=-3).head(5)
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
      <th>Pclass_3.0</th>
      <th>Pclass_1.0</th>
      <th>Pclass_2.0</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>1.111379</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>502.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>503.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.6292</td>
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
      <td>504.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>505.0</td>
      <td>-0.741789</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>354</td>
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
#查看到模块名为target_encoding的数据
ml.transform(x_test,run_to_layer="target_encoding").head(5)
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
      <th>Pclass_3.0</th>
      <th>Pclass_1.0</th>
      <th>Pclass_2.0</th>
      <th>Sex_male</th>
      <th>Sex_female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>500</th>
      <td>501.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>1.111379</td>
      <td>17.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>502.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>21.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>503.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
      <td>7.6292</td>
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
      <td>504.0</td>
      <td>0.482439</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
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
      <td>505.0</td>
      <td>-0.741789</td>
      <td>0</td>
      <td>-1.569990</td>
      <td>16.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>354</td>
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



### 4.4 中间层pipe模块获取 
有时候我们向获取指定pipe模块，并调用其函数接口,这里可以通过`下标索引`(从0开始),也可以通过`name`进行索引


```python
#比如调用WOEEncoding的show_detail函数
ml[3].show_detail()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>bin_value</th>
      <th>bad_num</th>
      <th>bad_rate</th>
      <th>good_num</th>
      <th>good_rate</th>
      <th>woe</th>
      <th>iv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Sex</td>
      <td>male</td>
      <td>54</td>
      <td>0.279793</td>
      <td>261</td>
      <td>0.850163</td>
      <td>1.111379</td>
      <td>0.633897</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Sex</td>
      <td>female</td>
      <td>139</td>
      <td>0.720207</td>
      <td>46</td>
      <td>0.149837</td>
      <td>-1.569990</td>
      <td>0.895475</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pclass</td>
      <td>3.0</td>
      <td>78</td>
      <td>0.404145</td>
      <td>201</td>
      <td>0.654723</td>
      <td>0.482439</td>
      <td>0.120889</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pclass</td>
      <td>1.0</td>
      <td>66</td>
      <td>0.341969</td>
      <td>50</td>
      <td>0.162866</td>
      <td>-0.741789</td>
      <td>0.132856</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Pclass</td>
      <td>2.0</td>
      <td>49</td>
      <td>0.253886</td>
      <td>56</td>
      <td>0.182410</td>
      <td>-0.330626</td>
      <td>0.023632</td>
    </tr>
  </tbody>
</table>
</div>




```python
#name="taget_encoding"的show_detail函数
ml["target_encoding"].show_detail().head()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>bin_value</th>
      <th>target_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Embarked</td>
      <td>C</td>
      <td>0.521739</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Embarked</td>
      <td>Q</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Embarked</td>
      <td>S</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Embarked</td>
      <td>nan</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cabin</td>
      <td>A14</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 5. Pipeline流程的训练&预测&持久化  

训练接口fit和批量预测接口transform上面demo以及运行多次就不介绍了，下面介绍当条数据的预测接口transform_single这个主要用于线上单条数据的预测，要求输入是字典格式，而且输出也是字典格式


```python
input_dict=x_test.to_dict("record")[0]
input_dict
```




    {'PassengerId': 501,
     'Pclass': 3,
     'Name': 'Calic, Mr. Petar',
     'Sex': 'male',
     'Age': 17.0,
     'SibSp': 0,
     'Parch': 0,
     'Ticket': '315086',
     'Fare': 8.6625,
     'Cabin': nan,
     'Embarked': 'S'}




```python
ml.transform_single(input_dict)
```




    {0: -395.47434383444215,
     1: 16.246479243154806,
     2: -12.52627328963649,
     3: -39.863261981794416,
     4: -3.5252367095105677,
     5: 1.17125734500937,
     6: -0.64250346850212,
     7: 0.4252039033015551}




```python
#也可以看看啥也不输入，能否得到一个结果，检验代码是否稳健
ml.transform_single({})
```

    (<class 'easymlops.ml.preprocessing.FixInput'>) module, please check these missing columns:[1;43m['Pclass', 'Name', 'Fare', 'SibSp', 'Parch', 'PassengerId', 'Ticket', 'Sex', 'Age', 'Embarked', 'Cabin'][0m, they will by filled by nan(number),None(category)
    




    {0: -96.9781043279831,
     1: -311.7235001595718,
     2: 219.46040409835868,
     3: -65.86569587701999,
     4: -17.895054639401152,
     5: 0.1821931041957648,
     6: -1.3061114366829252,
     7: -0.44054724091208186}



#### 持久化


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
  .pipe(OneHotEncoding(cols=["Pclass","Sex"],drop_col=False))\
  .pipe(WOEEncoding(cols=["Sex","Pclass"]))\
  .pipe(LabelEncoding(cols=["Name","Ticket"]))\
  .pipe(TargetEncoding(cols=["Embarked","Cabin"],name="target_encoding"))\
  .pipe(FillNa())\
  .pipe(PCADecomposition(n_components=8))
```




    <easymlops.pipeml.PipeML at 0x200e4883fc8>




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
      <td>-395.474344</td>
      <td>16.246479</td>
      <td>-12.526273</td>
      <td>-39.863262</td>
      <td>-3.525237</td>
      <td>1.171257</td>
      <td>-0.642503</td>
      <td>0.425204</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-396.034758</td>
      <td>17.057039</td>
      <td>-12.883940</td>
      <td>-40.405460</td>
      <td>0.505058</td>
      <td>-1.880799</td>
      <td>-0.632594</td>
      <td>0.421864</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-396.602256</td>
      <td>17.387752</td>
      <td>-13.794505</td>
      <td>-42.144470</td>
      <td>-20.413251</td>
      <td>-1.752624</td>
      <td>-0.926964</td>
      <td>0.291678</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-397.322960</td>
      <td>18.434067</td>
      <td>-13.508230</td>
      <td>-37.263258</td>
      <td>16.299650</td>
      <td>-1.968569</td>
      <td>-0.408951</td>
      <td>0.530000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-238.496005</td>
      <td>274.260464</td>
      <td>150.607307</td>
      <td>75.272168</td>
      <td>-21.145699</td>
      <td>-1.725589</td>
      <td>0.589471</td>
      <td>-0.513614</td>
    </tr>
  </tbody>
</table>
</div>




```python
ml.transform_single(input_dict)
```




    {0: -395.4743438344421,
     1: 16.246479243154795,
     2: -12.526273289636507,
     3: -39.863261981794416,
     4: -3.5252367095105663,
     5: 1.1712573450093697,
     6: -0.6425034685021197,
     7: 0.42520390330155505}



备注：对于分拆后组合的pipe模块，也需要按照训练时候样子申明，然后嵌套，然后load

## 6.自定义pipe模块及其接口扩展

把需求分为如下几个层级：

- 最低需求，只做数据探索工作，只需要实现fit和transform接口
- 模型持久化需求，需要实现set_params和get_params来告诉PipeML,你的模型预测需要保留那些参数
- 生产上线需求，需要实现transform_single接口，实现与transform一样的预测结果，但处理的数据格式不一样，transform是dataframe，而transform_single是字典，而且transform_single的性能要求通常比transform高  
- 自定义扩展函数，可以添加自定义的其他函数方法，比较监测线上数据分布的变化，然后通过`4.4`介绍的方法调用

下面看一下TargetEncoding的简化版实现


```python
#注意下面继承的是object
class TargetEncoding(object):
    def __init__(self,name="", y=None,cols=None, error_value=0):
        self.name=name
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def show_detail(self):
        data = []
        for col, map_detail in self.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])

    def fit(self, s):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def transform(self, s):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].apply(lambda x: self._user_defined_function(col, x))
        return s
    
    def transform_single(self, s):
        for col in self.cols:
            if col in s.keys():
                s[col] = self._user_defined_function(col, s[col])
        return s

    def _user_defined_function(self, col, x):
        map_detail_ = self.target_map_detail.get(col, dict())
        return map_detail_.get(x, self.error_value)

    def get_params(self):
        #获取父类的params
        params=super().get_params()
        #加入当前的参数
        params.update({"target_map_detail": self.target_map_detail, "error_value": self.error_value})
        return params

    def set_params(self, params):
        #设置父类的params
        super().set_params(params)
        #再设置当前层的params
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
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
ml.transform_single(x_test.to_dict("record")[0])
```




    {'Age': 17.0, 'Fare': 8.6625, 'Embarked': 0.3342541436464088}




```python
ml[-2].show_detail()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>bin_value</th>
      <th>target_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Embarked</td>
      <td>C</td>
      <td>0.521739</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Embarked</td>
      <td>Q</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Embarked</td>
      <td>S</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Embarked</td>
      <td>nan</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



#### 进阶接口
上面是简化版的，在实现fit\transform\transform_single\get_params\set_params时可能需要考虑更多：  

- 输入输出数据的类型是否需要校验一下？  
- 输入输出数据的顺序是否需要一致？  
- set_params和get_params时搞忘了父类可咋整？
- 当前命名的参数与底层的参数名称的冲突检测？    
- 在transform时候是否需要把数据拷贝一下？  

建议自定义时继承PipeObject对象，然后实现_fit\ _transform\ _transform_single\ _get_params\ _set_params，这样自定义的Pipe模块更稳健，如下，调整后的TargetEncoding，代码几乎一样


```python
from easymlops.base import PipeObject
class TargetEncoding(PipeObject):
    def __init__(self,y=None, cols=None, error_value=0):
        super().__init__()
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def show_detail(self):
        data = []
        for col, map_detail in self.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])

    def _fit(self, s):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def _transform(self, s):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].apply(lambda x: self._user_defined_function(col, x))
        return s
    
    def _transform_single(self, s):
        for col in self.cols:
            if col in s.keys():
                s[col] = self._user_defined_function(col, s[col])
        return s

    def _user_defined_function(self, col, x):
        map_detail_ = self.target_map_detail.get(col, dict())
        return map_detail_.get(x, self.error_value)

    def _get_params(self):
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value}

    def _set_params(self, params):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
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
ml.transform_single(x_test.to_dict("record")[0])
```




    {'Age': 17.0, 'Fare': 8.6625, 'Embarked': 0.3342541436464088}




```python
ml[-2].show_detail()
```




<div>
 
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>col</th>
      <th>bin_value</th>
      <th>target_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Embarked</td>
      <td>C</td>
      <td>0.521739</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Embarked</td>
      <td>Q</td>
      <td>0.511111</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Embarked</td>
      <td>S</td>
      <td>0.334254</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Embarked</td>
      <td>nan</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



## 7. 支持生产部署:数据一致性测试&性能测试&日志记录

通常，在生产线上使用pandas效率并不高，且生产的输入格式通常是字典格式(json)，所以如果需要部署生产，我们需要额外添加一个函数：  
- transform_single:实现与transform一致的功能，而input和output需要修改为字典格式  

###  7.1 transform_single



```python
ml.transform_single({'PassengerId': 1,
 'Cabin': 0,
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




    {'Age': 22.0, 'Fare': 7.25, 'Embarked': 0.3342541436464088}




```python
ml.transform_single({'PassengerId': 1,
 'Cabin': 0,
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




    {'Age': 22.0, 'Fare': 7.25, 'Embarked': 'S'}



### 7.2 数据一致性测试&性能测试
部署生产环境之前，我们通常要关注两点：  
- 离线训练模型和在线预测模型的一致性，即tranform和transform_single的一致性；  
- transform_single对当条数据的预测性能  

这些可以通过调用如下函数，进行自动化测试：  
- auto_check_transform：只要有打印[success]，则表示一致性测试通过，性能测试表示为[*]毫秒/每条数据，如果有异常则会直接抛出，并中断后续pipe模块的测试


```python
ml_combine.auto_check_transform(x_test)
```

    (<class 'easymlops.ml.preprocessing.FixInput'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.preprocessing.FillNa'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.encoding.OneHotEncoding'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.encoding.LabelEncoding'>)  module transform check [success], single transform speed:[0.01]ms/it
    (<class 'easymlops.ml.encoding.TargetEncoding'>)  module transform check [success], single transform speed:[0.0]ms/it
    (<class 'easymlops.ml.decomposition.PCADecomposition'>)  module transform check [success], single transform speed:[1.26]ms/it
    (<class 'easymlops.ml.classification.LogisticRegressionClassification'>)  module transform check [success], single transform speed:[1.36]ms/it
    

### 7.3 日志记录 
日志通常只需要在生产中使用，所以只在transform_single可用


```python
import logging
logger=logging.getLogger()#logging的具体使用方法还请另行查资料
```


```python
extend_log_info={"user_id":1,"time":"2023-01-12 15:04:32"}
```


```python
ml.transform_single({'PassengerId': 1,
 'Cabin': 0,
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




    {'Age': 22.0, 'Fare': 7.25, 'Embarked': 0.3342541436464088}



## TODO  

- 扩展包装式特征选择方法，比如模拟退火、遗传算法、团伙检测等方法
- 添加数据监控模块：主要是数据偏移(p(x),p(y),p(y|x))，后续会逐步加入对这些偏移问题的建模优化模块
- 扩展更多NLP中关于文本分类的内容，比如TextCNN、Bert+BiLSTM等等  
- 添加数据采样模块  
- 扩展更多集成学习方法(优先考虑不平衡样本的集成建模)

希望大家来提PR，扩充内容~


```python

```
