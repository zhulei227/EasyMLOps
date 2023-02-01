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


### 7. 生产部署:日志记录&预测一致性测试&性能测试&空值测试&极端值测试

- 生产预测接口，pipeobj.transform_single(data)即可对生产数据(通常转换为dict)进行预测
- 日志记录，pipeobj.transform_single(data,logger)可以追踪记录pipeline预测中每一步信息
- 预测一致性&性能测试，pipeobj.check_transform_function(data)可以对transform/transform_single的一致性以及各个pipe模块的性能做测试  
- 空值测试，pipeobj.check_null_value(data)主要用于检测取各类空值时，比如直接删除，取值None,null,nan,np.nan...最终预测结果是否还能一致
- 极端值测试，pipeobj.check_extreme_value(data)用于检测输入极端值的情况下，还能否有正常的输出，比如你处理的某列数据是1~100范围，线上生产给你一个inf,0,max float,min float看看模块还能否正常输出结果
- 类型反转测试，pipeobj.check_inverse_dtype(data) 比如，你的模型训练的是数值类型，如果给你一个字符串的"1"，你的代码会不会报错，如果你训练的字符数据，给你一个数值的0.01你的程序会不会崩
- int转float测试，pipeobj.check_int_trans_float(data)，pandas会将某些特征自动推断为int，而线上传输的可能是float，需要测试这两种情况是否能一致

- 自动化测试接口，pipeobj.auto_test(data)，依次将上面的各个测试走一遍


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
  .pipe(FillNa(cols=["Cabin","Ticket","Parch","Fare","Sex"],fill_mode="mode"))\
  .pipe(FillNa(cols=["Age"],fill_mode="mean"))\
  .pipe(FillNa(fill_detail={"Embarked":"N"}))\
  .pipe(TransToCategory(cols=["Cabin","Embarked","Name"]))\
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
      <td>-0.518312</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
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
      <td>-0.539225</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>1.0</td>
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
      <td>7.6292</td>
      <td>n</td>
      <td>q</td>
      <td>0.402925</td>
      <td>-0.541994</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>1.0</td>
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
      <td>-0.497111</td>
      <td>5.0</td>
      <td>6.0</td>
      <td>1.0</td>
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
      <td>1.265641</td>
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
      <td>244.143134</td>
      <td>-26.433827</td>
      <td>-5.579019</td>
      <td>0.139935</td>
    </tr>
    <tr>
      <th>1</th>
      <td>244.149330</td>
      <td>-27.065848</td>
      <td>-1.535096</td>
      <td>0.032236</td>
    </tr>
    <tr>
      <th>2</th>
      <td>244.048654</td>
      <td>-28.613322</td>
      <td>-22.472506</td>
      <td>-0.393424</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244.250079</td>
      <td>-24.145769</td>
      <td>14.298703</td>
      <td>0.353986</td>
    </tr>
    <tr>
      <th>4</th>
      <td>245.211271</td>
      <td>51.165456</td>
      <td>-11.852851</td>
      <td>-1.108305</td>
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
      <td>244.142772</td>
      <td>-26.441827</td>
      <td>6.209714</td>
      <td>0.178558</td>
    </tr>
    <tr>
      <th>1</th>
      <td>244.148957</td>
      <td>-27.072988</td>
      <td>6.225051</td>
      <td>0.168702</td>
    </tr>
    <tr>
      <th>2</th>
      <td>244.048279</td>
      <td>-28.620891</td>
      <td>6.222640</td>
      <td>0.096320</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244.249710</td>
      <td>-24.152708</td>
      <td>6.260848</td>
      <td>0.267476</td>
    </tr>
    <tr>
      <th>4</th>
      <td>245.211899</td>
      <td>51.176116</td>
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
      <td>0.967995</td>
      <td>0.032005</td>
      <td>0.658180</td>
      <td>0.341820</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.992226</td>
      <td>0.007774</td>
      <td>0.661813</td>
      <td>0.338187</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.981467</td>
      <td>0.018533</td>
      <td>0.665298</td>
      <td>0.334702</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.712881</td>
      <td>0.287119</td>
      <td>0.660105</td>
      <td>0.339895</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.264412</td>
      <td>0.735588</td>
      <td>0.471488</td>
      <td>0.528512</td>
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
      <td>0.648002</td>
      <td>0.075899</td>
      <td>0.093497</td>
      <td>-0.183208</td>
      <td>-0.017150</td>
      <td>0.403337</td>
      <td>0.551337</td>
    </tr>
    <tr>
      <th>501</th>
      <td>2.034848</td>
      <td>-0.608227</td>
      <td>0.733364</td>
      <td>0.013151</td>
      <td>-0.085437</td>
      <td>-0.033206</td>
      <td>0.293308</td>
      <td>0.416924</td>
    </tr>
    <tr>
      <th>502</th>
      <td>2.040231</td>
      <td>-0.616310</td>
      <td>0.747684</td>
      <td>0.022635</td>
      <td>-0.051665</td>
      <td>0.000399</td>
      <td>0.277010</td>
      <td>0.309539</td>
    </tr>
    <tr>
      <th>503</th>
      <td>2.026293</td>
      <td>-0.579112</td>
      <td>0.735054</td>
      <td>-0.012189</td>
      <td>-0.140556</td>
      <td>0.009208</td>
      <td>0.393155</td>
      <td>0.443419</td>
    </tr>
    <tr>
      <th>504</th>
      <td>2.025096</td>
      <td>-0.573415</td>
      <td>0.720067</td>
      <td>-0.010234</td>
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
      <td>244.143134</td>
      <td>-26.433827</td>
      <td>-5.579019</td>
      <td>0.139935</td>
      <td>-1.089247</td>
      <td>-0.171164</td>
      <td>0.037629</td>
      <td>-0.129986</td>
    </tr>
    <tr>
      <th>1</th>
      <td>244.149330</td>
      <td>-27.065848</td>
      <td>-1.535096</td>
      <td>0.032236</td>
      <td>0.030136</td>
      <td>-1.150590</td>
      <td>-0.114188</td>
      <td>-1.025774</td>
    </tr>
    <tr>
      <th>2</th>
      <td>244.048654</td>
      <td>-28.613322</td>
      <td>-22.472506</td>
      <td>-0.393424</td>
      <td>-0.062435</td>
      <td>-0.962957</td>
      <td>-0.102700</td>
      <td>-1.061490</td>
    </tr>
    <tr>
      <th>3</th>
      <td>244.250079</td>
      <td>-24.145769</td>
      <td>14.298703</td>
      <td>0.353986</td>
      <td>0.085036</td>
      <td>-1.292375</td>
      <td>-0.120276</td>
      <td>-1.005569</td>
    </tr>
    <tr>
      <th>4</th>
      <td>245.211271</td>
      <td>51.165456</td>
      <td>-11.852851</td>
      <td>-1.108305</td>
      <td>0.057464</td>
      <td>-0.521142</td>
      <td>-0.345954</td>
      <td>-0.625369</td>
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
      <td>0.996155</td>
      <td>0.003845</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.980622</td>
      <td>0.019378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.973603</td>
      <td>0.026397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.985191</td>
      <td>0.014809</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.258010</td>
      <td>0.741990</td>
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
      <td>0.996155</td>
      <td>0.003845</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.980622</td>
      <td>0.019378</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.973603</td>
      <td>0.026397</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.985191</td>
      <td>0.014809</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.258010</td>
      <td>0.741990</td>
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
      <td>-387.121036</td>
      <td>13.979222</td>
      <td>-12.776600</td>
      <td>-38.933042</td>
      <td>-2.880351</td>
      <td>1.155796</td>
      <td>-0.564385</td>
      <td>-0.593809</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-387.087823</td>
      <td>14.134133</td>
      <td>-12.661127</td>
      <td>-39.437177</td>
      <td>1.162099</td>
      <td>-1.898521</td>
      <td>-0.592609</td>
      <td>-0.750671</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-387.059681</td>
      <td>13.808067</td>
      <td>-13.096079</td>
      <td>-41.560890</td>
      <td>-19.720113</td>
      <td>-1.761060</td>
      <td>-0.881661</td>
      <td>-0.562568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-387.184160</td>
      <td>14.199977</td>
      <td>-12.390480</td>
      <td>-36.093664</td>
      <td>16.911090</td>
      <td>-1.990755</td>
      <td>-0.372917</td>
      <td>-0.905934</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-228.588562</td>
      <td>268.592272</td>
      <td>148.668435</td>
      <td>77.511089</td>
      <td>-22.424479</td>
      <td>-1.651499</td>
      <td>0.376687</td>
      <td>0.360375</td>
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
      <td>None</td>
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
      <td>None</td>
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
      <td>None</td>
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
      <td>None</td>
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




    {0: -387.12103636413934,
     1: 13.97922161572029,
     2: -12.776599783138563,
     3: -38.93304205157552,
     4: -2.8803508317309654,
     5: 1.155795904439429,
     6: -0.5643848743244888,
     7: -0.5938089285211727}




```python
#也可以看看啥也不输入，能否得到一个结果，检验代码是否稳健
ml.transform_single({})
```

    (<class 'easymlops.ml.preprocessing.FixInput'>) module, please check these missing columns:[1;43m['Embarked', 'PassengerId', 'Cabin', 'Fare', 'Ticket', 'Age', 'Pclass', 'Name', 'Sex', 'Parch', 'SibSp'][0m, they will by filled by nan(number),None(category)
    




    {0: -95.64039657615285,
     1: -305.1318009424023,
     2: 215.2032019812796,
     3: -62.09476149164795,
     4: -17.02882296816116,
     5: 0.1713669364343205,
     6: -1.3024227323286315,
     7: 0.5629456659915981}



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




    <easymlops.pipeml.PipeML at 0x208a3b23748>




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
      <td>-387.121036</td>
      <td>13.979222</td>
      <td>-12.776600</td>
      <td>-38.933042</td>
      <td>-2.880351</td>
      <td>1.155796</td>
      <td>-0.564385</td>
      <td>-0.593809</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-387.087823</td>
      <td>14.134133</td>
      <td>-12.661127</td>
      <td>-39.437177</td>
      <td>1.162099</td>
      <td>-1.898521</td>
      <td>-0.592609</td>
      <td>-0.750671</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-387.059681</td>
      <td>13.808067</td>
      <td>-13.096079</td>
      <td>-41.560890</td>
      <td>-19.720113</td>
      <td>-1.761060</td>
      <td>-0.881661</td>
      <td>-0.562568</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-387.184160</td>
      <td>14.199977</td>
      <td>-12.390480</td>
      <td>-36.093664</td>
      <td>16.911090</td>
      <td>-1.990755</td>
      <td>-0.372917</td>
      <td>-0.905934</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-228.588562</td>
      <td>268.592272</td>
      <td>148.668435</td>
      <td>77.511089</td>
      <td>-22.424479</td>
      <td>-1.651499</td>
      <td>0.376687</td>
      <td>0.360375</td>
    </tr>
  </tbody>
</table>
</div>




```python
ml.transform_single(input_dict)
```




    {0: -387.1210363641393,
     1: 13.979221615720292,
     2: -12.776599783138565,
     3: -38.933042051575526,
     4: -2.8803508317309676,
     5: 1.155795904439429,
     6: -0.5643848743244889,
     7: -0.5938089285211727}



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



### 7.2 日志记录 
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



### 7.3 transform/transform_single一致性测试&性能测试:check_transform_function
部署生产环境之前，我们通常要关注两点：  
- 离线训练模型和在线预测模型的一致性，即tranform和transform_single的一致性；  
- transform_single对当条数据的预测性能  

这些可以通过调用如下函数，进行自动化测试：  
- check_transform_function：只要有打印[success]，则表示在当前测试数据上transform和transform_single的输出一致，性能测试表示为[*]毫秒/每条数据，如果有异常则会直接抛出，并中断后续pipe模块的测试


```python
ml_combine.check_transform_function(x_test)
```

    (<class 'easymlops.ml.preprocessing.FixInput'>)  module transform check [success], single transform speed:[0.08]ms/it
    (<class 'easymlops.ml.preprocessing.FillNa'>)  module transform check [success], single transform speed:[0.0]ms/it
    (<class 'easymlops.ml.encoding.OneHotEncoding'>)  module transform check [success], single transform speed:[0.04]ms/it
    (<class 'easymlops.ml.encoding.LabelEncoding'>)  module transform check [success], single transform speed:[0.04]ms/it
    (<class 'easymlops.ml.encoding.TargetEncoding'>)  module transform check [success], single transform speed:[0.04]ms/it
    (<class 'easymlops.ml.decomposition.PCADecomposition'>)  module transform check [success], single transform speed:[5.83]ms/it
    (<class 'easymlops.ml.classification.LogisticRegressionClassification'>)  module transform check [success], single transform speed:[1.41]ms/it
    

### 7.4 空值测试：check_null_value  

- 由于pandas在读取数据时会自动做类型推断，对空会有不同的处理，比如float设置为np.nan，对object设置为None或NaN  
- 而且pandas读取数据默认为批量读取批量推断，所以某一列数据空还不唯一，np.nan和None可能共存  

所以，这里对逐个column分别设置不同的空进行测试，测试内容：  
- 相同的空情况下，transform和transform_single是否一致  
- 不同的空的transform结果是否一致  

可通过`null_values=[None, np.nan, "null", "NULL", "nan", "NaN", "", "none", "None", " "]`(默认)设置自定义空值


```python
ml_combine.check_null_value(x_test)
```

    column: [PassengerId] check null value complete, total single transform speed:[7.75]ms/it
    column: [Pclass] check null value complete, total single transform speed:[7.47]ms/it
    column: [Name] check null value complete, total single transform speed:[7.77]ms/it
    column: [Sex] check null value complete, total single transform speed:[7.66]ms/it
    column: [Age] check null value complete, total single transform speed:[7.87]ms/it
    column: [SibSp] check null value complete, total single transform speed:[8.17]ms/it
    column: [Parch] check null value complete, total single transform speed:[8.04]ms/it
    column: [Ticket] check null value complete, total single transform speed:[7.94]ms/it
    column: [Fare] check null value complete, total single transform speed:[7.91]ms/it
    column: [Cabin] check null value complete, total single transform speed:[8.28]ms/it
    column: [Embarked] check null value complete, total single transform speed:[7.65]ms/it
    

### 7.5极端值测试：check_extreme_value  

通常用于训练的数据都是经过筛选的正常数据，但线上难免会有极端值混入，比如你训练的某列数据范围在`0~1`之间，如果传入一个`-1`，也许就会报错，目前

- 对两种类型的分别进行极端测试，设置如下：
  - 数值型:设置`number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max]`(默认)
  - 离散型:设置`category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"]`(默认)  

- 将全部特征设置为如上的极端值进行测试

注意：这里只检测了transform与transform_single的一致性，不要求各极端值输入下的输出一致性(注意和上面的空值检测不一样，空值检测要求所有类型的空的输出也要一致)


```python
ml_combine.check_extreme_value(x_test)
```

    column: [PassengerId] check extreme value complete, total single transform speed:[7.47]ms/it
    column: [Pclass] check extreme value complete, total single transform speed:[7.62]ms/it
    column: [Name] check extreme value complete, total single transform speed:[7.74]ms/it
    column: [Sex] check extreme value complete, total single transform speed:[7.77]ms/it
    column: [Age] check extreme value complete, total single transform speed:[7.55]ms/it
    column: [SibSp] check extreme value complete, total single transform speed:[7.68]ms/it
    column: [Parch] check extreme value complete, total single transform speed:[7.63]ms/it
    column: [Ticket] check extreme value complete, total single transform speed:[7.56]ms/it
    column: [Fare] check extreme value complete, total single transform speed:[7.61]ms/it
    column: [Cabin] check extreme value complete, total single transform speed:[7.77]ms/it
    column: [Embarked] check extreme value complete, total single transform speed:[7.65]ms/it
    [__all__] columns set the same extreme value complete,total single transform speed:[8.06]ms/it
    

### 7.6 数据类型反转测试：check_inverse_dtype  

某特征入模是数据是数值，但上线后传过来的是离散值，也有可能相反，这里就对这种情况做测试，对原是数值的替换为离散做测试，对原始离散值的替换为数值，替换规则如下：
- 原数值的，替换为：`number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"]`(默认)  
- 原离散的，替换为：`category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max]`(默认)  

同样，数据类型反转测试只对transform和transform_single的一致性有要求


```python
ml_combine.check_inverse_dtype(x_test)
```

    column: [PassengerId] check inverse value complete, total single transform speed:[7.96]ms/it
    column: [Pclass] check inverse value complete, total single transform speed:[7.83]ms/it
    column: [Name] check inverse value complete, total single transform speed:[7.69]ms/it
    column: [Sex] check inverse value complete, total single transform speed:[7.68]ms/it
    column: [Age] check inverse value complete, total single transform speed:[7.86]ms/it
    column: [SibSp] check inverse value complete, total single transform speed:[7.82]ms/it
    column: [Parch] check inverse value complete, total single transform speed:[7.82]ms/it
    column: [Ticket] check inverse value complete, total single transform speed:[7.77]ms/it
    column: [Fare] check inverse value complete, total single transform speed:[8.06]ms/it
    column: [Cabin] check inverse value complete, total single transform speed:[8.06]ms/it
    column: [Embarked] check inverse value complete, total single transform speed:[7.53]ms/it
    

### 7.7 int转float测试：check_int_trans_float  
pandas会将某些特征自动推断为int，而线上可能传输的是float，需要做如下测试：  
- 转float后transform和transform_single之间的一致性  
- int和float特征通过transform后的一致性


```python
ml_combine.check_int_trans_float(x_test)
```

    column: [PassengerId] check int trans float value complete, total single transform speed:[7.53]ms/it
    column: [Pclass] check int trans float value complete, total single transform speed:[7.68]ms/it
    column: [SibSp] check int trans float value complete, total single transform speed:[7.53]ms/it
    column: [Parch] check int trans float value complete, total single transform speed:[7.98]ms/it
    

### 7.8 自动测试：auto_test
就是把上面的所有测试，整合到auto_test一个函数中


```python
#ml_combine.auto_test(x_test)
```

## TODO  

- 扩展包装式特征选择方法，比如模拟退火、遗传算法、团伙检测等方法
- 添加数据监控模块：主要是数据偏移(p(x),p(y),p(y|x))，后续会逐步加入对这些偏移问题的建模优化模块
- 扩展更多NLP中关于文本分类的内容，比如TextCNN、Bert+BiLSTM等等  
- 添加数据采样模块  
- 扩展更多集成学习方法(优先考虑不平衡样本的集成建模)

希望大家来提PR，扩充内容~


```python

```
