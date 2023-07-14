# 特征存储&监控&分析

模型部署生产后，我们通常还有些后续需求：  

- 存储：将某些pipe的输出直接保存到数据库存储，比如特征工程后的数据等；
- 监控&分析：监控数据短期内的变化情况，分析异常指标等

## 存储的两种方式  

- 嵌入式：算法研发的同学也肩负了后续指标监控、报表统计等运维形式的工作，这样在研发过程就可以把存储pipe模块嵌入到pipeline中；   
- 挂载式：算法研发只需要关注算法相关的模块，后续上线时，运维同学再把存储模块挂载到需要保存的pipe模块后，这种方式更加灵活   

下面以本地存储为例运行这两种方式，还请`pip install sqlite3`


```python
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


```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *
from easymlops.table.storage import LocalStorage,HbaseStorage
```

### 嵌入式


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass", "Sex"], drop_col=False)) \
  .pipe(WOEEncoding(cols=["Ticket", "Embarked", "Cabin", "Sex", "Pclass"], y=y_train)) \
  .pipe(LabelEncoding(cols=["Name"]))\
  .pipe(LocalStorage(db_name="./local.db", table_name="label_encoding",cols=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch',
                                                                             'Ticket','Fare','Cabin','Embarked','Pclass_3','Pclass_1','Pclass_2','Sex_male','Sex_female']))\
  .pipe(LGBMClassification(y=y_train,native_init_params={"max_depth":2},native_fit_params={"num_boost_round":128},prefix="lgbm"))\
  .pipe(LocalStorage(db_name="./local.db", table_name="predict",cols=["lgbm_0","lgbm_1"]))\

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x23a242ae908>



上面，再LabelEncoding和LGBMClassfication后分别加了一个LocalStorage模块，用于保存它们的输出，下面模拟一条生产输入数据


```python
record=x_test.to_dict("record")[0]
record
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
table.transform_single(record,storage_base_dict={"key":record.get("PassengerId")})
```




    {'lgbm_0': 0.9233260451690832, 'lgbm_1': 0.0766739548309168}



查询key


```python
#查看label encoding的输出
print(table[-3].select_key(key=501).to_markdown())
```

    |    |   storage_key | storage_transform_time   |   PassengerId |   Pclass |   Name |     Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:-------------------------|--------------:|---------:|-------:|--------:|------:|--------:|--------:|---------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           501 | 2023-03-07 21:41:33      |           501 | 0.482439 |      0 | 1.11138 |    17 |       0 |       0 |        0 |  8.664 | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    


```python
#查看lgbm classification的输出
print(table[-1].select_key(key=501).to_markdown())
```

    |    |   storage_key | storage_transform_time   |   lgbm_0 |   lgbm_1 |
    |---:|--------------:|:-------------------------|---------:|---------:|
    |  0 |           501 | 2023-03-07 21:41:33      | 0.923326 | 0.076674 |
    

### 挂载式 

如下，算法同学无需关心存储，正常建模就好


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass", "Sex"], drop_col=False)) \
  .pipe(WOEEncoding(cols=["Ticket", "Embarked", "Cabin", "Sex", "Pclass"], y=y_train)) \
  .pipe(LabelEncoding(cols=["Name"]))\
  .pipe(LGBMClassification(y=y_train,native_init_params={"max_depth":2},native_fit_params={"num_boost_round":128},prefix="lgbm"))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x2836874cb88>



运维同学通过如下方式挂载存储模块


```python
#记录label encoding的输出
table[-2].set_branch_pipe(LocalStorage(db_name="./local.db", table_name="label_encoding",cols=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch',
                                                                             'Ticket','Fare','Cabin','Embarked','Pclass_3','Pclass_1','Pclass_2','Sex_male','Sex_female']))
```


```python
#记录y的输出
table[-1].set_branch_pipe(LocalStorage(db_name="./local.db", table_name="predict",cols=["lgbm_0","lgbm_1"]))
```


```python
#模拟生产数据
record=x_test.to_dict("record")[1]
record
```




    {'PassengerId': 502,
     'Pclass': 3,
     'Name': 'Canavan, Miss. Mary',
     'Sex': 'female',
     'Age': 21.0,
     'SibSp': 0,
     'Parch': 0,
     'Ticket': '364846',
     'Fare': 7.75,
     'Cabin': nan,
     'Embarked': 'Q'}




```python
table.transform_single(record,storage_base_dict={"key":record.get("PassengerId")})
```




    {'lgbm_0': 0.3736520279722756, 'lgbm_1': 0.6263479720277243}



查询key，注意这里需要通过`.get_branch_pipe(index)`的方式获取到指定的存储模块，再进行查询key的操作


```python
print(table[-2].get_branch_pipe(0).select_key(key=502).to_markdown())
```

    |    |   storage_key | storage_transform_time   |   PassengerId |   Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:-------------------------|--------------:|---------:|-------:|---------:|------:|--------:|--------:|---------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           502 | 2023-03-07 21:41:39      |           502 | 0.482439 |      0 | -1.56999 |    21 |       0 |       0 |        0 |   7.75 | 0.299607 |  -0.508609 |          1 |          0 |          0 |          0 |            1 |
    


```python
print(table[-1].get_branch_pipe(0).select_key(key=502).to_markdown())
```

    |    |   storage_key | storage_transform_time   |   lgbm_0 |   lgbm_1 |
    |---:|--------------:|:-------------------------|---------:|---------:|
    |  0 |           502 | 2023-03-07 21:41:39      | 0.373652 | 0.626348 |
    

由于对当前pipe而言都只挂载了一个pipe，所以可以通过`get_branch_pipe(0)`获取，如果还继续挂载了其他pipe模块，通过增加index获取

## 本地存储

目前设计了如下几个函数做监控和分析用，接下来模拟更多的生产数据


```python
for record in tqdm(x_test.to_dict("record")[2:]):
    table.transform_single(record,storage_base_dict={"key":record.get("PassengerId")})
```

    100%|███████████████████████████████████████████████████████████████████████████████| 389/389 [00:01<00:00, 325.22it/s]
    

### 查询key:select_key


```python
print(table[-2].get_branch_pipe(0).select_key(key=502).to_markdown())
```

    |    |   storage_key | storage_transform_time   |   PassengerId |   Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:-------------------------|--------------:|---------:|-------:|---------:|------:|--------:|--------:|---------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           502 | 2023-03-07 21:41:39      |           502 | 0.482439 |      0 | -1.56999 |    21 |       0 |       0 |        0 |   7.75 | 0.299607 |  -0.508609 |          1 |          0 |          0 |          0 |            1 |
    

### 复杂查询:where


```python
print(table[-2].get_branch_pipe(0).where("Pclass>0.4 and Sex>1 and Sex_male=1",limit=5).to_markdown())
```

    |    |   storage_key | storage_transform_time   |   PassengerId |   Pclass |   Name |     Sex |   Age |   SibSp |   Parch |    Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:-------------------------|--------------:|---------:|-------:|--------:|------:|--------:|--------:|----------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           501 | 2023-03-07 21:41:33      |           501 | 0.482439 |      0 | 1.11138 |    17 |       0 |       0 |  0        |  8.664 | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  1 |           509 | 2023-03-07 21:41:41      |           509 | 0.482439 |      0 | 1.11138 |    28 |       0 |       0 |  0        | 22.53  | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  2 |           510 | 2023-03-07 21:41:41      |           510 | 0.482439 |      0 | 1.11138 |    26 |       0 |       0 | -0.464158 | 56.5   | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  3 |           511 | 2023-03-07 21:41:41      |           511 | 0.482439 |      0 | 1.11138 |    29 |       0 |       0 |  0        |  7.75  | 0.299607 |  -0.508609 |          1 |          0 |          0 |          1 |            0 |
    |  4 |           512 | 2023-03-07 21:41:41      |           512 | 0.482439 |      0 | 1.11138 |     0 |       0 |       0 |  0        |  8.05  | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    


```python
#存活率更高的乘客
print(table[-1].get_branch_pipe(0).where("lgbm_1>0.8",limit=5).to_markdown())
```

    |    |   storage_key | storage_transform_time   |    lgbm_0 |   lgbm_1 |
    |---:|--------------:|:-------------------------|----------:|---------:|
    |  0 |           505 | 2023-03-07 21:41:41      | 0.0684703 | 0.93153  |
    |  1 |           514 | 2023-03-07 21:41:41      | 0.0718778 | 0.928122 |
    |  2 |           517 | 2023-03-07 21:41:41      | 0.113928  | 0.886072 |
    |  3 |           519 | 2023-03-07 21:41:41      | 0.19458   | 0.80542  |
    |  4 |           521 | 2023-03-07 21:41:41      | 0.0710767 | 0.928923 |
    

### 聚合分析:group_agg_where

不同sex_male,pclass_3下的统计量


```python
return_df=table[-2].get_branch_pipe(0).group_agg_where(group_by="Sex_male,Pclass_3", agg_sql="Sex_male,Pclass_3,max(PassengerId) as PassengerId_max,sum(Sex)/count(Sex) as Sex_mean,count(Ticket) as Ticket_cnt", where_sql="storage_transform_time>='2023-03-03 01:01:01'", limit=10)
print(return_df.to_markdown())
```

    |    |   Sex_male |   Pclass_3 |   PassengerId_max |   Sex_mean |   Ticket_cnt |
    |---:|-----------:|-----------:|------------------:|-----------:|-------------:|
    |  0 |          0 |          0 |               888 |   -1.56999 |           75 |
    |  1 |          0 |          1 |               889 |   -1.56999 |           54 |
    |  2 |          1 |          0 |               890 |    1.11138 |          104 |
    |  3 |          1 |          1 |               891 |    1.11138 |          158 |
    

### 纯SQL函数:sql
以上接口不支持复杂的sql嵌套，这里可以直接定义复杂的sql嵌套分析


```python
table_name="predict"
sql=f"""
-- 统计是否存活用户的最早最晚transform时间
select survied_pred,min(storage_transform_time) as transform_min,max(storage_transform_time) as transform_max from 
    -- 如果lgbm_1>0.5就视为存活
    (select storage_key,storage_transform_time,case when lgbm_1>0.5 then 1 else 0 end as survied_pred from {table_name}) as t 
group by survied_pred
"""
return_df=table[-1].get_branch_pipe(0).sql(sql)
```


```python
print(return_df.to_markdown())
```

    |    |   survied_pred | transform_min       | transform_max       |
    |---:|---------------:|:--------------------|:--------------------|
    |  0 |              0 | 2023-03-07 21:41:33 | 2023-03-07 21:41:42 |
    |  1 |              1 | 2023-03-07 21:41:39 | 2023-03-07 21:41:42 |
    

## Hbase存储

由于hbase存储自身的限制，目前只支持：  
- 查询指定key的数据；  
- 查询指定时间范围内的数据；
- scan进行复杂查询  

该部分模块需要：  

`pip install happybase` 和 `pip install func_timeout`


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass", "Sex"], drop_col=False)) \
  .pipe(WOEEncoding(cols=["Ticket", "Embarked", "Cabin", "Sex", "Pclass"], y=y_train)) \
  .pipe(LabelEncoding(cols=["Name"]))\
  .pipe(LGBMClassification(y=y_train,native_init_params={"max_depth":2},native_fit_params={"num_boost_round":128},prefix="lgbm"))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x1e6b88d5808>




```python
#记录label encoding的输出
table[-2].set_branch_pipe(HbaseStorage(host="192.168.244.131",port=9090,table_name="label_encoding", cf_name="cf1",cols=['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch',
                                                                             'Ticket','Fare','Cabin','Embarked','Pclass_3','Pclass_1','Pclass_2','Sex_male','Sex_female']))
```

    connect to 192.168.244.131:9090/label_encoding success!
    


```python
#模拟生产数据
record=x_test.to_dict("record")[0]
record
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
table.transform_single(record,storage_base_dict={"key":record.get("PassengerId")})
```




    {'lgbm_0': 0.9233260451690832, 'lgbm_1': 0.0766739548309168}



### 查询指定key:select_key


```python
print(table[-2].get_branch_pipe(0).select_key("600").to_markdown())
```

    |    |   storage_key | storage_transform_time     |   PassengerId |    Pclass |   Name |     Sex |   Age |   SibSp |   Parch |   Ticket |    Fare |   Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:---------------------------|--------------:|----------:|-------:|--------:|------:|--------:|--------:|---------:|--------:|--------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           600 | 2023-03-07 21:42:04.038874 |           600 | -0.741789 |      0 | 1.11138 |    49 |       1 |       0 |        0 | 56.9375 |       0 |  -0.551169 |          0 |          1 |          0 |          1 |            0 |
    


```python
for record in tqdm(x_test.to_dict("record")[2:]):
    table.transform_single(record,storage_base_dict={"key":record.get("PassengerId")})
```

    100%|███████████████████████████████████████████████████████████████████████████████| 389/389 [00:01<00:00, 234.24it/s]
    

### 查询指定时间区间:select_time


```python
print(table[-2].get_branch_pipe(0).select_time(start_time="2023-03-07 21:21:00",stop_time="2023-03-07 22:22:00",limit=5).to_markdown())
```

    |    |   storage_key | storage_transform_time     |   PassengerId |    Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |      Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:---------------------------|--------------:|----------:|-------:|---------:|------:|--------:|--------:|---------:|----------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           501 | 2023-03-07 21:42:00.584158 |           501 |  0.482439 |      0 |  1.11138 |    17 |       0 |       0 |        0 |   8.66406 | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  1 |           503 | 2023-03-07 21:42:03.694794 |           503 |  0.482439 |      0 | -1.56999 |     0 |       0 |       0 |        0 |   7.62891 | 0.299607 |  -0.508609 |          1 |          0 |          0 |          0 |            1 |
    |  2 |           504 | 2023-03-07 21:42:03.697801 |           504 |  0.482439 |      0 | -1.56999 |    37 |       0 |       0 |        0 |   9.58594 | 0.299607 |   0.224849 |          1 |          0 |          0 |          0 |            1 |
    |  3 |           505 | 2023-03-07 21:42:03.701797 |           505 | -0.741789 |      0 | -1.56999 |    16 |       0 |       0 |        0 |  86.5     | 0        |   0.224849 |          0 |          1 |          0 |          0 |            1 |
    |  4 |           506 | 2023-03-07 21:42:03.704802 |           506 | -0.741789 |      0 |  1.11138 |    18 |       1 |       0 |        0 | 108.875   | 0        |  -0.551169 |          0 |          1 |          0 |          1 |            0 |
    

### 复杂查询:scan

hbase可以支持一些复杂的组合查询，使用方式可以参考如下链接

https://www.jianshu.com/p/0bad3534186b  
https://hbase.apache.org/book.html#thrift  

比如下面，查询PassengerId>="503" 且 storage_transform_time>="2023-03-07 21:21:00" 的前5条数据


```python
scan_filter = "SingleColumnValueFilter('cf1', 'PassengerId', >=, 'binary:503', true, true) AND SingleColumnValueFilter('cf1', 'storage_transform_time', >=, 'binary:2023-03-07 21:21:00', true, true) "
print(table[-2].get_branch_pipe(0).scan(scan_filter=scan_filter,limit=5).to_markdown())
```

    |    |   storage_key | storage_transform_time     |   PassengerId |    Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |      Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|--------------:|:---------------------------|--------------:|----------:|-------:|---------:|------:|--------:|--------:|---------:|----------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |           503 | 2023-03-07 21:50:43.963905 |           503 |  0.482439 |      0 | -1.56999 |     0 |       0 |       0 |        0 |   7.62891 | 0.299607 |  -0.508609 |          1 |          0 |          0 |          0 |            1 |
    |  1 |           504 | 2023-03-07 21:50:43.966905 |           504 |  0.482439 |      0 | -1.56999 |    37 |       0 |       0 |        0 |   9.58594 | 0.299607 |   0.224849 |          1 |          0 |          0 |          0 |            1 |
    |  2 |           505 | 2023-03-07 21:50:43.970904 |           505 | -0.741789 |      0 | -1.56999 |    16 |       0 |       0 |        0 |  86.5     | 0        |   0.224849 |          0 |          1 |          0 |          0 |            1 |
    |  3 |           506 | 2023-03-07 21:50:43.974912 |           506 | -0.741789 |      0 |  1.11138 |    18 |       1 |       0 |        0 | 108.875   | 0        |  -0.551169 |          0 |          1 |          0 |          1 |            0 |
    |  4 |           507 | 2023-03-07 21:50:43.977907 |           507 | -0.330626 |      0 | -1.56999 |    33 |       0 |       2 |        0 |  26       | 0.299607 |   0.224849 |          0 |          0 |          1 |          0 |            1 |
    

## Faiss存储

faiss即Facebook AI Similarity Search，是一个由facebook开发以用于搜索相似向量的库，所以该存储模块所存储的必需为数值，而该模块所能提供的功能也主要是最相似向量的检索，faiss具有不同的索引构造方式，以及不同的相似度度量方式，这部分可以通过`create_index_param`和`create_index_measure`设置，各种方式的优缺点以及适合场景还请访问官方文档:https://github.com/facebookresearch/faiss  

该模块需要 `pip install faiss-cpu`或`pip install faiss-gpu`


```python
data=pd.read_csv("./data/demo2.csv",encoding="gbk").sample(frac=1)
print(data.head(5).to_markdown())
```

    |      |   label | review                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
    |-----:|--------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    | 3834 |       1 | 不错的地方，房间不错，服务也可。除了火车声音有点受不了。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
    | 2491 |       1 | 这次住的附楼（未挂星），硬件设施比较陈旧，空调也不太好，我还因此换了回房间．二楼还有很多蚊子．不过服务                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
    |  279 |       1 | 地理位置不错，房间环境也可以，洗衣速度快，值得称赞，唯一不足就是装修期间，很多楼层乱乱的。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
    | 6056 |       0 | 床发出吱嘎吱嘎的声音，房间隔音太差，赠送的早餐非常好吃。补充点评2008年7月22日：我们住的是度假村的标准间。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
    | 4199 |       1 | 据我昆山一个搞装潢的朋友讲,这家酒店的大堂和客房选用的石材、家具、卫浴设备都很高档,以性价比来说挺超值的.所以以前出差到昆山都住这家酒店.半年没来了,这次来发现他们员工都换了新制服,大堂的很多装饰也焕然一新,走进大堂还有淡淡的香味,还蛮惊喜的.之前几次一直住主楼,这次订房想换个风格,就选了他们的商务楼.这边房间和主楼配备几乎相同,只是卫生间是透明的,可以一边泡澡一边看电视,估计也是他们的一个卖点.今天上网无意中点开他们家的"所有房型"才发现,可能是夏天酒店业淡季的原因,现在携程上面订行政客房就能送1份108元早餐和50元足浴,还有鲜花水果报纸什么的.细算下来,搞促销的时候住行政客房还是很划算的,但愿下次过去的时候这个活动还没结束.这次入住唯一感觉不太满意的就是办入住的时间有点长,扫描证件弄了好几分钟,不过奥运期间好象酒店都这样.宾馆反馈2008年8月14日：尊敬的客人:您好感谢您选择君豪酒店,除了行政客房外,我们还有商务豪华客房也有优惠的促销,同样是订房就送早餐和足浴券.关于您所说的证件登记的问题,我们也是在配合奥运期间公安部门的安全规定,极力为客人提供一个安全舒适的环境,也请您能够理解.欢迎您下次入住,我们一定会为您提供优质的更好的服务. |
    


```python
x_train=data[:5000]
x_test=data[5000:]
y_train=x_train["label"]
y_test=x_test["label"]
del x_train["label"]
del x_test["label"]
```

构造一个模型


```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.nlp.preprocessing import *
from easymlops.nlp.representation import *
from easymlops.table.classification import *
from easymlops.table.ensemble import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(ExtractChineseWords())\
  .pipe(ExtractJieBaWords())\
  .pipe(Parallel([LsiTopicModel(num_topics=128),Word2VecModel(embedding_size=128)]))\
  .pipe(Normalizer())\
  .pipe(LGBMClassification(y=y_train,native_init_params={"max_depth":2},native_fit_params={"num_boost_round":128},prefix="lgbm"))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x18d88026088>



挂载


```python
from easymlops.table.storage import FaissStorage
```


```python
table[-2].set_branch_pipe(FaissStorage(raw_data=x_train["review"].values.tolist(),storage_path="./faiss_storage",
                                       create_index_param="IVF100,PQ16",create_index_measure="METRIC_INNER_PRODUCT"))
```

这里由于没有事先训练好index，所以要先训练一次，但如果已经训练好了，可以通过`load_index_raw_data`直接加载


```python
table[-2].get_branch_pipe(0).fit(table.transform(x_train,run_to_layer=-2))
```




    <FaissStorage(<class 'easymlops.table.storage.faiss_storage.FaissStorage'>, started 8940)>



### 检索最相似的k个数据


```python
input_data={"review":"地方偏僻，而且部根本是人服的，前天去住的候，空都有，酒店借口，已了，但是一都有，了省，搞的，..."}
result_df=table[-2].get_branch_pipe(0).search(table.transform_single(input_data,run_to_layer=-2))
print(result_df.to_markdown())
```

    |    | similarity_top_1_raw_data                              | similarity_top_2_raw_data                                                                                                                    | similarity_top_3_raw_data                                                                                                                                                                                                                                                               |   similarity_top_1_measure |   similarity_top_2_measure |   similarity_top_3_measure |
    |---:|:-------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------:|---------------------------:|---------------------------:|
    |  0 | 角房太冷了，tnnd。角房太冷了，tnnd。角房太冷了，tnnd。 | 房间太小.设施太垃圾.洗澡水半天都是凉水.放很久才有热水啊.洗澡间放5分钟就水漫金山了，所以原定住10天,结果住了2天就退定了,冰箱没接电,宽带还可以. | 交通不便是地方最最令我失望的地方，住了三天，第一天，打不到的士，酒店的司了商竟然收我40去一上打只要不到20的地方。不是的，而是趁火打劫的行在令我怒。房的第二晚上，竟然有空，只慢吹。而且房的隔音也不怎么，白天晚上都可以到KTV和夜的歌。本酒店的房，和格我喜的，不以上的，以後真的不敢恭。 |                    90.5594 |                    64.8806 |                    63.1392 |
    


```python
input_data=x_test[-5:]
result_df=table[-2].get_branch_pipe(0).search(table.transform(input_data,run_to_layer=-2))
print(result_df.to_markdown())
```

    |      | similarity_top_1_raw_data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | similarity_top_2_raw_data                                                                                                                                                                                              | similarity_top_3_raw_data                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |   similarity_top_1_measure |   similarity_top_2_measure |   similarity_top_3_measure |
    |-----:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------:|---------------------------:|---------------------------:|
    | 4566 | 交通很方便，服务态度不错，就是隔音条件不太好。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       | 总的来说,比较干净,而且地理位置很好,市区繁华地段.进出方便.                                                                                                                                                              | 服务不错，位置也比较方便，坐地铁、去虹桥机场都很方便。设施有点旧，隔音不好。房间还比较干净。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |                   247.216  |                   236.071  |                   233.069  |
    | 5590 | 10月19日上岛的，11点到大堂，房间还没有好要等到下午1点，于是寄存行李，步行欣赏沿途风光，快12点赶往冬季码头潜水，中午工作人员为了节省时间没有培训就把我们送上船了，好在我的东北教练很负责任给我额外培训，虽然受天气影响能见度不高只看到几只小丑鱼，海葵和珊瑚，但也是很美好的潜水回忆了。---上午的时间安排很成功，避免了等待及厌烦。下午1点结束潜水在冬季码头乘坐免费的电瓶车到大堂，直接拿房间钥匙，由电瓶车免费送上去。不过观海木楼在山顶，有多个台阶车子只送到一半，要自己走上去。房间是8626有很不错的角度看海景，房间很有特色，设施齐全室外还有独立的游泳池，蜜月旅行酒店还特意送了果盘，潜水耳朵进水要了棉签，服务生很快就爬上山送来了，感觉这里的服务很错的。不过因为出于“室外桃源，人间仙境”的岛上，人和小动物和谐共处，所以有很多小家伙出入，我们看到好几只壁虎，小虫，飞蛾还有蛤蟆，蛤蟆从门缝钻进房间一跳就找不到了，我紧张的打电话给前台问是否会有蛇，蛤蟆还不算可怕，如果有蛇就太恐怖了，服务生说房间附近的草木都洒了硫磺，绝对不会有蛇的，还热心的问是否需要来帮我抓蛤蟆，被我们谢绝了。害怕有别的小东西再进来，老公用报纸堵住门缝，喷了防蚊药水，还把墙上的几个木头洞用纸巾堵上了，不过我还是睡不踏实，半夜4点惊醒尖叫，原来蛤蟆跳到被窝里把我吓醒了！老公惊醒就来说赶蛤蟆，那场景真实难忘啊！回来的路上老公还戏称这为“美好的回忆”呢！心惊胆战的度过一夜人很累一直睡到早上10点多，错过了看日出，在木楼上景色很不错，我们享受好美景就去退房，速度很快，服务态度也很好。然后又帮我们把行李存好。退房后我们又继续沿海滩行走，玩水到中午12点去海鲜自助餐厅吃“早中饭”，40元一人，虽然海鲜种类不多，可无限量供应的和乐蟹让人过足了瘾，伴着碧海蓝天椰林进餐，真是享受啊。吃好饭继续边走边看，一直走到未开发禁止通行牌才打道回大堂，再乘免费电瓶车送到码头坐船。时间很充裕的，因为我们把无谓的等待时间都利用起来了，感觉很不错。 | 手机已收到短消息回复订房OK，前台自己没查到，却叫我打电话给携程，要我自己搞定，即使我给前台看确认的短消息，一样不理，后来我自己打了电话，让携程的客服跟前台通话后，他们就找到了，这就是锦江之星（上海张江店）的服务态度 | 上海浦东雅诗阁，宽带是要额外收费的，然而在携程的网站上（，房间价格的表格中，宽带下面只是写着“宽带”两个字，后来在酒店结账的时候发现宽带是额外收费的，我给携程打电话，才被告知，“宽带”这两个字仅仅表明这个酒店有宽带设施！其实在我使用宽带之前，已经和前台询问是否收费，前台当时可能忙，说让我先用，结帐的时候免去。结果结账的时候是另一个服务生，拒绝免除宽带费。让我向携程讨公道去。知道我把他们的大堂副经理叫了出来，我的外国同事用英语把这个经理数落了一番之后，他们才免除了这笔费用。其实上网费并不多，几百元钱，但是这种出尔反尔，先告诉我免费，再收费的做法实在是过分。而且，对中国人和外国人的态度明显不一样，也让我感到很不爽。 |                   158.007  |                   154.179  |                   145.682  |
    |  281 | 酒店服务意识很好，很热情，主动介绍酒店的特色。感觉不错。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 很好的地理位置，一蹋糊涂的服务，萧条的酒店。                                                                                                                                                                           | 非常好的酒店，四星的标准完全超值的享受，服务非常好                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |                    87.1075 |                    85.8245 |                    84.1975 |
    | 1875 | 酒店位置十分偏远，交通，购物，餐饮极不方便                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 交通挺方便的，离机场步行十分钟，离火车站车行十分钟。可能最近是淡季吧，有点冷清。                                                                                                                                       | 酒店有世外桃的感,境非常美,不如果有自,交通就不是很方便了.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |                   179.902  |                   174.152  |                   169.272  |
    | 5119 | 第二次入住这个酒店了，服务很好，让客人感到很舒服，很满意。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 酒店服务意识很好，很热情，主动介绍酒店的特色。感觉不错。                                                                                                                                                               | 是非常不错的酒店，服务也很周到，赞赏香港人的服务意识。                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |                   113.135  |                   112.554  |                   103.822  |
    

### 为生产数据创建索引

如果需要为生产数据创建索引，需要设置`add_index=True`，以及在`storage_base_dict`中添加`row_data`，而持久化生产上的`index`和`raw_data`，需要自己**手动**调用`save_index_raw_data`接口，比如下面模拟生产数据


```python
i=0
for record in tqdm(x_test[:10].to_dict("record")):
    table.transform_single(record,storage_base_dict={"key":i,"row_data":record.get("review")},add_index=True)
    i+=1
```

    100%|██████████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, 32.93it/s]
    


```python
input_data=x_test[:10].to_dict("record")[1]
result_df=table[-2].get_branch_pipe(0).search(table.transform_single(input_data,run_to_layer=-2))
print(result_df.to_markdown())
```

    |    | similarity_top_1_raw_data                                                                                                                                                                                                                                                                                                                                                       | similarity_top_2_raw_data                              | similarity_top_3_raw_data                                    |   similarity_top_1_measure |   similarity_top_2_measure |   similarity_top_3_measure |
    |---:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------------------------------------------|:-------------------------------------------------------------|---------------------------:|---------------------------:|---------------------------:|
    |  0 | 听说金仕顿已经是4星级了，感觉和3星没什么区别，最不能忍受的是上网问题，普通的标准间竟然还是电话线拨号上网，而且奇贵无比，只好多花点银子住了商务房，如果不是来交通厅办事，打死不会住这里，福州其他以前感觉不错的4星宾馆现在都升5星了，看来福州宾馆的水平在全国来讲算偏低水平了。还要提醒酒店的是，早餐自助餐的花样太少，每每转几圈竟然不知要吃什么。|1|2023-03-13 21:10:47.876759 | 住的是一楼，房间太小，窗户对着一堵墙。以后不会再住它了 | 性价比较高，我住了一晚之后从乐山回来又续订了两晚，比较满意。 |                     65.514 |                    65.3999 |                    62.6765 |
    


```python
input_data
```




    {'review': '听说金仕顿已经是4星级了，感觉和3星没什么区别，最不能忍受的是上网问题，普通的标准间竟然还是电话线拨号上网，而且奇贵无比，只好多花点银子住了商务房，如果不是来交通厅办事，打死不会住这里，福州其他以前感觉不错的4星宾馆现在都升5星了，看来福州宾馆的水平在全国来讲算偏低水平了。还要提醒酒店的是，早餐自助餐的花样太少，每每转几圈竟然不知要吃什么。'}



有时候你可能会发现，完全相同的两个向量未必能被检索为相似度最高,这与你创建索引的方式以及评估相似度方式相关，通常来说，如果想要更高的召回，那就需要更多的内存和计算时间开销，请根据你自己的使用场景合理选择

## ElasticSearch存储
you know, for search  

这里需要注意一下`pip install elasticsearch==x.x.x`的版本需要与服务端一致


```python
import pandas as pd
data=pd.read_csv("./data/demo.csv",encoding="gbk")
x_train=data[:500]
x_test=data[500:]
y_train=x_train["Survived"]
y_test=x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]
```


```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
```

构建模型


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x2262dd51148>



挂载存储模块


```python
from easymlops.table.storage import ElasticSearchStorage
table[-1].set_branch_pipe(ElasticSearchStorage(index_name="survived_index",es_args=["http://192.168.244.133:9200"],es_kwargs=dict(timeout=60)))
```

### 模拟生产数据流


```python
i=0
for record in tqdm(x_test[:100].to_dict("record")):
    table.transform_single(record,storage_base_dict={"key":i})
    i+=1
```

    100%|██████████████████████████████████████████████████████████████████████████████| 100/100 [00:00<00:00, 2630.66it/s]
    

### 查询key


```python
print(table[-1].get_branch_pipe(0).select_key(key="3").to_markdown())
```

    |    |   storage_key | storage_transform_time     |   Age |   Cabin | Embarked   |    Fare | Name                           |   Parch |   PassengerId |   Pclass | Sex    |   SibSp |   Ticket |
    |---:|--------------:|:---------------------------|------:|--------:|:-----------|--------:|:-------------------------------|--------:|--------------:|---------:|:-------|--------:|---------:|
    |  0 |             3 | 2023-03-25 10:13:11.044650 |    37 |     nan | S          | 9.58594 | Laitinen, Miss. Kristina Sofia |       0 |           504 |        3 | female |       0 |     4135 |
    

### query搜索 
前三条与`Name`最相似的记录


```python
body = {
    'query': {
        'match': {
            'Name': 'Lester, Mr. James'
        }
    },
    'sort': {
        '_score': {                    
            'order': 'desc'        
        }
    },
    'from':0,
    'size':3
}
```


```python
query_output=table[-1].get_branch_pipe(0).query(body=body)
print(query_output.to_markdown())
```

    |    |   PassengerId |   Pclass | Name                      | Sex   |   Age |   SibSp |   Parch | Ticket           |     Fare | Cabin   | Embarked   |   storage_key | storage_transform_time     |   hit_score_ |
    |---:|--------------:|---------:|:--------------------------|:------|------:|--------:|--------:|:-----------------|---------:|:--------|:-----------|--------------:|:---------------------------|-------------:|
    |  0 |           512 |        3 | Webber, Mr. James         | male  |   0   |       0 |       0 | SOTON/OQ 3101316 |  8.04688 | nan     | S          |            11 | 2023-03-25 09:52:43.008322 |      3.64815 |
    |  1 |           526 |        3 | Farrell, Mr. James        | male  |  40.5 |       0 |       0 | 367232           |  7.75    | nan     | Q          |            25 | 2023-03-25 09:52:43.013323 |      3.64815 |
    |  2 |           513 |        1 | McGough, Mr. James Robert | male  |  36   |       0 |       0 | PC 17473         | 26.2812  | E25     | S          |            12 | 2023-03-25 09:52:43.009328 |      3.27887 |
    

### query聚合  
Pclass分组下最大的Fare


```python
body = {
    '_source':['Pclass','Fare'],
    'aggs': {
        'Pclass': {
            'max': {
                'field':'Fare'
            }
        }
    },
    'sort': {
        'Pclass': {                    
            'order': 'asc'        
        }
    },
    'from':0,
    'size':3
}
```


```python
agg_output=table[-1].get_branch_pipe(0).query(body=body)
print(agg_output.to_markdown())
```

    |    |   Pclass |     Fare |
    |---:|---------:|---------:|
    |  0 |        1 |  86.5    |
    |  1 |        1 | 108.875  |
    |  2 |        1 |  26.5469 |
    

通过`body`设置不同的查询语法可以获得更加复杂的查询结果，具体语法还请自行查看资料，另外保证很大的自由度，下面`search`可以直接透传原始的es的`search`  
### search透传


```python
table[-1].get_branch_pipe(0).search(body=body)
```




    {'took': 4,
     'timed_out': False,
     '_shards': {'total': 1, 'successful': 1, 'skipped': 0, 'failed': 0},
     'hits': {'total': {'value': 101, 'relation': 'eq'},
      'max_score': None,
      'hits': [{'_index': 'survived_index2',
        '_type': '_doc',
        '_id': '4',
        '_score': None,
        '_source': {'Pclass': 1, 'Fare': 86.5},
        'sort': [1]},
       {'_index': 'survived_index2',
        '_type': '_doc',
        '_id': '5',
        '_score': None,
        '_source': {'Pclass': 1, 'Fare': 108.875},
        'sort': [1]},
       {'_index': 'survived_index2',
        '_type': '_doc',
        '_id': '7',
        '_score': None,
        '_source': {'Pclass': 1, 'Fare': 26.546875},
        'sort': [1]}]},
     'aggregations': {'Pclass': {'value': 227.5}}}



## Kafka存储  
这里主要作为生产者的角色往kafka发送数据，具体的消费者需要自行实现


```python
import pandas as pd
data=pd.read_csv("./data/demo.csv",encoding="gbk").sample(frac=1)
print(data.head(5).to_markdown())
```

    |     |   PassengerId |   Survived |   Pclass | Name                    | Sex    |   Age |   SibSp |   Parch | Ticket     |    Fare |   Cabin | Embarked   |
    |----:|--------------:|-----------:|---------:|:------------------------|:-------|------:|--------:|--------:|:-----------|--------:|--------:|:-----------|
    | 152 |           153 |          0 |        3 | Meo, Mr. Alfonzo        | male   |  55.5 |       0 |       0 | A.5. 11206 |  8.05   |     nan | S          |
    | 485 |           486 |          0 |        3 | Lefebre, Miss. Jeannie  | female | nan   |       3 |       1 | 4133       | 25.4667 |     nan | S          |
    |  57 |            58 |          0 |        3 | Novel, Mr. Mansouer     | male   |  28.5 |       0 |       0 | 2697       |  7.2292 |     nan | C          |
    | 344 |           345 |          0 |        2 | Fox, Mr. Stanley Hubert | male   |  36   |       0 |       0 | 229236     | 13      |     nan | S          |
    | 301 |           302 |          1 |        3 | McCoy, Mr. Bernard      | male   | nan   |       2 |       0 | 367226     | 23.25   |     nan | Q          |
    


```python
x_train=data[:500]
x_test=data[500:]
y_train=x_train["Survived"]
y_test=x_test["Survived"]
del x_train["Survived"]
del x_test["Survived"]
```


```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.classification import *
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(FillNa())\
  .pipe(OneHotEncoding(cols=["Pclass", "Sex"], drop_col=False)) \
  .pipe(WOEEncoding(cols=["Ticket", "Embarked", "Cabin", "Sex", "Pclass"], y=y_train)) \
  .pipe(LabelEncoding(cols=["Name"]))\
  .pipe(LGBMClassification(y=y_train,native_init_params={"max_depth":2},native_fit_params={"num_boost_round":128},prefix="lgbm"))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x27f2525b8c8>



挂载


```python
from easymlops.table.storage import KafkaStorage
table[-1].set_branch_pipe(KafkaStorage(bootstrap_servers="192.168.244.133:9092",topic_name="label_encoding",cols=['lgbm_0','lgbm_1']))
```

### 模拟线上数据流


```python
i=0
for record in tqdm(x_test[:5].to_dict("record")):
    table.transform_single(record,storage_base_dict={"key":i})
    i+=1
```

    100%|███████████████████████████████████████████████████████████████████████████████████| 5/5 [00:00<00:00, 166.62it/s]
    

### 模拟消费


```python
from kafka import KafkaConsumer
import json
consumer = KafkaConsumer('label_encoding',group_id="easymlops", bootstrap_servers="192.168.244.133:9092",value_deserializer=lambda m: json.loads(m.decode('ascii')))
```


```python
for message in consumer:
    print(message)
```

    ConsumerRecord(topic='label_encoding', partition=0, offset=100, timestamp=1680707688289, timestamp_type=0, key=None, value={'lgbm_0': '0.8178924384195481', 'lgbm_1': '0.182107561580452', 'storage_key': '0', 'storage_transform_time': '2023-04-05 23:14:48.271271'}, headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=139, serialized_header_size=-1)
    ConsumerRecord(topic='label_encoding', partition=0, offset=101, timestamp=1680707688290, timestamp_type=0, key=None, value={'lgbm_0': '0.7956256701057736', 'lgbm_1': '0.20437432989422644', 'storage_key': '1', 'storage_transform_time': '2023-04-05 23:14:48.275281'}, headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=141, serialized_header_size=-1)
    ConsumerRecord(topic='label_encoding', partition=0, offset=102, timestamp=1680707688290, timestamp_type=0, key=None, value={'lgbm_0': '0.7603010291586166', 'lgbm_1': '0.23969897084138336', 'storage_key': '2', 'storage_transform_time': '2023-04-05 23:14:48.280271'}, headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=141, serialized_header_size=-1)
    ConsumerRecord(topic='label_encoding', partition=0, offset=103, timestamp=1680707688290, timestamp_type=0, key=None, value={'lgbm_0': '0.010905936465706449', 'lgbm_1': '0.9890940635342935', 'storage_key': '3', 'storage_transform_time': '2023-04-05 23:14:48.284272'}, headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=142, serialized_header_size=-1)
    ConsumerRecord(topic='label_encoding', partition=0, offset=104, timestamp=1680707688299, timestamp_type=0, key=None, value={'lgbm_0': '0.6324117990867535', 'lgbm_1': '0.36758820091324645', 'storage_key': '4', 'storage_transform_time': '2023-04-05 23:14:48.289273'}, headers=[], checksum=None, serialized_key_size=-1, serialized_value_size=141, serialized_header_size=-1)
    


```python

```
