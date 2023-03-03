# 特征存储&监控&分析

模型部署生产后，我们通常还有些后续需求：  

- 存储：将某些pipe的输出直接保存到数据库存储，比如特征工程后的数据等；
- 监控&分析：监控数据短期内的变化情况，分析异常指标等

## 存储的两种方式  

- 嵌入式：算法研发的同学也肩负了后续指标监控、报表统计等运维形式的工作，这样在研发过程就可以把存储pipe模块嵌入到pipeline中；   
- 挂载式：算法研发只需要关注算法相关的模块，后续上线时，运维同学再把存储模块挂载到需要保存的pipe模块后，这种方式更加灵活   

**另外，目前存储模块只有本地存储，后面陆续会开发基于hbase,kafka,redis,es,mysql等存储模块，由于本地存储基于sqlite3，还请**   
`pip install sqlite3`


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
from easymlops.table.storage import LocalStorage
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




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x20130bb7f48>



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

    |    |   key | transform_time      |   PassengerId |   Pclass |   Name |     Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|------:|:--------------------|--------------:|---------:|-------:|--------:|------:|--------:|--------:|---------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |   501 | 2023-03-03 19:04:09 |           501 | 0.482439 |      0 | 1.11138 |    17 |       0 |       0 |        0 |  8.664 | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    


```python
#查看lgbm classification的输出
print(table[-1].select_key(key=501).to_markdown())
```

    |    |   key | transform_time      |   lgbm_0 |   lgbm_1 |
    |---:|------:|:--------------------|---------:|---------:|
    |  0 |   501 | 2023-03-03 19:04:09 | 0.923326 | 0.076674 |
    

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




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x201345a5cc8>



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

    |    |   key | transform_time      |   PassengerId |   Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|------:|:--------------------|--------------:|---------:|-------:|---------:|------:|--------:|--------:|---------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |   502 | 2023-03-03 19:04:20 |           502 | 0.482439 |      0 | -1.56999 |    21 |       0 |       0 |        0 |   7.75 | 0.299607 |  -0.508609 |          1 |          0 |          0 |          0 |            1 |
    


```python
print(table[-1].get_branch_pipe(0).select_key(key=502).to_markdown())
```

    |    |   key | transform_time      |   lgbm_0 |   lgbm_1 |
    |---:|------:|:--------------------|---------:|---------:|
    |  0 |   502 | 2023-03-03 19:04:20 | 0.373652 | 0.626348 |
    

由于对当前pipe而言都只挂载了一个pipe，所以可以通过`get_branch_pipe(0)`获取，如果还继续挂载了其他pipe模块，通过增加index获取

## 监控&分析  

目前设计了如下几个函数做监控和分析用，接下来模拟更多的生产数据


```python
for record in tqdm(x_test.to_dict("record")[2:]):
    table.transform_single(record,storage_base_dict={"key":record.get("PassengerId")})
```

    100%|███████████████████████████████████████████████████████████████████████████████| 389/389 [00:01<00:00, 298.94it/s]
    

### 查询key


```python
print(table[-2].get_branch_pipe(0).select_key(key=502).to_markdown())
```

    |    |   key | transform_time      |   PassengerId |   Pclass |   Name |      Sex |   Age |   SibSp |   Parch |   Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|------:|:--------------------|--------------:|---------:|-------:|---------:|------:|--------:|--------:|---------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |   502 | 2023-03-03 19:04:20 |           502 | 0.482439 |      0 | -1.56999 |    21 |       0 |       0 |        0 |   7.75 | 0.299607 |  -0.508609 |          1 |          0 |          0 |          0 |            1 |
    

### 复杂查询


```python
print(table[-2].get_branch_pipe(0).where("Pclass>0.4 and Sex>1 and Sex_male=1",limit=5).to_markdown())
```

    |    |   key | transform_time      |   PassengerId |   Pclass |   Name |     Sex |   Age |   SibSp |   Parch |    Ticket |   Fare |    Cabin |   Embarked |   Pclass_3 |   Pclass_1 |   Pclass_2 |   Sex_male |   Sex_female |
    |---:|------:|:--------------------|--------------:|---------:|-------:|--------:|------:|--------:|--------:|----------:|-------:|---------:|-----------:|-----------:|-----------:|-----------:|-----------:|-------------:|
    |  0 |   501 | 2023-03-03 19:04:09 |           501 | 0.482439 |      0 | 1.11138 |    17 |       0 |       0 |  0        |  8.664 | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  1 |   509 | 2023-03-03 19:04:26 |           509 | 0.482439 |      0 | 1.11138 |    28 |       0 |       0 |  0        | 22.53  | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  2 |   510 | 2023-03-03 19:04:26 |           510 | 0.482439 |      0 | 1.11138 |    26 |       0 |       0 | -0.464158 | 56.5   | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    |  3 |   511 | 2023-03-03 19:04:26 |           511 | 0.482439 |      0 | 1.11138 |    29 |       0 |       0 |  0        |  7.75  | 0.299607 |  -0.508609 |          1 |          0 |          0 |          1 |            0 |
    |  4 |   512 | 2023-03-03 19:04:26 |           512 | 0.482439 |      0 | 1.11138 |     0 |       0 |       0 |  0        |  8.05  | 0.299607 |   0.224849 |          1 |          0 |          0 |          1 |            0 |
    


```python
#存活率更高的乘客
print(table[-1].get_branch_pipe(0).where("lgbm_1>0.8",limit=5).to_markdown())
```

    |    |   key | transform_time      |    lgbm_0 |   lgbm_1 |
    |---:|------:|:--------------------|----------:|---------:|
    |  0 |   505 | 2023-03-03 19:04:26 | 0.0684703 | 0.93153  |
    |  1 |   514 | 2023-03-03 19:04:26 | 0.0718778 | 0.928122 |
    |  2 |   517 | 2023-03-03 19:04:26 | 0.113928  | 0.886072 |
    |  3 |   519 | 2023-03-03 19:04:26 | 0.19458   | 0.80542  |
    |  4 |   521 | 2023-03-03 19:04:26 | 0.0710767 | 0.928923 |
    

### 聚合分析

不同sex_male,pclass_3下的统计量


```python
return_df=table[-2].get_branch_pipe(0).group_agg_where(group_by="Sex_male,Pclass_3", agg_sql="Sex_male,Pclass_3,max(PassengerId) as PassengerId_max,sum(Sex)/count(Sex) as Sex_mean,count(Ticket) as Ticket_cnt", where_sql="transform_time>='2023-03-03 01:01:01'", limit=10)
print(return_df.to_markdown())
```

    |    |   Sex_male |   Pclass_3 |   PassengerId_max |   Sex_mean |   Ticket_cnt |
    |---:|-----------:|-----------:|------------------:|-----------:|-------------:|
    |  0 |          0 |          0 |               888 |   -1.56999 |           75 |
    |  1 |          0 |          1 |               889 |   -1.56999 |           54 |
    |  2 |          1 |          0 |               890 |    1.11138 |          104 |
    |  3 |          1 |          1 |               891 |    1.11138 |          158 |
    

### 纯SQL
以上接口不支持复杂的sql嵌套，这里可以直接定义复杂的sql嵌套分析


```python
table_name="predict"
sql=f"""
-- 统计是否存活用户的最早最晚transform时间
select survied_pred,min(transform_time) as transform_min,max(transform_time) as transform_max from 
    -- 如果lgbm_1>0.5就视为存活
    (select key,transform_time,case when lgbm_1>0.5 then 1 else 0 end as survied_pred from {table_name}) as t 
group by survied_pred
"""
return_df=table[-1].get_branch_pipe(0).sql(sql)
```


```python
print(return_df.to_markdown())
```

    |    |   survied_pred | transform_min       | transform_max       |
    |---:|---------------:|:--------------------|:--------------------|
    |  0 |              0 | 2023-03-03 19:04:09 | 2023-03-03 19:04:27 |
    |  1 |              1 | 2023-03-03 19:04:20 | 2023-03-03 19:04:27 |
    


```python

```
