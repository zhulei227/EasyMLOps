# 自定义pipe模块

由设计逻辑出发，逐步对pipe所支持的功能展开介绍

## pipe设计逻辑

pipe是构成pipeline的最小执行单元，它要做到如下特性：   

- 独立：每个pipe就是一个独立的个体，这样一旦出现问题，通常只需要修复问题pipe模块，不需要去review全局代码； 
- 全面：同时每个pipe又要做到**麻雀虽小、五脏俱全**，它需要再自己内部完成模型的训练、预测、生产预测等基本功能；  
- 通用：所有pipe应该遵循同一套标准，规范输入和输出，这样对外调用透明，pipeline可以方便的流转整合这些pipe  

功能层面，按需求层次可以拆分如下：   

- 最低需求-数据分析：只需要完成对数据的fit和transform即可  
- 持久化需求：还想要复用模块，那就需要实现set_params和get_params  
- 生产需求：需要实现transform_single，保证与transform一致  
- 安全需求：更稳健的函数实现方式
- 扩展需求：
    - 自行定义添加新函数：后续在pipeline中会介绍如果获取到pipe并调用  
    - 回调函数的方式：这样不会使得pipe越来越臃肿，只保留自身相关的功能

- 高阶需求：
    - 依赖需求：获取当前pipe的前置pipe以完成一些需求，比如调用前置pipe的transform 
    - 挂载需求：当前pipe需要后续执行一些额外的操作，但这些操作又不需要在作用在pipeline中，比如记录当前pipe的输出，这是可以将额外操作封装为一个pipe，并设置为branch_pipe即可


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

## 最低需求：数据分析  
如果只是临时性的数据分析，而且需要用到easymlops的其他pipe模块，那么只需要实现fit和transform即融入，下面以TargetEncoding举例


```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import *
from easymlops.table.encoding import *
from easymlops.table.perfopt import *
from easymlops.table.decomposition import *
```


```python
class UDFTargetEncoding(TablePipeObjectBase):
    def __init__(self,y=None,cols=None, error_value=0):
        super().__init__()
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def fit(self, s, **kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def transform(self, s, **kwargs):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].map(self.target_map_detail[col]).fillna(self.error_value)
        return s
```

我们看到这个类里面定义了3个函数，它们的功能如下：   
- `__init__`函数：这是对象初始化函数，在新建对象的时候就会调用的函数（如果不清楚，可以补充一下python面向对象编程方面的知识），然后有几个入参，`y`即是`target`，`cols`表示要对那些列做encoding，`error_value`表示预测阶段不被命中的target encoding的默认值；同时通过`self.xxx`将这些参数共享（变量名称前加`self.`表示为成员变量，可以在各成员函数中共享），新建立了一个`self.target_map_detail`用于记录各col的各个取值对应的targetencoding值，另外这里面有个`super().__init__()`它表示调用父类`TablePipeObjectBase`中的初始化方法 

- `fit`函数：fit阶段将cols中的col逐个取出，并与y拼接，对col中各取值做分组求y的均值，然后将其对应关系保存到`self.target_map_detail`

- `transform`函数：拿到各col的map关系，做map即可


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(UDFTargetEncoding(cols=["Embarked"],y=y_train))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Age |     Fare |   Embarked |
    |----:|------:|---------:|-----------:|
    | 500 |    17 |  8.66406 |   0.334254 |
    | 501 |    21 |  7.75    |   0.511111 |
    | 502 |   nan |  7.62891 |   0.511111 |
    | 503 |    37 |  9.58594 |   0.334254 |
    | 504 |    16 | 86.5     |   0.334254 |
    

## 持久化需求  
但这时我们的模型还只能临时用一用，想保留下来留着下次使用还得实现另外一组函数get_params和set_params，
- get_params：将告诉上层，需要持久化的参数有那些 
- set_params：是上层给你这个参数，你要怎么还原为成员变量   

具体要保留那些参数，最小要求是transform所需的那些，接下来加入get_params和set_params


```python
class UDFTargetEncoding(TablePipeObjectBase):
    def __init__(self,y=None,cols=None, error_value=0):
        super().__init__()
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def fit(self, s, **kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def transform(self, s, **kwargs):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].map(self.target_map_detail[col]).fillna(self.error_value)
        return s
    
    def get_params(self):
        #获取父类的params
        params=super().get_params()
        #加入当前的参数
        params.update({"target_map_detail": self.target_map_detail, "error_value": self.error_value,"cols":self.cols})
        return params
    
    def set_params(self, params):
        #设置父类的params
        super().set_params(params)
        #再设置当前层的params
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.cols=params["cols"]
```

这里需要注意父类里面还有参数需要保存，所以有`super().get_params()`和`super().set_params(params)`


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(UDFTargetEncoding(cols=["Embarked"],y=y_train))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Age |     Fare |   Embarked |
    |----:|------:|---------:|-----------:|
    | 500 |    17 |  8.66406 |   0.334254 |
    | 501 |    21 |  7.75    |   0.511111 |
    | 502 |   nan |  7.62891 |   0.511111 |
    | 503 |    37 |  9.58594 |   0.334254 |
    | 504 |    16 | 86.5     |   0.334254 |
    

### 保存


```python
table.save("table.pkl")
```

### 加载


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols())\
  .pipe(UDFTargetEncoding())

table.load("table.pkl")
```


```python
print(table.transform(x_test).head(5).to_markdown())
```

    |     |   Age |     Fare |   Embarked |
    |----:|------:|---------:|-----------:|
    | 500 |    17 |  8.66406 |   0.334254 |
    | 501 |    21 |  7.75    |   0.511111 |
    | 502 |   nan |  7.62891 |   0.511111 |
    | 503 |    37 |  9.58594 |   0.334254 |
    | 504 |    16 | 86.5     |   0.334254 |
    

## 生产需求
线上生产环境需要：   

1）更加快速的预测； 

2）输入数据格式通常是json； 

为了匹配生产，单独设计的了另一个函数transform_single，用于生产上的数据预测，它的输入是dict格式(解析json后),要求输出要与transform的一致


```python
class UDFTargetEncoding(TablePipeObjectBase):
    def __init__(self,y=None,cols=None, error_value=0):
        super().__init__()
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def fit(self, s,**kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def transform(self, s,**kwargs):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].map(self.target_map_detail[col]).fillna(self.error_value)
        return s
    
    def transform_single(self, s,**kwargs):
        for col in self.cols:
            if col in s.keys():
                s[col] = self.target_map_detail[col].get(s[col],self.error_value)
        return s
    
    def get_params(self):
        #获取父类的params
        params=super().get_params()
        #加入当前的参数
        params.update({"target_map_detail": self.target_map_detail, "error_value": self.error_value,"cols":self.cols})
        return params

    def set_params(self, params):
        #设置父类的params
        super().set_params(params)
        #再设置当前层的params
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.cols=params["cols"]
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(UDFTargetEncoding(cols=["Embarked"],y=y_train))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Age |     Fare |   Embarked |
    |----:|------:|---------:|-----------:|
    | 500 |    17 |  8.66406 |   0.334254 |
    | 501 |    21 |  7.75    |   0.511111 |
    | 502 |   nan |  7.62891 |   0.511111 |
    | 503 |    37 |  9.58594 |   0.334254 |
    | 504 |    16 | 86.5     |   0.334254 |
    


```python
#取第一条数据
record=x_test[:1].to_dict("record")[0]
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
table.transform_single(record)
```




    {'Age': 17.0, 'Fare': 8.664, 'Embarked': 0.3342541436464088}



## 安全需求  
以上5个函数的实现几乎可以让项目上线了，但还存在一些安全隐患，比如：

- 输入输出数据的类型是否有校验  
- 输入输出数据的顺序是否就校验  
- set_params和get_params时有可能搞忘记存储父类的params
- 当前命名的参数与父类的参数名称可能有冲突  

为了更加安全，可以将上面实现的fit,transform,transform_single,set_params和get_params修改为udf_fit,udf_transform,udf_transform_single,udf_set_params,udf_get_params，内置的方法会帮你自动完成这些检验


```python
class UDFTargetEncoding(TablePipeObjectBase):
    def __init__(self,y=None,cols=None, error_value=0):
        super().__init__()
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def udf_fit(self, s,**kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def udf_transform(self, s,**kwargs):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].map(self.target_map_detail[col]).fillna(self.error_value)
        return s
    
    def udf_transform_single(self, s,**kwargs):
        for col in self.cols:
            if col in s.keys():
                s[col] = self.target_map_detail[col].get(s[col],self.error_value)
        return s
    
    def udf_get_params(self):
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value,"cols":self.cols}

    def udf_set_params(self, params):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.cols=params["cols"]
```

这里，添加了udf前缀后，就不需要考虑父类的params了


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(UDFTargetEncoding(cols=["Embarked"],y=y_train))

x_test_new=table.fit(x_train).transform(x_test)
print(x_test_new.head(5).to_markdown())
```

    |     |   Age |     Fare |   Embarked |
    |----:|------:|---------:|-----------:|
    | 500 |    17 |  8.66406 |   0.334254 |
    | 501 |    21 |  7.75    |   0.511111 |
    | 502 |   nan |  7.62891 |   0.511111 |
    | 503 |    37 |  9.58594 |   0.334254 |
    | 504 |    16 | 86.5     |   0.334254 |
    

## 扩展需求 

上面这些函数是一些必需的函数，但我们同时还有一些额外自己的需求，比如我想要一个函数用来返回target encoding的具体map细节 ，这里有两种实现方式  
### 直接添加新函数 

这种方法最简单直接了，如下加了一个`show_detail`函数  


```python
class UDFTargetEncoding(TablePipeObjectBase):
    def __init__(self,y=None,cols=None, error_value=0):
        super().__init__()
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def udf_fit(self, s,**kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def show_detail(self):
        data = []
        for col, map_detail in self.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])
    
    def udf_transform(self, s,**kwargs):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].map(self.target_map_detail[col]).fillna(self.error_value)
        return s
    
    def udf_transform_single(self, s,**kwargs):
        for col in self.cols:
            if col in s.keys():
                s[col] = self.target_map_detail[col].get(s[col],self.error_value)
        return s
    
    def udf_get_params(self):
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value,"cols":self.cols}

    def udf_set_params(self, params):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.cols=params["cols"]
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(UDFTargetEncoding(cols=["Embarked"],y=y_train))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x261d04cdc08>




```python
print(table[-1].show_detail().to_markdown())
```

    |    | col      | bin_value   |   target_value |
    |---:|:---------|:------------|---------------:|
    |  0 | Embarked | C           |       0.521739 |
    |  1 | Embarked | Q           |       0.511111 |
    |  2 | Embarked | S           |       0.334254 |
    |  3 | Embarked | nan         |       1        |
    

这里`table[-1]`表示拿到`table`这个pipeline的`-1`层的pipe模块，具体的获取方式将会在后续pipeline中介绍

### 回调式  

上面这种方式有两个显著的缺点：  

- 如果这个TargetEncoding类是别人写好了，你没法直接修改别人的代码  
- 就算这个类是自己写的，随着添加的自定义方法越来越多，维护将会越来越困难  

所以，这里将一些额外功能剥离出去，统一通过回调的方式来增加新功能，回调定义以及实现在最顶层父类的方法中了，所以上面的类无需再自己实现即可进行回调，父类中的callback定义如下：  
```python
def callback(self, callback_func, data, return_callback_result=False, *args, **kwargs):
    result = callback_func(self, data, *args, **kwargs)
    if return_callback_result:
        return result
```

这里，`callback_func`即外部的回调函数定义，回调函数的第一个入场为`self`，即当前调用者这个pipe，第二个为`data`即需要传入的参数，后续`args`和`kwargs`为其他扩展参数，下面看看如果外部定义`show_detail`（可以发现只需要把`self`换成另外一个变量名称即可）


```python
def show_detail(pipe_object,data=None):
        data = []
        for col, map_detail in pipe_object.target_map_detail.items():
            for bin_value, target_value in map_detail.items():
                data.append([col, bin_value, target_value])
        return pd.DataFrame(data=data, columns=["col", "bin_value", "target_value"])
```


```python
#内部定义的show_detail去掉
class UDFTargetEncoding(TablePipeObjectBase):
    def __init__(self,y=None,cols=None, error_value=0,**kwargs):
        super().__init__(**kwargs)
        self.y=y
        self.cols=cols
        self.error_value = error_value
        self.target_map_detail = dict()

    def udf_fit(self, s,**kwargs):
        assert self.y is not None and len(self.y) == len(s)
        s["y_"] = self.y
        for col in self.cols:
            tmp_ = s[[col, "y_"]]
            col_map = list(tmp_.groupby([col]).agg({"y_": ["mean"]}).to_dict().values())[0]
            self.target_map_detail[col] = col_map
        del s["y_"]
        return self
    
    def udf_transform(self, s,**kwargs):
        for col in self.cols:
            if col in s.columns:
                s[col] = s[col].map(self.target_map_detail[col]).fillna(self.error_value)
        return s
    
    def udf_transform_single(self, s,**kwargs):
        for col in self.cols:
            if col in s.keys():
                s[col] = self.target_map_detail[col].get(s[col],self.error_value)
        return s
    
    def udf_get_params(self):
        return {"target_map_detail": self.target_map_detail, "error_value": self.error_value,"cols":self.cols}

    def udf_set_params(self, params):
        self.target_map_detail = params["target_map_detail"]
        self.error_value = params["error_value"]
        self.cols=params["cols"]
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(UDFTargetEncoding(cols=["Embarked"],y=y_train))

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x2b736dbe248>




```python
print(table[-1].callback(show_detail,data=None,return_callback_result=True).to_markdown())
```

    |    | col      | bin_value   |   target_value |
    |---:|:---------|:------------|---------------:|
    |  0 | Embarked | C           |       0.521739 |
    |  1 | Embarked | Q           |       0.511111 |
    |  2 | Embarked | S           |       0.334254 |
    |  3 | Embarked | nan         |       1        |
    

## 高阶需求  

### 获取父类依赖 

有时候，我们需要拿到父类的pipe对象做一些操作，如果对新的x重复运行父类的transform，这里内置pipeline已经帮忙完成了依赖关系的设置(必需通过xxx.pipe(xxx)方式添加pipe才会自动设置依赖关系 )，可以通过如下俩函数，获取直接父类对象，以及所有父类对象的列表 

- get_parent_pipe 
- get_all_parent_pipes


```python
table[-1].get_parent_pipe()
```




    <easymlops.table.preprocessing.core.SelectCols at 0x2b736e35508>




```python
table[-1].get_all_parent_pipes()
```




    [<easymlops.table.preprocessing.core.FixInput at 0x2b736db8b88>,
     <easymlops.table.preprocessing.core.SelectCols at 0x2b736e35508>]



### 挂载需求
我们另外还需要做些额外的功能，比如保存当前模块的输出结果到某些数据库，或者做一些监控，或者做一些实时/定时的分析等等....这些主流程以外的操作，都可以通过挂载的方式追加到当前pipe之后，这里主要通过下面3个函数来实现  

- set_branch_pipe：为当前pipe挂载一个pipe，挂载pipe分别再宿主pipe的fit,transform,transform_single后，运行自己的fit,transform,transform_single
- get_branch_pipe：由于可以设置多个挂载pipe，通过get_branch_pipe传入一个index获取到想要的挂载pipe  
- remove_branch_pipe：更具传入的index，移除不需要的pipe  

比如下面，引入一个存储模块，将TargetEncoding的输出保存到本地


```python
from easymlops.table.storage import LocalStorage
```


```python
table[-1].set_branch_pipe(LocalStorage(db_name="./local.db", table_name="target_output",cols=["Age","Fare","Embarked"]))
```


```python
#模拟线上预测一条数据
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
table.transform_single(record,storage_base_dict={"key":501})#这里需要额外传入也给key，标识数据的唯一ID
```




    {'Age': 17.0, 'Fare': 8.664, 'Embarked': 0.3342541436464088}




```python
print(table[-1].get_branch_pipe(0).select_key(key=501).to_markdown())
```

    |    |   key | transform_time      |   Age |   Fare |   Embarked |
    |---:|------:|:--------------------|------:|-------:|-----------:|
    |  0 |   501 | 2023-03-02 15:02:11 |    17 |  8.664 |   0.334254 |
    


```python

```
