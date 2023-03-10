# 生产部署 

首先训练一个简单模型


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
```


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(TargetEncoding(cols=["Embarked"],y=y_train))\
  .pipe(FillNa())

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x1a71f8f3ac8>




```python
table.transform_single({})
```




    {'Age': 0, 'Fare': 0, 'Embarked': 1.0}



## flask部署  

将transform_single的结果返回即可，如下极简单的方式即可发布rest接口
```python

from flask import Flask,request
app = Flask(__name__)
 
@app.route('/predict',methods=['GET', "POST"])
def predict():
    request_params=dict(request.args) if request.method=="GET" else request.get_data()
    return table.transform_single(request_params)

if __name__ == '__main__':
    app.run()
``` 

然后，在浏览器中... 

输入:http://localhost:5000/predict 可以得到`{"Age":0,"Embarked":1.0,"Fare":0}`  

输入:http://localhost:5000/predict?Age=12&Fare=8.5&Embarked=S 可以得到 `{"Age": 12.0, "Fare": 8.5, "Embarked": 0.3342541436464088}
`

## 日志记录 

记录日志可以帮助我们监控系统的运行情况，以及后续做一些分析，使用方式如下


```python
table=TablePipeLine()
table.pipe(FixInput())\
  .pipe(SelectCols(cols=["Age","Fare","Embarked"]))\
  .pipe(TablePipeLine().pipe(TargetEncoding(cols=["Embarked"],y=y_train)))\
  .pipe(FillNa())

table.fit(x_train)
```




    <easymlops.table.core.pipeline_object.TablePipeLine at 0x1c07f979988>




```python
#定义日志记录格式
import logging
logger = logging.getLogger("EasyMLOps")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
```

需要一个字典额外补充数据的关键信息比如`user_id`,`手机号`等...方便追踪数据


```python
base_log_info={"key":"user123"}
```


```python
output=table.transform_single({'PassengerId': 1,
 'Cabin': 0,
 'Pclass': 3,
 'Name': 'Braund, Mr. Owen Harris',
 'Sex': 'male',
 'Age': 22.0,
 'SibSp': 1,
 'Parch': 0,
 'Ticket': 'A/5 21171',
 'Fare': 7.25,
 'Embarked': 'S'},logger=logger,log_base_dict=base_log_info)
```

    2023-03-03 19:02:51,631 - EasyMLOps - INFO - {'step': 'step-0', 'pipe_name': <class 'easymlops.table.preprocessing.core.FixInput'>, 'transform': {'PassengerId': 1, 'Pclass': 3, 'Name': 'Braund, Mr. Owen Harris', 'Sex': 'male', 'Age': 22.0, 'SibSp': 1, 'Parch': 0, 'Ticket': 'A/5 21171', 'Fare': 7.25, 'Cabin': '0', 'Embarked': 'S'}, 'key': 'user123'}
    2023-03-03 19:02:51,634 - EasyMLOps - INFO - {'step': 'step-1', 'pipe_name': <class 'easymlops.table.preprocessing.core.SelectCols'>, 'transform': {'Age': 22.0, 'Fare': 7.25, 'Embarked': 'S'}, 'key': 'user123'}
    2023-03-03 19:02:51,634 - EasyMLOps - INFO - {'step': 'step-2-0', 'pipe_name': <class 'easymlops.table.encoding.TargetEncoding'>, 'transform': {'Age': 22.0, 'Fare': 7.25, 'Embarked': 0.3342541436464088}, 'key': 'user123'}
    2023-03-03 19:02:51,635 - EasyMLOps - INFO - {'step': 'step-3', 'pipe_name': <class 'easymlops.table.preprocessing.onevar_operation.FillNa'>, 'transform': {'Age': 22.0, 'Fare': 7.25, 'Embarked': 0.3342541436464088}, 'key': 'user123'}
    

如上是pipeline记录日志的基本格式，包括这些信息：  

- step：当前是pipeline的第几层，如果嵌套，会表示为 `step-x-x`这样的格式，比如上面的`step-2-0`，表示第2层的第0层，即TargetEncoding层  
- pipe_name：当前层所对应的pipe模块名称 
- transform：当前层的输出 
- 额外信息：即上面的字典补充内容，如用key来标识数据

## 测试

对于表格型模型上线主要关注如下问题：  

- 性能测试：测试每个pipe模块的transform_single性能情况，包括平均耗时、cpu、内存消耗等；
- 一致性测试：测试transform函数和transform_single函数的输出是否一致；
- 空值测试：空值有很多表现形式，比如没有、None、np.nan、null等都是空的表现，空值测试要求不同的空的情况下的输出要一致；
- 极端值测试：模型训练时的数据分布相对比较正常，而实际生产可能会有意外的极端值情况，比如训练时某变量都>0，而生产环境传过来的值=0等等，极端值测试就是测试各种极端取值情况下，pipeline能不能正常运行  
- 类型反转测试：离线训练时，某变量是浮点数，而到了生产它变成了字符串，检测这种情况下，pipeline能不能正常运行；
- int转float测试：主要检验精度变化时，pipeline能否保持一致的输出；


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
    

### 一致性测试&性能测试 

这里把一致性测试和性能测试放到一起了，主要检验如下：  
- 离线训练模型和在线预测模型的一致性，即tranform和transform_single的一致性，包括：  
    - 输出的shape是否一致
    - 输出的数据类型是否一致 
    - 输出的column名称
    - 相同column的数值是否相等 
- transform_single对当条数据的预测性能，包括： 
    - 平均预测一条数据的耗时 
    - 在运行期间cpu最高使用率 
    - 在运行期间内存的最大变化

这些可以通过调用如下函数，进行自动化测试：  
- check_transform_function：只要有打印complete，则表示在当前测试数据上transform和transform_single的输出一致，性能测试表示为speed:[*]毫秒/每条数据，以及运行过程中cpu的最大使用率和内存变化(最大内存-最小内存)，如果有异常则会直接抛出，并中断后续pipe模块的测试  

**注意：如果对于下面回调方式不理解，可以往前看：自定义pipe模块 那里的介绍**


```python
from easymlops.table.callback import check_transform_function_pipeline
table.callback(check_transform_function_pipeline,x_test,sample=10)
```

    (<class 'easymlops.table.preprocessing.core.FixInput'>) module check [transform] complete,speed:[0.3ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.preprocessing.onevar_operation.FillNa'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.encoding.OneHotEncoding'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.encoding.WOEEncoding'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.encoding.LabelEncoding'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.classification.LGBMClassification'>) module check [transform] complete,speed:[0.5ms]/it,cpu:[0%],memory:[0K]
    

### 空值测试

- 由于pandas在读取数据时会自动做类型推断，对空会有不同的处理，比如float设置为np.nan，对object设置为None或NaN  
- 而且pandas读取数据默认为批量读取批量推断，所以某一列数据空还不唯一，np.nan和None可能共存  

所以，这里对逐个column分别设置不同的空进行测试，测试内容：  
- 相同的空情况下，transform和transform_single是否一致  
- 不同的空的transform结果是否一致  

可通过`null_values=[None, np.nan, "null", "NULL", "nan", "NaN", "", "none", "None", " "]`(默认)设置自定义空值


```python
from easymlops.table.callback import check_null_value
table.callback(check_null_value,x_test,sample=10)
```

    column:[PassengerId] check [null value] complete,speed:[2.23ms]/it,cpu:[100%],memory:[0K]
    column:[Pclass] check [null value] complete,speed:[2.31ms]/it,cpu:[0%],memory:[0K]
    column:[Name] check [null value] complete,speed:[2.62ms]/it,cpu:[0%],memory:[0K]
    column:[Sex] check [null value] complete,speed:[2.42ms]/it,cpu:[0%],memory:[0K]
    column:[Age] check [null value] complete,speed:[2.04ms]/it,cpu:[0%],memory:[0K]
    column:[SibSp] check [null value] complete,speed:[2.47ms]/it,cpu:[0%],memory:[0K]
    column:[Parch] check [null value] complete,speed:[2.26ms]/it,cpu:[100%],memory:[0K]
    column:[Ticket] check [null value] complete,speed:[2.9ms]/it,cpu:[0%],memory:[0K]
    column:[Fare] check [null value] complete,speed:[2.42ms]/it,cpu:[100%],memory:[0K]
    column:[Cabin] check [null value] complete,speed:[2.52ms]/it,cpu:[0%],memory:[0K]
    column:[Embarked] check [null value] complete,speed:[2.63ms]/it,cpu:[0%],memory:[0K]
    

### 极端值测试

通常用于训练的数据都是经过筛选的正常数据，但线上难免会有极端值混入，比如你训练的某列数据范围在`0~1`之间，如果传入一个`-1`，也许就会报错，目前

- 对两种类型的分别进行极端测试，设置如下：
  - 数值型:设置`number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max]`(默认)
  - 离散型:设置`category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"]`(默认)  

- 将全部特征设置为如上的极端值进行测试

注意：这里只检测了transform与transform_single的一致性，不要求各极端值输入下的输出一致性(注意和上面的空值检测不一样，空值检测要求所有类型的空的输出也要一致)


```python
from easymlops.table.callback import check_extreme_value
table.callback(check_extreme_value,x_test,sample=10)
```

    column:[PassengerId] check [extreme value] complete,speed:[2.29ms]/it,cpu:[10%],memory:[5205900K]
    column:[Pclass] check [extreme value] complete,speed:[2.45ms]/it,cpu:[100%],memory:[5205728K]
    column:[Name] check [extreme value] complete,speed:[2.6ms]/it,cpu:[100%],memory:[5205700K]
    column:[Sex] check [extreme value] complete,speed:[2.14ms]/it,cpu:[100%],memory:[5205624K]
    column:[Age] check [extreme value] complete,speed:[2.25ms]/it,cpu:[100%],memory:[5205468K]
    column:[SibSp] check [extreme value] complete,speed:[2.6ms]/it,cpu:[100%],memory:[5205468K]
    column:[Parch] check [extreme value] complete,speed:[2.58ms]/it,cpu:[100%],memory:[5205240K]
    column:[Ticket] check [extreme value] complete,speed:[2.5ms]/it,cpu:[100%],memory:[5205172K]
    column:[Fare] check [extreme value] complete,speed:[2.54ms]/it,cpu:[100%],memory:[5205260K]
    column:[Cabin] check [extreme value] complete,speed:[2.57ms]/it,cpu:[100%],memory:[5205324K]
    column:[Embarked] check [extreme value] complete,speed:[2.7ms]/it,cpu:[100%],memory:[5205208K]
    column:[__all__] check [extreme value] complete,speed:[2.05ms]/it,cpu:[100%],memory:[5205228K]
    

极端测试的覆盖场景其实还不够，上面仅测试了单个变量取极端值的情况，而任意K(k>=2)个变量取极端值的情况并没有测试（测试成本太高了），所以上面的`__all__`这个场景是直接对全部变量取不同极端值做了一次测试

### 数据类型反转测试

某特征入模是数据是数值，但上线后传过来的是离散值，也有可能相反，这里就对这种情况做测试，对原是数值的替换为离散做测试，对原始离散值的替换为数值，替换规则如下：
- 原数值的，替换为：`number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"]`(默认)  
- 原离散的，替换为：`category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max]`(默认)  

同样，数据类型反转测试只对transform和transform_single的一致性有要求


```python
from easymlops.table.callback import check_inverse_dtype
table.callback(check_inverse_dtype,x_test,sample=10)
```

    column:[PassengerId] check [inverse type] complete,speed:[2.02ms]/it,cpu:[0%],memory:[0K]
    column:[Pclass] check [inverse type] complete,speed:[2.78ms]/it,cpu:[100%],memory:[5539488K]
    column:[Name] check [inverse type] complete,speed:[2.44ms]/it,cpu:[100%],memory:[5539492K]
    column:[Sex] check [inverse type] complete,speed:[2.7ms]/it,cpu:[100%],memory:[5539480K]
    column:[Age] check [inverse type] complete,speed:[2.66ms]/it,cpu:[100%],memory:[5539432K]
    column:[SibSp] check [inverse type] complete,speed:[2.99ms]/it,cpu:[100%],memory:[5539428K]
    column:[Parch] check [inverse type] complete,speed:[2.79ms]/it,cpu:[100%],memory:[5539432K]
    column:[Ticket] check [inverse type] complete,speed:[2.42ms]/it,cpu:[100%],memory:[5539432K]
    column:[Fare] check [inverse type] complete,speed:[2.14ms]/it,cpu:[100%],memory:[5539416K]
    column:[Cabin] check [inverse type] complete,speed:[2.55ms]/it,cpu:[100%],memory:[5539436K]
    column:[Embarked] check [inverse type] complete,speed:[2.32ms]/it,cpu:[100%],memory:[5538952K]
    

### int转float测试

pandas会将某些特征自动推断为int，而线上可能传输的是float，需要做如下测试：  
- 转float后transform和transform_single之间的一致性  
- int和float特征通过transform后的一致性


```python
from easymlops.table.callback import check_int_trans_float
table.callback(check_int_trans_float,x_test,sample=10)
```

    column:[PassengerId] check [int trans float] complete,speed:[3.13ms]/it,cpu:[0%],memory:[0K]
    column:[Pclass] check [int trans float] complete,speed:[1.56ms]/it,cpu:[0%],memory:[0K]
    column:[SibSp] check [int trans float] complete,speed:[3.13ms]/it,cpu:[0%],memory:[0K]
    column:[Parch] check [int trans float] complete,speed:[3.13ms]/it,cpu:[0%],memory:[0K]
    

### 自动测试：auto_test
就是把上面的所有测试，整合到auto_test一个函数中


```python
table.auto_test(x_test,sample=10)
```

    
    ###################################################################
     1.一致性测试和性能测试:check_transform_function                      
    ###################################################################
    (<class 'easymlops.table.preprocessing.core.FixInput'>) module check [transform] complete,speed:[0.3ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.preprocessing.onevar_operation.FillNa'>) module check [transform] complete,speed:[0.1ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.encoding.OneHotEncoding'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.encoding.WOEEncoding'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.encoding.LabelEncoding'>) module check [transform] complete,speed:[0.0ms]/it,cpu:[0%],memory:[0K]
    (<class 'easymlops.table.classification.LGBMClassification'>) module check [transform] complete,speed:[1.9ms]/it,cpu:[0%],memory:[0K]
    
    #########################################################################################
     2.空值测试:check_null_value                                                           
     null_values=[None, np.nan, "null", "NULL", "nan", "NaN", "", "none", "None", " "](默认)
    ########################################################################################
    column:[PassengerId] check [null value] complete,speed:[2.73ms]/it,cpu:[0%],memory:[0K]
    column:[Pclass] check [null value] complete,speed:[2.49ms]/it,cpu:[100%],memory:[0K]
    column:[Name] check [null value] complete,speed:[2.78ms]/it,cpu:[0%],memory:[0K]
    column:[Sex] check [null value] complete,speed:[2.7ms]/it,cpu:[0%],memory:[0K]
    column:[Age] check [null value] complete,speed:[2.44ms]/it,cpu:[100%],memory:[0K]
    column:[SibSp] check [null value] complete,speed:[2.83ms]/it,cpu:[100%],memory:[36K]
    column:[Parch] check [null value] complete,speed:[3.16ms]/it,cpu:[100%],memory:[0K]
    column:[Ticket] check [null value] complete,speed:[2.51ms]/it,cpu:[0%],memory:[0K]
    column:[Fare] check [null value] complete,speed:[2.47ms]/it,cpu:[100%],memory:[0K]
    column:[Cabin] check [null value] complete,speed:[2.63ms]/it,cpu:[0%],memory:[0K]
    column:[Embarked] check [null value] complete,speed:[2.43ms]/it,cpu:[0%],memory:[0K]
    
    ############################################################################################################
     3.极端值测试:check_extreme_value                                                                            
     number_extreme_values = [np.inf, 0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max](默认)
     category_extreme_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1", "NaN", "None"](默认)                            
    ###########################################################################################################
    column:[PassengerId] check [extreme value] complete,speed:[2.7ms]/it,cpu:[100%],memory:[5539868K]
    column:[Pclass] check [extreme value] complete,speed:[2.45ms]/it,cpu:[100%],memory:[5539864K]
    column:[Name] check [extreme value] complete,speed:[2.54ms]/it,cpu:[100%],memory:[5539880K]
    column:[Sex] check [extreme value] complete,speed:[2.57ms]/it,cpu:[100%],memory:[5539876K]
    column:[Age] check [extreme value] complete,speed:[2.66ms]/it,cpu:[100%],memory:[5539860K]
    column:[SibSp] check [extreme value] complete,speed:[2.68ms]/it,cpu:[100%],memory:[5539804K]
    column:[Parch] check [extreme value] complete,speed:[2.39ms]/it,cpu:[100%],memory:[5539420K]
    column:[Ticket] check [extreme value] complete,speed:[2.63ms]/it,cpu:[100%],memory:[5539364K]
    column:[Fare] check [extreme value] complete,speed:[2.38ms]/it,cpu:[100%],memory:[5539352K]
    column:[Cabin] check [extreme value] complete,speed:[2.57ms]/it,cpu:[100%],memory:[5539292K]
    column:[Embarked] check [extreme value] complete,speed:[2.54ms]/it,cpu:[100%],memory:[5539268K]
    column:[__all__] check [extreme value] complete,speed:[1.67ms]/it,cpu:[0%],memory:[0K]
    
    ###############################################################################################################
     4.数据类型反转测试:check_inverse_dtype                                                                          
     category_inverse_values = [0.0, -1, 1, -1e-7, 1e-7, np.finfo(np.float64).min, np.finfo(np.float64).max](默认)
     number_inverse_values = ["", "null", None, "1.0", "0.0", "-1.0", "-1"](默认)                                
    #############################################################################################################
    column:[PassengerId] check [inverse type] complete,speed:[2.36ms]/it,cpu:[100%],memory:[5539252K]
    column:[Pclass] check [inverse type] complete,speed:[2.7ms]/it,cpu:[100%],memory:[5539256K]
    column:[Name] check [inverse type] complete,speed:[2.62ms]/it,cpu:[100%],memory:[5539256K]
    column:[Sex] check [inverse type] complete,speed:[2.65ms]/it,cpu:[100%],memory:[5539240K]
    column:[Age] check [inverse type] complete,speed:[2.23ms]/it,cpu:[100%],memory:[5539212K]
    column:[SibSp] check [inverse type] complete,speed:[2.88ms]/it,cpu:[100%],memory:[5539236K]
    column:[Parch] check [inverse type] complete,speed:[2.57ms]/it,cpu:[100%],memory:[5539228K]
    column:[Ticket] check [inverse type] complete,speed:[2.78ms]/it,cpu:[100%],memory:[5539208K]
    column:[Fare] check [inverse type] complete,speed:[2.22ms]/it,cpu:[100%],memory:[5539208K]
    column:[Cabin] check [inverse type] complete,speed:[2.48ms]/it,cpu:[100%],memory:[5539204K]
    column:[Embarked] check [inverse type] complete,speed:[2.95ms]/it,cpu:[100%],memory:[5539208K]
    
    ############################################
     5.int数据转float测试:check_int_trans_float                                                                                            
    ############################################
    column:[PassengerId] check [int trans float] complete,speed:[1.56ms]/it,cpu:[100%],memory:[0K]
    column:[Pclass] check [int trans float] complete,speed:[3.45ms]/it,cpu:[0%],memory:[0K]
    column:[SibSp] check [int trans float] complete,speed:[2.72ms]/it,cpu:[100%],memory:[0K]
    column:[Parch] check [int trans float] complete,speed:[3.32ms]/it,cpu:[0%],memory:[0K]
    


```python

```
