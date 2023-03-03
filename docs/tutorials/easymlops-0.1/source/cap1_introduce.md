# 简介&安装&反馈
  
## 简介 
`EasyMLOps`包以`Pipline`的方式构建建模任务，可直接进行模型训练、预测(离线，在线)，测试(离线在线预测一致性、预测性能)等功能，通过外套一层Flask或FastApi即可直接部署生产，介绍流程如下：

- Table任务：包括数据清洗、特征编码、特征降维、特征选择、分类回归、stacking等
- NLP任务：文本清洗、分词、关键词提取、ngram提取、bow/tfidf特征、fastext/word2vec/doc2vec特征提取、文本分类等
- 内存优化：优化数据类型，稠密矩阵转稀疏矩阵
- 持久化：保存及加载功能的原理和实现
- 自定义pipe模块：包括pipe设计逻辑、数据分析需求、持久化需求、生产需求、安全需求、扩展需求以及高阶需求的实现
- pipeline操作：包括pipeline也是pipe、聚合pipe为pipeline、运行到指定位置、获取指定的pipe模块、切分式调用等
- 生产部署：部署接口、日志记录、性能测试、一致性测试、空值极端值测试等
- 特征存储&监控&分析：包括嵌入式、挂载式这两种存储方式，以及数据查询、聚合分析等操作

## 安装
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

**注意：这里只安装必需的一些依赖包，需要用到扩展功能还请自行安装，建议先运行小批量数据，根据提示安装所缺的pip包**

## 问题反馈&建议  

大家在使用过程中遇到的问题或有好的建议，欢迎到github提交issue:  
 https://github.com/zhulei227/EasyMLOps


```python

```
