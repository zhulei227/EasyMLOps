EasyMLOps
=========

介绍
----

``EasyMLOps``\ 包以\ ``Pipline``\ 的方式构建建模任务，可直接进行模型训练、预测(离线，在线)，测试(离线在线预测一致性、预测性能)、特征存储、监控、分析等功能，通过外套一层Flask或FastApi即可直接部署生产，更多内容请访问\ `使用文档1>> <http://www.easymlops.com>`__
或 `使用文档2>> <http://43.153.61.105>`__

安装
----

.. code:: bash

   pip install easymlops

使用
----

.. code:: ipython3

    import pandas as pd
    data=pd.read_csv("./data/demo.csv")
    print(data.head(5).to_markdown())


.. parsed-literal::

    |    |   PassengerId |   Survived |   Pclass | Name                                                | Sex    |   Age |   SibSp |   Parch | Ticket           |    Fare | Cabin   | Embarked   |
    |---:|--------------:|-----------:|---------:|:----------------------------------------------------|:-------|------:|--------:|--------:|:-----------------|--------:|:--------|:-----------|
    |  0 |             1 |          0 |        3 | Braund, Mr. Owen Harris                             | male   |    22 |       1 |       0 | A/5 21171        |  7.25   | nan     | S          |
    |  1 |             2 |          1 |        1 | Cumings, Mrs. John Bradley (Florence Briggs Thayer) | female |    38 |       1 |       0 | PC 17599         | 71.2833 | C85     | C          |
    |  2 |             3 |          1 |        3 | Heikkinen, Miss. Laina                              | female |    26 |       0 |       0 | STON/O2. 3101282 |  7.925  | nan     | S          |
    |  3 |             4 |          1 |        1 | Futrelle, Mrs. Jacques Heath (Lily May Peel)        | female |    35 |       1 |       0 | 113803           | 53.1    | C123    | S          |
    |  4 |             5 |          0 |        3 | Allen, Mr. William Henry                            | male   |    35 |       0 |       0 | 373450           |  8.05   | nan     | S          |
    

.. code:: ipython3

    # 拆分训练测试
    x_train=data[:500]
    x_test=data[500:]
    y_train=x_train["Survived"]
    y_test=x_test["Survived"]
    del x_train["Survived"]
    del x_test["Survived"]

表格型任务
~~~~~~~~~~

.. code:: ipython3

    from easymlops import TablePipeLine
    from easymlops.table.preprocessing import *
    from easymlops.table.encoding import *
    from easymlops.table.decomposition import *
    from easymlops.table.classification import *
    from easymlops.table.ensemble import *

.. code:: ipython3

    table = TablePipeLine()
    table.pipe(FixInput()) \
      .pipe(FillNa()) \
      .pipe(Parallel([OneHotEncoding(cols=["Pclass", "Sex"]), LabelEncoding(cols=["Sex", "Pclass"]),
                        TargetEncoding(cols=["Name", "Ticket", "Embarked", "Cabin", "Sex"], y=y_train)])) \
      .pipe(Parallel([PCADecomposition(n_components=2, prefix="pca"), NMFDecomposition(n_components=2, prefix="nmf")]))\
      .pipe(Parallel([LGBMClassification(y=y_train, prefix="lgbm"), LogisticRegressionClassification(y=y_train, prefix="lr")]))
    
    table.fit(x_train)




.. parsed-literal::

    <easymlops.table.core.pipeline_object.TablePipeLine at 0x213de1f9988>



.. code:: ipython3

    print(table.transform(x_test).head(5).to_markdown())


.. parsed-literal::

    |     |   lgbm_0 |    lgbm_1 |     lr_0 |     lr_1 |
    |----:|---------:|----------:|---------:|---------:|
    | 500 | 0.965218 | 0.0347825 | 0.651417 | 0.348583 |
    | 501 | 0.98153  | 0.0184698 | 0.65506  | 0.34494  |
    | 502 | 0.979139 | 0.0208607 | 0.647266 | 0.352734 |
    | 503 | 0.808796 | 0.191204  | 0.656613 | 0.343387 |
    | 504 | 0.184484 | 0.815516  | 0.449149 | 0.550851 |
    

.. code:: ipython3

    table.transform_single(x_test.to_dict("record")[0])




.. parsed-literal::

    {'lgbm_0': 0.9652175316373408,
     'lgbm_1': 0.03478246836265923,
     'lr_0': 0.6514166695229407,
     'lr_1': 0.3485833304770593}



文本特征提取
~~~~~~~~~~~~

.. code:: ipython3

    from easymlops import NLPPipeline
    from easymlops.nlp.preprocessing import *
    from easymlops.nlp.representation import *

.. code:: ipython3

    nlp = NLPPipeline()
    nlp.pipe(FixInput()) \
      .pipe(FillNa()) \
      .pipe(SelectCols(cols=["Name"]))\
      .pipe(ReplaceDigits())\
      .pipe(RemovePunctuation())\
      .pipe(Parallel([Word2VecModel(embedding_size=4), FastTextModel(embedding_size=4)]))
    
    nlp.fit(x_train)




.. parsed-literal::

    <easymlops.nlp.core.pipeline_object.NLPPipeline at 0x213e69dda88>



.. code:: ipython3

    print(nlp.transform(x_test).head(5).to_markdown())


.. parsed-literal::

    |     |   w2v_Name_0 |   w2v_Name_1 |   w2v_Name_2 |   w2v_Name_3 |   fasttext_Name_0 |   fasttext_Name_1 |   fasttext_Name_2 |   fasttext_Name_3 |
    |----:|-------------:|-------------:|-------------:|-------------:|------------------:|------------------:|------------------:|------------------:|
    | 500 |   -0.0160304 |   0.00341222 |     0.131879 |    0.227311  |        0.0470462  |       -0.00434371 |        0.0662056  |        0.0268283  |
    | 501 |   -0.168437  |  -0.0802843  |     0.188844 |    0.0576391 |        0.00569692 |       -0.00443027 |        0.00559943 |        0.0177013  |
    | 502 |   -0.168437  |  -0.0802843  |     0.188844 |    0.0576391 |       -0.00195689 |       -0.0279951  |        0.00245144 |        0.0229158  |
    | 503 |   -0.233626  |  -0.181061   |     0.165193 |    0.226518  |        0.00192385 |       -0.0182485  |       -0.005737   |        0.032779   |
    | 504 |   -0.233626  |  -0.181061   |     0.165193 |    0.226518  |        0.010348   |       -0.0144759  |        0.028474   |        0.00600368 |
    

.. code:: ipython3

    nlp.transform_single(x_test.to_dict("record")[0])




.. parsed-literal::

    {'w2v_Name_0': -0.01603037677705288,
     'w2v_Name_1': 0.003412220859900117,
     'w2v_Name_2': 0.13187919557094574,
     'w2v_Name_3': 0.227310910820961,
     'fasttext_Name_0': 0.04704619571566582,
     'fasttext_Name_1': -0.004343707114458084,
     'fasttext_Name_2': 0.06620561331510544,
     'fasttext_Name_3': 0.026828348636627197}



