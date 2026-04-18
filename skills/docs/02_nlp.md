# NLP 任务 (NLPPipeline)

## 2.1 基础类

### NLPPipeObjectBase

NLP Pipe 基类，继承自 TablePipeObjectBase，专门用于处理 NLP 任务。允许使用 pandas DataFrame 进行文本处理。

```python
from easymlops.nlp.core import NLPPipeObjectBase

class MyNLPPipe(NLPPipeObjectBase):
    def udf_transform(self, s, **kwargs):
        return s
```

### PreprocessBase

文本预处理基类，所有文本预处理类的父类，提供列选择的标准框架。

```python
from easymlops.nlp.preprocessing import PreprocessBase

class MyPreprocess(PreprocessBase):
    def __init__(self, cols="all", **kwargs):
        super().__init__(cols=cols, **kwargs)
    
    def udf_transform(self, s, **kwargs):
        return s
```

## 2.2 文本预处理

```python
from easymlops import NLPPipeline
from easymlops.nlp.preprocessing import *

nlp = NLPPipeline()

# 大小写转换
nlp.pipe(Lower())          # 转小写
nlp.pipe(Upper())          # 转大写

# 数字处理
nlp.pipe(RemoveDigits())   # 删除数字
nlp.pipe(ReplaceDigits()) # 替换数字为 *

# 标点处理
nlp.pipe(RemovePunctuation())  # 删除标点
nlp.pipe(ReplacePunctuation()) # 替换标点为空格

# 空白处理
nlp.pipe(RemoveWhitespace())   # 删除空白
nlp.pipe(ExpandWhitespace())   # 展开空白为空格

# 停用词
nlp.pipe(RemoveStopWords())    # 删除停用词

# 分词
nlp.pipe(ExtractJieBaWords())  # jieba分词
nlp.pipe(ExtractChineseWords()) # 中文分词
nlp.pipe(ExtractNGramWords(n=2)) # N-gram词

# 关键词提取
nlp.pipe(ExtractKeyWords(topk=10))

# 词汇索引
nlp.pipe(VocabIndex())

# 执行
x_train_processed = nlp.fit(x_train).transform(x_test)
```

---

## 2.2 文本特征表示

```python
from easymlops.nlp.representation import BagOfWords, TFIDF, Word2VecModel, Doc2VecModel, FastTextModel

# 词袋模型
nlp.pipe(BagOfWords(max_features=5000))

# TF-IDF
nlp.pipe(TFIDF(max_features=5000, ngram_range=(1, 2)))

# Word2Vec
nlp.pipe(Word2VecModel(vector_size=100, window=5, min_count=1))

# Doc2Vec
nlp.pipe(Doc2VecModel(vector_size=100, window=5, min_count=1))

# FastText
nlp.pipe(FastTextModel(vector_size=100, window=5, min_count=1))

# LSI主题模型
nlp.pipe(LsiTopicModel(num_topics=10))

# LDA主题模型
nlp.pipe(LdaTopicModel(num_topics=10))
```

---

## 2.3 文本分类

```python
from easymlops.nlp.text_classification import TextCNNClassification, TextRNNClassification, HANClassification

# TextCNN分类
nlp.pipe(TextCNNClassification(
    label="label",
    vocab_size=5000,
    embedding_dim=128,
    num_filters=100,
    kernel_sizes=[2, 3, 4]
))

# TextRNN分类
nlp.pipe(TextRNNClassification(
    label="label",
    vocab_size=5000,
    embedding_dim=128,
    hidden_dim=128
))

# HAN分类 (Hierarchical Attention Network)
nlp.pipe(HANClassification(
    label="label",
    vocab_size=5000,
    embedding_dim=128
))
```

---

## 2.4 文本回归

```python
from easymlops.nlp.text_regression import TextCNNRegression

nlp.pipe(TextCNNRegression(
    label="score",
    vocab_size=5000,
    embedding_dim=128
))
```

---

## 2.5 相似文本检索

```python
from easymlops.nlp.similarity import FaissSimilarity, ElasticSearchSimilarity

# Faiss相似度检索
nlp.pipe(FaissSimilarity(index_type="IVF", metric="L2"))

# ElasticSearch相似度检索
nlp.pipe(ElasticSearchSimilarity(index_name="texts"))
```
