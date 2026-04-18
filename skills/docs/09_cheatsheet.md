# 常用类速查表

## 预处理 (preprocessing) - 一元操作

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| FixInput | 固定输入列顺序和数据类型 | cols |
| DoNoThing | 空操作 | - |
| FillNa | 空值填充 | cols, strategy |
| TransToCategory | 转换为类别 | cols |
| TransToFloat | 转换为浮点 | cols |
| TransToInt | 转换为整数 | cols |
| TransToLower | 转小写 | cols |
| TransToUpper | 转大写 | cols |
| SelectCols | 选择列 | cols |
| DropCols | 删除列 | cols |
| ReName | 重命名 | cols |
| Replace | 替换值 | cols, old, new |
| ClipString | 字符串裁剪 | cols, start, end |
| Clip | 数值裁剪 | cols, lower, upper |
| IsNull | 空值判断 | cols |
| IsNotNull | 非空判断 | cols |
| Abs | 绝对值 | cols |
| MapValues | 值映射 | cols, mapping |
| MinMaxScaler | 归一化 | cols |
| Normalizer | 标准化 | cols |
| Bins | 分箱 | cols, n_bins |
| Tanh | Tanh激活 | cols |
| Relu | ReLU激活 | cols |
| Sigmoid | Sigmoid激活 | cols |
| Swish | Swish激活 | cols |
| DateMonthInfo | 月份信息 | cols |
| DateHourInfo | 小时信息 | cols |
| DateMinuteInfo | 分钟信息 | cols |
| DateTotalMinuteInfo | 总分钟数 | cols |

## 预处理 (preprocessing) - 二元操作

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| Add | 加法 | cols, new_col |
| Subtract | 减法 | cols, new_col |
| Multiply | 乘法 | cols, new_col |
| Divide | 除法 | cols, new_col |
| DivideExact | 精确除法 | cols, new_col |
| Mod | 取模 | cols, new_col |
| Equal | 等于 | cols, new_col |
| GreaterThan | 大于 | cols, new_col |
| GreaterEqualThan | 大于等于 | cols, new_col |
| LessThan | 小于 | cols, new_col |
| LessEqualThan | 小于等于 | cols, new_col |
| And | 与运算 | cols, new_col |
| Or | 或运算 | cols, new_col |
| DateDayDiff | 日期天数差 | cols, new_col |

## 预处理 (preprocessing) - 多列操作

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| Sum | 求和 | cols, new_col |
| Mean | 平均值 | cols, new_col |
| Median | 中位数 | cols, new_col |
| CrossCategoryWithNumber | 类别×数值 | cols, new_col |
| CrossNumberWithNumber | 数值×数值 | cols, new_col |
| ExtractLastName | 提取姓氏 | name_col, new_col, split_char |

## 编码 (encoding)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LabelEncoding | 标签编码 | cols |
| OneHotEncoding | 独热编码 | cols |
| TargetEncoding | 目标编码 | cols, label |
| WOEEncoding | WOE编码 | cols, label |

## 降维 (decomposition)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| PCADecomposition | PCA降维 | n_components |
| NMFDecomposition | NMF分解 | n_components |
| LDADecomposition | LDA主题 | n_components |
| KernelPCADecomposition | Kernel PCA | n_components |
| FastICADecomposition | FastICA | n_components |
| DictionaryLearningDecomposition | 字典学习 | n_components |
| MiniBatchDictionaryLearningDecomposition | 迷你批字典学习 | n_components |
| TSNEDecomposition | t-SNE | n_components |
| MDSDecomposition | MDS | n_components |
| IsomapDecomposition | Isomap | n_components |
| SpectralEmbeddingDecomposition | 谱嵌入 | n_components |
| LocallyLinearEmbeddingDecomposition | LLE | n_components |
| TCADecomposition | TCA迁移学习 | n_components, kernel_type |

## 特征选择 (feature_selection) - 过滤式

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| MissRateFilter | 缺失率过滤 | threshold |
| VarianceFilter | 方差过滤 | threshold |
| PersonCorrFilter | Pearson相关 | threshold |
| Chi2Filter | 卡方检验 | k |
| PValueFilter | P值过滤 | k |
| MutualInfoFilter | 互信息 | k |
| IVFilter | IV值过滤 | threshold |
| PSIFilter | PSI稳定性 | threshold |

## 特征选择 (feature_selection) - 嵌入式

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LREmbed | 逻辑回归嵌入 | label, C |
| LGBMEmbed | LightGBM嵌入 | label, n_estimators |

## 分类 (classification)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LGBMClassification | LightGBM | label, n_estimators |
| LogisticRegressionClassification | 逻辑回归 | label |
| SVMClassification | SVM | label |
| DecisionTreeClassification | 决策树 | label |
| RandomForestClassification | 随机森林 | label, n_estimators |
| KNeighborsClassification | KNN | label, n_neighbors |
| GaussianNBClassification | 高斯朴素贝叶斯 | label |
| MultinomialNBClassification | 多项式朴素贝叶斯 | label |
| BernoulliNBClassification | 伯努利朴素贝叶斯 | label |

## 回归 (regression)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LGBMRegression | LightGBM | label, n_estimators |
| LogisticRegression | 逻辑回归 | label |
| LinearRegression | 线性回归 | label |
| RidgeRegression | 岭回归 | label |
| RidgeCVRegression | 岭CV回归 | label |
| SVMRegression | SVM回归 | label |

## 集成学习 (ensemble)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| Parallel | 并行集成 | models, n_jobs |

## 扩展模块 (extend)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| Normalization | 高级归一化 | cols |
| MapValues | 值映射 | cols, mapping |

## 性能优化 (perfopt)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| ReduceMemUsage | 内存优化，通过修改数据类型减少内存使用量 | skip_check_transform_type |
| Dense2Sparse | 稠密转稀疏，将稠密矩阵转换为稀疏矩阵 | - |

## 评估 (eval)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| Eval | 模型评估，使用SQL表达式对数据进行计算评估 | label, metrics, sql |

## NLP 预处理 (nlp.preprocessing)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| PreprocessBase | 文本预处理基类，提供列选择的标准框架 | cols |
| Lower | 转小写，将所有英文字符转换为小写 | cols |
| Upper | 转大写，将所有英文字符转换为大写 | cols |
| RemoveDigits | 移除数字字符，从文本中移除所有数字字符 | cols |
| ReplaceDigits | 替换数字字符，将数字替换为指定字符 | cols, symbols |
| RemovePunctuation | 移除标点符号，从文本中移除所有标点符号 | cols |
| ReplacePunctuation | 替换标点 | cols |
| RemoveWhitespace | 删除空白 | cols |
| ExpandWhitespace | 展开空白 | cols |
| Replace | 替换 | old, new |
| RemoveStopWords | 删除停用词 | stop_words |
| ExtractKeyWords | 关键词提取 | topk |
| AppendKeyWords | 追加关键词 | topk |
| ExtractChineseWords | 中文分词 | cols |
| ExtractNGramWords | N-gram词 | n |
| ExtractJieBaWords | jieba分词 | cols |
| VocabIndex | 词汇索引 | vocab_size |
| ExtractJieBaWordsWithSentSplit | 分词+句子分割 | cols |
| VocabIndexWithSentSplit | 词汇索引+句子分割 | vocab_size |

## NLP 表示 (nlp.representation)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| BagOfWords | 词袋模型 | max_features |
| TFIDF | TF-IDF | max_features, ngram_range |
| Word2VecModel | Word2Vec | vector_size |
| Doc2VecModel | Doc2Vec | vector_size |
| FastTextModel | FastText | vector_size |
| LdaTopicModel | LDA主题 | num_topics |
| LsiTopicModel | LSI主题 | num_topics |

## NLP 分类 (nlp.text_classification)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| TextCNNClassification | TextCNN | label, vocab_size |
| TextRNNClassification | TextRNN | label, vocab_size |
| HANClassification | HAN | label, vocab_size |

## NLP 回归 (nlp.text_regression)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| TextCNNRegression | TextCNN回归 | label, vocab_size |

## NLP 相似度 (nlp.similarity)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| FaissSimilarity | Faiss相似度 | index_type, metric |
| ElasticSearchSimilarity | ES相似度 | index_name |

## 存储 (storage)

| 类名 | 说明 |
|------|------|
| LocalStorage | 本地文件存储 |
| FaissStorage | Faiss向量存储 |
| ElasticSearchStorage | ES存储 |
| HBaseStorage | HBase存储 |
| KafkaStorage | Kafka消息队列 |

## 策略 (strategy)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| HillClimbingStackingRegression | Hill Climbing + Stacking | label, n_folds |
| AutoResidualRegression | 自动残差回归 | label, base_model, n_estimators |
| FactorStrategy | 因子策略 | - |
| VarRuler | 变异性规则 | - |
| LGBMRegressionLayers | LGBM回归层 | label, n_estimators |
| Climber | 爬山法基类 | X, y, metric, n_folds |
| ClimberCV | CV爬山法 | X, y, metric, n_folds |
| FasterLgbSinglePredictor | 快速LGB单类预测 | model_path, input_names |
| ForestCache | 森林缓存 | - |
| TreeCache | 树缓存 | - |
| MathUtils | 数学工具 | - |
| LGBMRegression4Ruler | 用于VarRuler的LGBM | - |
| COLORS | 命令行颜色工具 | - |

## 工具 (utils)

| 类名 | 说明 |
|------|------|
| EvalFunction | 评估函数 |
| FasterLgbMulticlassPredictor | 快速LGB多类预测 |
| FasterLgbPredictorSingle | 快速LGB单类预测 |
| CpuMemDetector | CPU/内存检测 |
| PandasUtils | Pandas工具 |

## 核心基类

| 类名 | 说明 |
|------|------|
| PipeObjectBase | Pipe对象基类，所有Pipe的根基类 |
| TablePipeObjectBase | Table Pipe对象基类，表格数据处理基类，继承自PipeObjectBase |
| TablePipeLine | Table Pipeline，表格数据处理流水线 |
| NLPPipeline | NLP Pipeline，NLP任务处理流水线 |
| TSPipeLine | 时序 Pipeline，时序任务处理流水线 |
| NLPPipeObjectBase | NLP Pipe对象基类，继承自TablePipeObjectBase |
| PreprocessBase | 预处理基类，数据预处理基础类 |
| EncodingBase | 编码基类，特征编码基础类 |
| Decomposition | 降基基类，特征降维基础类 |
| FilterBase | 特征选择基类 |
| ClassificationBase | 分类基类，分类模型基础类 |
| RegressionBase | 回归基类，回归模型基础类 |

## AutoML

| 类名 | 说明 |
|------|------|
| AutoMLTab | AutoML主类 |
| AutoML | AutoML (新版) |
| LLMToolManager | LLM工具管理器 |
| LLMSessionManager | LLM会话管理器 |
| LLM | LLM基类 |
| OllamaLLM | Ollama模型 |
| SparkLLM | 讯飞Spark模型 |
| ZhiPuLLM | 智谱模型 |
| KimiLLM | 月之暗面模型 |

## YOLO 视觉任务 (yolo)

| 类名 | 说明 |
|------|------|
| YOLOPipeObjectBase | YOLO Pipe 基类 |
| YOLODetection | 目标检测 |
| YOLOSegmentation | 实例分割 |
| YOLOClassification | 图像分类 |
| YOLOPose | 姿态估计 |
| YOLOOBB | 旋转目标检测 |
| YOLOTrain | 模型训练 |
| YOLOVal | 模型验证 |
| YOLOExport | 模型导出 |
| YOLOPipeline | YOLO Pipeline |

## OCR 任务 (ocr)

| 类名 | 说明 |
|------|------|
| OCRPipeObjectBase | OCR Pipe 基类 |
| EasyOCRText | 文本检测与识别 |
| OCRPipeLine | OCR Pipeline |
