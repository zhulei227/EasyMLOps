# 常用类速查表

## 预处理 (preprocessing) - 一元操作

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| FixInput | 固定输入列顺序和数据类型 | cols |
| DoNoThing | 空操作 | - |
| FillNa | 空值填充 | cols, fill_mode, fill_detail |
| TransToCategory | 转换为类别 | cols |
| TransToFloat | 转换为浮点 | cols |
| TransToInt | 转换为整数 | cols |
| TransToLower | 转小写 | cols |
| TransToUpper | 转大写 | cols |
| SelectCols | 选择列 | cols |
| DropCols | 删除列 | cols |
| ReName | 重命名 | cols |
| Replace | 替换值 | cols, source_values, target_value |
| ClipString | 字符串裁剪 | cols, clip_detail |
| Clip | 数值裁剪 | clip_detail, percent_range |
| IsNull | 空值判断 | cols |
| IsNotNull | 非空判断 | cols |
| Abs | 绝对值 | cols |
| MapValues | 值映射 | cols, map_detail |
| MinMaxScaler | 归一化 | decimals（作用于全部数值列） |
| Normalizer | 标准化 | decimals（作用于全部数值列） |
| Bins | 分箱 | n_bins, strategy |
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
| Add | 加法 | left_col_name, right_col_name, output_col_name |
| Subtract | 减法 | left_col_name, right_col_name, output_col_name |
| Multiply | 乘法 | left_col_name, right_col_name, output_col_name |
| Divide | 除法 | left_col_name, right_col_name, output_col_name |
| DivideExact | 精确除法 | left_col_name, right_col_name, output_col_name |
| Mod | 取模 | left_col_name, right_col_name, output_col_name |
| Equal | 等于 | left_col_name, right_col_name, output_col_name |
| GreaterThan | 大于 | left_col_name, right_col_name, output_col_name |
| GreaterEqualThan | 大于等于 | left_col_name, right_col_name, output_col_name |
| LessThan | 小于 | left_col_name, right_col_name, output_col_name |
| LessEqualThan | 小于等于 | left_col_name, right_col_name, output_col_name |
| And | 与运算 | left_col_name, right_col_name, output_col_name |
| Or | 或运算 | left_col_name, right_col_name, output_col_name |
| DateDayDiff | 日期天数差 | left_col_name, right_col_name, output_col_name |

## 预处理 (preprocessing) - 多列操作

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| Sum | 求和 | cols, output_col_name |
| Mean | 平均值 | cols, output_col_name |
| Median | 中位数 | cols, output_col_name |
| CrossCategoryWithNumber | 类别×数值 | left_col_name, right_col_name, operate_name |
| CrossNumberWithNumber | 数值×数值 | left_col_name, right_col_name, operate_name |
| ExtractLastName | 提取姓氏 | name_col, new_col, split_char |

## 编码 (encoding)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LabelEncoding | 标签编码 | cols |
| OneHotEncoding | 独热编码 | cols |
| TargetEncoding | 目标编码 | cols, y |
| WOEEncoding | WOE编码 | cols, y |

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
| MissRateFilter | 缺失率过滤 | max_threshold |
| VarianceFilter | 方差过滤 | min_threshold |
| PersonCorrFilter | Pearson相关 | y, min/max_threshold |
| Chi2Filter | 卡方检验 | y, min/max_threshold |
| PValueFilter | P值过滤 | y, min/max_threshold |
| MutualInfoFilter | 互信息 | y, min/max_threshold |
| IVFilter | IV值过滤 | y, min/max_threshold |
| PSIFilter | PSI稳定性 | oot_x, min/max_threshold |

## 特征选择 (feature_selection) - 嵌入式

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LREmbed | 逻辑回归嵌入 | y, min_threshold |
| LGBMEmbed | LightGBM嵌入 | y, objective, importance_type |

## 分类 (classification)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LGBMClassification | LightGBM | y, use_faster_predictor, weight（样本权重，长度=y） |
| LogisticRegressionClassification | 逻辑回归 | label |
| SVMClassification | SVM | label |
| DecisionTreeClassification | 决策树 | label |
| RandomForestClassification | 随机森林 | y, n_estimators |
| KNeighborsClassification | KNN | label, n_neighbors |
| GaussianNBClassification | 高斯朴素贝叶斯 | label |
| MultinomialNBClassification | 多项式朴素贝叶斯 | label |
| BernoulliNBClassification | 伯努利朴素贝叶斯 | label |

## 回归 (regression)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| LGBMRegression | LightGBM | y, use_faster_predictor, weight（样本权重，长度=y） |
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
| MapValues | 值映射 | cols, map_detail |

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
| BagOfWords | 词袋模型 | native_init_params |
| TFIDF | TF-IDF | native_init_params |
| Word2VecModel | Word2Vec | embedding_size, min_count |
| Doc2VecModel | Doc2Vec | embedding_size, min_count |
| FastTextModel | FastText | embedding_size, min_count |
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
| FaissSimilarity | Faiss相似度 | search_top_k, create_index_param |
| ElasticSearchSimilarity | ES相似度 | index_name |

## 存储 (storage)

| 类名 | 说明 |
|------|------|
| LocalStorage | 本地文件存储 |
| FaissStorage | Faiss向量存储 |
| ElasticSearchStorage | ES存储 |
| HbaseStorage | HBase存储 |
| KafkaStorage | Kafka消息队列 |

## 策略 (strategy)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| HillClimbingStackingRegression | Hill Climbing + Stacking | y, eval_function, n_splits |
| AutoResidualRegression | 自动残差回归 | y |
| FactorStrategy | 因子策略 | - |
| VarRuler | 变异性规则 | - |
| LGBMRegressionLayers | LGBM回归层 | y, pred_name |
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
| FasterLgbSinglePredictor | 快速LGB单类预测 |
| CpuMemDetector | CPU/内存检测 |
| PandasUtils | Pandas工具 |

## 诊断工具 (utils，本仓库扩展，实测见 tests/test_diag_tools.py)

| 函数 | 说明 | 关键参数 |
|------|------|---------|
| calc_psi | 两分布间 PSI（以 expected 分位数分箱），<0.1 稳定 / >0.25 显著漂移 | expected, actual, bins |
| calc_ks | KS 统计量及对应分数阈值，返回 {"ks", "ks_threshold"} | y_true, y_score |
| calc_lift | 按分数降序等频分箱的 Lift/增益表（bin/samples/bads/bad_rate/lift/cum_gain） | y_true, y_score, bins |
| prob2score | 概率转评分卡刻度（分数越高风险越低） | prob, base_score=600, pdo=20, base_odds=1/19 |
| resample | 类不平衡重采样，返回行对齐的 (x, y)；**须在构造 pipeline 前调用**（y 在构造时绑定） | x, y, method(oversample/undersample/smote), random_state |

## 校准 (calibration，本仓库扩展)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| IsotonicCalibrationPipe | 保序回归概率校准（按类别列 one-vs-rest 逐类） | y, cols, out_of_bounds="clip", null_fill=0.5 |

推荐用法（无偏）：主 pipeline 先 fit，再 `calib.fit(table.transform(x_val)); table.pipe(calib)`；
简便用法（有偏）：`IsotonicCalibrationPipe(y=y_train)` 直接挂入 pipeline 后统一 fit。

## 漂移监控 (drift，本仓库扩展)

| 类名 | 说明 | 关键参数 |
|------|------|---------|
| DriftMonitorPipe | 透传式逐列 PSI 监控，transform 后读 `last_psi_report_`/`last_drift_cols_`/`psi_history_` | cols(默认全部数值列), bins=10, psi_threshold=0.25 |

注意：pipe 的 `name` 默认是类对象而非字符串，检索用
`[m for m in pipeline.models if isinstance(m, DriftMonitorPipe)]` 或整数索引；
模型分数监控可传 `cols=[正类列名]` 挂在模型 pipe 之后。

## 时序预处理 (ts.preprocessing)

| 类名 | 说明 |
|------|------|
| FixTimeInput | 时间列解析/去重/排序 |
| DateFeatureExtract | 日期特征提取（year/month/weekday + sin/cos 周期编码） |
| TSFillNa | 时序感知填充（ffill/linear/value，组内不跨组） |
| TSDiff | 一阶差分（可逆，periods） |
| TSPctChange | 百分比变化（可逆） |
| TSLogReturn | 对数收益（可逆） |
| TSZScoreScaler | Z-Score 标准化（可逆，per_group） |
| TSMinMaxScaler | Min-Max 标准化（可逆，per_group） |
| TimeResample | 时间重采样（会改变行数） |
| TimeGapsFill | 时间网格补齐（会改变行数） |
| TSBoxCox | Box-Cox/Yeo-Johnson 幂变换（可逆，lambda 持久化） |

## 时序特征 (ts.features)

| 类名 | 说明 |
|------|------|
| LagFeature | 滞后特征（lags） |
| RollingFeature | 滚动统计（mean/std/min/max/skew/kurt，window/min_periods） |
| EWMAFeature | 指数加权移动平均（span/halflife） |
| ExpandingFeature | 累积扩展统计 |
| MovingAverageTrend | 移动平均趋势（center 可选） |
| SeasonalMean | 季节均值 profile（seasonal/deseasonal/both） |
| FourierFeature | 傅里叶周期特征（period, order） |
| RollingCorrCov | 双列滚动相关/协方差 |
| STLDecompose | STL 分解（trend/seasonal/resid） |

## 时序统计模型 (ts.statistical)

| 类名 | 说明 |
|------|------|
| ArimaRegression | ARIMA（group_cols/time_col 分组；save/load 持久化模型对象） |
| SarimaRegression | SARIMA 季节性 |
| ArimaxRegression | ARIMAX（exog_cols 外生变量） |
| SarimaxRegression | SARIMAX 季节性 |
| GarchRegression | GARCH 波动率（arch） |

## 时序深度模型（需 torch）

| 模块 | 类名 |
|------|------|
| ts.deep_learning | NBeatsRegression, NHiTSRegression, DeepARRegression, GPRegression |
| ts.transformer | TFTRegression, InformerRegression, AutoformerRegression, FEDformerRegression, PatchTSTRegression, TimesNetRegression, iTransformerRegression |
| ts.state_space | DeepStateRegression, MambaRegression, LiquidS4Regression |
| ts.generative | VAERegression, NormalizingFlowRegression, DiffusionRegression |

> 深度模型面板分组用 `GroupForecast(model=..., group_cols=..., time_col=...)` 包装；save/load 持久化 state_dict；`transform_single` 输出首步预测。

## 时序工具 (ts.utils)

| 函数 | 说明 |
|------|------|
| split_time_series | 防泄漏切分（time_col/group_cols/test_size） |
| walk_forward_eval | 滚动回测（组内展开） |
| calc_adf / calc_kpss | 平稳性检验 |
| calc_acf_pacf | 自相关/偏自相关 |
| calc_mape / calc_smape / calc_mase | 评估指标 |
| calc_sample_entropy / calc_hurst | 复杂度/趋势持续性 |

## 核心基类

| 类名 | 说明 |
|------|------|
| PipeObjectBase | Pipe对象基类，所有Pipe的根基类 |
| TablePipeObjectBase | Table Pipe对象基类，表格数据处理基类，继承自PipeObjectBase |
| TablePipeLine | Table Pipeline，表格数据处理流水线 |
| NLPPipeline | NLP Pipeline，NLP任务处理流水线 |
| TSPipeLine | 时序 Pipeline，时序任务处理流水线 |
| NLPPipeObjectBase | NLP Pipe对象基类，继承自TablePipeObjectBase |
| GroupTSBase | 分组时序操作基类（组内排序/组键/写回） |
| GroupForecast | 分组预测包装器（每组独立拟合一份模型副本） |
| TorchForecastMixin | torch 预测模型 Mixin（权重持久化 + 单条推理） |
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
