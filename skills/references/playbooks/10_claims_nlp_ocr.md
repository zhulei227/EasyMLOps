# Playbook：理赔材料/病历文本分类 + 单证发票 OCR 自动审核

## 业务目标

- **文本分类**：理赔描述/病历/投诉内容自动分派与风险标记
- **OCR 自动审核**：单证/发票/身份材料识别与要素校验，减少人工录入审核

## ① 文本分类（NLPPipeLine）

```python
from easymlops import NLPPipeline
from easymlops.nlp.preprocessing import Lower, RemovePunctuation, ExtractJieBaWords, RemoveStopWords
from easymlops.nlp.representation import TFIDF

nlp = NLPPipeline()
nlp.pipe(Lower()) \
   .pipe(RemovePunctuation()) \
   .pipe(ExtractJieBaWords()) \
   .pipe(RemoveStopWords()) \
   .pipe(TFIDF(native_init_params={"max_features": 5000}))
nlp.fit(x_train_text)
```

- 深度分类模型（TextCNN/TextRNN/HAN，`easymlops.nlp.text_classification`）依赖 torch，签名未在本环境验证，使用前先 `inspect.signature` 确认
- 传统路线：TFIDF 输出拼接 LGBM 分类（表格路），工程可控性更好

## ② OCR 自动审核（OCRPipeLine）

```python
from easymlops.ocr import OCRPipeLine, EasyOCRText

ocr = OCRPipeLine(pipes=[EasyOCRText(lang=["en", "ch_sim"], gpu=False)])
result = ocr.fit(df).transform(df)     # df 须含 image_path 列
single = ocr.transform_single({"image_path": "invoice.jpg"})
```

- 要素校验：识别文本 + 正则/规则校验金额、税号、日期一致性（自实现）
- 首次运行自动下载模型（约 140MB）；依赖 Pillow<10、numpy<2、torch

## ③ 迭代路径

1. 文本规则分派（关键词）→ 2. TFIDF+LGBM 分类 → 3. 增加病历专用词典
4. OCR 要素校验规则 → 5. 识别置信度低于阈值转人工 → 6. 错误样本回流

## ④ 工具缺口

版面分析/表格结构还原、医疗实体识别（NER）、大模型抽费——外补。
