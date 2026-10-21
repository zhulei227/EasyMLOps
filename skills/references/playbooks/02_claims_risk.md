# Playbook：理赔风控（欺诈识别 / 渗漏检测）

## 业务目标

识别欺诈案件与理赔渗漏（多赔/错赔），支持"自动通过 / 调查 / 拒赔"分流。特点：**表格特征 + 理赔描述文本双路**，欺诈样本极少。

## ① 标签与样本 checklist

- [ ] 标签：确认欺诈/渗漏案件（以调查结论为准，注意标注延迟）
- [ ] 极不平衡（欺诈常 <1%）：重采样外补 imblearn；评估用 PR-AUC、查准率优先
- [ ] 时间切分按报案日；标注成本高时可先用规则命中集做弱标签
- [ ] 合规：模型输出仅作调查线索，不作拒赔唯一依据

## ② 特征设计 checklist

- [ ] 表格：报案间隔/出险报案间隔、金额（申请额/免赔/历史均值比）、出险地与投保地一致性、历史报案频次
- [ ] 交叉派生：`Divide`(申请额/历史均赔)、`CrossCategoryWithNumber`(险种×出险地)、`SQL(sql="...")`
- [ ] 文本：理赔描述/事故经过 → NLPPipeLine（见下）
- [ ] 关联特征（重复电话/地址/账户）→ **图特征外补**（networkx）
- [ ] 泄漏红线：调查报告结论、赔付审批字段不可入模

## ③ 双路 Pipeline 模板（实测 API）

```python
from easymlops import TablePipeLine, NLPPipeline
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import TargetEncoding, OneHotEncoding
from easymlops.table.classification import LGBMClassification
from easymlops.table.ensemble import Parallel
from easymlops.nlp.preprocessing import Lower, RemovePunctuation, ExtractJieBaWords
from easymlops.nlp.representation import TFIDF

# 表格路
tab = TablePipeLine()
tab.pipe(FixInput(reduce_mem_usage=False)) \
   .pipe(FillNa(cols=[数值列], fill_mode="mean")) \
   .pipe(Parallel([
       OneHotEncoding(cols=[低基数列]),
       TargetEncoding(cols=[高基数列], y=y_train),
   ])) \
   .pipe(LGBMClassification(y=y_train, use_faster_predictor=False, prefix="tab"))

# 文本路（理赔描述）
nlp = NLPPipeline()
nlp.pipe(Lower()) \
   .pipe(RemovePunctuation()) \
   .pipe(ExtractJieBaWords()) \
   .pipe(TFIDF(native_init_params={"max_features": 5000})) \
   .pipe(LGBMClassification(y=y_train, prefix="txt"))
nlp.fit(x_train_text)
```

- 两路分数在业务侧按权重融合；或把 TFIDF 输出拼接进表格管道后统一建模（需自行拼 DataFrame）

## ④ 相似案件检索（辅助调查）

```python
from easymlops.nlp.similarity import FaissSimilarity
recall = TablePipeLine()
recall.pipe(Lower()).pipe(ExtractJieBaWords()) \
      .pipe(TFIDF(native_init_params={"max_features": 1000})) \
      .pipe(FaissSimilarity(search_top_k=3))
recall.fit(历史案件文本表)   # transform 返回 top-k 相似历史案件
```

## ⑤ 迭代路径

1. 规则引擎打底（黑名单/高频报案），模型补增量
2. 表格基线 → 文本路引入 → 双路融合
3. 阈值分级：`calc_precision_recall_at_thresholds` 定"自动通过/调查/拒赔"三档
4. 调查反馈闭环：新确认欺诈样本回流再训练（save/load 增量更新）
5. 稳定性：按月监控分数分布与调查命中率

## ⑥ 工具缺口

- 图关联特征（团伙识别）→ networkx / 图数据库
- 不平衡采样 → imblearn
- 调查优先级排序的收益曲线 → 自实现（按 top-k 命中率画图）
