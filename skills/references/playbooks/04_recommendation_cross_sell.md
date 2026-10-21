# Playbook：推荐系统 + 交叉销售/加保推荐（FM/CTR + 相似召回）

## 业务目标与边界

推荐是多阶段工程（召回→粗排→重排→A/B）。easymlops 只覆盖两个环节：

- **排序/CTR 预估**：`FM/FFM/DeepFM`（推荐经典模型）
- **相似召回**：`FaissSimilarity`（交叉销售"买过 X 的人还看 Y"）

多路召回调度、NDCG/MRR、在线 A/B 均为**外补**，本篇不承诺端到端推荐系统。

## ① 样本与标签 checklist

- [ ] CTR 样本：曝光-点击对；交叉销售：客户×候选险种"是否购买"对
- [ ] 时间切分按曝光/推荐时间；user/item 冷启动样本单独标记
- [ ] 大基数类别列（user_id/item_id）→ FM 的 field_dims 必须与 LabelEncoding 后基数一致

## ② CTR/交叉销售排序模板（实测 API）

```python
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import LabelEncoding
from easymlops.table.fm import FMClassification
from easymlops.table.classification import LGBMClassification
from easymlops.table.preprocessing import CrossCategoryWithNumber
from easymlops.table.encoding import TargetEncoding

# 路线A：FM（显式二阶交叉，稀疏场景）
fm = TablePipeLine()
fm.pipe(FixInput(reduce_mem_usage=False)) \
  .pipe(FillNa(fill_mode="mean")) \
  .pipe(LabelEncoding(cols=[类别列])) \
  .pipe(FMClassification(y=y_train, cols=[类别列], field_dims=[各列基数],
                         embed_dim=8, epochs=10))
fm.fit(x_train)

# 路线B：LGBM + 统计交叉（工程可控性更强）
lgb = TablePipeLine()
lgb.pipe(FixInput(reduce_mem_usage=False)) \
   .pipe(FillNa(fill_mode="mean")) \
   .pipe(CrossCategoryWithNumber(left_col_name="user_city", right_col_name="item_premium",
                                 operate_name="city_premium")) \
   .pipe(TargetEncoding(cols=[user_id, item_id], y=y_train)) \
   .pipe(LGBMClassification(y=y_train, use_faster_predictor=False))
```

- DeepFM 参数风格同 FM（多 hidden_layers），依赖 torch（签名未在本环境验证）

## ③ 相似召回模板（实测 API）

```python
from easymlops import NLPPipeline
from easymlops.nlp.preprocessing import Lower, ExtractJieBaWords
from easymlops.nlp.representation import TFIDF
from easymlops.nlp.similarity import FaissSimilarity

recall = NLPPipeline()
recall.pipe(Lower()).pipe(ExtractJieBaWords()) \
      .pipe(TFIDF(native_init_params={"max_features": 1000})) \
      .pipe(FaissSimilarity(search_top_k=3))
recall.fit(商品/险种文案表)   # transform(查询文本) 返回 top-k 相似条目
```

## ④ 迭代路径

1. 召回打底：Faiss 内容相似（冷启动可用）→ 2. 排序基线：历史 CTR 编码 + LGBM
3. 交叉升级：FM 二阶交叉或统计交叉 → 4. 离线 AUC → 5. 上线打分服务

评估缺口外补：NDCG/MRR、position-bias 校正、在线 A/B。

## ⑤ 工具缺口

多路召回调度与重排多样性、实时特征平台、双塔向量召回——均需外补。
