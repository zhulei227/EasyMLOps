# 特征存储

## 7.1 本地存储

```python
from easymlops.table.storage import LocalStorage

# 初始化存储
storage = LocalStorage(path="./storage")

# 保存特征
storage.save_features(features, "feature_name")

# 加载特征
features = storage.load_features("feature_name")

# 删除特征
storage.delete_features("feature_name")

# 列出所有特征
feature_list = storage.list_features()
```

---

## 7.2 Faiss 存储

```python
from easymlops.table.storage import FaissStorage

# 初始化存储
storage = FaissStorage(dim=128, index_type="IVF", metric="L2")

# 添加向量
storage.add_vectors(vectors, ids)

# 搜索
results = storage.search(query_vector, k=10)

# 删除
storage.delete_vectors(ids)

# 保存索引
storage.save_index("faiss.index")

# 加载索引
storage.load_index("faiss.index")
```

---

## 7.3 ElasticSearch 存储

```python
from easymlops.table.storage import ElasticSearchStorage

# 初始化存储
storage = ElasticSearchStorage(
    hosts=["localhost:9200"],
    index="features",
    doc_type="feature"
)

# 保存文档
documents = [
    {"id": "1", "text": "hello world", "vector": [0.1, 0.2, 0.3]},
    {"id": "2", "text": "foo bar", "vector": [0.4, 0.5, 0.6]}
]
storage.save_documents(documents)

# 搜索
results = storage.search(query, size=10)

# 批量查询
results = storage.mget(["1", "2", "3"])

# 删除
storage.delete_document("1")

# 创建索引
storage.create_index(mappings={...})
```

---

## 7.4 HBase 存储

```python
from easymlops.table.storage import HBaseStorage

# 初始化存储
storage = HBaseStorage(
    table="features",
    host="localhost",
    port=9090
)

# 保存特征
features_dict = {
    "feature1": {...},
    "feature2": {...}
}
storage.save_features(features_dict, row_key="user_001")

# 加载特征
features = storage.load_features(row_key="user_001")

# 扫描
features = storage.scan(start_row="user_001", stop_row="user_100")

# 删除
storage.delete_features(row_key="user_001")
```

---

## 7.5 Kafka 存储

```python
from easymlops.table.storage import KafkaStorage

# 初始化存储
storage = KafkaStorage(
    bootstrap_servers=["localhost:9092"],
    topic="features"
)

# 发送特征
features = {...}
storage.send_features(features)

# 批量发送
features_list = [{...}, {...}]
storage.send_batch_features(features_list)

# 消费特征
for message in storage.consume():
    features = message.value
    # 处理

# 消费者组
storage = KafkaStorage(
    bootstrap_servers=["localhost:9092"],
    topic="features",
    group_id="consumer_group_1"
)
```

---

## 7.6 存储对比

| 存储类型 | 适用场景 | 优点 | 缺点 |
|---------|---------|------|------|
| LocalStorage | 小规模数据 | 简单、无需额外依赖 | 不支持分布式 |
| FaissStorage | 向量检索 | 高效的向量相似度搜索 | 需要Faiss |
| ESStorage | 全文检索 | 支持复杂查询 | 资源消耗大 |
| HBaseStorage | 大规模KV存储 | 高扩展性 | 需要HBase集群 |
| KafkaStorage | 消息队列 | 异步、解耦 | 需要Kafka集群 |
