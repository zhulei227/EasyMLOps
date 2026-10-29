# -*- coding: utf-8 -*-
"""
表格神经网络 pipe（v0.2.6 新增，源自 Kaggle 优胜方案高频技术：NN on tabular +
entity/categorical embedding）。

约定:
    - 惰性导入 torch（未安装时给出明确报错）
    - 数值列自动标准化(缺失回填均值)；类别列自动 Embedding(未见/缺失→id 0)
    - 分类输出与 LGBMClassification 对齐(概率列列名=类别标签)；
      回归输出与 RegressionBase 对齐(pred_name 列)
    - 权重随 save/load 持久化(先恢复超参/编码器，再重建权重)
"""
import copy
import io

import numpy as np
import pandas as pd

from easymlops.table.core import *

dataframe_type = pd.DataFrame
dict_type = dict
series_type = pd.Series


class NNBase(TablePipeObjectBase):
    """表格神经网络基类：数值标准化 + 类别 Embedding + MLP，torch 惰性导入。"""

    def __init__(self, y=None, cols="all", categorical_cols=None, numeric_cols=None,
                 embedding_dim=8, hidden_dims=(256, 128), dropout=0.2,
                 epochs=20, batch_size=256, lr=1e-3, weight_decay=0.0,
                 random_state=42, loss="ce", focal_gamma=2.0, class_weight=None,
                 **kwargs):
        super().__init__(skip_check_transform_type=True, **kwargs)
        self.y = copy.deepcopy(y)
        self.cols = cols
        self.categorical_cols = list(categorical_cols) if categorical_cols else None
        self.numeric_cols = list(numeric_cols) if numeric_cols else None
        self.embedding_dim = int(embedding_dim)
        self.hidden_dims = list(hidden_dims)
        self.dropout = float(dropout)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.loss = loss if loss in ("ce", "focal") else "ce"
        self.focal_gamma = float(focal_gamma)
        self.class_weight = copy.deepcopy(class_weight)
        self.task_ = "regression"
        # fit 时填充
        self.feature_cols_ = []
        self.cat_cols_ = []
        self.num_cols_ = []
        self.cat_encoders_ = {}     # col -> {value: id}，0 保留给未见/缺失
        self.num_stats_ = {}        # col -> {"mean": m, "std": s}
        self.torch_model = None

    # ---- 数据准备 ----
    def _resolve_feature_cols(self, s: dataframe_type):
        if str(self.cols) == "all" or self.cols is None or \
                (isinstance(self.cols, list) and len(self.cols) == 0):
            return s.columns.tolist()
        if isinstance(self.cols, list) and self.cols and isinstance(self.cols[0], (tuple, list)):
            return [c[0] for c in self.cols]
        return list(self.cols)

    def _split_cols(self, s: dataframe_type, feature_cols):
        if self.categorical_cols is not None and self.numeric_cols is not None:
            return list(self.numeric_cols), list(self.categorical_cols)
        numeric, categorical = [], []
        for c in feature_cols:
            if np.issubdtype(s[c].dtype, np.number):
                numeric.append(c)
            else:
                categorical.append(c)
        if self.numeric_cols is not None:
            numeric = list(self.numeric_cols)
        if self.categorical_cols is not None:
            categorical = list(self.categorical_cols)
        return numeric, categorical

    def _learn_prep(self, s: dataframe_type):
        self.feature_cols_ = self._resolve_feature_cols(s)
        self.num_cols_, self.cat_cols_ = self._split_cols(s, self.feature_cols_)
        self.num_stats_ = {c: {"mean": float(s[c].mean()), "std": float(s[c].std() or 1.0)}
                           for c in self.num_cols_}
        self.cat_encoders_ = {}
        for c in self.cat_cols_:
            values = s[c].fillna("__nan__").astype(str).unique().tolist()
            self.cat_encoders_[c] = {v: i + 1 for i, v in enumerate(values)}

    def _num_array(self, s: dataframe_type) -> np.ndarray:
        cols = []
        for c in self.num_cols_:
            stats = self.num_stats_[c]
            values = pd.to_numeric(s[c], errors="coerce").fillna(stats["mean"])
            std = stats["std"] if stats["std"] else 1.0
            cols.append(((values - stats["mean"]) / std).to_numpy(dtype=np.float32))
        if not cols:
            return np.zeros((len(s), 0), dtype=np.float32)
        return np.stack(cols, axis=1)

    def _cat_array(self, s: dataframe_type) -> np.ndarray:
        cols = []
        for c in self.cat_cols_:
            encoder = self.cat_encoders_[c]
            values = s[c].fillna("__nan__").astype(str)
            cols.append(values.map(encoder).fillna(0).to_numpy(dtype=np.int64))
        if not cols:
            return np.zeros((len(s), 0), dtype=np.int64)
        return np.stack(cols, axis=1)

    # ---- 模型 ----
    def _build_model(self, out_dim: int):
        import torch
        import torch.nn as nn

        cat_vocabs = [len(self.cat_encoders_[c]) + 1 for c in self.cat_cols_]
        input_dim = len(self.num_cols_) + self.embedding_dim * len(cat_vocabs)

        class TabularMLP(nn.Module):
            def __init__(self, vocabs, embedding_dim, n_numeric, hidden_dims, dropout, out_dim):
                super().__init__()
                self.embeddings = nn.ModuleList([nn.Embedding(v, embedding_dim) for v in vocabs])
                net_input = n_numeric + embedding_dim * len(vocabs)
                layers = []
                prev = net_input
                for h in hidden_dims:
                    layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
                    prev = h
                layers.append(nn.Linear(prev, out_dim))
                self.mlp = nn.Sequential(*layers)

            def forward(self, num_x, cat_x):
                if len(self.embeddings) > 0:
                    embs = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
                    features = torch.cat([num_x] + embs, dim=1)
                else:
                    features = num_x
                return self.mlp(features)

        return TabularMLP(cat_vocabs, self.embedding_dim, len(self.num_cols_),
                          self.hidden_dims, self.dropout, out_dim)

    def _train(self, x_num: np.ndarray, x_cat: np.ndarray, targets: np.ndarray, out_dim: int):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch_model = self._build_model(out_dim).to(device)
        weight_t = None
        if self.task_ == "classification" and self.class_weight:
            weight_t = torch.FloatTensor(
                [self.class_weight.get(i, 1.0) for i in range(out_dim)]).to(device)
        if self.task_ == "classification":
            targets_ndim = np.asarray(targets).ndim
            if targets_ndim == 2:
                # 软标签交叉熵(蒸馏): -sum(p * log_softmax(output))
                targets_t = torch.FloatTensor(np.asarray(targets, dtype=np.float32))

                def criterion(output, target):
                    log_probs = torch.log_softmax(output, dim=1)
                    return -(target * log_probs).sum(dim=1).mean()
            else:
                if self.loss == "focal":
                    gamma = self.focal_gamma

                    def criterion(output, target_idx):
                        log_probs = torch.log_softmax(output, dim=1)
                        log_p_t = log_probs.gather(1, target_idx.unsqueeze(1)).squeeze(1)
                        p_t = log_p_t.exp()
                        batch_loss = -((1 - p_t) ** gamma) * log_p_t
                        if weight_t is not None:
                            batch_loss = batch_loss * weight_t[target_idx]
                        return batch_loss.mean()
                else:
                    criterion = nn.CrossEntropyLoss(weight=weight_t)
                targets_t = torch.LongTensor(np.asarray(targets, dtype=np.int64))
        else:
            criterion = nn.MSELoss()
            targets_t = torch.FloatTensor(targets)
        optimizer = torch.optim.Adam(self.torch_model.parameters(), lr=self.lr,
                                     weight_decay=self.weight_decay)
        num_t = torch.FloatTensor(x_num).to(device)
        cat_t = torch.LongTensor(x_cat).to(device)
        target_t = targets_t.to(device)
        n = len(x_num)
        self.torch_model.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                optimizer.zero_grad()
                output = self.torch_model(num_t[idx], cat_t[idx])
                loss = criterion(output, target_t[idx])
                loss.backward()
                optimizer.step()
        self.torch_model.eval()

    def udf_get_params(self) -> dict_type:
        return {"loss": self.loss, "focal_gamma": self.focal_gamma,
                "class_weight": self.class_weight}

    def udf_set_params(self, params: dict_type):
        self.loss = params["loss"]
        self.focal_gamma = params["focal_gamma"]
        self.class_weight = params["class_weight"]

    def _predict_output(self, s: dataframe_type) -> np.ndarray:
        import torch
        if self.torch_model is None:
            raise Exception("model is not fitted, call fit before transform")
        num_x = self._num_array(s)
        cat_x = self._cat_array(s)
        self.torch_model.eval()   # 关闭 Dropout, 保证加载后与训练结束行为一致
        with torch.no_grad():
            output = self.torch_model(torch.FloatTensor(num_x).to(self.device),
                                      torch.LongTensor(cat_x).to(self.device))
        if self.task_ == "classification":
            return torch.softmax(output, dim=1).cpu().numpy()
        return output.cpu().numpy().reshape(-1)

    # ---- 序列化 ----
    def _dl_state_bytes(self) -> bytes:
        import torch
        buffer = io.BytesIO()
        torch.save(self.torch_model.state_dict(), buffer)
        return buffer.getvalue()

    def _dl_load_state_bytes(self, data: bytes):
        import torch
        buffer = io.BytesIO(data)
        self.torch_model.load_state_dict(torch.load(buffer, map_location="cpu"))

    # ---- 训练入口 ----
    def udf_fit(self, s: dataframe_type, **kwargs):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise Exception("TabularMLP requires torch: pip install torch")
        self._learn_prep(s)
        if self.task_ == "classification":
            targets = self.y.map(self.label2id).to_numpy(dtype=np.int64)
            out_dim = self.num_class
        else:
            targets = np.asarray(self.y, dtype=np.float32)
            out_dim = 1
        self._train(self._num_array(s), self._cat_array(s), targets, out_dim)
        return self


class TabularMLPClassification(NNBase):
    """
    表格 MLP 分类（v0.2.6 新增）：数值列标准化 + 类别列 Embedding + MLP，
    输出各类别概率（列名=类别标签，与 LGBMClassification 对齐）。

    Args:
        y: 目标变量（构造时绑定）
        cols: 特征列；"all"/None 表示全部列
        categorical_cols / numeric_cols: 显式指定列类型；None 时按 dtype 自动划分
        embedding_dim: 类别 Embedding 维度
        hidden_dims: MLP 隐藏层维度
        dropout: Dropout 比例
        epochs / batch_size / lr / weight_decay: 训练超参
        random_state: 随机种子
        **kwargs: 其他父类参数

    Example:
        >>> from easymlops.table.nn import TabularMLPClassification
        >>> pipe = TabularMLPClassification(y=y_train, epochs=20)
        >>> pipe.fit(train_df)
        >>> probs = pipe.transform(test_df)   # 概率列（列名=类别标签）
    """

    def __init__(self, y=None, **kwargs):
        super().__init__(y=y, **kwargs)
        self.task_ = "classification"
        self.id2label = {}
        self.label2id = {}
        self.num_class = None
        if self.y is not None:
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        probs = self._predict_output(s)
        return pd.DataFrame(probs, columns=[self.id2label[i] for i in range(self.num_class)],
                            index=s.index)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        row = pd.DataFrame([s])
        probs = self._predict_output(row)[0]
        return {self.id2label[i]: float(probs[i]) for i in range(self.num_class)}

    def udf_get_params(self) -> dict_type:
        return {"id2label": self.id2label, "label2id": self.label2id,
                "num_class": self.num_class,
                "cat_cols_": self.cat_cols_, "num_cols_": self.num_cols_,
                "cat_encoders_": self.cat_encoders_, "num_stats_": self.num_stats_,
                "categorical_cols": self.categorical_cols, "numeric_cols": self.numeric_cols,
                "embedding_dim": self.embedding_dim, "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "model_state": self._dl_state_bytes() if self.torch_model is not None else None}

    def udf_set_params(self, params: dict_type):
        self.cat_cols_ = params["cat_cols_"]
        self.num_cols_ = params["num_cols_"]
        self.cat_encoders_ = params["cat_encoders_"]
        self.num_stats_ = params["num_stats_"]
        self.categorical_cols = params["categorical_cols"]
        self.numeric_cols = params["numeric_cols"]
        self.id2label = params["id2label"]
        self.label2id = params["label2id"]
        self.num_class = params["num_class"]
        self.embedding_dim = params["embedding_dim"]
        self.hidden_dims = params["hidden_dims"]
        self.dropout = params["dropout"]
        if params["model_state"] is not None:
            self.task_ = "classification"
            self.device = "cpu"
            self.torch_model = self._build_model(self.num_class)
            self._dl_load_state_bytes(params["model_state"])


class TabularMLPRegression(NNBase):
    """
    表格 MLP 回归（v0.2.6 新增）：输出单列预测（pred_name 列，默认 "pred"，
    与 RegressionBase 对齐）。

    Example:
        >>> from easymlops.table.nn import TabularMLPRegression
        >>> pipe = TabularMLPRegression(y=y_train, epochs=20)
        >>> pipe.fit(train_df)
        >>> preds = pipe.transform(test_df)   # pred 列
    """

    def __init__(self, y=None, pred_name="pred", **kwargs):
        super().__init__(y=y, **kwargs)
        self.task_ = "regression"
        self.pred_name = pred_name

    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        values = self._predict_output(s)
        return pd.DataFrame({self.pred_name: values}, index=s.index)

    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        row = pd.DataFrame([s])
        values = self._predict_output(row)
        return {self.pred_name: float(values[0])}

    def udf_get_params(self) -> dict_type:
        return {"pred_name": self.pred_name,
                "cat_cols_": self.cat_cols_, "num_cols_": self.num_cols_,
                "cat_encoders_": self.cat_encoders_, "num_stats_": self.num_stats_,
                "categorical_cols": self.categorical_cols, "numeric_cols": self.numeric_cols,
                "embedding_dim": self.embedding_dim, "hidden_dims": self.hidden_dims,
                "dropout": self.dropout,
                "model_state": self._dl_state_bytes() if self.torch_model is not None else None}

    def udf_set_params(self, params: dict_type):
        self.pred_name = params["pred_name"]
        self.cat_cols_ = params["cat_cols_"]
        self.num_cols_ = params["num_cols_"]
        self.cat_encoders_ = params["cat_encoders_"]
        self.num_stats_ = params["num_stats_"]
        self.categorical_cols = params["categorical_cols"]
        self.numeric_cols = params["numeric_cols"]
        self.embedding_dim = params["embedding_dim"]
        self.hidden_dims = params["hidden_dims"]
        self.dropout = params["dropout"]
        if params["model_state"] is not None:
            self.task_ = "regression"
            self.device = "cpu"
            self.torch_model = self._build_model(1)
            self._dl_load_state_bytes(params["model_state"])
