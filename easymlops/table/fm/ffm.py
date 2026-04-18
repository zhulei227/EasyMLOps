# -*- coding: utf-8 -*-
from easymlops.table.core import *
from easymlops.table.fm.base import FMBase
import pandas as pd
import numpy as np


pd.options.mode.copy_on_write = True


class FMBase(TablePipeObjectBase):
    """
    因子分解机基类。
    
    提供因子分解机模型的通用框架，包括FM、FFM、DeepFM等。
    
    Args:
        y: 目标变量
        cols: 用于模型训练的列
        task_type: 任务类型，"classification" 或 "regression"
        **kwargs: 其他父类参数
    """
    
    def __init__(self, y: series_type = None, cols="all", task_type="classification",
                 field_dims=None, embed_dim=8, **kwargs):
        super().__init__(**kwargs)
        self.y = copy.deepcopy(y)
        self.cols = cols if cols is not None else []
        self.task_type = task_type
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self.model = None
        self.id2label = {}
        self.label2id = {}
        self.num_class = None
        self.cols_ = None
        
        if self.y is not None and task_type == "classification":
            for idx, label in enumerate(self.y.value_counts().index):
                self.id2label[idx] = label
                self.label2id[label] = idx
            self.y = self.y.apply(lambda x: self.label2id.get(x))
            self.num_class = len(self.id2label)

    def before_fit(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_fit(s, **kwargs)
        assert self.y is not None
        if len(self.cols) == 0:
            self.cols = s.columns.tolist()
        if self.field_dims is None:
            self.field_dims = [s[col].nunique() for col in self.cols]
        return s

    def before_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s = super().before_transform(s, **kwargs)
        return s

    def transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        s_ = self.after_transform(self.udf_transform(self.before_transform(s, **kwargs), **kwargs), **kwargs)
        return s_

    def transform_single(self, s: dict_type, **kwargs) -> dict_type:
        s_ = copy.deepcopy(s)
        s_ = self.after_transform_single(
            self.udf_transform_single(self.before_transform_single(s_, **kwargs), **kwargs), **kwargs)
        return s_

    def udf_fit(self, s, **kwargs):
        raise Exception("need to implement")

    def udf_transform(self, s, **kwargs):
        raise Exception("need to implement")

    def udf_transform_single(self, s: dict_type, **kwargs):
        raise Exception("need to implement")

    def udf_get_params(self) -> dict_type:
        return {"cols": self.cols, "task_type": self.task_type, "field_dims": self.field_dims,
                "embed_dim": self.embed_dim, "id2label": self.id2label, "label2id": self.label2id,
                "num_class": self.num_class}

    def udf_set_params(self, params: dict):
        self.cols = params["cols"]
        self.task_type = params["task_type"]
        self.field_dims = params["field_dims"]
        self.embed_dim = params["embed_dim"]
        self.id2label = params.get("id2label", {})
        self.label2id = params.get("label2id", {})
        self.num_class = params.get("num_class", None)


class FFMClassification(FMBase):
    """
    FFM (Field-aware Factorization Machine) 分类模型。
    
    FFM在FM的基础上引入了field的概念，每个特征在不同field下有不同的隐向量。
    适用于需要进行特征域区分的点击率预估等场景。
    
    Args:
        y: 目标变量
        cols: 用于模型训练的列
        field_dims: 每个field的维度列表
        embed_dim: 嵌入维度
        learning_rate: 学习率
        reg_lambda: L2正则化系数
        epochs: 训练轮数
        batch_size: 批大小
        early_stopping_rounds: 早停轮数
        
    Example:
        >>> from easymlops.table.fm import FFMClassification
        >>> fm = FFMClassification(y=label, embed_dim=8, epochs=10)
        >>> fm.fit(df).transform(df)
    """
    
    def __init__(self, y: series_type = None, cols="all", field_dims=None, embed_dim=8,
                 learning_rate=0.01, reg_lambda=0.001, epochs=10, batch_size=256,
                 early_stopping_rounds=5, verbose=True, **kwargs):
        super().__init__(y=y, cols=cols, task_type="classification", 
                        field_dims=field_dims, embed_dim=embed_dim, **kwargs)
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.best_score = 0
        self.patience_counter = 0
        
    def _init_weights(self):
        np.random.seed(42)
        n_fields = len(self.field_dims)
        n_features = sum(self.field_dims)
        
        self.w0 = 0.0
        self.W = np.random.randn(n_features) * 0.01
        self.V = np.random.randn(n_features, self.embed_dim) * 0.01
        
    def _get_field_id(self, feat_idx):
        field_id = 0
        cumsum = 0
        for i, dim in enumerate(self.field_dims):
            if feat_idx < cumsum + dim:
                return field_id
            cumsum += dim
            field_id += 1
        return field_id
    
    def _transform_input(self, X):
        X_encoded = []
        for i in range(len(self.cols)):
            col_values = X[:, i]
            offset = sum(self.field_dims[:i])
            encoded = col_values + offset
            X_encoded.append(encoded)
        return np.column_stack(X_encoded)
    
    def _predict_one(self, x):
        linear = self.w0 + np.sum(self.W[x])
        
        interactions = 0.0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                field_i = self._get_field_id(x[i])
                field_j = self._get_field_id(x[j])
                vi_fj = self.V[x[j], field_i]
                vj_fi = self.V[x[i], field_j]
                interactions += np.dot(vi_fj, vj_fi)
        
        pred = linear + interactions
        return 1 / (1 + np.exp(-np.clip(pred, -500, 500)))
    
    def udf_fit(self, s: dataframe_type, **kwargs):
        X = s[self.cols].values
        X_encoded = self._transform_input(X)
        y = self.y.values if hasattr(self.y, 'values') else np.array(self.y)
        
        self._init_weights()
        
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                batch_X = X_encoded[batch_idx]
                batch_y = y[batch_idx]
                
                preds = np.array([self._predict_one(x) for x in batch_X])
                errors = preds - batch_y
                
                self.w0 -= self.learning_rate * np.mean(errors)
                
                for i, x_i in enumerate(batch_X):
                    error = errors[i]
                    self.W[x_i] -= self.learning_rate * (error + self.reg_lambda * self.W[x_i])
                
                for i, x_i in enumerate(batch_X):
                    error = errors[i]
                    for feat_idx in x_i:
                        self.V[feat_idx] -= self.learning_rate * (
                            error * (self.V[feat_idx] - self.W[x_i]) + 
                            self.reg_lambda * self.V[feat_idx]
                        )
                
                epoch_loss += np.mean(errors ** 2)
            
            if self.verbose and (epoch + 1) % 2 == 0:
                preds_all = np.array([self._predict_one(x) for x in X_encoded])
                preds_binary = (preds_all > 0.5).astype(int)
                accuracy = np.mean(preds_binary == y)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}")
                
                if accuracy > self.best_score:
                    self.best_score = accuracy
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        return self
    
    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        X = s[self.cols].values
        X_encoded = self._transform_input(X)
        
        preds = np.array([self._predict_one(x) for x in X_encoded])
        
        if self.num_class == 2:
            result = pd.DataFrame({
                self.id2label.get(0): 1 - preds,
                self.id2label.get(1): preds
            }, index=s.index)
        else:
            result = pd.DataFrame(preds, columns=[self.id2label.get(i) for i in range(self.num_class)], 
                                 index=s.index)
        
        return result
    
    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        X = np.array([[s.get(col, 0) for col in self.cols]])
        X_encoded = self._transform_input(X)
        pred = self._predict_one(X_encoded[0])
        
        if self.num_class == 2:
            return {self.id2label.get(0): 1 - pred, self.id2label.get(1): pred}
        else:
            return {self.id2label.get(i): pred for i in range(self.num_class)}
    
    def udf_get_params(self) -> dict_type:
        params = super().udf_get_params()
        params.update({
            "w0": self.w0, "W": self.W, "V": self.V,
            "learning_rate": self.learning_rate, "reg_lambda": self.reg_lambda,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "best_score": self.best_score
        })
        return params
    
    def udf_set_params(self, params: dict):
        super().udf_set_params(params)
        self.w0 = params.get("w0", 0.0)
        self.W = params.get("W", None)
        self.V = params.get("V", None)
        self.learning_rate = params.get("learning_rate", 0.01)
        self.reg_lambda = params.get("reg_lambda", 0.001)
        self.epochs = params.get("epochs", 10)
        self.batch_size = params.get("batch_size", 256)
        self.best_score = params.get("best_score", 0)


class FFMRegression(FMBase):
    """
    FFM (Field-aware Factorization Machine) 回归模型。
    
    Args:
        y: 目标变量
        cols: 用于模型训练的列
        field_dims: 每个field的维度列表
        embed_dim: 嵌入维度
        learning_rate: 学习率
        reg_lambda: L2正则化系数
        epochs: 训练轮数
        batch_size: 批大小
        
    Example:
        >>> from easymlops.table.fm import FFMRegression
        >>> fm = FFMRegression(y=target, embed_dim=8, epochs=10)
        >>> fm.fit(df).transform(df)
    """
    
    def __init__(self, y: series_type = None, cols="all", field_dims=None, embed_dim=8,
                 learning_rate=0.01, reg_lambda=0.001, epochs=10, batch_size=256,
                 early_stopping_rounds=5, verbose=True, **kwargs):
        super().__init__(y=y, cols=cols, task_type="regression", 
                        field_dims=field_dims, embed_dim=embed_dim, **kwargs)
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.best_mse = float('inf')
        self.patience_counter = 0
        
    def _init_weights(self):
        np.random.seed(42)
        self.w0 = 0.0
        self.W = np.random.randn(sum(self.field_dims)) * 0.01
        self.V = np.random.randn(sum(self.field_dims), self.embed_dim) * 0.01
        
    def _get_field_id(self, feat_idx):
        field_id = 0
        cumsum = 0
        for i, dim in enumerate(self.field_dims):
            if feat_idx < cumsum + dim:
                return field_id
            cumsum += dim
            field_id += 1
        return field_id
    
    def _transform_input(self, X):
        X_encoded = []
        for i in range(len(self.cols)):
            col_values = X[:, i]
            offset = sum(self.field_dims[:i])
            encoded = col_values + offset
            X_encoded.append(encoded)
        return np.column_stack(X_encoded)
    
    def _predict_one(self, x):
        linear = self.w0 + np.sum(self.W[x])
        
        interactions = 0.0
        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                field_i = self._get_field_id(x[i])
                field_j = self._get_field_id(x[j])
                vi_fj = self.V[x[j], field_i]
                vj_fi = self.V[x[i], field_j]
                interactions += np.dot(vi_fj, vj_fi)
        
        return linear + interactions
    
    def udf_fit(self, s: dataframe_type, **kwargs):
        X = s[self.cols].values
        X_encoded = self._transform_input(X)
        y = self.y.values if hasattr(self.y, 'values') else np.array(self.y)
        
        self._init_weights()
        
        n_samples = len(y)
        indices = np.arange(n_samples)
        
        for epoch in range(self.epochs):
            np.random.shuffle(indices)
            epoch_loss = 0.0
            
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                batch_X = X_encoded[batch_idx]
                batch_y = y[batch_idx]
                
                preds = np.array([self._predict_one(x) for x in batch_X])
                errors = preds - batch_y
                
                self.w0 -= self.learning_rate * np.mean(errors)
                
                for i, x_i in enumerate(batch_X):
                    error = errors[i]
                    self.W[x_i] -= self.learning_rate * (error + self.reg_lambda * self.W[x_i])
                
                for i, x_i in enumerate(batch_X):
                    error = errors[i]
                    for feat_idx in x_i:
                        self.V[feat_idx] -= self.learning_rate * (
                            error * (self.V[feat_idx] - self.W[x_i]) + 
                            self.reg_lambda * self.V[feat_idx]
                        )
                
                epoch_loss += np.mean(errors ** 2)
            
            if self.verbose and (epoch + 1) % 2 == 0:
                preds_all = np.array([self._predict_one(x) for x in X_encoded])
                mse = np.mean((preds_all - y) ** 2)
                rmse = np.sqrt(mse)
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {epoch_loss:.4f}, RMSE: {rmse:.4f}")
                
                if mse < self.best_mse:
                    self.best_mse = mse
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.early_stopping_rounds:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break
        
        return self
    
    def udf_transform(self, s: dataframe_type, **kwargs) -> dataframe_type:
        X = s[self.cols].values
        X_encoded = self._transform_input(X)
        
        preds = np.array([self._predict_one(x) for x in X_encoded])
        
        return pd.DataFrame({"prediction": preds}, index=s.index)
    
    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        X = np.array([[s.get(col, 0) for col in self.cols]])
        X_encoded = self._transform_input(X)
        pred = self._predict_one(X_encoded[0])
        
        return {"prediction": pred}
    
    def udf_get_params(self) -> dict_type:
        params = super().udf_get_params()
        params.update({
            "w0": self.w0, "W": self.W, "V": self.V,
            "learning_rate": self.learning_rate, "reg_lambda": self.reg_lambda,
            "epochs": self.epochs, "batch_size": self.batch_size,
            "best_mse": self.best_mse
        })
        return params
    
    def udf_set_params(self, params: dict):
        super().udf_set_params(params)
        self.w0 = params.get("w0", 0.0)
        self.W = params.get("W", None)
        self.V = params.get("V", None)
        self.learning_rate = params.get("learning_rate", 0.01)
        self.reg_lambda = params.get("reg_lambda", 0.001)
        self.epochs = params.get("epochs", 10)
        self.batch_size = params.get("batch_size", 256)
        self.best_mse = params.get("best_mse", float('inf'))
