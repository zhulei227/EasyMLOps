# -*- coding: utf-8 -*-
from easymlops.table.core import *
from easymlops.table.fm.base import FMBase
import pandas as pd
import numpy as np


pd.options.mode.copy_on_write = True


class FMClassification(FMBase):
    """
    FM (Factorization Machine) 分类模型。
    
    FM通过引入隐向量来建模特征间的二阶交互，能够在稀疏数据下进行有效的特征交叉。
    适用于点击率预估、推荐系统等场景。
    
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
        >>> from easymlops.table.fm import FMClassification
        >>> fm = FMClassification(y=label, embed_dim=8, epochs=10)
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
        n_features = sum(self.field_dims)
        
        self.w0 = 0.0
        self.W = np.random.randn(n_features) * 0.01
        self.V = np.random.randn(n_features, self.embed_dim) * 0.01
        
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
        
        sum_of_squares = np.zeros(self.embed_dim)
        square_of_sums = np.zeros(self.embed_dim)
        for feat_idx in x:
            sum_of_squares += self.V[feat_idx] ** 2
        
        for feat_idx in x:
            square_of_sums += self.V[feat_idx]
        square_of_sums = square_of_sums ** 2
        
        interactions = 0.5 * np.sum(sum_of_squares - square_of_sums)
        
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
                        grad_V = error * self.V[feat_idx] + self.reg_lambda * self.V[feat_idx]
                        self.V[feat_idx] -= self.learning_rate * grad_V
                
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


class FMRegression(FMBase):
    """
    FM (Factorization Machine) 回归模型。
    
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
        >>> from easymlops.table.fm import FMRegression
        >>> fm = FMRegression(y=target, embed_dim=8, epochs=10)
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
        n_features = sum(self.field_dims)
        
        self.w0 = 0.0
        self.W = np.random.randn(n_features) * 0.01
        self.V = np.random.randn(n_features, self.embed_dim) * 0.01
        
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
        
        sum_of_squares = np.zeros(self.embed_dim)
        square_of_sums = np.zeros(self.embed_dim)
        for feat_idx in x:
            sum_of_squares += self.V[feat_idx] ** 2
        
        for feat_idx in x:
            square_of_sums += self.V[feat_idx]
        square_of_sums = square_of_sums ** 2
        
        interactions = 0.5 * np.sum(sum_of_squares - square_of_sums)
        
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
                        grad_V = error * self.V[feat_idx] + self.reg_lambda * self.V[feat_idx]
                        self.V[feat_idx] -= self.learning_rate * grad_V
                
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
