# -*- coding: utf-8 -*-
from easymlops.table.core import *
from easymlops.table.fm.base import FMBase
import pandas as pd
import numpy as np


pd.options.mode.copy_on_write = True


class DeepFMClassification(FMBase):
    """
    DeepFM 分类模型。
    
    DeepFM结合了FM和深度神经网络，同时学习低阶和高阶特征交互。
    - FM组件：学习一阶特征和二阶特征交互
    - Deep组件：学习高阶特征交互
    
    适用于点击率预估、推荐系统等场景。
    
    Args:
        y: 目标变量
        cols: 用于模型训练的列
        field_dims: 每个field的维度列表
        embed_dim: 嵌入维度
        hidden_layers: DNN隐藏层维度列表
        learning_rate: 学习率
        reg_lambda: L2正则化系数
        dropout_rate: Dropout比例
        epochs: 训练轮数
        batch_size: 批大小
        early_stopping_rounds: 早停轮数
        verbose: 是否打印训练日志
        
    Example:
        >>> from easymlops.table.fm import DeepFMClassification
        >>> deepfm = DeepFMClassification(y=label, embed_dim=8, hidden_layers=[64, 32], epochs=10)
        >>> deepfm.fit(df).transform(df)
    """
    
    def __init__(self, y: series_type = None, cols="all", field_dims=None, embed_dim=8,
                 hidden_layers=None, learning_rate=0.001, reg_lambda=0.0001, dropout_rate=0.5,
                 epochs=10, batch_size=256, early_stopping_rounds=5, verbose=True, **kwargs):
        super().__init__(y=y, cols=cols, task_type="classification", 
                        field_dims=field_dims, embed_dim=embed_dim, **kwargs)
        self.hidden_layers = hidden_layers if hidden_layers else [64, 32]
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
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
        
        self.deep_layers = []
        self.deep_biases = []
        input_dim = len(self.cols) * self.embed_dim
        
        for hidden_dim in self.hidden_layers:
            self.deep_layers.append(np.random.randn(input_dim, hidden_dim) * 0.01)
            self.deep_biases.append(np.zeros(hidden_dim))
            input_dim = hidden_dim
        
        self.output_layer = np.random.randn(self.hidden_layers[-1] + len(self.cols), 1) * 0.01
        self.output_bias = 0.0
        
    def _transform_input(self, X):
        X_encoded = []
        for i in range(len(self.cols)):
            col_values = X[:, i]
            offset = sum(self.field_dims[:i])
            encoded = col_values + offset
            X_encoded.append(encoded)
        return np.column_stack(X_encoded)
    
    def _get_embeddings(self, x):
        embeddings = []
        for feat_idx in x:
            embeddings.append(self.V[feat_idx])
        return np.concatenate(embeddings)
    
    def _deep_forward(self, embeddings):
        x = embeddings
        for i, (layer, bias) in enumerate(zip(self.deep_layers, self.deep_biases)):
            x = np.dot(x, layer) + bias
            x = np.tanh(x)
            if i < len(self.deep_layers) - 1:
                mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
                x = x * mask / (1 - self.dropout_rate)
        return x
    
    def _predict_one(self, x):
        linear = self.w0 + np.sum(self.W[x])
        
        sum_of_squares = np.zeros(self.embed_dim)
        square_of_sums = np.zeros(self.embed_dim)
        for feat_idx in x:
            sum_of_squares += self.V[feat_idx] ** 2
        
        for feat_idx in x:
            square_of_sums += self.V[feat_idx]
        square_of_sums = square_of_sums ** 2
        
        fm_interactions = 0.5 * np.sum(sum_of_squares - square_of_sums)
        
        embeddings = self._get_embeddings(x)
        deep_output = self._deep_forward(embeddings)
        
        deep_interactions = np.dot(deep_output, self.output_layer[:len(deep_output)]) + self.output_bias
        
        pred = linear + fm_interactions + deep_interactions.flatten()[0]
        
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
            "deep_layers": self.deep_layers, "deep_biases": self.deep_biases,
            "output_layer": self.output_layer, "output_bias": self.output_bias,
            "learning_rate": self.learning_rate, "reg_lambda": self.reg_lambda,
            "dropout_rate": self.dropout_rate, "epochs": self.epochs,
            "hidden_layers": self.hidden_layers, "best_score": self.best_score
        })
        return params
    
    def udf_set_params(self, params: dict):
        super().udf_set_params(params)
        self.w0 = params.get("w0", 0.0)
        self.W = params.get("W", None)
        self.V = params.get("V", None)
        self.deep_layers = params.get("deep_layers", [])
        self.deep_biases = params.get("deep_biases", [])
        self.output_layer = params.get("output_layer", None)
        self.output_bias = params.get("output_bias", 0.0)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.reg_lambda = params.get("reg_lambda", 0.0001)
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.epochs = params.get("epochs", 10)
        self.hidden_layers = params.get("hidden_layers", [64, 32])
        self.best_score = params.get("best_score", 0)


class DeepFMRegression(FMBase):
    """
    DeepFM 回归模型。
    
    Args:
        y: 目标变量
        cols: 用于模型训练的列
        field_dims: 每个field的维度列表
        embed_dim: 嵌入维度
        hidden_layers: DNN隐藏层维度列表
        learning_rate: 学习率
        reg_lambda: L2正则化系数
        dropout_rate: Dropout比例
        epochs: 训练轮数
        batch_size: 批大小
        
    Example:
        >>> from easymlops.table.fm import DeepFMRegression
        >>> deepfm = DeepFMRegression(y=target, embed_dim=8, hidden_layers=[64, 32], epochs=10)
        >>> deepfm.fit(df).transform(df)
    """
    
    def __init__(self, y: series_type = None, cols="all", field_dims=None, embed_dim=8,
                 hidden_layers=None, learning_rate=0.001, reg_lambda=0.0001, dropout_rate=0.5,
                 epochs=10, batch_size=256, early_stopping_rounds=5, verbose=True, **kwargs):
        super().__init__(y=y, cols=cols, task_type="regression", 
                        field_dims=field_dims, embed_dim=embed_dim, **kwargs)
        self.hidden_layers = hidden_layers if hidden_layers else [64, 32]
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
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
        
        self.deep_layers = []
        self.deep_biases = []
        input_dim = len(self.cols) * self.embed_dim
        
        for hidden_dim in self.hidden_layers:
            self.deep_layers.append(np.random.randn(input_dim, hidden_dim) * 0.01)
            self.deep_biases.append(np.zeros(hidden_dim))
            input_dim = hidden_dim
        
        self.output_layer = np.random.randn(self.hidden_layers[-1] + len(self.cols), 1) * 0.01
        self.output_bias = 0.0
        
    def _transform_input(self, X):
        X_encoded = []
        for i in range(len(self.cols)):
            col_values = X[:, i]
            offset = sum(self.field_dims[:i])
            encoded = col_values + offset
            X_encoded.append(encoded)
        return np.column_stack(X_encoded)
    
    def _get_embeddings(self, x):
        embeddings = []
        for feat_idx in x:
            embeddings.append(self.V[feat_idx])
        return np.concatenate(embeddings)
    
    def _deep_forward(self, embeddings):
        x = embeddings
        for i, (layer, bias) in enumerate(zip(self.deep_layers, self.deep_biases)):
            x = np.dot(x, layer) + bias
            x = np.tanh(x)
            if i < len(self.deep_layers) - 1:
                mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
                x = x * mask / (1 - self.dropout_rate)
        return x
    
    def _predict_one(self, x):
        linear = self.w0 + np.sum(self.W[x])
        
        sum_of_squares = np.zeros(self.embed_dim)
        square_of_sums = np.zeros(self.embed_dim)
        for feat_idx in x:
            sum_of_squares += self.V[feat_idx] ** 2
        
        for feat_idx in x:
            square_of_sums += self.V[feat_idx]
        square_of_sums = square_of_sums ** 2
        
        fm_interactions = 0.5 * np.sum(sum_of_squares - square_of_sums)
        
        embeddings = self._get_embeddings(x)
        deep_output = self._deep_forward(embeddings)
        
        deep_interactions = np.dot(deep_output, self.output_layer[:len(deep_output)]) + self.output_bias
        
        return linear + fm_interactions + deep_interactions.flatten()[0]
    
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
            "deep_layers": self.deep_layers, "deep_biases": self.deep_biases,
            "output_layer": self.output_layer, "output_bias": self.output_bias,
            "learning_rate": self.learning_rate, "reg_lambda": self.reg_lambda,
            "dropout_rate": self.dropout_rate, "epochs": self.epochs,
            "hidden_layers": self.hidden_layers, "best_mse": self.best_mse
        })
        return params
    
    def udf_set_params(self, params: dict):
        super().udf_set_params(params)
        self.w0 = params.get("w0", 0.0)
        self.W = params.get("W", None)
        self.V = params.get("V", None)
        self.deep_layers = params.get("deep_layers", [])
        self.deep_biases = params.get("deep_biases", [])
        self.output_layer = params.get("output_layer", None)
        self.output_bias = params.get("output_bias", 0.0)
        self.learning_rate = params.get("learning_rate", 0.001)
        self.reg_lambda = params.get("reg_lambda", 0.0001)
        self.dropout_rate = params.get("dropout_rate", 0.5)
        self.epochs = params.get("epochs", 10)
        self.hidden_layers = params.get("hidden_layers", [64, 32])
        self.best_mse = params.get("best_mse", float('inf'))
