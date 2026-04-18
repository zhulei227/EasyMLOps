import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from easymlops.ts.core.pipe import PipeBase
import warnings

warnings.filterwarnings("ignore")


class NBeatsRegression(PipeBase):
    """
    N-BEATS时间序列预测模型
    """

    def __init__(self, output_col="nbeats_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_units=64, epochs=20, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_units: 隐藏层单元数
        :param epochs: 训练轮数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class SimpleNBeats(nn.Module):
            def __init__(self, input_dim, hidden_units, output_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, output_dim)
                )

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return SimpleNBeats(self.time_period, self.hidden_units, self.forecast_horizon)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
            numeric_cols = s.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.input_cols = [numeric_cols[0]]
            else:
                self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(np.float32)

        self.data_mean = np.mean(target)
        self.data_std = np.std(target) + 1e-8
        target_norm = (target - self.data_mean) / self.data_std

        self.time_period = min(self.time_period, len(target_norm) - self.forecast_horizon)

        X, y = [], []
        for i in range(len(target_norm) - self.time_period - self.forecast_horizon):
            X.append(target_norm[i:i + self.time_period])
            y.append(target_norm[i + self.time_period:i + self.time_period + self.forecast_horizon])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if len(X) == 0:
            return self

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.model is None:
            s = s.copy()
            s[self.output_col] = 0
            return s

        self.model.eval()
        with torch.no_grad():
            target = s[self.input_cols[0]].values.astype(np.float32)
            target_norm = (target - self.data_mean) / self.data_std

            if len(target_norm) >= self.time_period:
                input_seq = target_norm[-self.time_period:]
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                pred = self.model(input_tensor)
                pred = pred.cpu().numpy()[0]
            else:
                pred = np.zeros(self.forecast_horizon)

            pred = pred * self.data_std + self.data_mean

        s = s.copy()
        pred_value = pred[0] if len(pred) > 0 else 0
        s[self.output_col] = pred_value
        return s


class NHiTSRegression(PipeBase):
    """
    N-HiTS时间序列预测模型
    """

    def __init__(self, output_col="nhits_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_units=64, epochs=20, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_units: 隐藏层单元数
        :param epochs: 训练轮数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class SimpleNHiTS(nn.Module):
            def __init__(self, input_dim, hidden_units, output_dim):
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, hidden_units),
                    nn.ReLU(),
                    nn.Linear(hidden_units, output_dim)
                )

            def forward(self, x):
                x = x.view(x.size(0), -1)
                return self.fc(x)

        return SimpleNHiTS(self.time_period, self.hidden_units, self.forecast_horizon)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
            numeric_cols = s.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.input_cols = [numeric_cols[0]]
            else:
                self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(np.float32)

        self.data_mean = np.mean(target)
        self.data_std = np.std(target) + 1e-8
        target_norm = (target - self.data_mean) / self.data_std

        self.time_period = min(self.time_period, len(target_norm) - self.forecast_horizon)

        X, y = [], []
        for i in range(len(target_norm) - self.time_period - self.forecast_horizon):
            X.append(target_norm[i:i + self.time_period])
            y.append(target_norm[i + self.time_period:i + self.time_period + self.forecast_horizon])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        if len(X) == 0:
            return self

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.model is None:
            s = s.copy()
            s[self.output_col] = 0
            return s

        self.model.eval()
        with torch.no_grad():
            target = s[self.input_cols[0]].values.astype(np.float32)
            target_norm = (target - self.data_mean) / self.data_std

            if len(target_norm) >= self.time_period:
                input_seq = target_norm[-self.time_period:]
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                pred = self.model(input_tensor)
                pred = pred.cpu().numpy()[0]
            else:
                pred = np.zeros(self.forecast_horizon)

            pred = pred * self.data_std + self.data_mean

        s = s.copy()
        pred_value = pred[0] if len(pred) > 0 else 0
        s[self.output_col] = pred_value
        return s


class DeepARRegression(PipeBase):
    """
    DeepAR时间序列预测模型
    """

    def __init__(self, output_col="deepar_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_size=64, epochs=20, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_size: 隐藏层大小
        :param epochs: 训练轮数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class SimpleDeepAR(nn.Module):
            def __init__(self, input_dim, hidden_size, output_dim):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_dim)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                out = lstm_out[:, -1, :]
                out = self.fc(out)
                return out

        return SimpleDeepAR(1, self.hidden_size, self.forecast_horizon)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
            numeric_cols = s.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.input_cols = [numeric_cols[0]]
            else:
                self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(np.float32)

        self.data_mean = np.mean(target)
        self.data_std = np.std(target) + 1e-8
        target_norm = (target - self.data_mean) / self.data_std

        self.time_period = min(self.time_period, len(target_norm) - self.forecast_horizon)

        X, y = [], []
        for i in range(len(target_norm) - self.time_period - self.forecast_horizon):
            X.append(target_norm[i:i + self.time_period])
            y.append(target_norm[i + self.time_period:i + self.time_period + self.forecast_horizon])

        X = np.array(X, dtype=np.float32).reshape(-1, self.time_period, 1)
        y = np.array(y, dtype=np.float32)

        if len(X) == 0:
            return self

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if self.model is None:
            s = s.copy()
            s[self.output_col] = 0
            return s

        self.model.eval()
        with torch.no_grad():
            target = s[self.input_cols[0]].values.astype(np.float32)
            target_norm = (target - self.data_mean) / self.data_std

            if len(target_norm) >= self.time_period:
                input_seq = target_norm[-self.time_period:]
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(self.device)
                pred = self.model(input_tensor)
                pred = pred.cpu().numpy()[0]
            else:
                pred = np.zeros(self.forecast_horizon)

            pred = pred * self.data_std + self.data_mean

        s = s.copy()
        s[self.output_col] = pred[:len(s)]
        return s


class GPRegression(PipeBase):
    """
    Gaussian Process时间序列预测模型
    """

    def __init__(self, output_col="gp_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 length_scale=1.0, variance=1.0, **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param length_scale: 核函数长度尺度
        :param variance: 核函数方差
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.length_scale = length_scale
        self.variance = variance

    def _rbf_kernel(self, X1, X2, length_scale, variance):
        dists = np.sum(X1 ** 2, axis=1, keepdims=True) + np.sum(X2 ** 2, axis=1) - 2 * np.dot(X1, X2.T)
        return variance * np.exp(-0.5 * dists / length_scale ** 2)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
            numeric_cols = s.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) > 0:
                self.input_cols = [numeric_cols[0]]
            else:
                self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(np.float64)

        self.data_mean = np.mean(target)
        self.data_std = np.std(target) + 1e-8
        target_norm = (target - self.data_mean) / self.data_std

        self.time_period = min(self.time_period, len(target_norm) - self.forecast_horizon)

        X = np.arange(len(target_norm)).reshape(-1, 1)
        y = target_norm

        self.X_train = X
        self.y_train = y

        K = self._rbf_kernel(X, X, self.length_scale, self.variance) + 1e-6 * np.eye(len(X))
        self.K_inv = np.linalg.inv(K)

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
        target = s[self.input_cols[0]].values.astype(np.float64)
        target_norm = (target - self.data_mean) / self.data_std

        X_test = np.arange(len(target_norm), len(target_norm) + self.forecast_horizon).reshape(-1, 1)

        K_star = self._rbf_kernel(X_test, self.X_train, self.length_scale, self.variance)

        mu = K_star @ self.K_inv @ self.y_train
        pred = mu

        pred = pred * self.data_std + self.data_mean

        s = s.copy()
        s[self.output_col] = pred[:len(s)]
        return s
