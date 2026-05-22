import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from easymlops.ts.core.pipe import PipeBase
import warnings

warnings.filterwarnings("ignore")


class DeepStateRegression(PipeBase):
    """
    Deep State时间序列预测模型 (状态空间模型)
    """

    def __init__(self, output_col="deepstate_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_size=64, num_layers=2, latent_dim=16,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_size: 隐藏层大小
        :param num_layers: LSTM层数
        :param latent_dim: 潜在空间维度
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class DeepState(nn.Module):
            def __init__(self, input_dim, hidden_size, num_layers, latent_dim, output_dim):
                super().__init__()
                self.encoder = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
                self.emission = nn.Linear(hidden_size, latent_dim)
                self.transition = nn.Linear(latent_dim, latent_dim)
                self.observation = nn.Linear(latent_dim, output_dim)
                self.fc = nn.Linear(latent_dim, output_dim)

            def forward(self, x):
                _, (h_n, _) = self.encoder(x)
                h = h_n[-1]
                emission = self.emission(h)
                transition = self.transition(emission)
                out = self.fc(transition)
                return out

        return DeepState(1, self.hidden_size, self.num_layers, self.latent_dim, self.forecast_horizon)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
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

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
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


class MambaRegression(PipeBase):
    """
    Mamba时间序列预测模型 (状态空间模型)
    """

    def __init__(self, output_col="mamba_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, d_state=16, d_conv=4, expand=2,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param d_state: 状态维度
        :param d_conv: 卷积维度
        :param expand: 扩展因子
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class MambaBlock(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand):
                super().__init__()
                self.d_model = d_model
                self.d_inner = int(expand * d_model)
                self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
                self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner)
                self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
                self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
                self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
                self.D = nn.Parameter(torch.ones(self.d_inner))
                self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

            def forward(self, x):
                xz = self.in_proj(x)
                x_inner, z = xz.chunk(2, dim=-1)
                x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :-1].transpose(1, 2)
                x_ssm = torch.nn.functional.silu(x_conv + x_inner)
                y = self.out_proj(x_ssm * torch.nn.functional.silu(z))
                return y

        class MambaModel(nn.Module):
            def __init__(self, d_model, d_state, d_conv, expand, forecast_horizon):
                super().__init__()
                self.input_proj = nn.Linear(1, d_model)
                self.mamba_blocks = nn.ModuleList([MambaBlock(d_model, d_state, d_conv, expand) for _ in range(2)])
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = self.input_proj(x)
                for block in self.mamba_blocks:
                    x = block(x)
                out = self.fc(x[:, -1, :])
                return out

        return MambaModel(self.d_model, self.d_state, self.d_conv, self.expand, self.forecast_horizon)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
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

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
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


class LiquidS4Regression(PipeBase):
    """
    Liquid S4时间序列预测模型 (状态空间模型)
    """

    def __init__(self, output_col="liquids4_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, n_layers=2, dropout=0.1,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param n_layers: 层数
        :param dropout: Dropout比例
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class LiquidS4Layer(nn.Module):
            def __init__(self, d_model, dropout=0.1):
                super().__init__()
                self.d_model = d_model
                self.input_proj = nn.Linear(1, d_model)
                self.state_proj = nn.Linear(d_model, d_model)
                self.output_proj = nn.Linear(d_model, d_model)
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                x = self.input_proj(x)
                h = torch.tanh(self.state_proj(x))
                h = self.dropout(h)
                out = self.output_proj(h)
                return out

        class LiquidS4Model(nn.Module):
            def __init__(self, d_model, n_layers, forecast_horizon, dropout=0.1):
                super().__init__()
                self.layers = nn.ModuleList([LiquidS4Layer(d_model, dropout) for _ in range(n_layers)])
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                out = self.fc(x[:, -1, :])
                return out

        return LiquidS4Model(self.d_model, self.n_layers, self.forecast_horizon, self.dropout)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
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

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            output = self.model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def udf_transform(self, s: pd.DataFrame, **kwargs) -> pd.DataFrame:
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
