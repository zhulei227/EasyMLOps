import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
from easymlops.ts.core.pipe import PipeBase
import warnings

warnings.filterwarnings("ignore")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TFTRegression(PipeBase):
    """
    Temporal Fusion Transformer时间序列预测模型
    """

    def __init__(self, output_col="tft_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比例
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class TFT(nn.Module):
            def __init__(self, input_dim, d_model, nhead, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                           dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x[:, -1, :]
                out = self.fc(x)
                return out

        return TFT(1, self.d_model, self.nhead, self.num_layers, self.forecast_horizon, self.dropout)

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


class InformerRegression(PipeBase):
    """
    Informer时间序列预测模型
    """

    def __init__(self, output_col="informer_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比例
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class Informer(nn.Module):
            def __init__(self, d_model, nhead, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.input_projection = nn.Linear(1, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                           dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x.mean(dim=1)
                out = self.fc(x)
                return out

        return Informer(self.d_model, self.nhead, self.num_layers, self.forecast_horizon, self.dropout)

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


class AutoformerRegression(PipeBase):
    """
    Autoformer时间序列预测模型
    """

    def __init__(self, output_col="autoformer_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比例
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class Autoformer(nn.Module):
            def __init__(self, d_model, nhead, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.input_projection = nn.Linear(1, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                           dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.decomposition = nn.Linear(d_model, d_model)
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                seasonal, trend = x[:, :x.size(1)//2, :], x[:, x.size(1)//2:, :]
                x = seasonal + trend
                x = x.mean(dim=1)
                out = self.fc(x)
                return out

        return Autoformer(self.d_model, self.nhead, self.num_layers, self.forecast_horizon, self.dropout)

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


class FEDformerRegression(PipeBase):
    """
    FEDformer时间序列预测模型
    """

    def __init__(self, output_col="fedformer_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 mode_select="random", mode_len=3,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比例
        :param mode_select: 模式选择方式
        :param mode_len: 模式长度
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.mode_select = mode_select
        self.mode_len = mode_len
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class FEDformer(nn.Module):
            def __init__(self, d_model, nhead, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.input_projection = nn.Linear(1, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                           dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x.mean(dim=1)
                out = self.fc(x)
                return out

        return FEDformer(self.d_model, self.nhead, self.num_layers, self.forecast_horizon, self.dropout)

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


class PatchTSTRegression(PipeBase):
    """
    PatchTST时间序列预测模型
    """

    def __init__(self, output_col="patchtst_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=128, nhead=4, num_layers=3, dropout=0.1,
                 patch_len=16, stride=8,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比例
        :param patch_len: Patch长度
        :param stride: Patch步长
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.patch_len = patch_len
        self.stride = stride
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class PatchTST(nn.Module):
            def __init__(self, patch_len, stride, d_model, nhead, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.patch_len = patch_len
                self.stride = stride
                num_patches = (max(1, 128 - patch_len) // stride) + 1
                self.patch_projection = nn.Linear(patch_len, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                           dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                B, L, _ = x.shape
                x = x.unfold(1, self.patch_len, self.stride).transpose(1, 2)
                x = self.patch_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x.mean(dim=1)
                out = self.fc(x)
                return out

        return PatchTST(self.patch_len, self.stride, self.d_model, self.nhead,
                       self.num_layers, self.forecast_horizon, self.dropout)

    def udf_fit(self, s: pd.DataFrame, **kwargs):
        if self.input_cols is None or len(self.input_cols) == 0:
            self.input_cols = [s.columns[0]]

        target = s[self.input_cols[0]].values.astype(np.float32)

        self.data_mean = np.mean(target)
        self.data_std = np.std(target) + 1e-8
        target_norm = (target - self.data_mean) / self.data_std

        self.time_period = min(128, len(target_norm) - self.forecast_horizon)

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


class TimesNetRegression(PipeBase):
    """
    TimesNet时间序列预测模型
    """

    def __init__(self, output_col="timesnet_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, d_ff=128, num_kernels=3, num_layers=2, dropout=0.1,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param d_ff: 前馈网络维度
        :param num_kernels: 卷积核数量
        :param num_layers: 层数
        :param dropout: Dropout比例
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.d_ff = d_ff
        self.num_kernels = num_kernels
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class TimesNet(nn.Module):
            def __init__(self, d_model, d_ff, num_kernels, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.input_projection = nn.Linear(1, d_model)
                self.conv_layers = nn.ModuleList([
                    nn.Conv1d(d_model, d_ff, kernel_size=3, padding=1)
                    for _ in range(num_kernels)
                ])
                self.layers = nn.ModuleList([
                    nn.TransformerEncoderLayer(d_model, 4, dim_feedforward=d_ff, dropout=dropout, batch_first=True)
                    for _ in range(num_layers)
                ])
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = self.input_projection(x)
                x = x.transpose(1, 2)
                for conv in self.conv_layers:
                    x_conv = conv(x)
                    x = x + x_conv
                x = x.transpose(1, 2)
                for layer in self.layers:
                    x = layer(x)
                x = x.mean(dim=1)
                out = self.fc(x)
                return out

        return TimesNet(self.d_model, self.d_ff, self.num_kernels, self.num_layers,
                       self.forecast_horizon, self.dropout)

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


class iTransformerRegression(PipeBase):
    """
    iTransformer时间序列预测模型
    """

    def __init__(self, output_col="itransformer_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 d_model=64, nhead=4, num_layers=2, dropout=0.1,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param d_model: 模型维度
        :param nhead: 注意力头数
        :param num_layers: Transformer层数
        :param dropout: Dropout比例
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class iTransformer(nn.Module):
            def __init__(self, d_model, nhead, num_layers, forecast_horizon, dropout):
                super().__init__()
                self.input_projection = nn.Linear(time_period, d_model)
                self.pos_encoder = PositionalEncoding(d_model)
                encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model * 4,
                                                           dropout=dropout, batch_first=True)
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
                self.fc = nn.Linear(d_model, forecast_horizon)

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.input_projection(x)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x.mean(dim=1)
                out = self.fc(x)
                return out

        return iTransformer(self.d_model, self.nhead, self.num_layers, self.forecast_horizon, self.dropout)

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
