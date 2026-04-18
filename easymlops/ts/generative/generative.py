import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from easymlops.ts.core.pipe import PipeBase
import warnings

warnings.filterwarnings("ignore")


class VAERegression(PipeBase):
    """
    Variational Autoencoder (VAE) 时间序列预测模型
    """

    def __init__(self, output_col="vae_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_dim=64, latent_dim=16,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_dim: 隐藏层维度
        :param latent_dim: 潜在空间维度
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class VAE(nn.Module):
            def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU()
                )
                self.fc_mu = nn.Linear(hidden_dim, latent_dim)
                self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
                self.decoder = nn.Sequential(
                    nn.Linear(latent_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )

            def encode(self, x):
                h = self.encoder(x)
                return self.fc_mu(h), self.fc_logvar(h)

            def reparameterize(self, mu, logvar):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                return mu + eps * std

            def decode(self, z):
                return self.decoder(z)

            def forward(self, x):
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                return self.decode(z), mu, logvar

        return VAE(self.time_period, self.hidden_dim, self.latent_dim, self.forecast_horizon)

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

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            recon, mu, logvar = self.model(X_tensor)
            recon_loss = nn.MSELoss()(recon, y_tensor)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + kl_loss
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
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                recon, _, _ = self.model(input_tensor)
                pred = recon.cpu().numpy()[0]
            else:
                pred = np.zeros(self.forecast_horizon)

            pred = pred * self.data_std + self.data_mean

        s = s.copy()
        s[self.output_col] = pred[:len(s)]
        return s


class NormalizingFlowRegression(PipeBase):
    """
    Normalizing Flows 时间序列预测模型
    """

    def __init__(self, output_col="nf_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_dim=64, num_layers=3,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_dim: 隐藏层维度
        :param num_layers: 流层数量
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_model(self):
        class FlowLayer(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dim, dim * 2),
                    nn.ReLU(),
                    nn.Linear(dim * 2, dim * 2)
                )

            def forward(self, x):
                h = self.net(x)
                mu, logscale = h.chunk(2, dim=-1)
                scale = torch.exp(logscale)
                z = x * scale + mu
                log_det = torch.sum(scale, dim=-1)
                return z, log_det

        class NormalizingFlow(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
                super().__init__()
                self.flows = nn.ModuleList([FlowLayer(input_dim) for _ in range(num_layers)])
                self.fc = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )

            def forward(self, x):
                log_det_sum = 0
                for flow in self.flows:
                    x, log_det = flow(x)
                    log_det_sum += log_det
                out = self.fc(x)
                return out, log_det_sum

        return NormalizingFlow(self.time_period, self.hidden_dim, self.num_layers, self.forecast_horizon)

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

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        self.model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            pred, _ = self.model(X_tensor)
            loss = criterion(pred, y_tensor)
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
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)
                pred, _ = self.model(input_tensor)
                pred = pred.cpu().numpy()[0]
            else:
                pred = np.zeros(self.forecast_horizon)

            pred = pred * self.data_std + self.data_mean

        s = s.copy()
        s[self.output_col] = pred[:len(s)]
        return s


class DiffusionRegression(PipeBase):
    """
    Diffusion Model 时间序列预测模型
    """

    def __init__(self, output_col="diffusion_predict", input_cols=None,
                 time_period=30, forecast_horizon=7,
                 hidden_dim=64, num_steps=100,
                 **kwargs):
        """

        :param output_col: 输出列名
        :param input_cols: 输入列名
        :param time_period: 时间窗口大小
        :param forecast_horizon: 预测步数
        :param hidden_dim: 隐藏层维度
        :param num_steps: 扩散步数
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.output_col = output_col
        self.input_cols = input_cols
        self.time_period = time_period
        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.beta_start = 0.0001
        self.beta_end = 0.02

    def _build_model(self):
        class DiffusionModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super().__init__()
                self.input_proj = nn.Linear(input_dim, hidden_dim)
                self.time_embed = nn.Sequential(
                    nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                self.net = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim)
                )

            def forward(self, x, t):
                h = self.input_proj(x)
                t_emb = self.time_embed(t.unsqueeze(-1))
                h = torch.cat([h, t_emb], dim=-1)
                return self.net(h)

        return DiffusionModel(self.time_period, self.hidden_dim, self.forecast_horizon)

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

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)

        self.model = self._build_model().to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        betas = torch.linspace(self.beta_start, self.beta_end, self.num_steps).to(self.device)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.model.train()
        for epoch in range(30):
            optimizer.zero_grad()
            idx = torch.randint(0, self.num_steps, (X_tensor.size(0),)).to(self.device)
            alpha = alphas_cumprod[idx].view(-1, 1)
            noise = torch.randn_like(y_tensor)
            y_noisy = torch.sqrt(alpha) * y_tensor + torch.sqrt(1 - alpha) * noise
            t = idx.float() / self.num_steps
            pred = self.model(X_tensor, t)
            loss = nn.MSELoss()(pred, noise)
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
                input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)

                y_sample = torch.randn(1, self.forecast_horizon).to(self.device)
                for t in reversed(range(self.num_steps)):
                    t_tensor = torch.full((1,), t / self.num_steps).to(self.device)
                    pred = self.model(input_tensor, t_tensor)
                    y_sample = y_sample - pred * 0.01

                pred = y_sample.cpu().numpy()[0]
            else:
                pred = np.zeros(self.forecast_horizon)

            pred = pred * self.data_std + self.data_mean

        s = s.copy()
        s[self.output_col] = pred[:len(s)]
        return s
