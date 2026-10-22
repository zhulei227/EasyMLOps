# -*- coding: utf-8 -*-
"""
torch 预测模型公共 Mixin。

解决三个历史问题:
    1. save/load 只存基础参数, 权重丢失, load 后 transform 静默输出 0
       -> model_state(state_dict bytes) 随 get_params 持久化, load 后可直接预测
    2. 未实现 transform_single, pipeline 单条推理路径报错
       -> 统一实现: 以 fit 序列为历史起点, 每条记录推进滑动窗口后输出
          horizon 首步预测(与批量 transform 的首步预测同口径)
    3. 行数守恒: 单条输出标量, 与批量 transform 的标量广播约定一致

子类约定:
    - udf_fit 训练结束后调用 self._after_fit_(target_norm) 记录窗口历史
    - 覆写 _window_tensor 适配非 (1, L, 1) 的输入形状(如 NBeats 的 (1, L))
    - udf_get_params/udf_set_params 中合并 self._dl_get_params()/_dl_set_params()
"""
import io

import numpy as np
import pandas as pd
import torch

from easymlops.ts.core.pipe import PipeBase

dataframe_type = pd.DataFrame
dict_type = dict


class TorchForecastMixin(PipeBase):
    """torch 预测模型 Mixin: 权重持久化 + 单条推理(不改变既有 __init__/MRO 参数链)"""

    # ---- 子类钩子 ----
    def _window_tensor(self, seq: np.ndarray):
        """(L,) 序列 -> 模型输入张量; 默认 (1, L, 1)"""
        return torch.FloatTensor(np.asarray(seq, dtype=np.float32)) \
            .unsqueeze(0).unsqueeze(-1).to(self.device)

    def _after_fit_(self, target_norm: np.ndarray):
        """udf_fit 末尾调用: 记录归一化训练序列, 供单条推理使用"""
        self._train_tail = np.asarray(target_norm, dtype=np.float32).tolist()
        self._stream_window = list(self._train_tail)

    # ---- 预测 ----
    def _predict_first_step(self, window) -> float:
        self.model.eval()
        with torch.no_grad():
            out = self.model(self._window_tensor(window))
        if isinstance(out, tuple):
            out = out[0]   # VAE/NF 等 forward 返回 (recon, ...) 元组
        pred = out.cpu().numpy()[0]
        return float(pred[0]) * self.data_std + self.data_mean

    # ---- 持久化 ----
    def _serialize_model(self):
        if self.model is None:
            return None
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        return buf.getvalue()

    def _restore_model(self, state):
        if state is None:
            self.model = None
            return
        self.model = self._build_model().to(self.device)
        buf = io.BytesIO(state)
        self.model.load_state_dict(torch.load(buf, map_location=self.device))
        self.model.eval()

    def _dl_get_params(self) -> dict_type:
        # getattr 兜底: 模板模型未 fit 时(如作为 GroupForecast 模板)也可安全序列化
        return {"output_col": self.output_col, "input_cols": self.input_cols,
                "time_period": self.time_period, "forecast_horizon": self.forecast_horizon,
                "data_mean": getattr(self, "data_mean", 0.0),
                "data_std": getattr(self, "data_std", 1.0),
                "_train_tail": getattr(self, "_train_tail", None),
                "model_state": self._serialize_model()}

    def _dl_set_params(self, params: dict_type):
        self.output_col = params["output_col"]
        self.input_cols = params["input_cols"]
        self.time_period = params["time_period"]
        self.forecast_horizon = params["forecast_horizon"]
        self.data_mean = params["data_mean"]
        self.data_std = params["data_std"]
        self._train_tail = params["_train_tail"]
        self._stream_window = list(self._train_tail) if self._train_tail is not None else None
        self._restore_model(params["model_state"])

    # ---- 单条推理 ----
    def udf_transform_single(self, s: dict_type, **kwargs) -> dict_type:
        if self.model is None:
            s[self.output_col] = 0
            return s
        if self._stream_window is None:
            self._stream_window = list(self._train_tail or [])
        col = self.input_cols[0] if self.input_cols else None
        try:
            raw = float(s[col])
        except (TypeError, ValueError, KeyError):
            raw = 0.0
        self._stream_window.append((raw - self.data_mean) / self.data_std)
        if len(self._stream_window) > self.time_period:
            self._stream_window = self._stream_window[-self.time_period:]
        s[self.output_col] = self._predict_first_step(self._stream_window)
        return s
