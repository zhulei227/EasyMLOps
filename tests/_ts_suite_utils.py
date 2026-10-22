# -*- coding: utf-8 -*-
"""旧 TS 套件升级公共工具: check 聚合 + 预测模型验收模板(行数守恒/有限性/save-load 一致)"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))


class TSSuite:
    def __init__(self, name):
        self.name = name
        self.passed = []
        self.failed = []

    def check(self, name, cond, detail=""):
        if cond:
            self.passed.append(name)
            print("[PASS] {} {}".format(name, detail))
        else:
            self.failed.append(name)
            print("[FAIL] {} {}".format(name, detail))
        return cond

    def finish(self):
        print("\n================ {} SUMMARY ================".format(self.name))
        print("PASS: {}, FAIL: {}".format(len(self.passed), len(self.failed)))
        if self.failed:
            print("FAILED:", self.failed)
            sys.exit(1)
        print("ALL TESTS PASSED")

    def verify_forecast_model(self, factory, df, output_col, name, pkl_path,
                              stochastic=False):
        """
        预测模型统一验收: fit/transform 行数守恒+有限 -> save/load 预测一致 ->
        transform_single 有限。factory: 无参构造函数; stochastic: 需固定 torch 种子。
        """
        try:
            import torch
            seed = torch.manual_seed
        except ImportError:
            seed = None

        m = factory()
        m.fit(df)
        if stochastic and seed is not None:
            seed(42)
        out = m.transform(df)
        ok = self.check("{}_rows".format(name), len(out) == len(df) and
                        output_col in out.columns and
                        np.isfinite(out[output_col].astype(float)).all())
        if not ok:
            return

        m.save(pkl_path)
        m2 = factory()
        m2.load(pkl_path)
        if stochastic and seed is not None:
            seed(42)
        out2 = m2.transform(df)
        self.check("{}_save_load_consistent".format(name),
                   np.allclose(out[output_col].astype(float).to_numpy(),
                               out2[output_col].astype(float).to_numpy(), atol=1e-6))
        if os.path.exists(pkl_path):
            os.remove(pkl_path)

        single = m.transform_single(dict(df.iloc[-1]))
        self.check("{}_single_finite".format(name),
                   output_col in single and np.isfinite(float(single[output_col])))


def generate_test_data(n=200, seed=42):
    """与旧套件一致的数据生成(含 date 首列, 验证 input_cols 自动解析)"""
    np.random.seed(seed)
    dates = pd.date_range(start="2020-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "value": np.cumsum(np.random.randn(n)) + 100,
        "feature1": np.random.randn(n) * 10,
        "feature2": np.random.randn(n) * 5,
    })
