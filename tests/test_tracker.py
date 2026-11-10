# -*- coding: utf-8 -*-
# Tracker(swanlab 适配) 验收: 降级/本地记录/MLP 接入/流水线透传/ts 接入/disabled 模式 (v0.2.13)
import os
import shutil
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from easymlops.core import Tracker, pop_tracker

PASS, FAIL = [], []
LOG_DIR = "d:/datas/share/traes/easymlops-trae/tests/_tmp_tracker_logs"


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print("[PASS] {} {}".format(name, detail))
    else:
        FAIL.append(name)
        print("[FAIL] {} {}".format(name, detail))


try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import swanlab  # noqa: F401
    HAS_SWANLAB = True
except ImportError:
    HAS_SWANLAB = False

if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)

# ============ 数据 ============
rng = np.random.RandomState(7)
n = 400
x_clf, y_clf = make_classification(n_samples=n, n_features=6, n_informative=4,
                                   n_classes=2, random_state=7)
clf_df = pd.DataFrame(x_clf, columns=[f"f{i}" for i in range(6)])
clf_df["city"] = rng.choice(["bj", "sh", "sz"], n)
y_clf = pd.Series(y_clf)

# ============ 1. no-op 降级 + 本地 JSONL ============
print("=== 1. Tracker 降级与本地记录 ===")
t = Tracker(project="easymlops-test", experiment_name="noop-run",
            mode="offline", log_dir=LOG_DIR, config={"lr": 0.01})
t.log({"loss": 1.0}, step=1)
t.log({"loss": 0.5}, step=2)
t.log({"loss": 0.3, "acc": 0.8}, step=3)
t.log_params({"epochs": 5})
hist = t.read_history()
check("history_len", len(hist) == 3, len(hist))
check("history_fields", set(hist[0].keys()) == {"step", "time", "metrics"})
check("history_metrics", hist[2]["metrics"]["loss"] == 0.3 and
      hist[2]["metrics"]["acc"] == 0.8)
check("config_merged", t.config.get("lr") == 0.01 and t.config.get("epochs") == 5)
s = t.summary()
check("summary_first_last", s["loss"]["first"] == 1.0 and s["loss"]["last"] == 0.3)
check("summary_min_max", s["loss"]["min"] == 0.3 and s["loss"]["max"] == 1.0 and
      s["acc"]["max"] == 0.8)
t.finish()
t.finish()  # 幂等
check("finish_idempotent", t._finished)
check("jsonl_exists", os.path.exists(t.history_path))
with open(t.history_path, encoding="utf-8") as f:
    lines = [l for l in f.read().splitlines() if l.strip()]
check("jsonl_parse", len(lines) == 3 and '"loss": 0.3' in lines[2])
# 未安装 swanlab 时也不应抛异常(降级 no-op)
check("swanlab_optional", HAS_SWANLAB or t._run is None,
      "swanlab installed={}".format(HAS_SWANLAB))

# ============ 2. pop_tracker 工具函数 ============
print("=== 2. pop_tracker ===")
kwargs = {"tracker": t, "other": 1}
tr = pop_tracker(kwargs)
check("pop_get", tr is t and kwargs == {"other": 1})
check("pop_missing", pop_tracker({}) is None)

# ============ 3. context manager ============
print("=== 3. context manager ===")
with Tracker(project="easymlops-test", experiment_name="ctx-run",
             mode="disabled", log_dir=LOG_DIR) as t2:
    t2.log({"loss": 0.1}, step=1)
check("ctx_finished", t2._finished)
check("ctx_disabled_history", len(t2.read_history()) == 1)

if HAS_TORCH:
    from easymlops.table.nn import TabularMLPClassification
    from easymlops import TablePipeLine

    # ============ 4. MLP fit 接入 tracker ============
    print("=== 4. TabularMLP 接入 ===")
    t3 = Tracker(project="easymlops-test", experiment_name="mlp-run",
                 mode="offline", log_dir=LOG_DIR, config={"epochs": 6})
    m = TabularMLPClassification(y=y_clf[:320], epochs=6, hidden_dims=(32,),
                                 random_state=42)
    m.fit(clf_df[:320], tracker=t3)
    t3.finish()
    hist3 = t3.read_history()
    check("mlp_history_epochs", len(hist3) == 6,
          "records={}".format(len(hist3)))
    check("mlp_steps_monotonic", [r["step"] for r in hist3] == [1, 2, 3, 4, 5, 6])
    losses = [r["metrics"]["loss"] for r in hist3]
    check("mlp_loss_recorded", all(np.isfinite(losses)))
    out = m.transform(clf_df[320:].copy())
    check("mlp_fit_ok", len(out) == 80)

    # ============ 5. 流水线 fit 透传 tracker ============
    print("=== 5. TablePipeLine 透传 ===")
    t4 = Tracker(project="easymlops-test", experiment_name="pipe-run",
                 mode="offline", log_dir=LOG_DIR)
    pipe = TabularMLPClassification(y=y_clf[:320], epochs=3, hidden_dims=(16,),
                                    random_state=42)
    pl = TablePipeLine().pipe(pipe)
    pl.fit(clf_df[:320], tracker=t4)
    t4.finish()
    check("pipeline_tracker_history", len(t4.read_history()) == 3,
          "records={}".format(len(t4.read_history())))

    # ============ 6. ts Transformer 接入(kwargs 透传修复) ============
    print("=== 6. ts Transformer 接入 ===")
    from easymlops.ts.transformer import InformerRegression
    t5 = Tracker(project="easymlops-test", experiment_name="ts-run",
                 mode="offline", log_dir=LOG_DIR)
    ts_df = pd.DataFrame({"value": np.sin(np.linspace(0, 20, 120)) + 0.05 * rng.randn(120)})
    ts_pipe = InformerRegression(output_col="pred", time_period=8,
                                 forecast_horizon=2, epochs=4, d_model=16)
    ts_pipe.fit(ts_df, tracker=t5)
    t5.finish()
    hist5 = t5.read_history()
    check("ts_history_epochs", len(hist5) == 4, "records={}".format(len(hist5)))
    check("ts_transform_ok", "pred" in ts_pipe.transform(ts_df.copy()).columns)

# ============ 7. FM(numpy 手写 SGD)接入 ============
print("=== 7. FM 接入 ===")
from easymlops.table.fm import FMClassification
t6 = Tracker(project="easymlops-test", experiment_name="fm-run",
             mode="offline", log_dir=LOG_DIR)
fm_df = pd.DataFrame({
    "f1": rng.randint(0, 5, 200), "f2": rng.randint(0, 8, 200),
    "f3": rng.randint(0, 3, 200)})
fm_label = pd.Series(rng.randint(0, 2, 200))
fm = FMClassification(y=fm_label, cols=["f1", "f2", "f3"], field_dims=[5, 8, 3],
                      embed_dim=4, epochs=5, batch_size=64, verbose=False)
fm.fit(fm_df, tracker=t6)
t6.finish()
check("fm_history_epochs", len(t6.read_history()) == 5,
      "records={}".format(len(t6.read_history())))
check("fm_transform_ok", len(fm.transform(fm_df.head(20))) == 20)

# ============ 8. OptunaTuner 接入(trial 级指标) ============
print("=== 8. OptunaTuner 接入 ===")
try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

if HAS_OPTUNA:
    from easymlops.table.regression import LGBMRegression
    from easymlops.table.tuner import OptunaTuner
    from sklearn.datasets import make_regression
    x_reg, y_reg = make_regression(n_samples=200, n_features=6, n_informative=4,
                                   noise=10.0, random_state=7)
    reg_df = pd.DataFrame(x_reg, columns=[f"g{i}" for i in range(6)])
    y_reg = pd.Series(y_reg)
    t7 = Tracker(project="easymlops-test", experiment_name="tuner-run",
                 mode="offline", log_dir=LOG_DIR)
    tuner = OptunaTuner(
        model_class=LGBMRegression,
        fixed_params={"y": y_reg, "use_faster_predictor": False},
        search_space={
            "learning_rate": lambda t: t.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": lambda t: t.suggest_int("num_leaves", 8, 32),
        },
        n_trials=3, cv=2, task="regression", seed=7)
    tuner.fit(reg_df, tracker=t7)
    t7.finish()
    hist7 = t7.read_history()
    check("tuner_trials_logged", len(hist7) == 3,
          "records={}".format(len(hist7)))
    check("tuner_metric_name", all("rmse" in r["metrics"] for r in hist7))
    check("tuner_transform_ok", len(tuner.transform(reg_df)) == 200)
else:
    print("[SKIP] OptunaTuner tests (optuna not installed)")

# ============ 汇总 ============
print("\n===== RESULT: PASS={} FAIL={} =====".format(len(PASS), len(FAIL)))
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL GREEN")
