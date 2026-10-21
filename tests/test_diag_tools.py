# -*- coding: utf-8 -*-
# 新增诊断能力测试：PSI/KS/Lift/评分卡刻度、重采样、sample_weight、概率校准、漂移监控
# 重点：所有 pipe 均通过 auto_test 一致性检查
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
import pandas as pd
from easymlops import TablePipeLine
from easymlops.table.preprocessing import FixInput, FillNa
from easymlops.table.encoding import LabelEncoding
from easymlops.table.classification import LGBMClassification
from easymlops.table.regression import LGBMRegression
from easymlops.table.calibration import IsotonicCalibrationPipe
from easymlops.table.drift import DriftMonitorPipe
from easymlops.table.utils import calc_psi, calc_ks, calc_lift, prob2score, resample

np.random.seed(42)
n = 1000
PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"[PASS] {name} {detail}")
    else:
        FAIL.append(name)
        print(f"[FAIL] {name} {detail}")


# ============ 1. 指标函数 ============
print("=== 1. metric utils ===")
rng = np.random.RandomState(42)
base = rng.normal(0, 1, 5000)
psi_same = calc_psi(base, rng.normal(0, 1, 5000))
psi_shift = calc_psi(base, rng.normal(2.0, 1, 5000))
check("psi_same_dist_small", psi_same < 0.1, f"psi={psi_same:.4f}")
check("psi_shifted_dist_large", psi_shift > 0.5, f"psi={psi_shift:.4f}")

y_true = (rng.rand(1000) < (1 / (1 + np.exp(-rng.normal(0, 1, 1000) * 2)))).astype(int)
y_score = y_true * 0.7 + rng.rand(1000) * 0.3
ks = calc_ks(y_true, y_score)
check("ks_positive", ks["ks"] > 0.2, f"ks={ks['ks']:.3f}, thr={ks['ks_threshold']:.3f}")

lift = calc_lift(y_true, y_score, bins=5)
check("lift_first_bin_highest", lift["lift"].iloc[0] >= lift["lift"].iloc[-1],
      f"first={lift['lift'].iloc[0]:.2f}, last={lift['lift'].iloc[-1]:.2f}")
check("lift_cum_gain_last_is_1", abs(lift["cum_gain"].iloc[-1] - 1.0) < 1e-9)

scores = prob2score(pd.Series([0.01, 0.05, 0.2, 0.5]))
check("scorecard_monotonic_decreasing", np.all(np.diff(scores) < 0),
      f"scores={[round(s, 1) for s in scores]}")

# ============ 2. 重采样 ============
print("=== 2. resample ===")
x_imb = pd.DataFrame({"a": rng.rand(500), "b": rng.rand(500)})
y_imb = pd.Series([1] * 50 + [0] * 450)
x_over, y_over = resample(x_imb, y_imb, method="oversample")
x_under, y_under = resample(x_imb, y_imb, method="undersample")
check("oversample_balanced", y_over.value_counts().nunique() == 1 and len(y_over) == 900,
      f"counts={y_over.value_counts().to_dict()}")
check("undersample_balanced", y_under.value_counts().nunique() == 1 and len(y_under) == 100,
      f"counts={y_under.value_counts().to_dict()}")
try:
    x_rs, y_rs = resample(x_imb, y_imb, method="smote")
    check("smote_balanced", y_rs.value_counts().nunique() == 1, f"counts={y_rs.value_counts().to_dict()}")
except ImportError as e:
    print(f"[SKIP] smote_balanced ({e})")

# ============ 3. sample_weight ============
print("=== 3. sample_weight ===")
x = pd.DataFrame({"a": rng.rand(n), "b": rng.rand(n), "gender": rng.choice(["M", "F"], n)})
y = ((x["a"] + x["b"]) > 1.0).astype(int)
# 字符串列需先编码（模型 pipe 要求纯数值输入）
from easymlops.table.encoding import LabelEncoding as _LE
x_enc = _LE(cols=["gender"]).fit(x).transform(x)
w = np.where(y.values == 1, 2.0, 1.0)
m = LGBMClassification(y=y, use_faster_predictor=False, weight=w,
                       native_fit_params={"num_boost_round": 10})
m.fit(x_enc)
pred = m.transform(x)
check("lgbm_cls_weight_fit", pred.shape[0] == n)

mr = LGBMRegression(y=pd.Series(rng.rand(n)), use_faster_predictor=False,
                    weight=np.ones(n), native_fit_params={"num_boost_round": 10})
mr.fit(x)
check("lgbm_reg_weight_fit", mr.transform(x).shape[0] == n)

# ============ 4. 校准 pipe + 漂移监控 pipe（完整建模 pipeline）============
print("=== 4. calibration + drift in pipeline ===")
x_train, x_val, x_test = x[:600], x[600:800], x[800:]
y_train, y_val, y_test = y[:600], y[600:800], y[800:]

table = TablePipeLine()
table.pipe(FixInput(reduce_mem_usage=False)) \
     .pipe(FillNa(cols=["a", "b"], fill_mode="mean")) \
     .pipe(LabelEncoding(cols=["gender"])) \
     .pipe(DriftMonitorPipe(psi_threshold=0.25)) \
     .pipe(LGBMClassification(y=y_train, use_faster_predictor=False,
                              native_fit_params={"num_boost_round": 10}))
table.fit(x_train)

# 校准 pipe 在验证集预测上单独 fit 后 append（推荐用法，无偏）
calib = IsotonicCalibrationPipe(y=y_val, cols=[1])
calib.fit(table.transform(x_val))
table.pipe(calib)

calibrated = table.transform(x_test)
pos_col = [c for c in calibrated.columns if str(c) == "1"][0]
check("calibrated_in_range", calibrated[pos_col].between(0, 1).all(),
      f"min={calibrated[pos_col].min():.3f}, max={calibrated[pos_col].max():.3f}")
single_out = table.transform_single(x_test.iloc[0].to_dict())
check("calibrated_single_consistent",
      abs(single_out[pos_col] - calibrated[pos_col].iloc[0]) < 1e-5,
      f"batch={calibrated[pos_col].iloc[0]:.6f}, single={single_out[pos_col]:.6f}")

monitor = [m for m in table.models if isinstance(m, DriftMonitorPipe)][0]
check("drift_report_on_no_shift", len(monitor.last_psi_report_) == 3
      and all(p < 0.5 for p in monitor.last_psi_report_.values()),
      f"report={ {k: round(v, 3) for k, v in monitor.last_psi_report_.items()} }")

# 人造漂移数据
x_shift = x_test.copy()
x_shift["a"] = x_shift["a"] + 3.0
table.transform(x_shift)
check("drift_detected_on_shift", monitor.last_drift_cols_ == ["a"],
      f"drift_cols={monitor.last_drift_cols_}")

# save/load 一致性
table.save("d:/datas/share/traes/easymlops-trae/tests/diag_model.pkl")
online = TablePipeLine()
online.load("d:/datas/share/traes/easymlops-trae/tests/diag_model.pkl")
reload_out = online.transform(x_test)
check("save_load_consistent",
      np.allclose(reload_out[pos_col].values, calibrated[pos_col].values))

# ============ 5. auto_test（核心验收）============
print("=== 5. auto_test ===")
print("--- pipeline 1: preprocessing + drift monitor ---")
p1 = TablePipeLine()
p1.pipe(FixInput(reduce_mem_usage=False)) \
  .pipe(FillNa(cols=["a", "b"], fill_mode="mean")) \
  .pipe(DriftMonitorPipe())
p1.fit(x_train)
try:
    p1.auto_test(x_train, sample=20)
    check("auto_test_drift_pipe", True)
except Exception as e:
    check("auto_test_drift_pipe", False, str(e)[:200])

print("--- pipeline 2: full modeling + calibration ---")
try:
    table.auto_test(x_train, sample=20)
    check("auto_test_full_pipeline", True)
except Exception as e:
    check("auto_test_full_pipeline", False, str(e)[:200])

# ============ 汇总 ============
print("\n================ SUMMARY ================")
print(f"PASS: {len(PASS)}, FAIL: {len(FAIL)}")
if len(FAIL) > 0:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL TESTS PASSED")
