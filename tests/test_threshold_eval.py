# -*- coding: utf-8 -*-
import sys
sys.path.insert(0, "d:/datas/share/traes/easymlops-trae")

import numpy as np
from easymlops.table.utils import calc_precision_recall_at_thresholds, calc_precision_recall_at_quantiles, plot_pr_curve
from easymlops.table.utils import calc_roc_at_thresholds, plot_roc_curve

np.random.seed(42)

y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.7, 0.3, 0.2, 0.85, 0.15, 0.6, 0.4, 0.95, 0.75, 0.25, 0.35, 0.55])

print("=== calc_precision_recall_at_thresholds Test ===")
print(f"y_true: {y_true}")
print(f"y_pred: {y_pred}")
print()

result = calc_precision_recall_at_thresholds(y_true, y_pred, bins=5)
print("Result (bins=5):")
print(result.to_string(index=False))
print()

print("=== calc_precision_recall_at_quantiles Test ===")
result2 = calc_precision_recall_at_quantiles(y_true, y_pred, quantiles=5)
print("Result (quantiles=5):")
print(result2.to_string(index=False))
print()

print("=== plot_pr_curve Test ===")
y_preds_dict = {
    "Model_A": np.random.rand(15),
    "Model_B": np.random.rand(15),
    "Model_C": np.random.rand(15)
}
best_scores = plot_pr_curve(y_true, y_preds_dict, bins=5, save_path="d:/datas/share/traes/easymlops-trae/tests/pr_curve.png")
print("Best scores:")
for model_name, scores in best_scores.items():
    print(f"  {model_name}: F1={scores['best_f1']:.4f}, threshold={scores['best_threshold']:.4f}")
print()

print("=== calc_roc_at_thresholds Test ===")
result_roc = calc_roc_at_thresholds(y_true, y_pred, bins=5)
print("Result (bins=5):")
print(result_roc)
print()

print("=== plot_roc_curve Test ===")
y_preds_dict_roc = {
    "Model_A": np.random.rand(15),
    "Model_B": np.random.rand(15),
    "Model_C": np.random.rand(15)
}
aucs = plot_roc_curve(y_true, y_preds_dict_roc, bins=5, save_path="d:/datas/share/traes/easymlops-trae/tests/roc_curve.png")
print("AUCs:")
for model_name, auc in aucs.items():
    print(f"  {model_name}: AUC={auc:.4f}")
print()

print("=== Test PASSED ===")
