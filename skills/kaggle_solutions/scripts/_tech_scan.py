# -*- coding: utf-8 -*-
"""扫描 kaggle_solutions 离线语料, 统计候选 pipe 技术的提及频率。

用法: 在 TECHS 中增删关键词后运行 `python scripts/_tech_scan.py`。
"""
import re
from collections import Counter
from pathlib import Path

PAGES = Path(__file__).resolve().parent.parent / "data" / "pages"

# 候选技术 -> 正则变体
TECHS = {
    "count/frequency encoding": r"count encod|frequency encod|count of .*categor",
    "target encoding (CV/smooth)": r"target encod|mean encod|likelihood encod",
    "catboost/james-stein/m-estimate enc": r"catboost encod|james.stein|m.estimate|leave.one.out",
    "groupby aggregation feats": r"group.?by (features|statistics|aggregat)|aggregat.*group|groupby",
    "categorical combination/cross": r"categor.* combination|combin.*categor|interaction between categor|cross.*categor",
    "gauss rank / quantile transform": r"rank.gauss|gauss.*rank|quantile transform|quantile scaler",
    "polynomial features": r"polynomial feature|poly feature",
    "log/sqrt/power transform": r"log.?transform|log1p|sqrt transform|power transform|yeo.johnson|box.cox",
    "nan/missing indicator": r"missing.*indicator|nan.*indicator|is.?missing|mask.*nan",
    "outlier handling/clip": r"outlier|winsoriz|clip.*value",
    "text TF-IDF/SVD feats": r"tf.?idf|svd.*feature|truncated svd|word2vec|text.*embedding",
    "entity/nn embedding": r"entity embedding|categorical embedding|embedding layer|nn embedding",
    "pseudo-labeling": r"pseudo.?label",
    "adversarial validation": r"adversarial validation|adversarial valid",
    "domain/shift adaptation": r"domain adaptation|distribution shift|covariate shift",
    "corr-based feature drop": r"dropp.*correlat|correlat.*dropp|remove.*correlat|highly correlat",
    "permutation importance": r"permutation importance",
    "null importance/boruta": r"null importance|boruta",
    "RFE/stepwise selection": r"recursive feature elimination|rfe",
    "OOF stacking/blending": r"out.of.fold|oof|stacking|blend",
    "seed averaging/bagging": r"seed averaging|seed.*average|different seeds|bagging",
    "hyperopt/optuna/bayesian": r"optuna|hyperopt|bayesian optim|hyperparameter optim",
    "spline features": r"spline",
    "time-since / delta features": r"time since|days since|delta.*feature|time diff",
    "holiday/calendar features": r"holiday|calendar feature",
    "rolling window feats (tabular)": r"rolling (mean|window|average)",
    "pca/ica decomposition feats": r"pca|ica|principal component",
    "k-means cluster feats": r"k.?means|cluster.*feature|feature.*cluster",
    "prediction clipping/postproc": r"clip.*prediction|post.?process|postprocessing",
    "sample/fold weighting": r"sample weight|sample weight|weight.*sample",
    "ordinal encoding for trees": r"ordinal encod",
    "hashing encoder": r"hashing encoder|feature hashing",
    "difference/ratio features": r"ratio.*feature|difference between|dividing.*feature|feature.*ratio",
    "target-guided ordering": r"target.?guided|target order",
    "double validation/early stop": r"early stopping|double validation",
    "neural net on tabular": r"neural network|tabnet|deep learning model|fcn|mlp",
}

counts = Counter()
docs_with = Counter()
files = list(PAGES.rglob("*.txt"))
compiled = {k: re.compile(v, re.I) for k, v in TECHS.items()}
for f in files:
    try:
        text = f.read_text(encoding="utf-8", errors="ignore").lower()
    except Exception:
        continue
    for k, rx in compiled.items():
        n = len(rx.findall(text))
        if n:
            counts[k] += n
            docs_with[k] += 1

print(f"corpus: {len(files)} solution pages\n")
print(f"{'technique':<38}{'mentions':>9}{'docs':>7}")
for k, v in counts.most_common():
    print(f"{k:<38}{v:>9}{docs_with[k]:>7}")
