# -*- coding: utf-8 -*-
"""第二波技术扫描: 上一轮未覆盖的候选 pipe 技术族(词边界正则, 避免子串误匹配)"""
import re
from collections import Counter
from pathlib import Path

PAGES = Path(__file__).resolve().parent.parent / "data" / "pages"

TECHS = {
    "threshold/metric-aware postproc": r"threshold optimi[sz]|metric.aware|optimi[sz]e .{0,20}threshold|kappa.{0,20}round|optimi[sz]e .{0,15}rounding",
    "prediction interval/conformal": r"prediction interval|conformal|uncertainty estimat|quantile.{0,10}interval",
    "quantile regression": r"quantile regress",
    "monotonic constraints": r"monotonic",
    "shap-based selection": r"\bshap\b",
    "knowledge distillation": r"distill",
    "string statistical features": r"string length|word count|character count|text length|num.?words|char.?count",
    "anomaly detection features": r"isolation forest|anomaly (score|detection|feature)|outlier detection",
    "gaussian mixture features": r"gaussian mixture|\bgmm\b",
    "matrix factorization feats": r"matrix factori[sz]ation|\bals\b|alternating least",
    "group frequency encoding": r"frequency within|group.{0,15}frequency|count within",
    "leave-one-out encoding": r"leave.one.out",
    "ordinal target handling": r"ordinal target|ordinal regress|\bkappa\b",
    "time decay / recency weighting": r"time decay|exponential decay|recency",
    "missing pattern features": r"missing pattern|nan pattern|missingness|pattern of missing",
    "duplicate detection": r"duplicate row|near.duplicate|duplicate detect|drop duplicate",
    "email/url/domain extraction": r"\bemail\b|\burl\b|\bdomain name",
    "cyclic/fourier time encoding": r"cyclic encod|sin.{0,10}cos.{0,10}encod|fourier.{0,15}time|cyclical",
    "ordered target statistics": r"ordered target|ordered ts|target statistic",
    "bayesian target encoding": r"bayesian (target )?encod",
    "group-aware CV scheme": r"group ?kfold|group.{0,5}cv|timeseries split|time series split|purged",
    "pretrained text embeddings": r"pretrained.{0,20}embed|\bbert\b|sentence.?embed|use\b.{0,10}embed|word2vec|fasttext|glove",
    "feature hashing": r"feature hashing|hashing trick|hashingvectorizer",
    "weighted/focal loss": r"focal loss|weighted logloss|class weight",
    "ratio features within group": r"ratio to|relative to (the )?group|share of|proportion of",
    "svd on one-hot": r"svd.{0,20}(one.?hot|categor)|(one.?hot|categor).{0,20}svd",
    "NN embedding fine-tune": r"embedding.{0,15}(finetune|fine.tune|train)|learned embedding",
    "power averaging/blend weights": r"weighted blend|weight.{0,10}average.{0,10}model|rank averag",
    "model stacking with oof meta": r"meta.?model|meta.?learner|level 2|second.level",
    "semi-supervised consistency": r"semi.supervised|consistency regular",
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

print("corpus: {} solution pages\n".format(len(files)))
print("{:<38}{:>9}{:>7}".format("technique(wave2)", "mentions", "docs"))
for k, v in counts.most_common():
    print("{:<38}{:>9}{:>7}".format(k, v, docs_with[k]))
missing = set(TECHS) - set(counts)
for k in missing:
    print("{:<38}{:>9}{:>7}".format(k, 0, 0))
