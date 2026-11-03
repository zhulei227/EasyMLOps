# -*- coding: utf-8 -*-
"""第三波技术扫描: 波1/波2 未覆盖的候选 pipe 技术族(词边界正则)。
用法: python scripts/_tech_scan_wave3.py
"""
import re
from collections import Counter
from pathlib import Path

PAGES = Path(__file__).resolve().parent.parent / "data" / "pages"

TECHS = {
    "fft/spectral features": r"\bfft\b|fourier transform|spectrogram|spectral feature",
    "peak/turning point detection": r"peak detection|turning point|local maxima|local minima|find_peaks",
    "zero crossing / wavelet": r"zero crossing|wavelet",
    "knn distance features": r"\bknn\b.{0,25}(feature|distance)|k nearest neighb.{0,25}feature|distance to nearest",
    "umap/tsne features": r"\bumap\b|\btsne\b|t-sne",
    "rolling mode/mode stat": r"rolling mode|mode.{0,10}rolling|most common value",
    "date/calendar features": r"holiday feature|calendar feature|day of week|week of year|is_weekend|day of month",
    "binning/qcut features": r"\bqcut\b|quantile bucket|binning feature|age binning|bucketize",
    "pairwise arithmetic feats": r"multiply.{0,25}divide.{0,25}feature|subtract.{0,25}feature|ratio between|difference between.{0,20}column",
    "label smoothing": r"label smoothing",
    "mixup/SMOTE aug": r"\bmixup\b|\bsMOTE\b|cutmix|data augmentation",
    "custom objective/loss": r"custom objective|custom loss",
    "huber/quantile loss": r"huber loss|quantile loss",
    "snapshot/EMA ensemble": r"snapshot ensemble|\bema\b|exponential moving average.{0,15}weight",
    "feature subsample bagging": r"feature subsampl|random subspace|column subsampl",
    "voting ensemble": r"voting classifier|voting ensemble|soft voting|hard voting",
    "blend weight optimization": r"optimi[sz]e.{0,20}blend|blend weight|weight.{0,10}search.{0,15}ensemble|scipy.{0,15}optimi[sz]e.{0,15}weight",
    "hill climbing ensemble": r"hill climbing",
    "target encode continuous": r"target encod.{0,25}(continuous|numeric)|bin.{0,10}target encod",
    "rank/gauss normalize": r"rank gauss|rank data|gaussianize|quantile transform",
    "adversarial validation": r"adversarial validation",
    "leak check tools": r"leak.{0,20}(check|detect)|time travel|future leak",
    "string fuzzy match feats": r"fuzzy|levenshtein|edit distance|difflib",
    "n-gram char features": r"char.?ngram|character ngram|ngram range",
    "keyword/pattern match feats": r"keyword.{0,15}(match|feature|extract)|regex.{0,15}feature|pattern match",
    "language detect": r"language detection|langdetect|detect language",
    "graph/pagerank feats": r"pagerank|graph feature|node feature|networkx",
    "per-group normalize": r"group.{0,15}normal|within group.{0,15}(standard|mean)|per group.{0,15}scal",
    "count of rare values": r"rare value|rare label|grouping rare|other category",
    "ordinal encode high card": r"ordinal encod|label encod.{0,20}(high|category)",
    "embed cluster": r"embedding.{0,25}cluster|cluster.{0,25}embedding",
    "catboost native cat": r"catboost.{0,25}(categorical|cat_feature)|one_hot_max_size",
    "interaction constraints": r"interaction constraint",
    "pca reconstruction outlier": r"pca.{0,20}outlier|reconstruction error",
    "dup row removal": r"duplicated\(|drop duplicate",
    "woe/iv numeric binning": r"woe.{0,20}bin|iv.{0,20}bin",
    "first/last value feats": r"first value|last value|first occurrence|earliest.{0,15}latest",
    "shift diff pairs": r"shift.{0,10}diff|diff between consecutive",
    "exponential smoothing feats": r"exponential smoothing|ewm\(|holt.?winters",
    "aggregate on date parts": r"aggregate.{0,20}(month|weekday|hour)|groupby.{0,15}(month|hour|weekday)",
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
print("{:<38}{:>9}{:>7}".format("technique(wave3)", "mentions", "docs"))
for k, v in counts.most_common():
    print("{:<38}{:>9}{:>7}".format(k, v, docs_with[k]))
for k in set(TECHS) - set(counts):
    print("{:<38}{:>9}{:>7}".format(k, 0, 0))
