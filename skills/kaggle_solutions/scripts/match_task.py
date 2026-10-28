#!/usr/bin/env python3
"""Match the current business/task against past Kaggle competitions and derive
optimization recommendations from offline top solutions.

Pipeline:
  1. Profile the task   -- explicit --desc/--metric, or auto-scan a project dir
                         (competition URLs, metric keywords, domain terms).
  2. Match competitions -- score the 721-competition archive by task similarity.
  3. Mine solutions     -- read offline winner pages (data/pages/) and extract
                         techniques via a curated lexicon.
  4. Report             -- ranked references + technique table with quotes +
                         actionable checklist. --out saves a markdown report.

Usage:
  python scripts/match_task.py "detect fraud in card transactions" --metric auc
  python scripts/match_task.py --dir path/to/project --top 8 --out report.md
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_solutions as ks  # noqa: E402

try:
    import embedding_index as ei  # numpy-based TF-IDF semantic search
except ImportError:  # numpy missing -> keyword matching only
    ei = None

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCAN_EXTS = {".py", ".ipynb", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".cfg", ".csv"}
SKIP_DIRS = {".git", ".codebuddy", "node_modules", "__pycache__", ".venv", "venv",
             "dist", "build", ".playwright-cli", "logs"}
MAX_FILES = 500
MAX_BYTES = 512 * 1024

METRIC_SYNONYMS = {
    "auc": {"auc", "roc", "auroc", "曲线下面积", "roc曲线", "gini"},
    "logloss": {"logloss", "log loss", "cross entropy", "对数损失", "交叉熵"},
    "rmse": {"rmse", "mse", "root mean square", "均方根误差", "均方误差"},
    "mae": {"mae", "mean absolute", "平均绝对误差"},
    "f1": {"f1", "micro f1", "macro f1", "f1值", "f1分数"},
    "kappa": {"kappa", "quadratic weighted", "kappa系数"},
    "dice": {"dice", "iou", "jaccard", "dice coefficient", "骰子系数"},
    "map": {"map@", "mean average precision", "平均精度均值"},
    "spearman": {"spearman", "斯皮尔曼"},
    "accuracy": {"accuracy", "准确率"},
}

# (category, phrase) lexicon for mining winner pages
LEXICON = [
    ("validation", "cross validation"), ("validation", "k-fold"), ("validation", "5-fold"),
    ("validation", "stratified"), ("validation", "time series split"), ("validation", "group k-fold"),
    ("validation", "adversarial validation"), ("validation", "holdout"), ("validation", "leak"),
    ("gbdt", "lightgbm"), ("gbdt", "lgbm"), ("gbdt", "xgboost"), ("gbdt", "catboost"),
    ("gbdt", "gradient boosting"), ("gbdt", "random forest"),
    ("deep-learning", "transformer"), ("deep-learning", "bert"), ("deep-learning", "deberta"),
    ("deep-learning", "roberta"), ("deep-learning", "swin"), ("deep-learning", "convnext"),
    ("deep-learning", "efficientnet"), ("deep-learning", "resnet"), ("deep-learning", "unet"),
    ("deep-learning", "lstm"), ("deep-learning", "gnn"), ("deep-learning", "vision transformer"),
    ("deep-learning", "vit"), ("deep-learning", "timm"),
    ("ensemble", "ensemble"), ("ensemble", "blend"), ("ensemble", "stacking"),
    ("ensemble", "weighted average"), ("ensemble", "seed"), ("ensemble", "hill climbing"),
    ("ensemble", "caruana"), ("ensemble", "model averaging"),
    ("features", "feature engineering"), ("features", "target encoding"),
    ("features", "frequency encoding"), ("features", "aggregation"), ("features", "statistics"),
    ("features", "embedding"), ("features", "pca"), ("features", "polynomial"),
    ("training", "pseudo label"), ("training", "pseudo-label"), ("training", "distillation"),
    ("training", "tta"), ("training", "test time augmentation"), ("training", "mixup"),
    ("training", "label smoothing"), ("training", "focal loss"), ("training", "scheduler"),
    ("training", "cosine"), ("training", "warmup"), ("training", "mixed precision"),
    ("training", "fp16"), ("training", "swa"), ("training", "gradient accumulation"),
    ("training", "fine-tune"), ("training", "pretrained"), ("training", "checkpoint"),
    ("data", "augmentation"), ("data", "external data"), ("data", "clipping"),
    ("data", "normalization"), ("data", "missing value"), ("data", "imputation"),
    ("data", "outlier"), ("data", "dedup"),
    ("post-process", "threshold"), ("post-process", "calibration"), ("post-process", "isotonic"),
    ("post-process", "postprocess"), ("post-process", "post-processing"), ("post-process", "rounding"),
    ("llm", "llm"), ("llm", "retrieval"), ("llm", "rag"), ("llm", "prompt"), ("llm", "agent"),
]

CHECKLIST = {
    "validation": "复现优胜者的验证方案（CV 划分方式），确保本地验证与线上 LB 一致",
    "gbdt": "表格类任务优先 LightGBM/XGBoost/CatBoost 基线，再考虑神经网络",
    "deep-learning": "采用优胜者的骨干网络与预训练权重（timm/HF），而非从零训练",
    "ensemble": "多种子/多模型融合（blend、stacking、hill climbing）通常稳定涨分",
    "features": "迁移优胜者的特征工程思路（目标编码、聚合统计等）",
    "training": "借鉴训练技巧：伪标签、TTA、mixup、label smoothing、学习率调度等",
    "data": "检查外部数据、增强策略与清洗手段（优胜者常用外部数据源）",
    "post-process": "对齐优胜者的后处理（阈值、校准、取整）——常被忽视的高性价比环节",
    "llm": "LLM/Agent 类任务参考优胜者的检索、prompt 与微调策略",
}


def normalize_metric(text):
    low = text.lower()
    for canon, keys in METRIC_SYNONYMS.items():
        if any(k in low for k in keys):
            return canon
    return None


# --------------------------------------------------------------- task profile

def profile_from_desc(desc, metric):
    translated = ks.translate_cn(desc)
    return {
        "desc": desc,
        "terms": _bag(translated),
        "metric": normalize_metric(metric or translated),
        "slugs": set(re.findall(r"kaggle\.com/c/([a-z0-9\-_]+)", desc.lower())),
        "sources": ["--desc" + (" +中文映射" if translated != desc else "")],
    }


def profile_from_dir(root):
    root = Path(root)
    text_all, slugs = [], set()
    files = [p for p in root.rglob("*")
             if p.is_file() and p.suffix.lower() in SCAN_EXTS
             and not any(part in SKIP_DIRS for part in p.parts)
             and p.stat().st_size < MAX_BYTES]
    for p in files[:MAX_FILES]:
        try:
            if p.suffix.lower() == ".csv":  # header only
                with open(p, encoding="utf-8", errors="replace") as f:
                    text_all.append(f.read(2048))
            else:
                text_all.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    blob = "\n".join(text_all)
    translated = ks.translate_cn(blob[:50000])
    slugs = set(re.findall(r"kaggle\.com/c/([a-z0-9\-_]+)", blob.lower()))
    return {
        "desc": blob[:20000],
        "terms": _bag(translated),
        "metric": normalize_metric(blob[:20000]),
        "slugs": slugs,
        "sources": ["dir scan: %s (%d files)" % (root, min(len(files), MAX_FILES))],
    }


# ----------------------------------------------------------------- matching

def _bag(text):
    """Light-stemmed term set (fraud/frauds, transaction/transactions match)."""
    out = set()
    for t in ks._terms(text):
        out.add(t)
        if len(t) > 3:
            if t.endswith("ies"):
                out.add(t[:-3] + "y")
            if t.endswith("s") and not t.endswith("ss"):
                out.add(t[:-1])
            if t.endswith("ing") and len(t) > 5:
                out.add(t[:-3])
            if t.endswith("ed") and len(t) > 4:
                out.add(t[:-2])
    return out


def match_competitions(profile, comps, top, sem_boost=None):
    """sem_boost: {number: 0..1 semantic similarity} from the TF-IDF index."""
    scored = []
    terms = {t for t in _bag(" ".join(profile["terms"])) if len(t) > 2}
    pslugs = profile["slugs"]
    for c in comps:
        slug = ks.SOL_URL.search(c["link"])
        score = 0.0
        if slug and slug.group(1).lower() in pslugs:
            score += 100.0  # direct hit: the task IS this competition
        bag = _bag("%s %s %s %s" % (c["title"], c["desc"], c["metric"], c["kind"]))
        if terms:
            overlap = terms & bag
            if overlap:
                score += 60.0 * len(overlap) / len(terms)
        if profile["metric"] and profile["metric"] in (normalize_metric(c["metric"] or "") or ""):
            score += 15.0
        if c["curated"]:
            score += 3.0
        n_off = len(list((ks.PAGES_DIR / str(c["number"])).glob("*.txt"))) \
            if (ks.PAGES_DIR / str(c["number"])).exists() else 0
        if n_off:
            score += 5.0 + min(n_off, 3)
        if sem_boost:
            score += 20.0 * sem_boost.get(c["number"], 0.0)
        if score > 0:
            scored.append((score, n_off, c))
    scored.sort(key=lambda t: (-t[0], -t[1]))
    return scored[:top]


# ------------------------------------------------------------------- mining

def offline_pages(number):
    folder = ks.PAGES_DIR / str(number)
    if not folder.is_dir():
        return []
    pages = []
    for p in sorted(folder.glob("*.txt"), key=lambda x: (len(x.name), x.name)):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        body = text.split("\n\n", 1)[-1]
        if len(body) > 300 and not body.startswith("FETCH_FAILED"):
            pages.append((p.name, text.splitlines()[0] if text else "", body))
    return pages[:3]  # keep the best-ranked pages per competition


def mine_techniques(matches, deep):
    """Count lexicon phrases across offline winner pages of top matches."""
    stats = {}  # phrase -> {"cat", "count", "comps", "quote", "src"}
    for score, n_off, c in matches[:deep]:
        for fname, url, body in offline_pages(c["number"]):
            low = body.lower()
            lines = body.splitlines()
            for cat, phrase in LEXICON:
                if re.search(r"\b%s\b" % re.escape(phrase), low):
                    e = stats.setdefault(phrase, {"cat": cat, "count": 0,
                                                  "comps": set(), "quote": "", "src": ""})
                    e["count"] += 1
                    e["comps"].add(c["number"])
                    if not e["quote"]:
                        for line in lines:
                            if phrase in line.lower():
                                e["quote"] = line.strip()[:180]
                                e["src"] = "#%d %s" % (c["number"], c["title"])
                                break
    return stats


# ------------------------------------------------------------------ report

def build_report(profile, matches, stats, deep, top, sem_hits=None):
    L = []
    L.append("# Kaggle 方案匹配与优化建议报告\n")
    L.append("## 1. 当前业务画像\n")
    if profile["slugs"]:
        L.append("- 检测到 Kaggle 竞赛引用: %s" % ", ".join(sorted(profile["slugs"])))
    if profile["metric"]:
        L.append("- 推断评估指标类型: **%s**" % profile["metric"])
    L.append("- 关键词条目数: %d（来源: %s）\n" % (len(profile["terms"]), "; ".join(profile["sources"])))

    L.append("## 2. 匹配的历史竞赛 TOP %d\n" % top)
    L.append("| # | 竞赛 | 类型/年份 | 指标 | 匹配分 | 离线页 |")
    L.append("|---|---|---|---|---|---|")
    for score, n_off, c in matches:
        L.append("| #%d | [%s](%s) | %s %s | %s | %.0f | %d |"
                 % (c["number"], c["title"], c["link"], c["kind"], c["year"],
                    c["metric"] or "-", score, n_off))
    L.append("")

    n = 3
    if sem_hits:
        L.append("## 3. 语义检索的相关优胜方案（TF-IDF 向量匹配）\n")
        comps_by_number = {c["number"]: c for c in ks.load_competitions()}
        for score, m in sem_hits:
            c = comps_by_number.get(m["number"])
            title = c["title"] if c else m.get("title") or "#%d" % m["number"]
            read_hint = "`read %d %d`" % (m["number"], m["rank"]) if m["rank"] else ""
            L.append("- %.3f | #%d [%s] rank %d %s | %s"
                     % (score, m["number"], title, m["rank"], read_hint, m["url"]))
        L.append("")

    L.append("## %d. 优胜方案技术挖掘（TOP %d 竞赛的前 3 个方案页）\n" % (n + 1, deep))
    if not stats:
        L.append("（离线正文不足，先运行 download_solutions.py 补齐对应竞赛页面）\n")
    else:
        L.append("| 技术 | 类别 | 出现次数 | 涉及竞赛数 |")
        L.append("|---|---|---|---|")
        for phrase, e in sorted(stats.items(), key=lambda kv: (-kv[1]["count"], kv[0])):
            L.append("| %s | %s | %d | %d |" % (phrase, e["cat"], e["count"], len(e["comps"])))
        L.append("")
        L.append("### 原文依据（节选）\n")
        for phrase, e in sorted(stats.items(), key=lambda kv: -kv[1]["count"])[:12]:
            if e["quote"]:
                L.append("- **%s** (%s): “%s” —— %s" % (phrase, e["cat"], e["quote"], e["src"]))
        L.append("")

    L.append("## %d. 优化建议清单\n" % (n + 2))
    cats = sorted({e["cat"] for e in stats.values()},
                  key=lambda c: -max(e["count"] for e in stats.values() if e["cat"] == c))
    if not cats:
        cats = ["validation", "gbdt", "ensemble", "post-process"]
    for i, cat in enumerate(cats, 1):
        top_phrases = [p for p, e in sorted(stats.items(), key=lambda kv: -kv[1]["count"])
                       if e["cat"] == cat][:5]
        L.append("%d. **%s**: %s" % (i, cat, CHECKLIST.get(cat, "")))
        if top_phrases:
            L.append("   - 优胜者高频用法: %s" % ", ".join(top_phrases))
    L.append("")
    missing = [(c["number"], c["title"]) for _, _, c in matches
               if not (ks.PAGES_DIR / str(c["number"])).exists()]
    if missing:
        L.append("## %d. 补充离线方案\n" % (n + 3))
        for n, t in missing[:5]:
            L.append("- #%d %s 尚无离线正文，运行: `python scripts/download_solutions.py --max-rank 3`"
                     "（或定向补充后用 `read %d 1`）" % (n, t, n))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("desc", nargs="*", help="业务描述（省略则自动扫描 --dir）")
    ap.add_argument("--dir", default=".", help="当前项目目录（自动画像）")
    ap.add_argument("--metric", help="评估指标提示（auc/rmse/f1/...）")
    ap.add_argument("--top", type=int, default=8, help="匹配竞赛数量")
    ap.add_argument("--deep", type=int, default=4, help="深入挖掘方案正文的竞赛数")
    ap.add_argument("--out", help="报告输出路径（markdown）")
    ap.add_argument("--rebuild", action="store_true",
                    help="强制重建 TF-IDF 语义索引后继续")
    ap.add_argument("--no-sem", action="store_true",
                    help="禁用语义检索（仅关键词匹配）")
    args = ap.parse_args()

    comps = ks.load_competitions()
    if " ".join(args.desc).strip():
        profile = profile_from_desc(" ".join(args.desc), args.metric)
    else:
        profile = profile_from_dir(args.dir)

    sem_boost, sem_hits = None, None
    if ei is not None and not args.no_sem:
        try:
            if args.rebuild or not ei.INDEX_FILE.exists():
                ei.build(force=args.rebuild)
            sem = ei.search(profile["desc"][:4000], corpus="competitions", top_k=60)
            sem_boost = {m["number"]: s for s, m in sem}
            sem_hits = ei.search(profile["desc"][:4000], corpus="solutions", top_k=8)
        except SystemExit:
            pass  # index unavailable -> keyword matching only

    matches = match_competitions(profile, comps, args.top, sem_boost=sem_boost)
    if not matches:
        raise SystemExit("未能匹配到相似竞赛，请补充 --desc 描述或确认项目文件中包含任务信息。")
    stats = mine_techniques(matches, args.deep)
    report = build_report(profile, matches, stats, args.deep, args.top, sem_hits=sem_hits)

    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print("Report saved to %s\n" % args.out)
    print(report)


if __name__ == "__main__":
    main()
