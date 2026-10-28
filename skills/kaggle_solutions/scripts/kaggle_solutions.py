#!/usr/bin/env python3
"""Query the Kaggle Solutions archive (kaggle.farid.one / faridrashidi/kaggle-solutions).

Zero hard dependencies: stdlib only. PyYAML is used when available; otherwise the
site's competition-markup.json is parsed with regex as fallback.

Commands:
  search [terms...] [--kind K] [--year Y] [--status curated|missing|hot]
         [--metric SUBSTR] [--limit N] [--solutions] [--json]
  show <number|title-substring> [--json]
  similar "<problem description>" [--limit N] [--json]
  read <number> [rank] [--max-chars N]   # read offline solution text
  grep <terms...> [--limit N]            # full-text search offline pages
  stats [--json]
  refresh

Offline-first: metadata is read from data/competitions.{json,yml} inside the
skill directory (snapshot of kaggle.farid.one). Solution page texts downloaded
by download_solutions.py live under data/pages/<number>/<rank>.txt and are
surfaced by `show` (offline markers), `read` and `grep`. Only when no offline
data exists is the network used (temp cache 7 days; `refresh` re-downloads).
"""

import argparse
import json
import os
import re
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
DATA_DIR = SKILL_DIR / "data"
PAGES_DIR = DATA_DIR / "pages"
OFFLINE_JSON = DATA_DIR / "competitions.json"
OFFLINE_YML = DATA_DIR / "competitions.yml"

YML_URL = "https://raw.githubusercontent.com/faridrashidi/kaggle-solutions/main/data/competitions.yml"
MARKUP_URL = "https://kaggle.farid.one/competition-markup.json"
CACHE_DIR = os.path.join(tempfile.gettempdir(), "kaggle_solutions_cache")
CACHE_FILE = os.path.join(CACHE_DIR, "competitions.json")
CACHE_TTL = 7 * 86400

UA = {"User-Agent": "kaggle-solutions-skill/0.1 (+https://kaggle.farid.one/)"}

SOLUTION_KIND_LABEL = {"description": "write-up", "code": "code", "kernel": "notebook"}


def _parse_rank(value):
    """Extract leading digits from rank ("1", "21st", "all", None -> int)."""
    m = re.match(r"\s*(\d+)", str(value or ""))
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------- data loading

def _download(url):
    last = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:  # noqa: BLE001 - retried below
            last = e
            time.sleep(2 * (attempt + 1))
    raise last


def _load_yml(text):
    import yaml  # optional
    data = yaml.safe_load(text)["competitions"]
    out = []
    for c in data:
        out.append({
            "number": int(c.get("number") or 0),
            "title": (c.get("title") or "").strip(),
            "desc": (c.get("desc") or "").strip(),
            "kind": (c.get("kind") or "").strip(),
            "prize": (c.get("prize") or "").strip(),
            "team": (c.get("team") or "").strip(),
            "metric": (c.get("metric") or "").strip(),
            "link": (c.get("link") or "").strip(),
            "year": int(c.get("year") or 0),
            "hot": str(c.get("isHot")).lower() == "true",
            "curated": str(c.get("done")).lower() == "true",
            "solutions": [
                {
                    "rank": _parse_rank(s.get("rank")),
                    "link": (s.get("link") or "").strip(),
                    "kind": (s.get("kind") or "description").strip(),
                }
                for s in (c.get("solutions") or [])
            ],
        })
    return out


def _load_markup(text):
    """Fallback parser for kaggle.farid.one/competition-markup.json (HTML fragments)."""
    data = json.loads(text)
    out = []
    for key in sorted(data, key=lambda k: int(k)):
        html = data[key].get("card") or data[key].get("row") or ""
        if not html:
            continue

        def rx(pattern, default="", group=1, s=html):
            m = re.search(pattern, s)
            return m.group(group).strip() if m else default

        link = rx(r'class="competition-card__title"[^>]*href="([^"]+)"')
        title = rx(r'class="competition-card__title"[^>]*>([^<]+)<')
        if not title:
            link = rx(r'class="competition-main__title"[^>]*href="([^"]+)"')
            title = rx(r'class="competition-main__title"[^>]*>([^<]+)<')
        chips = re.findall(r'class="meta-chip">([^<]+)<', html)
        details = dict(
            (k.strip(), v.strip())
            for k, v in re.findall(r"<dt>([^<]+)</dt>\s*<dd>([^<]+)</dd>", html)
        )
        status = "Missing" if ">Missing<" in html else "Curated"
        solutions = [
            {
                "rank": int(re.match(r"(\d+)", label).group(1)) if re.match(r"(\d+)", label) else 0,
                "link": href,
                "kind": "description",
            }
            for href, label in re.findall(
                r'class="solution-badge[^"]*"\s+href="([^"]+)"[^>]*>.*?'
                r'class="solution-badge__label">([^<]+)<', html, re.S)
        ]
        out.append({
            "number": int(rx(r'data-id="(\d+)"', key)),
            "title": title,
            "desc": rx(r'class="competition-(?:card__|main__)description">([^<]+)<'),
            "kind": chips[0] if chips else "",
            "prize": details.get("Prize", ""),
            "team": details.get("Teams", "").replace(",", ""),
            "metric": details.get("Metric", ""),
            "link": link,
            "year": int(chips[1]) if len(chips) > 1 and chips[1].isdigit() else 0,
            "hot": False,
            "curated": status.lower() == "curated",
            "solutions": solutions,
        })
    return out


def load_competitions(force=False):
    """Return list of normalized competition dicts.

    Order: skill offline JSON -> skill offline YAML -> temp cache (7d TTL)
    -> network (yml, then markup fallback). Successful network fetches are
    written back to the offline snapshot so the skill keeps working offline.
    """
    if not force:
        if OFFLINE_JSON.exists():
            with open(OFFLINE_JSON, encoding="utf-8") as f:
                return json.load(f)
        if OFFLINE_YML.exists():
            try:
                data = _load_yml(OFFLINE_YML.read_text(encoding="utf-8"))
                _write_offline(data)
                return data
            except Exception:  # noqa: BLE001 - fall through to network
                pass
        if os.path.exists(CACHE_FILE) and \
                time.time() - os.path.getmtime(CACHE_FILE) < CACHE_TTL:
            with open(CACHE_FILE, encoding="utf-8") as f:
                return json.load(f)

    errors = []
    data = None
    try:
        data = _load_yml(_download(YML_URL))
    except Exception as e:  # noqa: BLE001 - report both attempts at the end
        errors.append("yml: %s" % e)
    if not data:
        try:
            data = _load_markup(_download(MARKUP_URL))
        except Exception as e:  # noqa: BLE001
            errors.append("markup: %s" % e)
    if not data:
        raise SystemExit("Failed to load data (offline snapshot unavailable too). "
                         + " | ".join(errors))

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    _write_offline(data)
    return data


def _write_offline(data):
    """Persist normalized data next to the offline YAML (best effort)."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(OFFLINE_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except OSError:
        pass


# ------------------------------------------------------------------ formatting

SOL_URL = re.compile(r"kaggle\.com/c/([^/]+)")


def _stored_urls(number):
    """Map of normalized offline page files for one competition: url -> path."""
    out = {}
    folder = PAGES_DIR / str(number)
    if not folder.is_dir():
        return out
    for p in sorted(folder.glob("*.txt")):
        try:
            first = p.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        except OSError:
            continue
        m = re.match(r"url:\s*(\S+)", first)
        if m:
            out[m.group(1).rstrip("/")] = p
    return out


def fmt_solution(s, offline=None):
    label = SOLUTION_KIND_LABEL.get(s["kind"], s["kind"])
    mark = ""
    if offline is not None and s["link"].rstrip("/") in offline:
        mark = "  **[offline]**"
    return "    %3d. [%s] %s%s" % (s["rank"], label, s["link"], mark)


def fmt_comp(c, with_solutions=False, index=None):
    status = "Curated" if c["curated"] else "Missing"
    head = "#%d [%s %s] %s (%s, %d links%s)" % (
        c["number"], c["kind"], c["year"], c["title"], status,
        len(c["solutions"]), ", HOT" if c["hot"] else "")
    if index is not None:
        head = "%2d. %s" % (index, head)
    lines = [head,
             "    %s" % c["desc"],
             "    Metric: %s | Prize: %s | Teams: %s" % (c["metric"], c["prize"], c["team"]),
             "    %s" % c["link"]]
    if with_solutions:
        offline = _stored_urls(c["number"])
        lines += [fmt_solution(s, offline)
                  for s in sorted(c["solutions"], key=lambda s: s["rank"])]
    return "\n".join(lines)


# ---------------------------------------------------------------------- search

def _terms(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# Chinese business/domain terms -> English equivalents used in the archive.
# Longest keys match first when scanning (dict order: longer first).
CN_EN_MAP = {
    "客户流失": "customer churn", "用户流失": "customer churn", "流失预测": "churn",
    "反欺诈": "fraud detection", "欺诈检测": "fraud detection", "欺诈": "fraud",
    "风控": "risk fraud detection", "信贷": "credit loan default",
    "违约": "loan default", "信用评分": "credit score",
    "销量预测": "sales forecasting demand", "需求预测": "demand forecasting",
    "销量": "sales", "库存": "inventory", "零售": "retail store",
    "时间序列": "time series forecasting", "时序预测": "time series forecasting",
    "时序": "time series", "预测": "prediction forecast",
    "推荐系统": "recommendation", "点击率": "click through rate ctr",
    "广告": "advertising ctr", "搜索排序": "search ranking",
    "异常检测": "anomaly detection", "垃圾邮件": "spam",
    "情感分析": "sentiment analysis", "文本分类": "text classification",
    "机器翻译": "machine translation", "问答": "question answering",
    "摘要": "summarization", "命名实体识别": "named entity recognition ner",
    "自然语言处理": "nlp text", "大模型": "llm", "智能体": "agent",
    "图像分类": "image classification", "目标检测": "object detection",
    "语义分割": "image segmentation", "实例分割": "image segmentation",
    "图像": "image", "人脸识别": "face recognition", "OCR": "ocr text",
    "语音识别": "speech recognition audio", "声纹": "audio",
    "医学影像": "medical imaging", "医疗": "medical", "癌症": "cancer",
    "肺癌": "lung cancer", "心电图": "ecg", "基因": "dna genomic",
    "蛋白质": "protein", "分子": "molecule", "药物": "drug molecular",
    "自动驾驶": "autonomous driving", "房价": "house price",
    "保险理赔": "insurance claim", "航空": "airline flight delay",
    "地震": "earthquake", "天气": "weather", "能源负荷": "energy load",
    "电力": "energy power", "降雨": "rainfall precipitation",
    "交叉验证": "cross validation", "特征工程": "feature engineering",
    "集成学习": "ensemble", "模型融合": "ensemble blend",
    "表格数据": "tabular", "缺失值": "missing value", "类别不平衡": "imbalanced",
}


def translate_cn(text):
    """Append English equivalents for Chinese business terms found in text."""
    low = text.lower()
    extra = [en for cn, en in CN_EN_MAP.items() if cn in low]
    return text + (" " + " ".join(extra) if extra else "")


def match_terms(c, terms):
    """All terms must appear in title+desc+metric+kind. Returns overlap score."""
    if not terms:
        return 0
    bag = _terms("%s %s %s %s" % (c["title"], c["desc"], c["metric"], c["kind"]))
    missing = terms - bag
    if missing:
        return 0
    return len(terms & bag)


def cmd_search(args):
    comps = load_competitions()
    terms = _terms(translate_cn(" ".join(args.terms)))
    results = []
    for c in comps:
        if args.kind and c["kind"].lower() != args.kind.lower():
            continue
        if args.year and str(c["year"]) != str(args.year):
            continue
        if args.metric and args.metric.lower() not in c["metric"].lower():
            continue
        if args.status == "curated" and not c["curated"]:
            continue
        if args.status == "missing" and c["curated"]:
            continue
        if args.status == "hot" and not c["hot"]:
            continue
        score = match_terms(c, terms)
        if terms and score == 0:
            continue
        results.append((score, c))

    results.sort(key=lambda t: (-t[0], -len(t[1]["solutions"]), -t[1]["number"]))
    results = results[: args.limit]

    if args.json:
        print(json.dumps([c for _, c in results], ensure_ascii=False, indent=2))
        return
    if not results:
        print("No competitions matched. Loosen filters or terms.")
        return
    for i, (_, c) in enumerate(results, 1):
        print(fmt_comp(c, with_solutions=args.solutions, index=i))
        print()


def cmd_show(args):
    comps = load_competitions()
    key = args.target.strip()
    if key.isdigit():
        matches = [c for c in comps if str(c["number"]) == key]
    else:
        t = _terms(key)
        matches = sorted((c for c in comps if t and t <= _terms(c["title"])),
                         key=lambda c: -len(c["solutions"]))
        if not matches:  # fall back to substring match
            matches = [c for c in comps if key.lower() in c["title"].lower()]
    if not matches:
        raise SystemExit("No competition found for %r. Use `search` to find one." % key)
    if args.json:
        print(json.dumps(matches, ensure_ascii=False, indent=2))
        return
    for c in matches[:3]:
        print(fmt_comp(c, with_solutions=True))
        print()


def cmd_similar(args):
    """Find past competitions similar to a problem description."""
    comps = load_competitions()
    q = _terms(translate_cn(args.description))
    stop = {"the", "a", "an", "of", "to", "and", "for", "in", "on", "with", "is",
            "are", "predict", "prediction", "predictions", "data", "dataset", "model"}
    q = {t for t in q if t not in stop and len(t) > 2}
    scored = []
    for c in comps:
        bag = _terms("%s %s %s" % (c["title"], c["desc"], c["metric"]))
        overlap = q & bag
        if not overlap:
            continue
        score = len(overlap) / len(q)
        if c["curated"]:
            score += 0.15 * min(len(c["solutions"]), 10) / 10
        scored.append((score, c))
    scored.sort(key=lambda t: -t[0])
    top = scored[: args.limit]
    if args.json:
        print(json.dumps([{"similarity": round(s, 3), **c} for s, c in top],
                         ensure_ascii=False, indent=2))
        return
    if not top:
        print("No similar competitions found. Try different keywords.")
        return
    for s, c in top:
        print(fmt_comp(c, with_solutions=True))
        print("    similarity: %.2f" % s)
        print()


def cmd_read(args):
    """Print offline solution text for one competition."""
    comps = load_competitions()
    key = str(args.target).strip()
    comp = None
    if key.isdigit():
        comp = next((c for c in comps if str(c["number"]) == key), None)
    if comp is None:
        t = _terms(key)
        comp = next((c for c in comps if t and t <= _terms(c["title"])), None)
        if comp is None:
            comp = next((c for c in comps if key.lower() in c["title"].lower()), None)
    if comp is None:
        raise SystemExit("No competition found for %r." % key)

    files = _stored_urls(comp["number"])
    if not files:
        raise SystemExit("No offline pages for #%d yet. Run download_solutions.py first."
                         % comp["number"])
    picked = []
    for u, p in files.items():
        rank = int(p.stem[:2])
        if args.rank and rank != args.rank:
            continue
        picked.append((rank, u, p))
    picked.sort(key=lambda t: (t[0], t[2].name))
    if not picked:
        raise SystemExit("No offline page for rank %s of #%d. Saved ranks: %s"
                         % (args.rank, comp["number"],
                            sorted({int(p.stem[:2]) for p in files.values()})))
    for rank, u, p in picked:
        text = p.read_text(encoding="utf-8", errors="replace")
        body = text.split("\n\n", 1)[-1]
        if args.max_chars and len(body) > args.max_chars:
            body = body[: args.max_chars] + "\n... [truncated, full text at %s]" % p
        print("=" * 70)
        print("#%d %s | rank %d | %s" % (comp["number"], comp["title"], rank, u))
        print("=" * 70)
        print(body.strip())
        print()


def cmd_grep(args):
    """Full-text search across offline solution pages."""
    terms = [t.lower() for t in args.terms]
    if not terms:
        raise SystemExit("Provide search terms.")
    comps = load_competitions()
    title_by_number = {c["number"]: c["title"] for c in comps}
    hits = 0
    if not PAGES_DIR.is_dir():
        raise SystemExit("No offline pages yet. Run download_solutions.py first.")
    for p in sorted(PAGES_DIR.glob("*/*.txt"),
                    key=lambda p: (int(p.parent.name), p.name)):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        low = text.lower()
        if not all(t in low for t in terms):
            continue
        hits += 1
        number = int(p.parent.name)
        rank = int(p.stem[:2])
        first = text.splitlines()[0] if text else ""
        print("%s | #%d %s | rank %d" % (p.relative_to(SKILL_DIR),
                                         number, title_by_number.get(number, "?"), rank))
        print("  %s" % first.strip())
        shown = 0
        for line in text.splitlines()[1:]:
            lt = line.lower()
            if any(t in lt for t in terms):
                print("  > %s" % line.strip()[:160])
                shown += 1
                if shown >= 3:
                    break
        print()
        if args.limit and hits >= args.limit:
            break
    print("%d offline page(s) matched." % hits)


def cmd_stats(args):
    comps = load_competitions()
    from collections import Counter
    by_kind = Counter(c["kind"] for c in comps)
    by_year = Counter(c["year"] for c in comps)
    curated = sum(1 for c in comps if c["curated"])
    sols = sum(len(c["solutions"]) for c in comps)
    if args.json:
        print(json.dumps({"total": len(comps), "curated": curated,
                          "solutions": sols,
                          "by_kind": dict(by_kind), "by_year": dict(sorted(by_year.items()))},
                         ensure_ascii=False, indent=2))
        return
    print("Total competitions: %d (curated: %d, missing solutions: %d)"
          % (len(comps), curated, len(comps) - curated))
    print("Total solution links: %d" % sols)
    print("\nBy kind:")
    for k, n in by_kind.most_common():
        print("  %-15s %d" % (k or "-", n))
    print("\nBy year:")
    for y, n in sorted(by_year.items(), reverse=True):
        print("  %d  %d" % (y, n))


def cmd_refresh(_args):
    comps = load_competitions(force=True)
    print("Refreshed cache: %d competitions, %d solution links"
          % (len(comps), sum(len(c["solutions"]) for c in comps)))


# ------------------------------------------------------------------------ main

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search competitions by keywords/filters")
    s.add_argument("terms", nargs="*", help="keywords (all must match title/desc/metric)")
    s.add_argument("--kind", help="Featured|Playground|Research|Analytics|Recruitment|Getting Started|Masters")
    s.add_argument("--year")
    s.add_argument("--status", choices=["curated", "missing", "hot"])
    s.add_argument("--metric", help="metric substring, e.g. auc, rmse")
    s.add_argument("--limit", type=int, default=10)
    s.add_argument("--solutions", action="store_true", help="also list solution links")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("show", help="show one competition with all solution links")
    s.add_argument("target", help="competition number or title substring")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_show)

    s = sub.add_parser("similar", help="find past competitions similar to a problem description")
    s.add_argument("description")
    s.add_argument("--limit", type=int, default=5)
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_similar)

    s = sub.add_parser("read", help="print offline solution text for a competition")
    s.add_argument("target", help="competition number or title substring")
    s.add_argument("rank", nargs="?", type=int, help="solution rank (omit = all saved)")
    s.add_argument("--max-chars", type=int, default=20000)
    s.set_defaults(func=cmd_read)

    s = sub.add_parser("grep", help="full-text search across offline pages")
    s.add_argument("terms", nargs="+")
    s.add_argument("--limit", type=int, default=20)
    s.set_defaults(func=cmd_grep)

    s = sub.add_parser("stats", help="archive statistics")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("refresh", help="force re-download of data")
    s.set_defaults(func=cmd_refresh)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
