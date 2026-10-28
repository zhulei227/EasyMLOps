---
name: kaggle-solutions
description: Search and retrieve Kaggle competition top solutions (write-ups, code repos, notebooks) from the kaggle.farid.one archive covering 721 competitions and 4700+ ranked solution links, with an offline snapshot of metadata and top-3 solution texts. This skill should be used when the user asks to find Kaggle competition solutions or winning approaches, look up a specific competition's top placements, find similar past competitions for a data science task, or benchmark approaches by metric/competition type.
---

# Kaggle Solutions

## Overview

Query the community-maintained Kaggle solutions archive (https://kaggle.farid.one, source repo `faridrashidi/kaggle-solutions`), read top solutions fully offline, and **automatically match the current business task to similar past competitions to derive optimization recommendations**.

## Primary Workflow: match current business → optimize

1. **Profile the task and match competitions** — run `scripts/match_task.py`:
   - Explicit description: `python scripts/match_task.py "churn prediction for telecom users" --metric auc --out report.md`
   - Auto-profile a project directory (scans code/docs for task terms, metric hints and kaggle.com/c/ slugs): `python scripts/match_task.py --dir path/to/project --out report.md`
   The report contains: task profile, ranked similar competitions, techniques mined from offline winner pages (with original quotes), and an actionable optimization checklist (validation / models / features / ensembling / post-processing).
2. **Read the winning approaches** — for the top matches use `read <number> <rank>` (offline) and distill concrete steps into the user's task; cite technique quotes from the report.
3. **Apply and iterate** — turn the checklist into concrete changes for the user's pipeline (CV scheme, model choice, features, ensembling, post-processing), highest-frequency winner techniques first.
4. If the best-matched competitions lack offline pages, run `download_solutions.py` (see below) and re-run `match_task.py`.

### Semantic search: `scripts/embedding_index.py` (numpy TF-IDF vectors)

Local vector index over both corpora (721 competitions + ~1280 offline winner pages),
stored at `data/index/tfidf.npz`. No model download; queries go through the CN→EN map
so Chinese works.

```bash
python scripts/embedding_index.py build          # (re)build — run after downloads grow
python scripts/embedding_index.py "churn prediction" --corpus solutions --top 10
python scripts/embedding_index.py "欺诈检测" --corpus competitions --top 5
python scripts/embedding_index.py --status
```

`match_task.py` uses it automatically: competitions are ranked by keyword score +
20×cosine similarity, and the report includes a "语义检索的相关优胜方案" section with
`read` hints. Rebuild the index after new pages are downloaded (or pass `--rebuild`);
with `--no-sem` to fall back to keyword-only matching. numpy is required; without it
the skill silently degrades to keyword matching.

### Chinese input support

Descriptions, project docs and search keywords may be written in Chinese. A built-in
CN→EN concept map (`CN_EN_MAP` in kaggle_solutions.py, ~60 business/domain terms:
客户流失/欺诈/销量预测/时间序列/图像分割/反欺诈...) translates them to English archive
vocabulary before matching; metric hints also accept Chinese (AUC曲线下面积/均方根误差/
准确率...). To extend coverage, add entries to `CN_EN_MAP`.

## Data Layout (offline-first)

```
data/
├── competitions.yml        # full metadata snapshot: 721 competitions, 4768 solution links
├── competitions.json       # normalized metadata (auto-generated from the yml)
└── pages/<number>/<NN>.txt # offline solution texts, NN = rank (01 = 1st place);
                            # header line "url: ..." records the source link
```

All metadata queries run entirely from the local snapshot — no network needed. Solution *texts* are available only for pages already downloaded by `download_solutions.py`; check availability with the `[offline]` markers in `show`/`search --solutions` output or `download_solutions.py --status`.

Normalized record schema:

```yaml
number: 721            # site index, newest = highest
title: "..."
desc: "..."            # one-line competition description
kind: Featured         # Featured | Playground | Research | Analytics | Recruitment | Getting Started | Masters
prize: "$60,000"       # or Swag / Knowledge / ...
team: "3575"           # team count (string, no commas)
metric: "Roc Auc Score"
link: "https://www.kaggle.com/c/<slug>"
year: 2026
hot: false             # site "Hot" flag
curated: true          # true = has solutions (Curated), false = Missing
solutions:             # ranked solution links
  - rank: 1            # placement number
    link: "https://www.kaggle.com/c/<slug>/writeups/..."   # or /discussion/... or external blog
    kind: description  # description = write-up, code = codebase, kernel = notebook
```

Data is cached in `<tempdir>/kaggle_solutions_cache/competitions.json` for 7 days.

## Scripts

### Task matching: `scripts/match_task.py` (stdlib-only)

See "Primary Workflow" above. Technique mining uses a curated lexicon (validation, gbdt,
deep-learning, ensemble, features, training, data, post-process, llm) over offline winner
pages; stemming handles singular/plural. Use `--top 8 --deep 4` defaults; `--out report.md`
saves the markdown report.

### Query: `scripts/kaggle_solutions.py` (stdlib-only)

```bash
# Search by keywords (all terms must appear in title/desc/metric)
python scripts/kaggle_solutions.py search image segmentation --status curated --limit 5

# Filter by kind / year / metric; --solutions lists ranked links with [offline] markers
python scripts/kaggle_solutions.py search --kind Featured --year 2024 --metric auc --solutions

# Full detail of one competition (by site number or title substring)
python scripts/kaggle_solutions.py show 721

# Read OFFLINE solution text (omit rank to list every saved page)
python scripts/kaggle_solutions.py read 721 1 --max-chars 20000

# Full-text search across all downloaded solution pages
python scripts/kaggle_solutions.py grep "transformer ensemble"

# Find past competitions similar to a problem description
python scripts/kaggle_solutions.py similar "predict machine failure from sensor time series"

# stats [--json] / refresh (re-download metadata) / --json on any command
```

### Downloader: `scripts/download_solutions.py` (needs playwright-cli + Edge/Chrome)

Fetches solution page texts with a real browser (kaggle.com blocks plain HTTP) into
`data/pages/`. Resumable — re-running skips already-saved URLs. Default priority:
rank 1 across all competitions first, then rank 2, then rank 3.

```bash
python scripts/download_solutions.py --status                # progress: wanted vs saved
python scripts/download_solutions.py --dry-run --limit 20    # preview queue
python scripts/download_solutions.py --max-rank 1            # only 1st places
python scripts/download_solutions.py --max-rank 3            # full medal set (~2-3h)
# options: --limit N (stop after N pages), --delay S (rate limit), --browser chrome
```

Long runs: launch in background, log to `logs/download.log`, and poll `--status`.

## Workflow

1. **User asks for solutions to a known competition** — `show <title or number>`; rank 1–3 are medal approaches. Links marked **[offline]** have full text stored: read with `read <number> <rank>` instead of fetching the web.
2. **User wants the actual approach/methodology** — check offline coverage with `--status`. If the needed pages are saved, use `read`; otherwise run `download_solutions.py` for those competitions (or `--max-rank 1`), or fetch the URL with the web fetch tool as a fallback. Pages contain site navigation noise at the top — skip to the title line for the actual write-up.
3. **User is working on a task and wants proven approaches** — `similar "<task description>"`, then `show`/`read` the best matches. Filter with `--kind`/`--metric` to match the task type (tabular → Playground/Featured with AUC/RMSE; NLP → Research).
4. **Search inside downloaded solutions** — `grep "terms..."` finds techniques across all saved pages (e.g. `grep pseudo-labeling`).
5. **Stale metadata** — `refresh` re-downloads the yml snapshot; new competitions' solution texts still need a `download_solutions.py` run.

## Output Guidance

- Present competitions as: `#number [Kind Year] Title — desc (metric, N solution links)` followed by the competition URL.
- Group solution links by medal tier (1st / 2nd / 3rd / other) and label each as write-up, code, or notebook based on `kind`.
- Use `--json` output when downstream processing (tables, further filtering) is needed.
