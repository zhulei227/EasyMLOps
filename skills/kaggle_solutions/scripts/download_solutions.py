#!/usr/bin/env python3
"""Offline snapshot downloader for Kaggle competition solutions.

Downloads solution page content (write-ups, discussion posts, external pages)
into <skill>/data/pages/<competition_number>/<NN>.txt so the skill works fully
offline. Resumable: pages already saved with a matching URL are skipped.

Requires playwright-cli (https://www.npmjs.com/package/@playwright/cli) with a
system browser (Edge by default) because kaggle.com blocks plain HTTP clients.

Usage:
  python scripts/download_solutions.py --status              # progress report
  python scripts/download_solutions.py --dry-run --limit 20  # preview queue
  python scripts/download_solutions.py --max-rank 1          # rank-1 for all
  python scripts/download_solutions.py --max-rank 3 --limit 50
  python scripts/download_solutions.py --max-rank 3          # full run (~2-3h)
"""

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kaggle_solutions as ks  # noqa: E402

PAGES_DIR = SKILL_DIR / "data" / "pages"
SESSION = "ks"
if hasattr(sys.stdout, "reconfigure"):  # GBK consoles choke on page content
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
JS_TEXT = ("(() => { const el = document.querySelector('main') || document.body;"
           " return el.innerText; })()")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"}
BLOCK_MARKERS = ("checking your browser", "recaptcha", "unusual traffic")


PCLI = shutil.which("playwright-cli") or "playwright-cli"


def run(args, timeout=90):
    args = [PCLI if a == "playwright-cli" else a for a in args]
    p = subprocess.run(args, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=timeout)
    return p.stdout + "\n" + p.stderr


def norm_url(url):
    """Canonical key for resume comparison."""
    u = url.split("#")[0].rstrip("/")
    u = re.sub(r"^http://", "https://", u)
    return u


def page_path(number, rank, suffix):
    return PAGES_DIR / str(number) / ("%02d%s.txt" % (rank, suffix))


def stored_url(path):
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
        m = re.match(r"url:\s*(\S+)", first)
        return norm_url(m.group(1)) if m else ""
    except OSError:
        return ""


def save_page(number, rank, url, text):
    suffix = ""
    path = page_path(number, rank, suffix)
    n = 0
    while path.exists() and stored_url(path) not in ("", norm_url(url)):
        n += 1
        suffix = chr(96 + n)  # a, b, c...
        path = page_path(number, rank, suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("url: %s\nfetched: %s\nchars: %d\n\n%s"
                    % (url, time.strftime("%Y-%m-%d"), len(text), text),
                    encoding="utf-8")
    return path


def build_queue(comps, max_rank, include_unranked):
    """rank ascending across all competitions so interruptions keep coverage fair."""
    queue = []
    for rank in range(1, max_rank + 1):
        for c in sorted(comps, key=lambda x: -x["number"]):
            for idx, s in enumerate(c["solutions"]):
                if s["rank"] != rank:
                    continue
                queue.append((c["number"], rank, idx, s["link"]))
    if include_unranked:
        for c in sorted(comps, key=lambda x: -x["number"]):
            for idx, s in enumerate(c["solutions"]):
                if s["rank"] == 0:
                    queue.append((c["number"], 0, idx, s["link"]))
    return queue


GITHUB_REPO = re.compile(r"https://github\.com/([^/]+)/([^/#?]+)")


def _github_readme(url):
    """Fetch a GitHub repo README via the API (repo home pages are JS-heavy)."""
    m = GITHUB_REPO.match(url)
    if not m or "/blob/" in url or "/tree/" in url:
        return None
    import base64
    api = "https://api.github.com/repos/%s/%s/readme" % (m.group(1), m.group(2))
    try:
        req = urllib.request.Request(api, headers={**UA, "Accept": "application/vnd.github+json"})
        data = json.loads(urllib.request.urlopen(req, timeout=30).read())
        return base64.b64decode(data["content"]).decode("utf-8", "replace")[:200000]
    except Exception:  # noqa: BLE001 - fall back to HTML fetch
        return None


def fetch_external(url):
    """Best-effort plain HTTP fetch for non-kaggle links (blogs, GitHub...)."""
    readme = _github_readme(url)
    if readme is not None:
        return readme
    try:
        html = urllib.request.urlopen(
            urllib.request.Request(url, headers=UA), timeout=30).read()
        for enc in ("utf-8", "latin-1"):
            try:
                html = html.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
        return text.strip()[:200000]
    except Exception as e:  # noqa: BLE001
        return "FETCH_FAILED: %s" % e


def browser_fetch(url, browser, delays):
    out = run(["playwright-cli", "-s=" + SESSION, "goto", url], timeout=90)
    if "Error" in out and "Page URL" not in out:
        return None, "goto: %s" % out.strip()[:200]
    time.sleep(1.5 + delays * 0.5)
    out = run(["playwright-cli", "-s=" + SESSION, "eval", JS_TEXT], timeout=60)
    m = re.search(r"### Result\s*\n(.*?)\n### Ran Playwright code", out, re.S)
    if not m:
        return None, "eval failed: %s" % out.strip()[:200]
    try:
        text = json.loads(m.group(1))
    except json.JSONDecodeError:
        return None, "eval parse failed"
    low = text[:2000].lower()
    if any(b in low for b in BLOCK_MARKERS):
        return None, "blocked by anti-bot page"
    if len(text) < 400:  # SPA may still be rendering
        time.sleep(3)
        out = run(["playwright-cli", "-s=" + SESSION, "eval", JS_TEXT], timeout=60)
        m = re.search(r"### Result\s*\n(.*?)\n### Ran Playwright code", out, re.S)
        if m:
            try:
                t2 = json.loads(m.group(1))
                if len(t2) > len(text):
                    text = t2
            except json.JSONDecodeError:
                pass
    return text, None


def cmd_dedupe(_args):
    """Remove duplicate page files whose normalized URL already exists."""
    if not PAGES_DIR.is_dir():
        return
    removed = 0
    for folder in sorted(PAGES_DIR.iterdir()):
        if not folder.is_dir():
            continue
        seen = {}
        for p in sorted(folder.glob("*.txt"),
                        key=lambda x: (len(x.name), x.name)):  # 01.txt before 01a.txt
            u = stored_url(p)
            if u and u in seen:
                p.unlink()
                removed += 1
            elif u:
                seen[u] = p
    print("Removed %d duplicate file(s)." % removed)


def cmd_status(_args):
    comps = ks.load_competitions()
    want = sum(1 for c in comps for s in c["solutions"] if 0 < s["rank"] <= 3)
    have = sum(1 for _ in PAGES_DIR.glob("*/*.txt")) if PAGES_DIR.exists() else 0
    size = sum(p.stat().st_size for p in PAGES_DIR.glob("*/*.txt")) \
        if PAGES_DIR.exists() else 0
    print("Competitions: %d | rank<=3 solution pages: %d wanted, %d saved (%.1f MB)"
          % (len(comps), want, have, size / 1e6))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-rank", type=int, default=3,
                    help="download solutions with rank <= N (default 3: medals)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max pages to fetch this run (0 = no limit)")
    ap.add_argument("--browser", default="msedge",
                    help="system browser for playwright-cli (msedge/chrome)")
    ap.add_argument("--delay", type=float, default=2.0,
                    help="base seconds between pages (jitter added)")
    ap.add_argument("--include-unranked", action="store_true",
                    help="also fetch rank-0/unranked links")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dedupe", action="store_true",
                    help="remove duplicate page files and exit")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.dedupe:
        cmd_dedupe(args)
        return
    if args.status:
        cmd_status(args)
        return

    comps = ks.load_competitions()
    queue = build_queue(comps, args.max_rank, args.include_unranked)
    todo = [(n, r, i, u) for n, r, i, u in queue
            if stored_url(page_path(n, r, "")) != norm_url(u)]
    print("Queue: %d links (rank<=%d), %d already saved, %d to fetch"
          % (len(queue), args.max_rank, len(queue) - len(todo), len(todo)))
    if args.dry_run:
        for n, r, i, u in todo[: args.limit or 20]:
            print("  #%d rank %d: %s" % (n, r, u))
        return
    if not todo:
        print("Nothing to do.")
        return

    run(["playwright-cli", "-s=" + SESSION, "open", "--browser=" + args.browser,
         "about:blank"])
    ok = fail = 0
    try:
        for done, (number, rank, idx, url) in enumerate(todo, 1):
            if args.limit and ok >= args.limit:
                break
            if norm_url(url).startswith("https://www.kaggle.com"):
                text, err = browser_fetch(url, args.browser, args.delay)
                if text and len(text) >= 400:
                    path = save_page(number, rank, url, text)
                    ok += 1
                    print("[%d/%d] #%d r%d OK %d chars -> %s"
                          % (done, len(todo), number, rank, len(text),
                             path.relative_to(SKILL_DIR)))
                else:
                    fail += 1
                    print("[%d/%d] #%d r%d FAIL %s: %s"
                          % (done, len(todo), number, rank, url, err))
            else:
                text = fetch_external(url)
                path = save_page(number, rank, url, text)
                if text.startswith("FETCH_FAILED"):
                    fail += 1
                    print("[%d/%d] #%d r%d FAIL %s" % (done, len(todo), number, rank, url))
                else:
                    ok += 1
                    print("[%d/%d] #%d r%d OK (external) %d chars -> %s"
                          % (done, len(todo), number, rank, len(text),
                             path.relative_to(SKILL_DIR)))
            time.sleep(args.delay + random.random() * args.delay)
    finally:
        run(["playwright-cli", "-s=" + SESSION, "close"])
    print("Done. saved=%d failed=%d. Re-run to resume." % (ok, fail))


if __name__ == "__main__":
    main()
