#!/usr/bin/env python3
"""Local TF-IDF vector index for semantic search over the kaggle-solutions skill.

Two corpora:
  competitions — one vector per competition (title+desc+metric+kind), 721 docs
  solutions    — one vector per offline winner page (data/pages/), ~1280 docs

Pure numpy (no model download, fully offline). Queries go through the skill's
CN->EN concept map, so Chinese queries work for mapped terms.

Usage:
  python scripts/embedding_index.py build                  # (re)build index
  python scripts/embedding_index.py "sales forecasting"    # search both corpora
  python scripts/embedding_index.py "欺诈检测" --corpus solutions --top 10
  python scripts/embedding_index.py --status
"""

import argparse
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SKILL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
import kaggle_solutions as ks  # noqa: E402

INDEX_FILE = SKILL_DIR / "data" / "index" / "tfidf.npz"

STOP = {
    "the", "a", "an", "of", "to", "and", "for", "in", "on", "with", "is", "are",
    "was", "were", "be", "been", "this", "that", "it", "as", "at", "by", "from",
    "or", "we", "our", "i", "my", "you", "your", "they", "their", "he", "she",
    "his", "her", "not", "but", "if", "then", "than", "so", "can", "will",
    "would", "could", "should", "have", "has", "had", "do", "does", "did",
    "used", "use", "using", "also", "very", "more", "most", "some", "all",
    "data", "dataset", "model", "predict", "prediction", "competition",
}


def stem(t):
    if len(t) > 3:
        if t.endswith("ies"):
            return t[:-3] + "y"
        if t.endswith("sses"):
            return t[:-2]
        if t.endswith("s") and not t.endswith("ss"):
            return t[:-1]
        if t.endswith("ing") and len(t) > 5:
            return t[:-3]
        if t.endswith("ed") and len(t) > 4:
            return t[:-2]
    return t


def tokenize(text):
    return [stem(t) for t in re.findall(r"[a-z0-9]+", ks.translate_cn(text).lower())
            if len(t) > 1 and t not in STOP]


# ------------------------------------------------------------------ building

def _iter_docs(comps):
    for c in comps:
        yield {"corpus": "competitions", "number": c["number"], "rank": 0,
               "url": c["link"], "title": c["title"]}, \
              "%s %s %s %s" % (c["title"], c["desc"], c["metric"], c["kind"])
    if ks.PAGES_DIR.is_dir():
        for p in sorted(ks.PAGES_DIR.glob("*/*.txt")):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            body = text.split("\n\n", 1)[-1]
            if len(body) < 200 or body.startswith("FETCH_FAILED"):
                continue
            url = ""
            m = re.match(r"url:\s*(\S+)", text)
            if m:
                url = m.group(1)
            yield {"corpus": "solutions", "number": int(p.parent.name),
                   "rank": int(p.stem[:2]), "url": url,
                   "title": ""}, body[:8000]


def build(force=False):
    if INDEX_FILE.exists() and not force:
        print("Index exists (%s), use build --force to rebuild." % INDEX_FILE)
        return
    comps = ks.load_competitions()
    metas, texts, tfs = [], [], []
    df = Counter()
    for meta, text in _iter_docs(comps):
        tokens = tokenize(text)
        if not tokens:
            continue
        metas.append(meta)
        tf = Counter(tokens)
        tfs.append(tf)
        df.update(tf.keys())
    n = len(metas)
    vocab = {t: i for i, t in enumerate(sorted(df))}
    idf = np.array([math.log((n + 1) / (df[t] + 1)) + 1.0 for t in sorted(df)],
                   dtype=np.float32)

    indptr, indices, data = [0], [], []
    for tf in tfs:
        ids = np.array(sorted(vocab[t] for t in tf), dtype=np.int32)
        vals = np.array([(1 + math.log(c)) * idf[vocab[t]] for t, c in tf.items()],
                        dtype=np.float32)
        norm = float(np.linalg.norm(vals)) or 1.0
        indices.append(ids)
        data.append(vals / norm)
        indptr.append(indptr[-1] + len(ids))

    INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        INDEX_FILE,
        indptr=np.array(indptr, dtype=np.int64),
        indices=np.concatenate(indices) if indices else np.array([], np.int32),
        data=np.concatenate(data) if data else np.array([], np.float32),
        idf=idf,
        vocab=np.array(json.dumps(vocab)),
        meta=np.array(json.dumps(metas, ensure_ascii=False)),
    )
    print("Built index: %d docs (%d competitions, %d solution pages), vocab %d -> %s"
          % (n, sum(1 for m in metas if m["corpus"] == "competitions"),
             sum(1 for m in metas if m["corpus"] == "solutions"),
             len(vocab), INDEX_FILE))


# ------------------------------------------------------------------ querying

def _load():
    if not INDEX_FILE.exists():
        raise SystemExit("Index not found. Run: python scripts/embedding_index.py build")
    z = np.load(INDEX_FILE, allow_pickle=False)
    return (z["indptr"], z["indices"], z["data"], z["idf"],
            json.loads(str(z["vocab"])), json.loads(str(z["meta"])))


def search(query, corpus=None, top_k=10):
    """Return [(score, meta)] sorted by cosine similarity."""
    indptr, indices, data, idf, vocab, metas = _load()
    qtokens = [t for t in tokenize(query) if t in vocab]
    if not qtokens:
        return []
    q = np.zeros(len(idf), dtype=np.float32)
    for t, c in Counter(qtokens).items():
        q[vocab[t]] = (1 + math.log(c)) * idf[vocab[t]]
    q /= float(np.linalg.norm(q)) or 1.0

    scores = np.zeros(len(metas), dtype=np.float32)
    for i in range(len(metas)):
        seg = slice(indptr[i], indptr[i + 1])
        if seg.start == seg.stop:
            continue
        scores[i] = float(np.dot(q[indices[seg]], data[seg]))
    out = []
    for i in np.argsort(-scores):
        if scores[i] <= 0 or len(out) >= top_k:
            break
        if corpus and metas[i]["corpus"] != corpus:
            continue
        out.append((float(scores[i]), metas[i]))
    return out


# ----------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("query", nargs="*", help="search query (omit to build/status)")
    ap.add_argument("--corpus", choices=["competitions", "solutions"])
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--force", action="store_true", help="rebuild even if index exists")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()

    if args.build:
        build(force=args.force)
        return
    if args.status:
        if INDEX_FILE.exists():
            z = np.load(INDEX_FILE, allow_pickle=False)
            metas = json.loads(str(z["meta"]))
            print("Index: %s (%.2f MB, built for %d docs, mtime %s)"
                  % (INDEX_FILE, INDEX_FILE.stat().st_size / 1e6, len(metas),
                     time.strftime("%Y-%m-%d %H:%M", time.localtime(INDEX_FILE.stat().st_mtime))))
        else:
            print("No index yet. Run: python scripts/embedding_index.py build")
        return
    query = " ".join(args.query).strip()
    if query == "build":
        args.build = True
        query = ""
    if args.build:
        build(force=args.force)
        return
    if not query:
        ap.error("provide a query, or 'build' / --status")

    comps = {c["number"]: c for c in ks.load_competitions()}
    for corpus in ([args.corpus] if args.corpus else ["competitions", "solutions"]):
        print("== %s ==" % corpus)
        hits = search(query, corpus=corpus, top_k=args.top)
        if not hits:
            print("  (no hits)")
            continue
        for score, m in hits:
            c = comps.get(m["number"], {})
            title = c.get("title") or m["title"] or "?"
            print("  %.3f  #%d r%d [%s] %s" % (score, m["number"], m["rank"],
                                               corpus[:4], title))
            if m["url"]:
                print("         %s" % m["url"])


if __name__ == "__main__":
    main()
