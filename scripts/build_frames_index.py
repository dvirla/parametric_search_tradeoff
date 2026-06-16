"""
Build a local search corpus for the FRAMES benchmark.

Collects every gold Wikipedia page referenced by FRAMES questions (`wiki_links`),
adds a configurable number of random Wikipedia articles as distractors, fetches
full plain-text extracts via the free MediaWiki API (cached + resumable), chunks
each article into paragraph-sized passages, and writes:

    <index-dir>/corpus.jsonl      one passage per line {doc_id,title,passage_id,text}
    <index-dir>/manifest.json     build stats / config
    <index-dir>/embeddings.npy    (only with --backend dense|both) passage embeddings

The corpus is then served by src/services/local_index_search.LocalIndexSearchService.

Usage:
    uv run python scripts/build_frames_index.py
    uv run python scripts/build_frames_index.py --distractor-ratio 1.0 --backend bm25
    uv run python scripts/build_frames_index.py --backend both          # also embed
"""

import os
import re
import sys
import ast
import json
import time
import hashlib
import argparse
from urllib.parse import unquote
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {
    "User-Agent": "parametric-search-tradeoff/0.1 (research; https://github.com/dvirla)",
    "Accept": "application/json",
}
_HEADER_RE = re.compile(r"^=+.*=+$")  # explaintext section headers like "== History =="


def setup_args():
    p = argparse.ArgumentParser(description="Build the local FRAMES search corpus.")
    p.add_argument("--index-dir", default="data/frames_index", help="Output directory.")
    p.add_argument("--distractor-ratio", type=float, default=1.0,
                   help="Random distractor articles as a multiple of #gold pages (default 1.0).")
    p.add_argument("--max-chars", type=int, default=1000, help="Max chars per passage chunk.")
    p.add_argument("--min-chars", type=int, default=120, help="Drop passages shorter than this.")
    p.add_argument("--workers", type=int, default=6, help="Parallel fetch workers (keep modest; extracts is throttled under bursts).")
    p.add_argument("--backend", choices=["bm25", "dense", "both"], default="bm25",
                   help="bm25 needs no extra artifacts; dense/both also writes embeddings.npy.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--seed", type=int, default=0, help="Seed for distractor sampling order.")
    p.add_argument("--refresh-distractors", action="store_true",
                   help="Re-sample random distractors even if titles.json exists.")
    return p.parse_args()


def parse_links(raw):
    if isinstance(raw, (list, tuple)):
        return list(raw)
    try:
        return json.loads(raw)
    except Exception:
        try:
            return ast.literal_eval(raw)
        except Exception:
            return []


def url_to_title(url: str) -> str:
    return unquote(url.rstrip("/").split("/wiki/")[-1]).split("#")[0].replace("_", " ")


def gold_titles():
    df = load_dataset("google/frames-benchmark", split="test").to_pandas()
    titles = set()
    for raw in df["wiki_links"]:
        for u in parse_links(raw):
            if "/wiki/" in u:
                titles.add(url_to_title(u))
    return sorted(titles)


def random_titles(n: int, exclude: set, batch: int = 500):
    """Fetch n random article titles (namespace 0) not already in `exclude`."""
    out = []
    seen = set(exclude)
    while len(out) < n:
        params = {"action": "query", "list": "random", "rnnamespace": 0,
                  "rnlimit": min(batch, 500), "format": "json"}
        try:
            data = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=15).json()
        except requests.exceptions.RequestException:
            time.sleep(1)
            continue
        for r in data.get("query", {}).get("random", []):
            t = r["title"]
            if t not in seen:
                seen.add(t)
                out.append(t)
                if len(out) >= n:
                    break
    return out


def _cache_path(cache_dir: str, title: str) -> str:
    key = hashlib.sha256(title.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, f"{key}.json")


def fetch_extract(title: str, cache_dir: str) -> str:
    """Fetch full plain-text extract for one article (disk-cached, resumable).

    Only *non-empty* extracts are cached, and empty cache entries are ignored on
    read — so transient empty responses (e.g. API throttling under concurrency)
    are retried on the next run rather than poisoning the corpus permanently.
    """
    title = title.split("#")[0]  # drop section anchors like "2018 FIFA World Cup#Officiating"
    cp = _cache_path(cache_dir, title)
    if os.path.exists(cp):
        try:
            with open(cp) as f:
                cached = json.load(f).get("extract", "")
            if cached:
                return cached  # only trust non-empty cache
        except (json.JSONDecodeError, OSError):
            pass
    params = {"action": "query", "prop": "extracts", "explaintext": 1,
              "redirects": 1, "titles": title, "format": "json"}
    backoff = 1.0
    for attempt in range(4):
        try:
            data = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=20).json()
            pages = data.get("query", {}).get("pages", {})
            extract = ""
            for _, page in pages.items():
                extract = page.get("extract", "") or ""
            if extract:  # only cache successful, non-empty extracts
                try:
                    with open(cp, "w") as f:
                        json.dump({"title": title, "extract": extract}, f)
                except OSError:
                    pass
            return extract
        except requests.exceptions.RequestException:
            if attempt == 3:
                return ""
            time.sleep(backoff)
            backoff *= 2
    return ""


def chunk(text: str, max_chars: int, min_chars: int):
    """Split an explaintext article into paragraph-sized passages."""
    passages, buf = [], ""
    for line in text.split("\n"):
        line = line.strip()
        if not line or _HEADER_RE.match(line):
            continue
        if buf and len(buf) + len(line) + 1 > max_chars:
            passages.append(buf)
            buf = line
        else:
            buf = f"{buf} {line}".strip() if buf else line
    if buf:
        passages.append(buf)
    return [p for p in passages if len(p) >= min_chars]


def main():
    args = setup_args()
    os.makedirs(args.index_dir, exist_ok=True)
    cache_dir = os.path.join(args.index_dir, "pages_cache")
    os.makedirs(cache_dir, exist_ok=True)

    # Persist the chosen title list so reruns are reproducible and converge
    # (recovering stragglers) instead of re-sampling a fresh distractor set.
    titles_path = os.path.join(args.index_dir, "titles.json")
    if os.path.exists(titles_path) and not args.refresh_distractors:
        saved = json.load(open(titles_path))
        gold, distract = saved["gold"], saved["distractors"]
        print(f"Reusing title list from {titles_path}: {len(gold)} gold + {len(distract)} distractors.")
    else:
        print("Collecting FRAMES gold page titles...")
        gold = gold_titles()
        print(f"  {len(gold)} unique gold pages.")
        n_distract = int(round(args.distractor_ratio * len(gold)))
        print(f"Fetching {n_distract} random distractor titles...")
        distract = random_titles(n_distract, exclude=set(gold)) if n_distract > 0 else []
        print(f"  {len(distract)} distractor pages.")
        with open(titles_path, "w") as f:
            json.dump({"gold": gold, "distractors": distract}, f)

    all_titles = gold + distract
    is_gold = {t: (i < len(gold)) for i, t in enumerate(all_titles)}

    print(f"Fetching {len(all_titles)} article extracts ({args.workers} workers, cached)...")
    extracts = {}
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_extract, t, cache_dir): t for t in all_titles}
        for fut in as_completed(futs):
            t = futs[fut]
            extracts[t] = fut.result()
            done += 1
            if done % 250 == 0:
                print(f"  fetched {done}/{len(all_titles)}")

    print("Chunking into passages + writing corpus.jsonl...")
    corpus_path = os.path.join(args.index_dir, "corpus.jsonl")
    n_passages = n_empty = 0
    with open(corpus_path, "w") as out:
        for doc_id, title in enumerate(all_titles):
            chunks = chunk(extracts.get(title, ""), args.max_chars, args.min_chars)
            if not chunks:
                n_empty += 1
            for pid, text in enumerate(chunks):
                out.write(json.dumps({
                    "doc_id": doc_id, "title": title, "passage_id": pid,
                    "text": text, "is_gold": is_gold[title],
                }) + "\n")
                n_passages += 1

    manifest = {
        "n_gold_pages": len(gold), "n_distractor_pages": len(distract),
        "n_total_pages": len(all_titles), "n_pages_no_text": n_empty,
        "n_passages": n_passages, "max_chars": args.max_chars,
        "min_chars": args.min_chars, "distractor_ratio": args.distractor_ratio,
    }
    print(f"  wrote {n_passages} passages ({n_empty} pages had no usable text).")

    if args.backend in ("dense", "both"):
        print(f"Embedding passages with {args.dense_model} (this is slow on CPU)...")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        # E5-family models require asymmetric prefixes: "passage: " on documents
        # and "query: " on queries. Auto-detect by model name.
        is_e5 = "e5" in args.dense_model.lower()
        passage_prefix = "passage: " if is_e5 else ""
        query_prefix = "query: " if is_e5 else ""
        texts = [passage_prefix + json.loads(l)["text"] for l in open(corpus_path)]
        model = SentenceTransformer(args.dense_model)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True).astype("float32")
        np.save(os.path.join(args.index_dir, "embeddings.npy"), emb)
        # Record the embedding model, dim, and query prefix so the search service
        # encodes queries with the *same* model + convention (otherwise FAISS
        # raises a dimension mismatch / retrieval silently degrades).
        manifest["dense_model"] = args.dense_model
        manifest["dense_dim"] = int(emb.shape[1])
        manifest["query_prefix"] = query_prefix
        print(f"  saved embeddings.npy {emb.shape} (query_prefix={query_prefix!r})")

    with open(os.path.join(args.index_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Corpus at {args.index_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
