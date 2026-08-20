"""
Validate a local HotpotQA BM25/dense index before running behavioral experiments.

Purpose: prove the corpus (built by scripts/build_hotpotqa_index.py) actually supports HotpotQA
retrieval, so a retrieval hole can never be mistaken for a behavioral effect downstream.

Unlike MedQA, HotpotQA's `supporting_facts` gives the EXACT gold title(s) per question -- so
unlike validate_medqa_index.py's phrasing-bound substring recall (a regression tripwire only,
never a pass/fail bar), this script's title-recall@k is a genuine, content-bound metric: it
should clear a random-title baseline by a wide margin for a healthy index.

Loads HotpotQA the same way src/services/qa_eval.py does in its `hotpotqa` branch
(hotpotqa/hotpot_qa, distractor config, split=validation), takes a seeded sample of --n
questions, retrieves top-k passages per question via LocalIndexSearchService, and reports:

  1. Sanity        - n_passages, n_titles, manifest agreement with corpus.jsonl.
  2. Title recall@k - fraction of sampled questions where >=1 gold title (from
                       supporting_facts) appears among the top-k retrieved passage titles,
                       vs. the same check against a random title set of equal size. THE
                       PRIMARY QUANTITATIVE SIGNAL (real, not phrasing-bound).
  3. Latency        - mean seconds per search() call.
  4. Eyeball        - 5 sampled (question, gold titles, top-3 retrieved titles) triples.

Usage:
    uv run python scripts/validate_hotpotqa_index.py --index-dir data/hotpotqa_index --n 200 --k 10
"""

import os
import sys
import json
import time
import random
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.services.local_index_search import LocalIndexSearchService


def setup_args():
    p = argparse.ArgumentParser(
        description="Validate a local HotpotQA BM25/dense index against the real HotpotQA validation set."
    )
    p.add_argument("--index-dir", default="data/hotpotqa_index", help="Index directory (corpus.jsonl + manifest.json).")
    p.add_argument("--n", type=int, default=200, help="Number of sampled HotpotQA validation questions.")
    p.add_argument("--k", type=int, default=10, help="Top-k passages retrieved per search() call.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the question sample + random-title baseline.")
    p.add_argument("--backend", choices=["bm25", "dense"], default="bm25",
                   help="Which LocalIndexSearchService backend to query.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Only used when --backend dense.")
    p.add_argument("--n-dump", type=int, default=5, help="How many sampled examples to print in the eyeball dump.")
    return p.parse_args()


def load_hotpotqa():
    """Load HotpotQA exactly as src/services/qa_eval.py does in its `hotpotqa` branch, plus supporting_facts."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    examples = []
    for row in ds:
        gold_titles = sorted(set(row["supporting_facts"]["title"]))
        examples.append({
            "example_id": row["id"], "problem": row["question"],
            "gold answer": row["answer"], "gold_titles": gold_titles,
        })
    return examples


def read_corpus_stats(index_dir: str):
    corpus_path = os.path.join(index_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}")
    n_lines = 0
    titles = set()
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_lines += 1
            titles.add(row.get("title"))
    return {"n_passages": n_lines, "n_titles": len(titles)}, titles


def read_manifest(index_dir: str):
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f)


def main():
    args = setup_args()

    print("=" * 78)
    print("HotpotQA local index validation")
    print("=" * 78)
    print(f"index_dir = {args.index_dir}")
    print(f"n = {args.n}   k = {args.k}   seed = {args.seed}   backend = {args.backend}")
    print()

    # ---- 1. Sanity ---------------------------------------------------------
    print("-" * 78)
    print("1. SANITY")
    print("-" * 78)
    stats, corpus_titles = read_corpus_stats(args.index_dir)
    manifest = read_manifest(args.index_dir)
    print(f"n_passages (corpus.jsonl lines) = {stats['n_passages']}")
    print(f"n_titles (distinct)             = {stats['n_titles']}")
    if manifest is None:
        print("manifest.json                   NOT FOUND (skipping agreement check)")
    else:
        manifest_n = manifest.get("n_passages")
        agree = manifest_n == stats["n_passages"]
        print(f"manifest.json n_passages        = {manifest_n}  ({'MATCH' if agree else 'MISMATCH vs corpus.jsonl line count!'})")
        print(f"manifest.json (full)            = {json.dumps(manifest, indent=2)}")
    print()

    # ---- Load index + HotpotQA sample --------------------------------------
    kwargs = {"backend": args.backend}
    if args.backend == "dense":
        kwargs["dense_model"] = args.dense_model
    svc = LocalIndexSearchService(args.index_dir, **kwargs)

    print("Loading HotpotQA distractor validation split (hotpotqa/hotpot_qa)...")
    examples = load_hotpotqa()
    print(f"  {len(examples)} total validation examples.")
    rng = random.Random(args.seed)
    n_sample = min(args.n, len(examples))
    sampled = rng.sample(examples, n_sample)
    print(f"  sampled {n_sample} examples with seed={args.seed}.")
    print()

    # ---- 2. Title recall@k --------------------------------------------------
    corpus_titles_list = sorted(corpus_titles)
    gold_hits = 0
    random_hits = 0
    latencies = []
    dump_rows = []

    for ex in sampled:
        t0 = time.time()
        results = svc.search(ex["problem"], max_results=args.k)
        latencies.append(time.time() - t0)
        retrieved_titles = {r["title"] for r in results}

        gold_titles = set(ex["gold_titles"])
        gold_hit = bool(gold_titles & retrieved_titles)
        gold_hits += int(gold_hit)

        # Random baseline: same-size random title set in place of the gold titles, same
        # question's retrieved set -- isolates "did retrieval find the gold titles specifically"
        # from "does the corpus happen to contain a lot of titles that overlap any k-set".
        random_titles = set(rng.sample(corpus_titles_list, min(len(gold_titles), len(corpus_titles_list))))
        random_hits += int(bool(random_titles & retrieved_titles))

        if len(dump_rows) < args.n_dump:
            dump_rows.append({
                "example_id": ex["example_id"],
                "question": ex["problem"][:220],
                "gold_titles": ex["gold_titles"],
                "gold_hit": gold_hit,
                "top3_retrieved_titles": [r["title"] for r in results[:3]],
            })

    gold_recall = gold_hits / n_sample if n_sample else 0.0
    random_recall = random_hits / n_sample if n_sample else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("-" * 78)
    print(f"2. GOLD-TITLE RECALL@{args.k}  [PRIMARY METRIC -- content-bound, not phrasing-bound]")
    print("-" * 78)
    print(f"gold recall@{args.k}    = {gold_recall:.3f}  ({gold_hits}/{n_sample} questions had >=1 gold title retrieved)")
    print(f"random baseline@{args.k} = {random_recall:.3f}  ({random_hits}/{n_sample} questions, same-size random title set)")
    print(f"lift over random     = {gold_recall - random_recall:+.3f}")
    print("   Unlike MedQA's substring recall, this DOES have discriminative power: a healthy")
    print("   index should clear the random baseline by a wide margin.")
    print()

    print("-" * 78)
    print("3. LATENCY")
    print("-" * 78)
    print(f"mean seconds per search() call = {mean_latency:.4f}s over {len(latencies)} calls")
    print()

    print("-" * 78)
    print(f"4. EYEBALL DUMP ({len(dump_rows)} sampled examples)")
    print("-" * 78)
    for row in dump_rows:
        print(f"[{row['example_id']}]")
        print(f"  question: {row['question']}")
        print(f"  gold titles: {row['gold_titles']}  (hit={row['gold_hit']})")
        print(f"  top-3 retrieved titles: {row['top3_retrieved_titles']}")
        print()

    print("=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
