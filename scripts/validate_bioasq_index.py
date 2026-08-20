"""
Validate a local BioASQ BM25/dense index before running behavioral experiments.

Purpose: prove the corpus (built by scripts/build_bioasq_index.py) actually supports BioASQ
retrieval, so a retrieval hole can never be mistaken for a behavioral effect downstream.

rag-mini-bioasq's `relevant_passage_ids` gives EXACT gold passage ids per question (doc_id in
the corpus is the corpus's own numeric passage id, not resequenced -- see build script) -- so
this is an even stronger signal than HotpotQA's title match: real doc-id recall@k, no fuzzy
matching at all.

CAVEAT: ~30% of relevant_passage_ids references point to passages whose source text is the
literal string "nan" (missing) and were dropped by the build script -- so recall is computed
only against the SURVIVING gold ids per question (reported as "gold coverage" below), and a
question with zero surviving gold ids is excluded from the recall denominator. This caps
achievable recall below 100% by data-quality construction, not retrieval quality -- see
build_bioasq_index.py's docstring.

Loads the QA set the same way src/services/qa_eval.py does in its `bioasq` branch
(rag-datasets/rag-mini-bioasq, question-answer-passages config, split=test), takes a seeded
sample of --n questions, retrieves top-k passages per question via LocalIndexSearchService, and
reports:

  1. Sanity          - n_passages, manifest agreement with corpus.jsonl.
  2. Gold coverage    - fraction of gold passage-id references that survived the build's
                        null-passage filter (sets the ceiling for section 3).
  3. Doc-id recall@k  - fraction of sampled questions where >=1 surviving gold id appears
                        among the top-k retrieved doc_ids, vs. a random-id baseline. THE
                        PRIMARY QUANTITATIVE SIGNAL.
  4. Latency          - mean seconds per search() call.
  5. Eyeball          - 5 sampled (question, gold ids, top-3 retrieved ids) triples.

Usage:
    uv run python scripts/validate_bioasq_index.py --index-dir data/bioasq_index --n 200 --k 10
"""

import os
import sys
import json
import time
import random
import argparse
import ast

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.services.local_index_search import LocalIndexSearchService


def setup_args():
    p = argparse.ArgumentParser(
        description="Validate a local BioASQ BM25/dense index against the real rag-mini-bioasq QA set."
    )
    p.add_argument("--index-dir", default="data/bioasq_index", help="Index directory (corpus.jsonl + manifest.json).")
    p.add_argument("--n", type=int, default=200, help="Number of sampled BioASQ QA questions.")
    p.add_argument("--k", type=int, default=10, help="Top-k passages retrieved per search() call.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the question sample + random-id baseline.")
    p.add_argument("--backend", choices=["bm25", "dense"], default="bm25",
                   help="Which LocalIndexSearchService backend to query.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Only used when --backend dense.")
    p.add_argument("--n-dump", type=int, default=5, help="How many sampled examples to print in the eyeball dump.")
    return p.parse_args()


def parse_ids(raw) -> list[int]:
    try:
        return json.loads(raw)
    except Exception:
        return ast.literal_eval(raw)


def load_bioasq_qa():
    """Load rag-mini-bioasq QA exactly as src/services/qa_eval.py does in its `bioasq` branch, plus relevant_passage_ids."""
    from datasets import load_dataset

    ds = load_dataset("rag-datasets/rag-mini-bioasq", "question-answer-passages", split="test")
    examples = []
    for row in ds:
        examples.append({
            "example_id": row["id"], "problem": row["question"],
            "gold answer": row["answer"], "gold_ids": parse_ids(row["relevant_passage_ids"]),
        })
    return examples


def read_corpus_stats(index_dir: str):
    corpus_path = os.path.join(index_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}")
    n_lines = 0
    doc_ids = set()
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_lines += 1
            doc_ids.add(row.get("doc_id"))
    return {"n_passages": n_lines, "n_doc_ids": len(doc_ids)}, doc_ids


def read_manifest(index_dir: str):
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f)


def main():
    args = setup_args()

    print("=" * 78)
    print("BioASQ local index validation")
    print("=" * 78)
    print(f"index_dir = {args.index_dir}")
    print(f"n = {args.n}   k = {args.k}   seed = {args.seed}   backend = {args.backend}")
    print()

    # ---- 1. Sanity ---------------------------------------------------------
    print("-" * 78)
    print("1. SANITY")
    print("-" * 78)
    stats, corpus_doc_ids = read_corpus_stats(args.index_dir)
    manifest = read_manifest(args.index_dir)
    print(f"n_passages (corpus.jsonl lines) = {stats['n_passages']}")
    print(f"n_doc_ids (distinct)            = {stats['n_doc_ids']}")
    if manifest is None:
        print("manifest.json                   NOT FOUND (skipping agreement check)")
    else:
        manifest_n = manifest.get("n_passages")
        agree = manifest_n == stats["n_passages"]
        print(f"manifest.json n_passages        = {manifest_n}  ({'MATCH' if agree else 'MISMATCH vs corpus.jsonl line count!'})")
        print(f"manifest.json (full)            = {json.dumps(manifest, indent=2)}")
    print()

    # ---- Load index + BioASQ sample -----------------------------------------
    kwargs = {"backend": args.backend}
    if args.backend == "dense":
        kwargs["dense_model"] = args.dense_model
    svc = LocalIndexSearchService(args.index_dir, **kwargs)

    print("Loading rag-mini-bioasq QA test split...")
    examples = load_bioasq_qa()
    print(f"  {len(examples)} total QA examples.")
    rng = random.Random(args.seed)
    n_sample = min(args.n, len(examples))
    sampled = rng.sample(examples, n_sample)
    print(f"  sampled {n_sample} examples with seed={args.seed}.")
    print()

    # ---- 2. Gold coverage (data-quality ceiling) ----------------------------
    total_gold = 0
    surviving_gold = 0
    for ex in sampled:
        total_gold += len(ex["gold_ids"])
        surviving_gold += sum(1 for i in ex["gold_ids"] if i in corpus_doc_ids)
    coverage = surviving_gold / total_gold if total_gold else 0.0

    print("-" * 78)
    print("2. GOLD-ID COVERAGE (fraction of relevant_passage_ids that survived the null-passage filter)")
    print("-" * 78)
    print(f"coverage = {coverage:.3f}  ({surviving_gold}/{total_gold} gold references)")
    print("   This is a DATA-QUALITY ceiling, not a retrieval-quality number -- see build script")
    print("   CAVEAT. Section 3's recall denominator only counts questions with >=1 surviving id.")
    print()

    # ---- 3. Doc-id recall@k --------------------------------------------------
    corpus_doc_ids_list = sorted(corpus_doc_ids)
    gold_hits = 0
    random_hits = 0
    n_eligible = 0
    latencies = []
    dump_rows = []

    for ex in sampled:
        surviving = [i for i in ex["gold_ids"] if i in corpus_doc_ids]
        if not surviving:
            continue
        n_eligible += 1

        t0 = time.time()
        results = svc.search(ex["problem"], max_results=args.k)
        latencies.append(time.time() - t0)
        retrieved_ids = {r["title"].removeprefix("PubMed passage ") for r in results}
        retrieved_ids = {int(x) for x in retrieved_ids if x.isdigit()}

        gold_hit = bool(set(surviving) & retrieved_ids)
        gold_hits += int(gold_hit)

        random_ids = set(rng.sample(corpus_doc_ids_list, min(len(surviving), len(corpus_doc_ids_list))))
        random_hits += int(bool(random_ids & retrieved_ids))

        if len(dump_rows) < args.n_dump:
            dump_rows.append({
                "example_id": ex["example_id"],
                "question": ex["problem"][:220],
                "surviving_gold_ids": surviving,
                "gold_hit": gold_hit,
                "top3_retrieved_ids": sorted(retrieved_ids)[:3] if retrieved_ids else list(retrieved_ids)[:3],
            })

    gold_recall = gold_hits / n_eligible if n_eligible else 0.0
    random_recall = random_hits / n_eligible if n_eligible else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("-" * 78)
    print(f"3. GOLD DOC-ID RECALL@{args.k}  [PRIMARY METRIC -- exact id match, no fuzzy matching at all]")
    print("-" * 78)
    print(f"eligible questions (>=1 surviving gold id) = {n_eligible}/{n_sample}")
    print(f"gold recall@{args.k}     = {gold_recall:.3f}  ({gold_hits}/{n_eligible})")
    print(f"random baseline@{args.k} = {random_recall:.3f}  ({random_hits}/{n_eligible})")
    print(f"lift over random      = {gold_recall - random_recall:+.3f}")
    print()

    print("-" * 78)
    print("4. LATENCY")
    print("-" * 78)
    print(f"mean seconds per search() call = {mean_latency:.4f}s over {len(latencies)} calls")
    print()

    print("-" * 78)
    print(f"5. EYEBALL DUMP ({len(dump_rows)} sampled examples)")
    print("-" * 78)
    for row in dump_rows:
        print(f"[{row['example_id']}]")
        print(f"  question: {row['question']}")
        print(f"  surviving gold ids: {row['surviving_gold_ids']}  (hit={row['gold_hit']})")
        print(f"  top-3 retrieved ids: {row['top3_retrieved_ids']}")
        print()

    print("=" * 78)
    print("Done.")
    print("=" * 78)


if __name__ == "__main__":
    main()
