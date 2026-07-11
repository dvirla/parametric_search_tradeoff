"""
Validate a local MedQA BM25/dense index before running behavioral experiments.

Purpose: prove the corpus (built by scripts/build_medqa_index.py, see
docs/PLAN_medqa_local_index.md) actually supports MedQA retrieval, so a
retrieval hole can never be mistaken for a behavioral effect downstream.

Loads MedQA the same way src/services/qa_eval.py does in its `medqa` branch
(GBaker/MedQA-USMLE-4-options, split=test, question->problem, answer->gold
answer, carrying options + answer_idx), takes a seeded sample of --n
questions, retrieves top-k passages per question via LocalIndexSearchService,
and reports:

  1. Sanity     - n_passages, n_books, manifest agreement with corpus.jsonl.
  2. Coverage   - fraction of gold answers whose text occurs ANYWHERE in the
                  corpus, vs. the same for the three wrong options. This
                  separates "the corpus lacks the content" from "the query
                  was bad" -- the two failures that a plain recall number
                  silently conflates.
  3. Recall@k   - fraction of sampled questions where the gold answer occurs
     (diagnostic) as a substring in the top-k passages retrieved for the raw
                  question text, alongside the same for the wrong options
                  (reusing the SAME retrieved passages, no extra search()).
  4. Latency    - mean seconds per search() call.
  5. Eyeball    - 5 sampled (question, top-3 titles, top-1 snippet head)
                  triples. THE PRIMARY QUALITATIVE SIGNAL.

MEASURED, 2026-07-10 (n=100, k=10, seed=0, bm25, data/medqa_index):
  gold present anywhere in corpus  0.54   |  distractors  0.56
  gold recall@10 (raw vignette)    0.010  |  distractors  0.027
  gold recall@10 (focused query)   0.06   |  (n=50)

READ THIS BEFORE TRUSTING SECTION 3. Substring recall is *phrasing-bound*, not
content-bound, and for MedQA it is uninformative in both directions. Gold
answers are compositional management phrases ("Moxifloxacin and admission to
the medical floor", "Reassure the mother") that a textbook states in different
words; only ~54% appear verbatim anywhere in 98MB of source text. Crucially
the wrong options are drawn from the SAME phrasing distribution and appear at
the SAME rate (56%), so the gold-vs-distractor contrast has NO discriminative
power by construction -- it cannot come out positive even for a perfect index,
and its sign at these floor-level counts (1 gold hit, 8 distractor hits/300) is
noise. Focusing the query barely moves it (0.00 -> 0.06). Section 3 is retained
only as a regression tripwire: if it ever jumps far above these numbers, the
corpus changed. It is NOT a pass criterion, and this script never exits nonzero
on it.

What actually establishes index health: the structural checks in section 1, the
topical plausibility of the eyeball dump in section 5, and -- decisively -- an
end-to-end eval run (`run_qa_eval_experiment.py --dataset medqa
--search-backend local`) showing nonzero sampler_search_calls and non-collapsed
accuracy. See docs/PLAN_medqa_local_index.md.

Usage:
    uv run python scripts/validate_medqa_index.py --index-dir data/medqa_index --n 100 --k 10
    uv run python scripts/validate_medqa_index.py --index-dir /path/to/fixture --n 20 --k 10 --seed 0
"""

import os
import sys
import json
import time
import random
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.services.local_index_search import LocalIndexSearchService

CAVEAT = """\
CAVEAT: Substring recall (section 3) is PHRASING-BOUND, not content-bound, and
for MedQA it is uninformative in both directions. Gold answers are compositional
management phrases a textbook states in other words; the wrong options are drawn
from the same phrasing distribution and appear at the same rate, so the
gold-vs-distractor contrast CANNOT come out positive even for a perfect index.
Measured 2026-07-10: gold present-anywhere 0.54 vs distractors 0.56; gold
recall@10 0.010 vs distractors 0.027 (floor-level counts -> the sign is noise).
Neither a low recall nor a negative contrast is grounds to reject the corpus.
Section 3 is a regression tripwire, not a pass criterion. Index health rests on
section 1 (structure), section 5 (topical plausibility), and an end-to-end eval
run showing nonzero sampler_search_calls with non-collapsed accuracy.\
"""


def setup_args():
    p = argparse.ArgumentParser(
        description="Validate a local MedQA BM25/dense index against the real MedQA-USMLE test set."
    )
    p.add_argument("--index-dir", default="data/medqa_index", help="Index directory (corpus.jsonl + manifest.json).")
    p.add_argument("--n", type=int, default=100, help="Number of sampled MedQA test questions.")
    p.add_argument("--k", type=int, default=10, help="Top-k passages retrieved per search() call.")
    p.add_argument("--seed", type=int, default=0, help="Seed for the question sample.")
    p.add_argument("--backend", choices=["bm25", "dense"], default="bm25",
                   help="Which LocalIndexSearchService backend to query.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2",
                   help="Only used when --backend dense.")
    p.add_argument("--n-dump", type=int, default=5, help="How many sampled examples to print in the eyeball dump.")
    return p.parse_args()


def load_medqa():
    """Load MedQA exactly as src/services/qa_eval.py does in its `medqa` branch."""
    from datasets import load_dataset

    mdf = load_dataset("GBaker/MedQA-USMLE-4-options", split="test").to_pandas()
    mdf = mdf.rename(columns={"question": "problem", "answer": "gold answer"})
    mdf["example_id"] = ["medqa_test_%04d" % i for i in range(len(mdf))]
    return mdf[["example_id", "problem", "gold answer", "options", "answer_idx"]].to_dict("records")


def distractor_options(ex):
    """The three options that are NOT the keyed-correct answer_idx."""
    opts = ex.get("options") or {}
    if hasattr(opts, "items"):
        items = list(opts.items())
    elif isinstance(opts, (list, tuple)):
        items = [(chr(65 + i), v) for i, v in enumerate(opts)]
    else:
        return []
    correct = str(ex.get("answer_idx", "")).strip()
    return [v for k, v in items if str(k).strip() != correct]


def contains_answer(passages, answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return False
    return any(a in (p.get("snippet") or "").lower() for p in passages)


def read_corpus_stats(index_dir: str):
    corpus_path = os.path.join(index_dir, "corpus.jsonl")
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"corpus.jsonl not found at {corpus_path}")
    n_lines = 0
    doc_ids = set()
    titles = set()
    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n_lines += 1
            doc_ids.add(row.get("doc_id"))
            titles.add(row.get("title"))
    return {"n_passages": n_lines, "n_books_by_doc_id": len(doc_ids), "n_books_by_title": len(titles)}


def read_manifest(index_dir: str):
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path) as f:
        return json.load(f)


def main():
    args = setup_args()

    print("=" * 78)
    print("MedQA local index validation")
    print("=" * 78)
    print(f"index_dir = {args.index_dir}")
    print(f"n = {args.n}   k = {args.k}   seed = {args.seed}   backend = {args.backend}")
    print()
    print(CAVEAT)
    print()

    # ---- 1. Sanity ---------------------------------------------------------
    print("-" * 78)
    print("1. SANITY")
    print("-" * 78)
    stats = read_corpus_stats(args.index_dir)
    manifest = read_manifest(args.index_dir)
    print(f"n_passages (corpus.jsonl lines)     = {stats['n_passages']}")
    print(f"n_books (distinct doc_id)           = {stats['n_books_by_doc_id']}")
    print(f"n_books (distinct title, sanity chk)= {stats['n_books_by_title']}")
    if manifest is None:
        print("manifest.json                       NOT FOUND (skipping agreement check)")
    else:
        manifest_n = manifest.get("n_passages")
        agree = manifest_n == stats["n_passages"]
        print(f"manifest.json n_passages            = {manifest_n}  "
              f"({'MATCH' if agree else 'MISMATCH vs corpus.jsonl line count!'})")
        if "n_books" in manifest:
            books_agree = manifest["n_books"] == stats["n_books_by_doc_id"]
            print(f"manifest.json n_books               = {manifest['n_books']}  "
                  f"({'MATCH' if books_agree else 'MISMATCH vs corpus doc_id count!'})")
        print(f"manifest.json (full)                = {json.dumps(manifest, indent=2)}")
    print()

    # ---- Load index + MedQA sample -----------------------------------------
    kwargs = {"backend": args.backend}
    if args.backend == "dense":
        kwargs["dense_model"] = args.dense_model
    svc = LocalIndexSearchService(args.index_dir, **kwargs)

    print(f"Loading MedQA-USMLE test split (GBaker/MedQA-USMLE-4-options)...")
    examples = load_medqa()
    print(f"  {len(examples)} total test examples.")
    rng = random.Random(args.seed)
    n_sample = min(args.n, len(examples))
    sampled = rng.sample(examples, n_sample)
    print(f"  sampled {n_sample} examples with seed={args.seed}.")
    print()

    # ---- 2. Corpus coverage -----------------------------------------------
    # One lowercased blob of every passage. ~98MB for the real corpus; substring
    # scans are fast enough (~ms) and this is the only way to separate "corpus
    # lacks the content" from "the query was bad".
    blob = "\n".join(p["text"].lower() for p in svc.passages)
    gold_cov_hits = 0
    dist_cov_hits = 0
    dist_cov_total = 0
    for ex in sampled:
        if str(ex["gold answer"]).strip().lower() in blob:
            gold_cov_hits += 1
        for d in distractor_options(ex):
            dist_cov_total += 1
            if str(d).strip().lower() in blob:
                dist_cov_hits += 1
    gold_cov = gold_cov_hits / n_sample if n_sample else 0.0
    dist_cov = dist_cov_hits / dist_cov_total if dist_cov_total else 0.0
    del blob

    # ---- 3/4. Recall diagnostic + latency ---------------------------------
    gold_hits = 0
    distractor_hits = 0
    distractor_total = 0
    latencies = []
    dump_rows = []

    for i, ex in enumerate(sampled):
        t0 = time.time()
        results = svc.search(ex["problem"], max_results=args.k)
        latencies.append(time.time() - t0)

        gold = ex["gold answer"]
        gold_hit = contains_answer(results, gold)
        gold_hits += int(gold_hit)

        distractors = distractor_options(ex)
        d_hit_flags = []
        for d in distractors:
            hit = contains_answer(results, d)
            d_hit_flags.append(hit)
            distractor_total += 1
            distractor_hits += int(hit)

        if len(dump_rows) < args.n_dump:
            top3_titles = [r["title"] for r in results[:3]]
            top1_head = (results[0]["snippet"][:200] + "...") if results else "(no hits)"
            dump_rows.append({
                "example_id": ex["example_id"],
                "question_head": ex["problem"][:220].replace("\n", " "),
                "gold_answer": gold,
                "gold_hit": gold_hit,
                "distractor_hits": d_hit_flags,
                "top3_titles": top3_titles,
                "top1_snippet_head": top1_head,
            })

    gold_recall = gold_hits / n_sample if n_sample else 0.0
    distractor_recall = distractor_hits / distractor_total if distractor_total else 0.0
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0

    print("-" * 78)
    print("2. CORPUS COVERAGE (answer text present ANYWHERE in the corpus)")
    print("-" * 78)
    print(f"gold present-anywhere         = {gold_cov:.3f}  ({gold_cov_hits}/{n_sample} questions)")
    print(f"distractor present-anywhere   = {dist_cov:.3f}  ({dist_cov_hits}/{dist_cov_total} option-checks)")
    print("   Gold and distractors are drawn from the same phrasing distribution, so these")
    print("   two rates should be ~equal. They measure how often an option happens to be")
    print("   stated verbatim in a textbook -- NOT whether the corpus covers the topic.")
    print()

    print("-" * 78)
    print(f"3. ANSWER-STRING RECALL@{args.k}  [DIAGNOSTIC / REGRESSION TRIPWIRE ONLY]")
    print("-" * 78)
    print(f"gold recall@{args.k}              = {gold_recall:.3f}  ({gold_hits}/{n_sample} questions)")
    print(f"mean distractor recall@{args.k}   = {distractor_recall:.3f}  ({distractor_hits}/{distractor_total} option-checks)")
    print(f"contrast (gold - distractor) = {gold_recall - distractor_recall:+.3f}")
    print("   This contrast has NO discriminative power for MedQA (see caveat) -- do not")
    print("   read its sign as a verdict. Expected near floor: ~0.01 gold, ~0.03 distractor.")
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
        print(f"  question: {row['question_head']}")
        print(f"  gold answer: {row['gold_answer']!r}  (hit={row['gold_hit']})")
        print(f"  distractor hits: {row['distractor_hits']}")
        print(f"  top-3 titles: {row['top3_titles']}")
        print(f"  top-1 snippet head: {row['top1_snippet_head']!r}")
        print()

    print("=" * 78)
    print("Done. Reminder:")
    print(CAVEAT)
    print("=" * 78)


if __name__ == "__main__":
    main()
