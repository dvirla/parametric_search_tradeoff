"""
LLM-judge semantic-entropy clustering for the PLAIN baseline condition (no cue),
within a single model's own N repeated runs per example (never across models).

Companion to cluster_cues_llm_judge.py: reuses the exact same clusterer
(gpt-oss:120b), prompt, and entropy formula, importing the shared pieces
directly so the two can never silently drift apart. Targets the plain
condition's un-cued filenames (results/*_parametric/<model>/<prefix>_<tag>_run_N.json
-> ..._<tag>_llm_clusters(.json|_5run.json)), which cluster_cues_llm_judge.py's
CUES list deliberately excludes.

Resumable the same two ways as cluster_cues_llm_judge.py: per-combo (skips a
combo once its output file exists for that n_runs) and per-example (a
.gradecache.jsonl cache next to the output).

Usage:
  uv run python scripts/cluster_plain_llm_judge.py --workers 8                # 3-run (default)
  uv run python scripts/cluster_plain_llm_judge.py --workers 8 --n-runs 5     # 5-run
  uv run python scripts/cluster_plain_llm_judge.py --workers 8 --best-available  # 5-run if ready else 3-run
"""
import argparse
import asyncio
import json
import os
import sys

REPO = os.environ.get("REPO_ROOT", "/home/dvirla/parametric_search_tradeoff")
sys.path.append(REPO)
sys.path.append(os.path.join(REPO, "scripts"))

from cluster_cues_llm_judge import (  # noqa: E402
    DATASETS, SLUG_TO_TAG, _out_suffix, build_clusterer, cluster_all,
)


def load_runs(model_dir, file_prefix, tag, n_runs: int):
    files = [os.path.join(model_dir, f"{file_prefix}_{tag}_run_{i}.json") for i in range(1, n_runs + 1)]
    runs = [json.load(open(f)) for f in files]
    by_id = {}
    for r_idx, run in enumerate(runs):
        for ex in run:
            eid = ex["example_id"]
            by_id.setdefault(eid, {"problem": ex["problem"], "correct_answer": ex["correct_answer"], "responses": {}})
            by_id[eid]["responses"][r_idx] = ex["sampler_response"]
    ids = sorted(by_id.keys(), key=lambda x: str(x))
    ids = [i for i in ids if len(by_id[i]["responses"]) == n_runs]
    return by_id, ids


def _combo_ready(model_dir, file_prefix, tag, n_runs, target, min_frac):
    suffix = _out_suffix(n_runs)
    out_path = os.path.join(model_dir, f"{file_prefix}_{tag}_llm_clusters{suffix}.json")
    if os.path.exists(out_path):
        return None
    files = [os.path.join(model_dir, f"{file_prefix}_{tag}_run_{i}.json") for i in range(1, n_runs + 1)]
    if not all(os.path.exists(f) for f in files):
        return None
    try:
        counts = [len(json.load(open(f))) for f in files]
    except Exception:
        return None
    if all(c >= target * min_frac for c in counts):
        return counts
    return None


def discover_ready_combos(n_runs: int, min_frac: float = 0.95):
    ready = []
    for ds, cfg in DATASETS.items():
        base = os.path.join(REPO, cfg["result_dir"])
        if not os.path.isdir(base):
            continue
        for model_slug in sorted(os.listdir(base)):
            tag = SLUG_TO_TAG.get(model_slug)
            if not tag:
                continue
            model_dir = os.path.join(base, model_slug)
            counts = _combo_ready(model_dir, cfg["file_prefix"], tag, n_runs, cfg["target"], min_frac)
            if counts is not None:
                ready.append((ds, model_slug, tag, counts))
    return ready


def discover_ready_combos_best_available(min_frac: float = 0.95):
    ready = []
    for ds, cfg in DATASETS.items():
        base = os.path.join(REPO, cfg["result_dir"])
        if not os.path.isdir(base):
            continue
        for model_slug in sorted(os.listdir(base)):
            tag = SLUG_TO_TAG.get(model_slug)
            if not tag:
                continue
            model_dir = os.path.join(base, model_slug)
            other_5run_done = os.path.exists(os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_llm_clusters_5run.json"))
            other_3run_done = os.path.exists(os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_llm_clusters.json"))
            if other_5run_done or other_3run_done:
                continue
            counts5 = _combo_ready(model_dir, cfg["file_prefix"], tag, 5, cfg["target"], min_frac)
            if counts5 is not None:
                ready.append((ds, model_slug, tag, 5, counts5))
                continue
            counts3 = _combo_ready(model_dir, cfg["file_prefix"], tag, 3, cfg["target"], min_frac)
            if counts3 is not None:
                ready.append((ds, model_slug, tag, 3, counts3))
    return ready


async def process_combo(ds, model_slug, tag, workers, n_runs: int):
    cfg = DATASETS[ds]
    model_dir = os.path.join(REPO, cfg["result_dir"], model_slug)
    by_id, ids = load_runs(model_dir, cfg["file_prefix"], tag, n_runs)
    print(f"\n=== {ds} / {model_slug} / plain: {len(ids)} examples with complete {n_runs}/{n_runs} runs ===")

    examples = [
        {"id": eid, "question": by_id[eid]["problem"],
         "answers": [by_id[eid]["responses"][i] for i in range(n_runs)]}
        for eid in ids
    ]

    suffix = _out_suffix(n_runs)
    clusterer, max_chars = build_clusterer()
    cache_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_llm_clusters{suffix}.gradecache.jsonl")
    results = await cluster_all(clusterer, max_chars, examples, cache_path, workers)

    out = []
    for eid in ids:
        ent, cids = results[eid]
        out.append({
            "example_id": eid,
            "problem": by_id[eid]["problem"],
            "correct_answer": by_id[eid]["correct_answer"],
            "semantic_entropy": ent,
            "cluster_ids": cids,
        })
    out_path = os.path.join(model_dir, f"{cfg['file_prefix']}_{tag}_llm_clusters{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    mean_ent = sum(r["semantic_entropy"] for r in out) / max(len(out), 1)
    n_zero = sum(1 for r in out if r["semantic_entropy"] == 0.0)
    print(f"    wrote {out_path}  (mean entropy={mean_ent:.3f}, {n_zero}/{len(out)} zero-entropy)")


async def main_async(args):
    if args.best_available:
        ready = discover_ready_combos_best_available()
        print(f"Found {len(ready)} ready (dataset, model) combos not yet clustered (best-available: 5-run else 3-run):")
        for ds, model_slug, tag, n_runs, counts in ready:
            print(f"  {ds:8s} {model_slug:26s} n_runs={n_runs}  {counts}")
        if args.dry_run:
            return
        for ds, model_slug, tag, n_runs, counts in ready:
            await process_combo(ds, model_slug, tag, args.workers, n_runs)
        return

    ready = discover_ready_combos(args.n_runs)
    print(f"Found {len(ready)} ready (dataset, model) combos not yet clustered (n_runs={args.n_runs}):")
    for ds, model_slug, tag, counts in ready:
        print(f"  {ds:8s} {model_slug:26s} {counts}")
    if args.dry_run:
        return
    for ds, model_slug, tag, counts in ready:
        await process_combo(ds, model_slug, tag, args.workers, args.n_runs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-runs", type=int, default=3)
    ap.add_argument("--best-available", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
