"""
Materialize a fixed, stratified, NESTED subset of HotpotQA for the cue-sensitivity experiments.

Why a materialized file instead of `--dataset hotpotqa --num_examples 50`:
  * `EvaluationService` samples with `random.Random(seed).sample(...)`, which is deterministic
    across conditions but NOT nested -- a later 300-example run draws a different set, so the
    50 rollouts already paid for can't be reused, and the two tiers aren't comparable.
  * It's also unstratified. The distractor VALIDATION split is entirely level='hard' (easy/medium
    exist only in train), so the one structural axis is `type`: 5918 bridge / 1487 comparison
    (79.9% / 20.1%). We hold that ratio at every prefix.
  * Materializing lets us carry per-example fields the loader would otherwise drop -- `type`, and
    `answer_is_boolean` for the ~6.2% yes/no answers, which are the ones offline regex/EM grading
    handles worst (search-call analysis keeps them; accuracy analysis can report with/without).

NESTING: the emitted order is such that EVERY prefix is ~type-balanced, so
    hotpotqa_50.jsonl  == first 50  rows of hotpotqa_500.jsonl
    hotpotqa_300.jsonl == first 300 rows of hotpotqa_500.jsonl
Growing the pilot to a bigger tier therefore only runs the ids you haven't run yet (the driver's
seed_reuse splices the smaller tier's results into the bigger tier's file first).

Writes (all gitignored -- rsync to the remote alongside data/hotpotqa_index):
    data/hotpotqa_50.jsonl
    data/hotpotqa_300.jsonl
    data/hotpotqa_500.jsonl
    data/hotpotqa_subset_manifest.json

Usage:
    uv run python scripts/build_hotpotqa_subset.py
    uv run python scripts/build_hotpotqa_subset.py --tiers 50 300 500 --seed 0
"""

import os
import sys
import json
import random
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset


def setup_args():
    p = argparse.ArgumentParser(description="Build the nested, type-stratified HotpotQA subsets.")
    p.add_argument("--out-dir", default="data", help="Directory to write hotpotqa_<n>.jsonl into.")
    p.add_argument("--tiers", type=int, nargs="+", default=[50, 300, 500],
                   help="Nested tier sizes; the largest is sampled and the rest are its prefixes.")
    p.add_argument("--split", default="validation", choices=["train", "validation"],
                   help="Must match the split build_hotpotqa_index.py pooled its corpus from.")
    p.add_argument("--seed", type=int, default=0, help="Sampling seed (kept in the manifest).")
    return p.parse_args()


def interleave_stratified(groups: dict[str, list], rng: random.Random) -> list:
    """Order examples so every prefix preserves the population type-ratio.

    Emits from each group at a rate proportional to its share, by giving each group a running
    'credit' and always emitting from whichever group is furthest ahead of its quota. This keeps
    e.g. a 50-row prefix at ~40 bridge / ~10 comparison rather than whatever a plain shuffle gives.
    """
    for g in groups.values():
        rng.shuffle(g)
    total = sum(len(g) for g in groups.values())
    share = {k: len(g) / total for k, g in groups.items()}
    emitted = {k: 0 for k in groups}
    cursor = {k: 0 for k in groups}
    order = []
    for i in range(total):
        # Pick the group with the largest deficit against its quota at this position.
        best, best_deficit = None, None
        for k, g in groups.items():
            if cursor[k] >= len(g):
                continue
            deficit = (i + 1) * share[k] - emitted[k]
            if best_deficit is None or deficit > best_deficit:
                best, best_deficit = k, deficit
        order.append(groups[best][cursor[best]])
        cursor[best] += 1
        emitted[best] += 1
    return order


def main():
    args = setup_args()
    tiers = sorted(args.tiers)
    largest = tiers[-1]

    print(f"Loading hotpotqa/hotpot_qa (distractor, {args.split})...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)

    levels = set(ds["level"])
    if levels != {"hard"}:
        print(f"NOTE: split carries levels {sorted(levels)} -- stratifying on `type` only "
              f"(the validation split is uniformly 'hard').")

    rows = []
    for r in ds:
        answer = str(r["answer"]).strip()
        rows.append({
            "example_id": r["id"],
            "problem": r["question"],
            "gold answer": answer,
            "type": r["type"],
            "answer_is_boolean": answer.lower() in ("yes", "no"),
        })

    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["type"], []).append(r)
    print("population by type: " + ", ".join(
        f"{k}={len(v)} ({100*len(v)/len(rows):.1f}%)" for k, v in sorted(groups.items())))

    rng = random.Random(args.seed)
    # Draw the largest tier stratified, then order it so smaller tiers are balanced prefixes.
    drawn: dict[str, list] = {}
    alloc = {k: round(largest * len(v) / len(rows)) for k, v in groups.items()}
    diff = largest - sum(alloc.values())
    if diff:
        alloc[max(alloc, key=lambda k: len(groups[k]))] += diff
    for k, g in groups.items():
        drawn[k] = rng.sample(g, alloc[k])
    ordered = interleave_stratified(drawn, rng)
    assert len(ordered) == largest

    os.makedirs(args.out_dir, exist_ok=True)
    manifest = {"source_dataset": "hotpotqa/hotpot_qa (distractor)", "split": args.split,
                "seed": args.seed, "tiers": {}, "nested": True}
    for n in tiers:
        subset = ordered[:n]
        path = os.path.join(args.out_dir, f"hotpotqa_{n}.jsonl")
        with open(path, "w") as f:
            for r in subset:
                f.write(json.dumps(r) + "\n")
        by_type = {}
        for r in subset:
            by_type[r["type"]] = by_type.get(r["type"], 0) + 1
        n_bool = sum(1 for r in subset if r["answer_is_boolean"])
        manifest["tiers"][str(n)] = {"path": path, "by_type": by_type, "n_boolean": n_bool}
        print(f"wrote {path}: n={n}, by_type={by_type}, yes/no answers={n_bool}")

    mpath = os.path.join(args.out_dir, "hotpotqa_subset_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {mpath}")


if __name__ == "__main__":
    main()
