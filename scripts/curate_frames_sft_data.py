"""
Curate FRAMES on-policy rollouts into a cue-robustness SFT set + held-out split.

Reads the raw rollouts produced by create_frames_sft_data.py and keeps, per
question, the rollouts that are (a) correct and (b) whose search-call count is
close to the SAME question's plain-original (verbose_plain) reference. Correct
plain-original rollouts are always kept as a neutral anchor. A deterministic
~20% question-level split is held out for the robustness eval and never written
to the SFT file.

Closeness metric = absolute search-call distance from the per-question plain
reference (median search_calls over correct plain rollouts). Run --stats first
to see the kept-rollout yield as a function of --threshold, then pick it.

Output:
    <output-dir>/procedure1_onpolicy_sft_rewired.jsonl   (ChatML, train questions only)
    <output-dir>/test_ids.json                            (held-out example_ids)

Usage:
    # 1) inspect yield vs threshold
    uv run python scripts/curate_frames_sft_data.py --rollouts data/sft/frames/rollouts.jsonl --stats
    # 2) emit the SFT file + split
    uv run python scripts/curate_frames_sft_data.py --rollouts data/sft/frames/rollouts.jsonl \
        --threshold 1 --output-dir data/sft/frames
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

PLAIN_CONDITION = "verbose_plain"
SFT_FILENAME = "procedure1_onpolicy_sft_rewired.jsonl"  # matches train_sft.py _ARM_FILES


def load_rollouts(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_by_example(rollouts: list[dict]) -> dict[str, dict[str, list[dict]]]:
    """example_id -> condition -> [rollout, ...]"""
    g: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rollouts:
        g[str(r["example_id"])][r["condition"]].append(r)
    return g


def plain_reference(cond_map: dict[str, list[dict]]) -> float | None:
    """Median search_calls over CORRECT plain rollouts (fallback: all plain)."""
    plain = cond_map.get(PLAIN_CONDITION, [])
    correct = [r["search_calls"] for r in plain if r["is_correct"]]
    pool = correct or [r["search_calls"] for r in plain]
    return statistics.median(pool) if pool else None


def is_test(example_id: str, test_pct: int) -> bool:
    """Deterministic ~test_pct% question-level hold-out."""
    h = int(hashlib.md5(str(example_id).encode()).hexdigest(), 16)
    return (h % 100) < test_pct


def kept_for_example(cond_map, ref, threshold, max_per_condition):
    """Return list of kept rollout records for one (train) example."""
    kept = []
    # Anchor: correct plain-original rollouts.
    plain_correct = [r for r in cond_map.get(PLAIN_CONDITION, []) if r["is_correct"]]
    if max_per_condition:
        plain_correct = plain_correct[:max_per_condition]
    kept.extend(plain_correct)
    # Cue conditions: correct AND search-close to plain reference.
    for cond, rollouts in cond_map.items():
        if cond == PLAIN_CONDITION:
            continue
        cand = [r for r in rollouts if r["is_correct"]]
        if ref is not None:
            cand = [r for r in cand if abs(r["search_calls"] - ref) <= threshold]
        if max_per_condition:
            cand = cand[:max_per_condition]
        kept.extend(cand)
    return kept


def run_stats(grouped, test_pct, max_thr=6):
    """Print kept-rollout yield vs threshold (train questions only)."""
    train_ids = [e for e in grouped if not is_test(e, test_pct)]
    test_ids = [e for e in grouped if is_test(e, test_pct)]
    n_noref = sum(1 for e in train_ids if plain_reference(grouped[e]) is None)
    print(f"questions: {len(grouped)} total | {len(train_ids)} train | {len(test_ids)} test "
          f"(~{test_pct}% held out)")
    print(f"train questions with NO plain reference: {n_noref}")

    # Correct anchors are threshold-independent.
    anchors = sum(len([r for r in grouped[e].get(PLAIN_CONDITION, []) if r["is_correct"]])
                  for e in train_ids)
    print(f"\nplain-anchor correct rollouts (train): {anchors}")
    print("\nthreshold | kept cue rollouts | total SFT rollouts | cue by condition")
    conds = sorted({c for e in train_ids for c in grouped[e] if c != PLAIN_CONDITION})
    for thr in range(0, max_thr + 1):
        by_cond = defaultdict(int)
        cue_total = 0
        for e in train_ids:
            ref = plain_reference(grouped[e])
            for cond in conds:
                cand = [r for r in grouped[e].get(cond, []) if r["is_correct"]]
                if ref is not None:
                    cand = [r for r in cand if abs(r["search_calls"] - ref) <= thr]
                by_cond[cond] += len(cand)
                cue_total += len(cand)
        dist = " ".join(f"{c.replace('verbose_','v_').replace('terse_','t_')}={by_cond[c]}" for c in conds)
        print(f"    {thr:>5} | {cue_total:>17} | {anchors + cue_total:>18} | {dist}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", default="data/sft/frames/rollouts.jsonl")
    p.add_argument("--output-dir", default="data/sft/frames")
    p.add_argument("--threshold", type=int, default=1,
                   help="Max |search_calls - plain_ref| for a cue rollout to be kept.")
    p.add_argument("--test-pct", type=int, default=20, help="Percent of questions held out for eval.")
    p.add_argument("--max-per-condition", type=int, default=None,
                   help="Optional cap on kept rollouts per (question, condition) to avoid over-weighting.")
    p.add_argument("--stats", action="store_true", help="Print yield vs threshold and exit (no write).")
    args = p.parse_args()

    rollouts = load_rollouts(args.rollouts)
    grouped = group_by_example(rollouts)
    print(f"Loaded {len(rollouts)} raw rollouts across {len(grouped)} questions.")

    if args.stats:
        run_stats(grouped, args.test_pct)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    test_ids = sorted(e for e in grouped if is_test(e, args.test_pct))
    train_ids = [e for e in grouped if not is_test(e, args.test_pct)]

    sft_path = os.path.join(args.output_dir, SFT_FILENAME)
    n_written = 0
    per_cond = defaultdict(int)
    with open(sft_path, "w") as out:
        for e in train_ids:
            ref = plain_reference(grouped[e])
            for rec in kept_for_example(grouped[e], ref, args.threshold, args.max_per_condition):
                out.write(json.dumps({"messages": rec["messages"]}) + "\n")
                per_cond[rec["condition"]] += 1
                n_written += 1

    test_path = os.path.join(args.output_dir, "test_ids.json")
    with open(test_path, "w") as f:
        json.dump(test_ids, f, indent=2)

    print(f"\nWrote {n_written} SFT rollouts -> {sft_path}")
    print(f"  by condition: {dict(sorted(per_cond.items()))}")
    print(f"Held out {len(test_ids)} test questions -> {test_path}")
    print(f"Train questions: {len(train_ids)} | threshold={args.threshold}")


if __name__ == "__main__":
    main()
