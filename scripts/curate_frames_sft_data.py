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

from __future__ import annotations  # defer annotation eval so `X | None` works on py<3.10

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


def plain_reference(cond_map: dict[str, list[dict]], require_correct: bool = False) -> float | None:
    """Median search_calls of the plain-original rollouts used as the closeness reference.

    Default: median over CORRECT plain rollouts, falling back to all plain rollouts when
    none are correct. With require_correct=True there is NO fallback: a question with no
    correct plain rollout returns None and is dropped from curation entirely (cleaner signal).
    """
    plain = cond_map.get(PLAIN_CONDITION, [])
    correct = [r["search_calls"] for r in plain if r["is_correct"]]
    if require_correct:
        pool = correct
    else:
        pool = correct or [r["search_calls"] for r in plain]
    return statistics.median(pool) if pool else None


def is_test(example_id: str, test_pct: int) -> bool:
    """Deterministic ~test_pct% question-level hold-out."""
    h = int(hashlib.md5(str(example_id).encode()).hexdigest(), 16)
    return (h % 100) < test_pct


def kept_for_example(cond_map, ref, threshold, max_per_condition):
    """Return list of kept rollout records for one (train) example."""
    # No valid plain reference (e.g. --require-correct-plain-ref and no correct plain
    # rollout) -> the question contributes nothing: no anchor, no cue rollouts.
    if ref is None:
        return []
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


CONDITION_ORDER = ["verbose_plain", "verbose_polite", "terse_plain", "verbose_natural",
                   "verbose_elaborate", "verbose_query", "verbose_direct"]


def describe_rollouts(rollouts):
    """Descriptive summary of the raw rollouts (split-independent), printed before curation."""
    by = defaultdict(list)
    for r in rollouts:
        by[r["condition"]].append(r)
    qids = {str(r["example_id"]) for r in rollouts}
    n_corr = sum(1 for r in rollouts if r["is_correct"])

    print("=== ROLLOUTS: TOTALS ===")
    print(f"rollouts={len(rollouts)}  questions={len(qids)}  conditions={len(by)}  "
          f"overall_correct={n_corr} ({100.0*n_corr/max(1,len(rollouts)):.1f}%)")

    conds = [c for c in CONDITION_ORDER if c in by] + sorted(c for c in by if c not in CONDITION_ORDER)
    print("\n=== PER CONDITION ===")
    print(f"{'condition':18} {'n':>6} {'%corr':>7} {'sc_med':>7} {'sc_mean':>8} {'sc_max':>7} {'%sc=0':>7}")
    for c in conds:
        rs = by[c]
        sc = [r["search_calls"] for r in rs]
        ncorr = sum(1 for r in rs if r["is_correct"])
        zero = sum(1 for x in sc if x == 0)
        print(f"{c:18} {len(rs):>6} {100.0*ncorr/len(rs):>6.1f}% "
              f"{statistics.median(sc):>7.1f} {statistics.mean(sc):>8.2f} {max(sc):>7} "
              f"{100.0*zero/len(rs):>6.1f}%")

    # Plain reference availability (curation keeps cue rollouts close to this per question).
    plain_by_q = defaultdict(list)
    for r in by.get(PLAIN_CONDITION, []):
        plain_by_q[str(r["example_id"])].append(r)
    q_corr_plain = sum(1 for rs in plain_by_q.values() if any(x["is_correct"] for x in rs))
    print("\n=== PLAIN REFERENCE (verbose_plain) ===")
    print(f"questions with a plain rollout: {len(plain_by_q)} | "
          f"with a CORRECT plain rollout: {q_corr_plain} "
          f"({100.0*q_corr_plain/max(1,len(plain_by_q)):.1f}%)")

    print("\n=== CORRECT-ROLLOUT POOL (ceiling before closeness filter) ===")
    tot = 0
    for c in conds:
        k = sum(1 for r in by[c] if r["is_correct"])
        tot += k
        print(f"  {c:18} {k:>5}")
    print(f"  {'TOTAL':18} {tot:>5}")

    buckets = defaultdict(int)
    for r in rollouts:
        s = r["search_calls"]
        buckets[str(s) if s <= 5 else "6+"] += 1
    print("\n=== SEARCH-CALL DISTRIBUTION (all rollouts) ===")
    for k in ["0", "1", "2", "3", "4", "5", "6+"]:
        v = buckets.get(k, 0)
        print(f"  {k:>3} calls: {v:>6} ({100.0*v/max(1,len(rollouts)):.1f}%)")
    print()


def run_stats(grouped, test_pct, require_correct=False, max_thr=6):
    """Print kept-rollout yield vs threshold (train questions only)."""
    train_ids = [e for e in grouped if not is_test(e, test_pct)]
    test_ids = [e for e in grouped if is_test(e, test_pct)]
    refs = {e: plain_reference(grouped[e], require_correct) for e in train_ids}
    n_noref = sum(1 for e in train_ids if refs[e] is None)
    print(f"questions: {len(grouped)} total | {len(train_ids)} train | {len(test_ids)} test "
          f"(~{test_pct}% held out)")
    print(f"require_correct_plain_ref={require_correct} | "
          f"train questions dropped (no valid plain ref): {n_noref} "
          f"| usable train questions: {len(train_ids) - n_noref}")

    # Anchors count only for questions that have a valid reference (kept in curation).
    anchors = sum(len([r for r in grouped[e].get(PLAIN_CONDITION, []) if r["is_correct"]])
                  for e in train_ids if refs[e] is not None)
    print(f"\nplain-anchor correct rollouts (train): {anchors}")
    print("\nthreshold | kept cue rollouts | total SFT rollouts | cue by condition")
    conds = sorted({c for e in train_ids for c in grouped[e] if c != PLAIN_CONDITION})
    for thr in range(0, max_thr + 1):
        by_cond = defaultdict(int)
        cue_total = 0
        for e in train_ids:
            ref = refs[e]
            if ref is None:
                continue
            for cond in conds:
                cand = [r for r in grouped[e].get(cond, []) if r["is_correct"]]
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
    p.add_argument("--require-correct-plain-ref", action="store_true",
                   help="Drop questions with no CORRECT verbose_plain rollout (no all-plain fallback "
                        "reference); yields a cleaner closeness signal on a smaller set.")
    p.add_argument("--conditions", nargs="+", default=None,
                   help="Restrict curation to this subset of conditions (e.g. to build a controlled "
                        "N-condition arm from a superset rollouts.jsonl that has more collected than "
                        "you want in this particular training set). verbose_plain is always required "
                        "regardless of whether it's listed, since it's the closeness reference/anchor. "
                        "Default: use every condition present in --rollouts.")
    p.add_argument("--stats", action="store_true", help="Print yield vs threshold and exit (no write).")
    args = p.parse_args()

    rollouts = load_rollouts(args.rollouts)
    if args.conditions:
        wanted = set(args.conditions) | {PLAIN_CONDITION}
        before = len(rollouts)
        rollouts = [r for r in rollouts if r["condition"] in wanted]
        print(f"--conditions filter: kept {len(rollouts)}/{before} rollouts "
              f"(conditions={sorted(wanted)})")
    grouped = group_by_example(rollouts)
    print(f"Loaded {len(rollouts)} raw rollouts across {len(grouped)} questions.")

    if args.stats:
        describe_rollouts(rollouts)
        run_stats(grouped, args.test_pct, require_correct=args.require_correct_plain_ref)
        return

    os.makedirs(args.output_dir, exist_ok=True)
    test_ids = sorted(e for e in grouped if is_test(e, args.test_pct))
    train_ids = [e for e in grouped if not is_test(e, args.test_pct)]

    sft_path = os.path.join(args.output_dir, SFT_FILENAME)
    n_written = 0
    n_dropped_q = 0
    per_cond = defaultdict(int)
    with open(sft_path, "w") as out:
        for e in train_ids:
            ref = plain_reference(grouped[e], require_correct=args.require_correct_plain_ref)
            if ref is None:
                n_dropped_q += 1
                continue
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
    print(f"Train questions: {len(train_ids)} | used: {len(train_ids) - n_dropped_q} | "
          f"dropped (no valid plain ref): {n_dropped_q} | threshold={args.threshold} | "
          f"require_correct_plain_ref={args.require_correct_plain_ref}")


if __name__ == "__main__":
    main()
