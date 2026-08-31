"""
Phase 1 (no LLM calls) of the modal-answer redirection check: does the cue's
CANONICAL (largest-cluster) answer change relative to the model's own
cue-free baseline, on the same question -- distinct from whether entropy
(the SIZE of the spread) changes, which analyze_entropy_under_cue.py already
tests. Two examples can have identical entropy under a cue yet completely
different modal answers (redirection) or identical modal answers yet
different entropy (narrowing/widening) -- these are orthogonal questions.

This script does the free half: for every example, in both the plain and cue
condition, determine how many distinct clusters exist (n_clusters, derived
from `cluster_ids`) and -- when a majority exists (n_clusters in {1, 2}, i.e.
at least 2 of the 3 no-search rollouts agree) -- extract the majority
cluster's representative response text (the lowest-indexed run in that
cluster, deterministic, not random, so reruns are reproducible) from the raw
per-run rollout files.

Three-way ties (n_clusters == 3, all runs mutually disagree, entropy =
log2(3) bits) have no modal answer to extract -- excluded from the pairwise
text-comparison population, but NOT silently dropped: the plain -> cue
TRANSITION in n_clusters is tracked for every example regardless, because a
question moving from a majority (1 or 2 clusters) under `plain` to a 3-way
tie under the cue is itself a serious finding -- the model had a working
consensus answer and the cue broke it, even though there's no single "new"
answer to compare against the old one. The reverse (3-way tie -> majority)
is tracked too, for completeness.

Phase 2 (the actual judging) is NOT done here -- it was implemented independently as
scripts/compare_modal_plain_vs_cue.py (same equivalence criterion, same gpt-oss:120b
judge, its own modal-answer selection), consumed by analyze_modal_answer_shift_judged.py.
This script's job is narrower and doesn't overlap with that pipeline: it only reports
how many examples WOULD be eligible for judging per cell (a sanity-check number, matched
against compare_modal_plain_vs_cue.py's actual output row counts), and the cluster-count
TRANSITION table below, which nothing else computes.

Usage:
    uv run python scripts/analyze_modal_answer_shift.py                # 3-run (default)
    uv run python scripts/analyze_modal_answer_shift.py --n-runs 5     # 5-run
"""
import argparse
import csv
import glob
import json
import os
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "modal_answer_shift")
os.makedirs(OUT_DIR, exist_ok=True)

TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
        "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b",
        "nemotron-cascade-2_30b": "nemotron-cascade-2:30b", "qwen3.5_122b": "qwen3.5:122b"}

DATASETS = {
    "frames": dict(dir="results/frames_parametric", prefix="frames-cues"),
    "medqa": dict(dir="results/medqa_parametric", prefix="medqa-500"),
}


def load_clusters(path):
    data = json.load(open(path))
    return {row["example_id"]: row["cluster_ids"] for row in data}


def load_responses(model_dir, prefix, tag, cue, n_runs=3):
    """example_id -> [response_run1, response_run2, response_run3], in run order."""
    out = {}
    for run in range(1, n_runs + 1):
        fname = f"{prefix}_no_search_{tag}_{cue}_run_{run}.json" if cue else f"{prefix}_no_search_{tag}_run_{run}.json"
        path = os.path.join(model_dir, fname)
        if not os.path.exists(path):
            return None
        for row in json.load(open(path)):
            out.setdefault(row["example_id"], []).append(row.get("sampler_response") or "")
    return out


def majority_representative(cluster_ids, responses):
    """Return (n_clusters, representative_text_or_None). Representative = the
    response from the LOWEST-INDEXED run belonging to the majority cluster
    (deterministic). None if no majority (every run in its own distinct cluster)."""
    n_clusters = len(set(cluster_ids))
    counts = Counter(cluster_ids)
    top_id, top_count = counts.most_common(1)[0]
    if top_count < 2:
        return n_clusters, None
    rep_idx = next(i for i, cid in enumerate(cluster_ids) if cid == top_id)
    return n_clusters, responses[rep_idx]


def main(n_runs=3):
    suffix = "" if n_runs == 3 else f"_{n_runs}run"
    transition_rows = []

    for ds, cfg in DATASETS.items():
        for model, tag in TAGS.items():
            model_dir = os.path.join(REPO, cfg["dir"], model)
            plain_cluster_path = os.path.join(model_dir, f"{cfg['prefix']}_no_search_{tag}_llm_clusters{suffix}.json")
            if not os.path.exists(plain_cluster_path):
                continue
            clusters_plain = load_clusters(plain_cluster_path)
            responses_plain = load_responses(model_dir, cfg["prefix"], tag, None, n_runs=n_runs)
            if responses_plain is None:
                continue

            cue_cluster_paths = glob.glob(os.path.join(model_dir, f"{cfg['prefix']}_no_search_{tag}_*_llm_clusters{suffix}.json"))
            for cue_path in sorted(cue_cluster_paths):
                fname = os.path.basename(cue_path)
                marker = f"{cfg['prefix']}_no_search_{tag}_"
                tail = f"_llm_clusters{suffix}.json"
                cue = fname[len(marker):-len(tail)]
                clusters_cue = load_clusters(cue_path)
                responses_cue = load_responses(model_dir, cfg["prefix"], tag, cue, n_runs=n_runs)
                if responses_cue is None:
                    continue

                common = sorted(set(clusters_plain) & set(clusters_cue) & set(responses_plain) & set(responses_cue), key=str)
                if len(common) < 20:
                    continue

                transition_counts = Counter()
                consensus_breakdown = 0   # plain had a majority (n_clusters < n_runs), cue is a full N-way tie
                consensus_formed = 0      # plain was a full N-way tie, cue has a majority
                n_eligible = 0

                for eid in common:
                    n_plain, rep_plain = majority_representative(clusters_plain[eid], responses_plain[eid])
                    n_cue, rep_cue = majority_representative(clusters_cue[eid], responses_cue[eid])
                    transition_counts[(n_plain, n_cue)] += 1
                    if n_plain < n_runs and n_cue == n_runs:
                        consensus_breakdown += 1
                    if n_plain == n_runs and n_cue < n_runs:
                        consensus_formed += 1
                    if rep_plain is not None and rep_cue is not None:
                        n_eligible += 1

                n = len(common)
                transition_rows.append(dict(
                    dataset=ds, model=model, cue=cue, n=n,
                    n_eligible_for_judge=n_eligible,
                    pct_eligible=round(100 * n_eligible / n, 1),
                    consensus_breakdown=consensus_breakdown,
                    pct_consensus_breakdown=round(100 * consensus_breakdown / n, 1),
                    consensus_formed=consensus_formed,
                    pct_consensus_formed=round(100 * consensus_formed / n, 1),
                    **{f"plain{p}_cue{c}": transition_counts.get((p, c), 0)
                       for p in range(1, n_runs + 1) for c in range(1, n_runs + 1)},
                ))

    if not transition_rows:
        print("No cells found.")
        return

    trans_path = os.path.join(OUT_DIR, f"cluster_count_transitions{suffix}.csv")
    with open(trans_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(transition_rows[0].keys()))
        w.writeheader()
        w.writerows(transition_rows)
    print(f"wrote {trans_path}  ({len(transition_rows)} rows)\n")

    print(f"=== cluster-count (consensus structure) transitions, plain -> cue (n_runs={n_runs}) ===")
    print(f"n_clusters: 1 = all {n_runs} runs agree, ..., {n_runs} = every run in its own cluster (no modal answer)\n")
    for r in sorted(transition_rows, key=lambda r: (r["dataset"], r["model"], r["cue"])):
        flag = " <-- consensus breakdown present" if r["consensus_breakdown"] > 0 else ""
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:22s} n={r['n']:4d}  "
              f"eligible for judge: {r['n_eligible_for_judge']}/{r['n']} ({r['pct_eligible']}%)  "
              f"breakdown(majority->{n_runs}way): {r['consensus_breakdown']} ({r['pct_consensus_breakdown']}%)  "
              f"formed({n_runs}way->majority): {r['consensus_formed']} ({r['pct_consensus_formed']}%){flag}")

    total_eligible = sum(r["n_eligible_for_judge"] for r in transition_rows)
    total_breakdown = sum(r["consensus_breakdown"] for r in transition_rows)
    print(f"\nTotal judge-eligible population across all {len(transition_rows)} cells: {total_eligible} pairs "
          f"(cross-check against scripts/compare_modal_plain_vs_cue.py's actual output row counts).")
    print(f"Total consensus-breakdown examples (majority -> {n_runs}-way tie) across all cells: {total_breakdown}.")
    print("\nActual Phase-2 judging: scripts/compare_modal_plain_vs_cue.py -> "
          "scripts/analyze_modal_answer_shift_judged.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-runs", type=int, default=3)
    args = ap.parse_args()
    main(n_runs=args.n_runs)
