"""
Paired analysis of the HotpotQA cue pilot: search-call counts under plain vs. each cue,
joined on example_id (NOT on problem text -- the cue templates rewrite the prompt, and the
stored `problem` is the raw question, but example_id is the only guaranteed-unique key).

The pilot's decision question is NOT "is the effect significant" -- it is "does this dataset
have usable paired power at all". MedQA's ~95.8% zero-search floor made cue effects
unmeasurable there at ANY n, because pairs that are 0-vs-0 are TIES and ties carry no signal
in a signed-rank test. So the headline numbers here are the tie rate and the observed SD of
the paired difference, which together give the n actually required for a full tier.

Reported per contrast:
  * Wilcoxon signed-rank (zero_method="wilcox": ties dropped, which is the honest treatment)
  * rank-biserial r -- scale-free, so it does not inflate when the baseline sits at a floor
    (this is the statistic that inverted the MedQA-transfer verdict; see
    project_frames_cue_robustness_sft)
  * required n for 80% power at alpha=.05, from the OBSERVED paired SD

Usage:
    uv run python scripts/analyze_hotpotqa_cue_pilot.py \
        --results-dir results/hotpotqa_cue_pilot/gemma4_31b --dataset hotpotqa-50
"""

import os
import sys
import json
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
from scipy import stats


def setup_args():
    p = argparse.ArgumentParser(description="Paired search-call analysis for the HotpotQA cue pilot.")
    p.add_argument("--results-dir", required=True, help="e.g. results/hotpotqa_cue_pilot/gemma4_31b")
    p.add_argument("--dataset", default="hotpotqa-50", help="Dataset tier the runs used.")
    p.add_argument("--baseline", default="plain", help="Condition every other one is compared against.")
    p.add_argument("--subset-file", default=None,
                   help="Tier JSONL for the type/boolean breakdown (default: derived from --dataset).")
    p.add_argument("--out-csv", default=None, help="Optional path for the per-example wide CSV.")
    return p.parse_args()


def load_conditions(results_dir: str, dataset: str) -> dict[str, pd.DataFrame]:
    """condition -> DataFrame[example_id, search_calls, correct]. Condition is the filename tail."""
    pattern = os.path.join(results_dir, f"{dataset}_baseline_*_*.json")
    out = {}
    for path in sorted(glob.glob(pattern)):
        stem = os.path.basename(path)[:-len(".json")]
        # <dataset>_baseline_<model>_<condition> -- model may contain '_', so strip the known
        # prefix and take the condition as the tail after the LAST model segment. The run_name is
        # what the driver passed, so match it against the known cue vocabulary instead of guessing.
        rest = stem[len(f"{dataset}_baseline_"):]
        cond = None
        for known in ("confident_parametric", "elaborate", "natural", "polite", "direct",
                      "query", "plain"):
            if rest.endswith("_" + known):
                cond = known
                break
        if cond is None:
            print(f"  [skip] can't parse condition from {stem}")
            continue
        rows = json.load(open(path))
        recs = [{"example_id": r.get("example_id"),
                 "search_calls": r.get("sampler_search_calls", 0),
                 "correct": r.get("sampler_correct")} for r in rows]
        df = pd.DataFrame(recs)
        if df["example_id"].isna().any():
            raise SystemExit(f"{path}: rows without example_id -- cannot join safely.")
        out[cond] = df.set_index("example_id")
        print(f"  loaded {cond}: n={len(df)} from {os.path.basename(path)}")
    return out


def required_n(diffs: np.ndarray, alpha: float = 0.05, power: float = 0.80) -> float | None:
    """Paired-t n for the OBSERVED mean/SD. Wilcoxon costs ~5% more; reported as a floor."""
    sd = diffs.std(ddof=1)
    if sd == 0 or not np.isfinite(sd):
        return None
    dz = abs(diffs.mean()) / sd
    if dz == 0:
        return None
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z ** 2 / dz ** 2)


def main():
    args = setup_args()
    print(f"Loading {args.dataset} conditions from {args.results_dir} ...")
    conds = load_conditions(args.results_dir, args.dataset)
    if args.baseline not in conds:
        raise SystemExit(f"baseline condition '{args.baseline}' not found (have: {sorted(conds)})")

    # Inner-join every condition on example_id so all contrasts use the SAME example set.
    common = set.intersection(*(set(df.index) for df in conds.values()))
    print(f"\ncommon example_ids across {len(conds)} conditions: {len(common)}")
    wide = pd.DataFrame(index=sorted(common))
    for cond, df in conds.items():
        wide[f"sc_{cond}"] = df.loc[sorted(common), "search_calls"]

    subset_file = args.subset_file or f"data/{args.dataset.replace('-', '_')}.jsonl"
    if os.path.exists(subset_file):
        meta = pd.read_json(subset_file, lines=True).set_index("example_id")
        wide["type"] = meta.loc[wide.index, "type"]
        wide["answer_is_boolean"] = meta.loc[wide.index, "answer_is_boolean"]
    else:
        print(f"  [note] {subset_file} absent -- skipping type breakdown")

    print("\n=== marginal search-call distribution ===")
    for cond in conds:
        v = wide[f"sc_{cond}"].to_numpy()
        print(f"  {cond:22s} mean={v.mean():5.2f}  median={np.median(v):4.1f}  "
              f"max={v.max():3d}  zero-search={100*(v==0).mean():5.1f}%")

    base = wide[f"sc_{args.baseline}"].to_numpy()
    print(f"\n=== paired contrasts vs '{args.baseline}' (n={len(wide)}) ===")
    for cond in conds:
        if cond == args.baseline:
            continue
        cur = wide[f"sc_{cond}"].to_numpy()
        d = cur - base
        n_tie = int((d == 0).sum())
        n_eff = len(d) - n_tie
        print(f"\n  {args.baseline} -> {cond}")
        print(f"    mean {base.mean():.2f} -> {cur.mean():.2f}   "
              f"(mean paired diff {d.mean():+.2f}, median {np.median(d):+.1f})")
        print(f"    ties (identical search count): {n_tie}/{len(d)} "
              f"({100*n_tie/len(d):.1f}%)  -> effective n = {n_eff}")
        if n_eff == 0:
            print("    all pairs tied -- no signal at ANY n on this dataset/model.")
            continue
        w = stats.wilcoxon(cur, base, zero_method="wilcox")
        # rank-biserial from the signed-rank statistic over the non-tied pairs.
        rb = 1 - (2 * w.statistic) / (n_eff * (n_eff + 1) / 2)
        print(f"    Wilcoxon W={w.statistic:.1f}  p={w.pvalue:.4g}  rank-biserial r={rb:+.3f}")
        req = required_n(d)
        if req is not None:
            print(f"    observed paired SD={d.std(ddof=1):.2f} -> n for 80% power @a=.05: "
                  f"~{int(np.ceil(req))} (paired-t floor; Wilcoxon ~5% more)")
        if "type" in wide.columns:
            for t, g in wide.groupby("type"):
                gd = g[f"sc_{cond}"].to_numpy() - g[f"sc_{args.baseline}"].to_numpy()
                print(f"      [{t:11s} n={len(g):3d}] mean diff {gd.mean():+.2f}, "
                      f"ties {int((gd==0).sum())}")

    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        wide.to_csv(args.out_csv)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
