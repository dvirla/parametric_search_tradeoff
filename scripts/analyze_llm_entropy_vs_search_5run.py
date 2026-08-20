"""
5-run version of analyze_llm_entropy_vs_search.py. Relationship between LLM-judge
semantic entropy (over 5 no_search parametric-uncertainty runs, gpt-oss:120b
clusterer -- results/{frames,medqa}_parametric/<model>/*_llm_clusters_5run.json)
and search-call volume on the paired plain-original search run, per dataset.

5 samples give up to 7 distinct entropy levels (0, 0.722, 0.971, 1.371, 1.522,
1.922, 2.322 bits) vs. only 3 for the original probe -- finer resolution, same
join logic. See docs/PARAMETRIC_UNCERTAINTY_HANDOFF.md for provenance.

Usage:
    uv run python scripts/analyze_llm_entropy_vs_search_5run.py
"""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "param_vs_search_llm_5run")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]

DATASETS = {
    "frames": dict(
        entropy_dir="results/frames_parametric",
        entropy_glob="frames-cues_no_search_*_llm_clusters_5run.json",
        search_dir="results/frames_cues_full",
        search_glob="frames-cues_baseline_*_verbose_plain.json",
    ),
    "medqa": dict(
        entropy_dir="results/medqa_parametric",
        entropy_glob="medqa-500_no_search_*_llm_clusters_5run.json",
        search_dir="results/medqa_grid",
        search_glob="medqa-500_baseline_*_orig_plain.json",
    ),
}


def load_entropy(model_dir, glob_pat):
    files = glob.glob(os.path.join(REPO, model_dir, "*", glob_pat))
    assert len(files) == 1, f"expected 1 file, got {files} in {model_dir}/*/{glob_pat}"
    data = json.load(open(files[0]))
    return {row["example_id"]: row["semantic_entropy"] for row in data}


def load_search_calls(model_dir, glob_pat):
    files = glob.glob(os.path.join(REPO, model_dir, "*", glob_pat))
    assert len(files) == 1, f"expected 1 file, got {files} in {model_dir}/*/{glob_pat}"
    data = json.load(open(files[0]))
    return {row["example_id"]: row["sampler_search_calls"] for row in data}


CANONICAL_5RUN_LEVELS = (0.0, 0.7219280948873623, 0.9709505944546686, 1.3709505944546687,
                          1.5219280948873623, 1.9219280948873623, 2.321928094887362)


def round_entropy(e):
    """Snap floating semantic entropy to its nearest canonical 5-sample level."""
    for lvl in CANONICAL_5RUN_LEVELS:
        if abs(e - lvl) < 1e-6:
            return round(lvl, 3)
    return round(e, 3)


def main():
    by_rows = []
    stats_rows = []

    for ds, cfg in DATASETS.items():
        pooled_ent, pooled_calls = [], []
        for model in MODELS:
            model_entropy_dir = os.path.join(cfg["entropy_dir"], model)
            model_search_dir = os.path.join(cfg["search_dir"], model)
            ent_glob = cfg["entropy_glob"]
            search_glob = cfg["search_glob"]
            ent_files = glob.glob(os.path.join(REPO, model_entropy_dir, ent_glob))
            search_files = glob.glob(os.path.join(REPO, model_search_dir, search_glob))
            if not ent_files or not search_files:
                print(f"  ! skipping {ds}/{model}: entropy={len(ent_files)} search={len(search_files)}")
                continue
            entropy = {row["example_id"]: row["semantic_entropy"] for row in json.load(open(ent_files[0]))}
            calls = {row["example_id"]: row["sampler_search_calls"] for row in json.load(open(search_files[0]))}

            common = sorted(set(entropy) & set(calls), key=str)
            ent_vals = np.array([entropy[e] for e in common])
            call_vals = np.array([calls[e] for e in common])
            n = len(common)

            rho, rho_p = stats.spearmanr(ent_vals, call_vals)
            zero_mask = ent_vals == 0.0
            high_mask = ent_vals >= 1.9  # 4-or-5-of-5 samples distinct (max cluster spread)
            mwu_p = stats.mannwhitneyu(call_vals[high_mask], call_vals[zero_mask],
                                        alternative="greater").pvalue if zero_mask.any() and high_mask.any() else float("nan")
            stats_rows.append({
                "dataset": ds, "model": model, "n": n,
                "rho": round(rho, 3), "rho_p": f"{rho_p:.2g}",
                "mwu_p_high_gt_zero": f"{mwu_p:.2g}" if mwu_p == mwu_p else "",
                "calls_entropy0": round(call_vals[zero_mask].mean(), 3) if zero_mask.any() else float("nan"),
                "calls_entropy_high": round(call_vals[high_mask].mean(), 3) if high_mask.any() else float("nan"),
            })

            for lvl in sorted(set(round_entropy(e) for e in ent_vals)):
                mask = np.array([round_entropy(e) == lvl for e in ent_vals])
                by_rows.append({
                    "dataset": ds, "model": model, "entropy_level": lvl, "n": int(mask.sum()),
                    "mean_search_calls": round(call_vals[mask].mean(), 4),
                    "pct_searched": round(100 * (call_vals[mask] > 0).mean(), 1),
                })

            pooled_ent.append(ent_vals)
            pooled_calls.append(call_vals)

        if pooled_ent:
            ent_all = np.concatenate(pooled_ent)
            calls_all = np.concatenate(pooled_calls)
            rho, rho_p = stats.spearmanr(ent_all, calls_all)
            zero_mask = ent_all == 0.0
            high_mask = ent_all >= 1.9  # 4-or-5-of-5 samples distinct (max cluster spread)
            mwu_p = stats.mannwhitneyu(calls_all[high_mask], calls_all[zero_mask], alternative="greater").pvalue
            stats_rows.append({
                "dataset": ds, "model": "POOLED", "n": len(ent_all),
                "rho": round(rho, 3), "rho_p": f"{rho_p:.2g}",
                "mwu_p_high_gt_zero": f"{mwu_p:.2g}",
                "calls_entropy0": round(calls_all[zero_mask].mean(), 3),
                "calls_entropy_high": round(calls_all[high_mask].mean(), 3),
            })

    by_path = os.path.join(OUT_DIR, "llm_entropy_vs_search_by_level.csv")
    stats_path = os.path.join(OUT_DIR, "llm_entropy_vs_search_stats.csv")
    with open(by_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_rows[0].keys()))
        w.writeheader()
        w.writerows(by_rows)
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)

    print(f"wrote {by_path}")
    print(f"wrote {stats_path}")
    print()
    for row in stats_rows:
        print(row)


if __name__ == "__main__":
    main()
