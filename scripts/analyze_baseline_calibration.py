"""
Is a model's search-call decision calibrated to its own uncertainty -- checked
with REPLICATION, using two independent no-cue baseline rollouts of the exact
same prompt (no cue manipulation at all, unlike analyze_necessity_vs_template_search*).

This is a correlational/reliability analysis, not a causal one: nothing is
manipulated here, entropy is an observed covariate and so is the search-call
outcome. What it adds over the single-rollout entropy-vs-search correlation
(analyze_llm_entropy_vs_search*.py) is a replication check: if the entropy-search
coupling shows up independently in TWO separate rollouts of the identical prompt
(run A = results/{frames_cues_full,medqa_grid}, run B =
results/{frames_cues_rerun,medqa_grid_rerun} -- the literal repeat used
elsewhere as a "noise floor"), that rules out the coupling being a fluke of one
particular rollout's sampling noise, and the averaged (A+B) estimate is a less
noisy readout of the true relationship than either alone.

Necessity proxy: 5-run LLM-judge semantic entropy (see docs/PARAMETRIC_UNCERTAINTY_HANDOFF.md).

Usage:
    uv run python scripts/analyze_baseline_calibration.py
"""
import csv
import glob
import json
import os

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "baseline_calibration")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b", "nemotron-cascade-2_30b"]

# Recent reclustering added per-cue 5run cluster files alongside the plain one (e.g.
# frames-cues_no_search_gemma4:31b_direct_llm_clusters_5run.json), so a bare "*" wildcard
# in entropy_glob now matches multiple files per model instead of just the cue-free
# baseline -- pin the tag explicitly and anchor the plain filename with {model} formatting.
TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
        "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b",
        "nemotron-cascade-2_30b": "nemotron-cascade-2:30b"}

DATASETS = {
    "frames": dict(
        entropy_dir="results/frames_parametric",
        entropy_glob="frames-cues_no_search_{tag}_llm_clusters_5run.json",
        run_a_dir="results/frames_cues_full",
        run_a_glob="frames-cues_baseline_*_verbose_plain.json",
        run_b_dir="results/frames_cues_rerun",
        run_b_glob="frames-cues_baseline_*_verbose_plain.json",
    ),
    "medqa": dict(
        entropy_dir="results/medqa_parametric",
        entropy_glob="medqa-500_no_search_{tag}_llm_clusters_5run.json",
        run_a_dir="results/medqa_grid",
        run_a_glob="medqa-500_baseline_*_orig_plain.json",
        run_b_dir="results/medqa_grid_rerun",
        run_b_glob="medqa-500_baseline_*_orig_plain.json",
    ),
}

CANONICAL_5RUN_LEVELS = (0.0, 0.7219280948873623, 0.9709505944546686, 1.3709505944546687,
                          1.5219280948873623, 1.9219280948873623, 2.321928094887362)


def round_entropy(e):
    for lvl in CANONICAL_5RUN_LEVELS:
        if abs(e - lvl) < 1e-6:
            return round(lvl, 3)
    return round(e, 3)


def load_one(model_dir, glob_pat, key):
    files = glob.glob(os.path.join(REPO, model_dir, glob_pat))
    if len(files) != 1:
        return None
    data = json.load(open(files[0]))
    return {row["example_id"]: row.get(key) for row in data}


def main():
    stats_rows = []
    bin_rows = []

    for ds, cfg in DATASETS.items():
        for model in MODELS:
            entropy = load_one(os.path.join(cfg["entropy_dir"], model), cfg["entropy_glob"].format(tag=TAGS[model]), "semantic_entropy")
            calls_a = load_one(os.path.join(cfg["run_a_dir"], model), cfg["run_a_glob"], "sampler_search_calls")
            calls_b = load_one(os.path.join(cfg["run_b_dir"], model), cfg["run_b_glob"], "sampler_search_calls")
            if entropy is None or calls_a is None or calls_b is None:
                print(f"  ! skip {ds}/{model}: entropy={entropy is not None} A={calls_a is not None} B={calls_b is not None}")
                continue

            common = sorted(set(entropy) & set(calls_a) & set(calls_b), key=str)
            common = [e for e in common if entropy[e] is not None]
            n = len(common)
            ent = np.array([entropy[e] for e in common])
            a = np.array([calls_a[e] for e in common], dtype=float)
            b = np.array([calls_b[e] for e in common], dtype=float)
            avg = (a + b) / 2.0

            rho_a, p_a = stats.spearmanr(ent, a)
            rho_b, p_b = stats.spearmanr(ent, b)
            rho_avg, p_avg = stats.spearmanr(ent, avg)
            test_retest_r, test_retest_p = stats.spearmanr(a, b)

            stats_rows.append(dict(
                dataset=ds, model=model, n=n,
                rho_run_a=round(rho_a, 3), p_run_a=f"{p_a:.2g}",
                rho_run_b=round(rho_b, 3), p_run_b=f"{p_b:.2g}",
                replicates_both_sig=bool(p_a < 0.05 and p_b < 0.05 and np.sign(rho_a) == np.sign(rho_b)),
                rho_averaged=round(rho_avg, 3), p_averaged=f"{p_avg:.2g}",
                test_retest_r=round(test_retest_r, 3), test_retest_p=f"{test_retest_p:.2g}",
                mean_calls_a=round(a.mean(), 3), mean_calls_b=round(b.mean(), 3),
            ))

            for lvl in sorted(set(round_entropy(e) for e in ent)):
                mask = np.array([round_entropy(e) == lvl for e in ent])
                bin_rows.append(dict(
                    dataset=ds, model=model, entropy_level=lvl, n=int(mask.sum()),
                    mean_calls_a=round(a[mask].mean(), 3), mean_calls_b=round(b[mask].mean(), 3),
                    mean_calls_avg=round(avg[mask].mean(), 3),
                    pct_searched_a=round(100 * (a[mask] > 0).mean(), 1),
                    pct_searched_b=round(100 * (b[mask] > 0).mean(), 1),
                ))

    stats_path = os.path.join(OUT_DIR, "baseline_calibration_stats.csv")
    bin_path = os.path.join(OUT_DIR, "baseline_calibration_by_level.csv")
    with open(stats_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
        w.writeheader()
        w.writerows(stats_rows)
    with open(bin_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(bin_rows[0].keys()))
        w.writeheader()
        w.writerows(bin_rows)
    print(f"wrote {stats_path}")
    print(f"wrote {bin_path}\n")

    for r in stats_rows:
        rep = "REPLICATES" if r["replicates_both_sig"] else "does not replicate"
        print(f"  {r['dataset']:6s} {r['model']:20s} n={r['n']:4d}  "
              f"rho_A={r['rho_run_a']:+.3f}(p={r['p_run_a']})  rho_B={r['rho_run_b']:+.3f}(p={r['p_run_b']})  "
              f"[{rep}]  rho_avg={r['rho_averaged']:+.3f}(p={r['p_averaged']})  "
              f"test-retest(A,B)={r['test_retest_r']:+.3f}")


if __name__ == "__main__":
    main()
