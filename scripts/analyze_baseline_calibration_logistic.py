"""
Logistic companion to analyze_baseline_calibration.py. That script reports
Spearman rho(entropy, calls) as evidence entropy tracks search behavior -- but
Spearman rho is a DISCRIMINATION statistic only (does higher entropy rank
higher on search behavior). It says nothing about CALIBRATION in the strict
forecasting-literature sense: if entropy implied a probability of searching,
would that probability match the observed search rate? This script fits that
probability model directly: searched ~ entropy (logistic), split-half
replicated on the same two independent no-cue rollouts (run A vs run B) used
in analyze_baseline_calibration.py, and reports the actual calibration
statistics -- Brier score and a reliability curve -- alongside AUC
(discrimination) and McFadden pseudo-R^2, so "does entropy predict search" and
"is that prediction well-calibrated as a probability" are no longer conflated
into one correlation number.

Usage:
    uv run python scripts/analyze_baseline_calibration_logistic.py
"""
import csv
import glob
import json
import os
import warnings

import numpy as np
import statsmodels.formula.api as smf
from sklearn.metrics import brier_score_loss, roc_auc_score

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "baseline_calibration_logistic")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]

DATASETS = {
    "frames": dict(
        entropy_dir="results/frames_parametric",
        entropy_glob="frames-cues_no_search_*_llm_clusters_5run.json",
        run_a_dir="results/frames_cues_full",
        run_a_glob="frames-cues_baseline_*_verbose_plain.json",
        run_b_dir="results/frames_cues_rerun",
        run_b_glob="frames-cues_baseline_*_verbose_plain.json",
    ),
    "medqa": dict(
        entropy_dir="results/medqa_parametric",
        entropy_glob="medqa-500_no_search_*_llm_clusters_5run.json",
        run_a_dir="results/medqa_grid",
        run_a_glob="medqa-500_baseline_*_orig_plain.json",
        run_b_dir="results/medqa_grid_rerun",
        run_b_glob="medqa-500_baseline_*_orig_plain.json",
    ),
}

CANONICAL_5RUN_LEVELS = (0.0, 0.7219280948873623, 0.9709505944546686, 1.3709505944546687,
                          1.5219280948873623, 1.9219280948873623, 2.321928094887362)

MIN_POSITIVES = 15


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


def fit_run(ent, searched):
    """Fit searched ~ entropy, return (fit_or_None, predicted_probs_or_None)."""
    if int(searched.sum()) < MIN_POSITIVES or int((1 - searched).sum()) < MIN_POSITIVES:
        return None, None
    df = {"entropy": ent, "searched": searched}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.logit("searched ~ entropy", data=df).fit(disp=False, maxiter=200)
        if not fit.mle_retvals.get("converged", True):
            return None, None
        return fit, fit.predict(df)
    except Exception:
        return None, None


def main():
    stats_rows = []
    bin_rows = []

    for ds, cfg in DATASETS.items():
        for model in MODELS:
            entropy = load_one(os.path.join(cfg["entropy_dir"], model), cfg["entropy_glob"], "semantic_entropy")
            calls_a = load_one(os.path.join(cfg["run_a_dir"], model), cfg["run_a_glob"], "sampler_search_calls")
            calls_b = load_one(os.path.join(cfg["run_b_dir"], model), cfg["run_b_glob"], "sampler_search_calls")
            if entropy is None or calls_a is None or calls_b is None:
                print(f"  ! skip {ds}/{model}: entropy={entropy is not None} A={calls_a is not None} B={calls_b is not None}")
                continue

            common = sorted(set(entropy) & set(calls_a) & set(calls_b), key=str)
            common = [e for e in common if entropy[e] is not None]
            n = len(common)
            ent = np.array([entropy[e] for e in common])
            sa = (np.array([calls_a[e] for e in common], dtype=float) > 0).astype(int)
            sb = (np.array([calls_b[e] for e in common], dtype=float) > 0).astype(int)

            fit_a, pred_a = fit_run(ent, sa)
            fit_b, pred_b = fit_run(ent, sb)

            def summarize(fit, pred, y):
                if fit is None:
                    return dict(b_entropy="", p_entropy="", or_entropy="", pseudo_r2="",
                                auc="", brier="", brier_null="")
                auc = roc_auc_score(y, ent) if len(set(y)) > 1 else np.nan
                brier = brier_score_loss(y, pred)
                brier_null = brier_score_loss(y, np.full_like(pred, y.mean()))
                return dict(
                    b_entropy=round(fit.params["entropy"], 4),
                    p_entropy=f"{fit.pvalues['entropy']:.2g}",
                    or_entropy=round(float(np.exp(fit.params["entropy"])), 4),
                    pseudo_r2=round(fit.prsquared, 4),
                    auc=round(auc, 3) if auc == auc else "",
                    brier=round(brier, 4),
                    brier_null=round(brier_null, 4),
                )

            sum_a = summarize(fit_a, pred_a, sa)
            sum_b = summarize(fit_b, pred_b, sb)
            both_sig = (fit_a is not None and fit_b is not None
                        and float(sum_a["p_entropy"]) < 0.05 and float(sum_b["p_entropy"]) < 0.05
                        and np.sign(sum_a["b_entropy"]) == np.sign(sum_b["b_entropy"]))

            stats_rows.append(dict(
                dataset=ds, model=model, n=n,
                pct_searched_a=round(100 * sa.mean(), 1), pct_searched_b=round(100 * sb.mean(), 1),
                **{f"{k}_a": v for k, v in sum_a.items()},
                **{f"{k}_b": v for k, v in sum_b.items()},
                replicates_both_sig=bool(both_sig),
            ))

            for lvl in sorted(set(round_entropy(e) for e in ent)):
                mask = np.array([round_entropy(e) == lvl for e in ent])
                row = dict(dataset=ds, model=model, entropy_level=lvl, n=int(mask.sum()),
                           pct_searched_a=round(100 * sa[mask].mean(), 1),
                           pct_searched_b=round(100 * sb[mask].mean(), 1))
                row["predicted_p_a"] = round(float(pred_a[mask].mean()), 4) if pred_a is not None else ""
                row["predicted_p_b"] = round(float(pred_b[mask].mean()), 4) if pred_b is not None else ""
                bin_rows.append(row)

    stats_path = os.path.join(OUT_DIR, "baseline_calibration_logistic_stats.csv")
    bin_path = os.path.join(OUT_DIR, "baseline_calibration_logistic_by_level.csv")
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
        print(f"  {r['dataset']:6s} {r['model']:20s} n={r['n']:4d}  [{rep}]")
        print(f"    run A: AUC={r['auc_a']}  Brier={r['brier_a']} (null={r['brier_null_a']})  "
              f"pseudo_r2={r['pseudo_r2_a']}  OR={r['or_entropy_a']} (p={r['p_entropy_a']})")
        print(f"    run B: AUC={r['auc_b']}  Brier={r['brier_b']} (null={r['brier_null_b']})  "
              f"pseudo_r2={r['pseudo_r2_b']}  OR={r['or_entropy_b']} (p={r['p_entropy_b']})")


if __name__ == "__main__":
    main()
