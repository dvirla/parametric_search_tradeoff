"""
Logistic companion to analyze_necessity_vs_template_search_5run.py: same causal
design (paired plain/cue, entropy as a pre-treatment necessity covariate,
cluster-robust SEs by example_id, FDR correction across cells), but the outcome
is the BINARY decision to search at all (`sampler_search_calls > 0`), not the
raw call count.

Why this is a different (and often better-specified) question than the OLS
volume model:
  - On MedQA search is rare (4-20% of examples ever search at all, per
    scripts/analyze_medqa_search_conditional.py) -- an OLS model on the raw
    count is fit almost entirely to a mass of zeros with a thin positive tail;
    a logistic model on "searched at all" targets the actual margin of
    variation directly instead of averaging over it.
  - It isolates the EXTENSIVE margin (does the agent decide to search at all)
    from the INTENSIVE margin (how many times, conditional on searching) that
    the OLS model conflates. A cue could suppress search entirely for
    borderline-necessity examples (extensive) while leaving call counts
    unchanged for examples that still search (intensive) -- the OLS model on
    raw counts would show this as one soft downward volume shift; the logistic
    model shows it as a clean threshold effect.
  - The same level-shift-vs-slope-change decomposition used for the OLS model
    and the mechanism taxonomy (analyze_cue_suppression_mechanism.py) applies
    directly here, just on the log-odds scale: b_is_cue = does the cue shift
    the probability of searching at every necessity level equally; b_interaction
    = does the cue change how much necessity itself matters to that probability.

Usage:
    uv run python scripts/analyze_necessity_vs_template_search_logistic.py
"""
import csv
import glob
import json
import os
import re
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score
from statsmodels.stats.contingency_tables import mcnemar

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "necessity_vs_template_logistic")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b", "nemotron-cascade-2_30b"]

# Recent reclustering added per-cue 5run cluster files alongside the plain one (e.g.
# frames-cues_no_search_gemma4:31b_direct_llm_clusters_5run.json), so a bare "*" wildcard
# in entropy_glob now matches multiple files per model instead of just the cue-free
# baseline -- pin the tag explicitly and anchor the plain filename with {tag} formatting.
TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
        "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b",
        "nemotron-cascade-2_30b": "nemotron-cascade-2:30b"}

# Minimum number of positive outcomes (searched==1) required, summed across
# BOTH plain and cue conditions, before trusting a cluster-robust logistic fit
# -- below this, quasi-separation / huge SEs are common (MedQA cells especially).
MIN_POSITIVES = 15


def frames_cond(fname):
    m = re.match(r"^frames-cues_baseline_(?P<model>.+?)_(?P<condition>(?:terse|verbose|epi_strong)_[a-z0-9_]+)\.json$", fname)
    return None if m is None else m.group("condition")


def medqa_cond(fname):
    m = re.match(r"^medqa-(?:500|terse)_baseline_(?P<model>.+?)_(?P<phrasing>orig|terse)_(?P<cue>[a-z0-9_]+)\.json$", fname)
    return None if m is None else f"{m.group('phrasing')}_{m.group('cue')}"


def plain_cond_for(condition):
    if condition.startswith("epi_strong_"):
        return "verbose_plain"
    phrasing = condition.split("_", 1)[0]
    return f"{phrasing}_plain"


DATASETS = {
    "frames": dict(
        entropy_dir="results/frames_parametric",
        entropy_glob="frames-cues_no_search_{tag}_llm_clusters_5run.json",
        search_dir="results/frames_cues_full",
        cond_fn=frames_cond,
    ),
    "medqa": dict(
        entropy_dir="results/medqa_parametric",
        entropy_glob="medqa-500_no_search_{tag}_llm_clusters_5run.json",
        search_dir="results/medqa_grid",
        cond_fn=medqa_cond,
    ),
}

PHRASING_MISMATCH_PREFIXES = ("terse_",)

# See analyze_necessity_vs_template_search_5run.py for the full explanation:
# AgentAsSampler.acall() counts search calls over pydantic-ai's all_messages(),
# which includes the injected message_history -- so raw sampler_search_calls for
# these history-injected cues is inflated by exactly this many FAKE search calls
# from the mocked history itself. This matters MORE here than in the continuous
# model: the binary searched=calls>0 threshold sits exactly where the offset
# lands for most MedQA examples (true live search rate there is very low), so
# without this correction a purely fake historical call can flip `searched` from
# False to True for examples where the model never actually searched at all.
MOCK_HISTORY_OFFSET = {"searchmulti": 1, "searchmulti2": 2, "searchmulti3": 3}


def strip_phrasing(cue):
    for p in ("verbose_", "orig_", "terse_"):
        if cue.startswith(p):
            return cue[len(p):]
    return cue


def load_one(model_dir, glob_pat, key="semantic_entropy"):
    files = glob.glob(os.path.join(REPO, model_dir, glob_pat))
    if len(files) != 1:
        return None
    data = json.load(open(files[0]))
    return {row["example_id"]: row.get(key) for row in data}


def discover_conditions(search_dir, cond_fn):
    out = {}
    for path in glob.glob(os.path.join(REPO, search_dir, "*.json")):
        cond = cond_fn(os.path.basename(path))
        if cond is not None:
            out[cond] = path
    return out


def load_calls(path):
    data = json.load(open(path))
    return {row["example_id"]: row.get("sampler_search_calls") for row in data}


def safe_auc(y, x):
    """AUC of x predicting y; undefined (nan) if y has only one class."""
    if len(set(y)) < 2:
        return np.nan
    return roc_auc_score(y, x)


def bh_fdr(pvals):
    pvals = np.asarray(pvals, dtype=float)
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(pvals) + 1)
    fdr = pvals * len(pvals) / ranks
    fdr_sorted = np.minimum.accumulate(fdr[order][::-1])[::-1]
    fdr_final = np.empty_like(fdr_sorted)
    fdr_final[order] = np.minimum(fdr_sorted, 1.0)
    return fdr_final


def main():
    per_cue_rows = []
    interaction_rows = []

    for ds, cfg in DATASETS.items():
        for model in MODELS:
            ent_dir = os.path.join(cfg["entropy_dir"], model)
            entropy = load_one(ent_dir, cfg["entropy_glob"].format(tag=TAGS[model]))
            search_dir = os.path.join(cfg["search_dir"], model)
            if entropy is None:
                print(f"  ! skip {ds}/{model}: missing entropy file")
                continue

            conditions = discover_conditions(search_dir, cfg["cond_fn"])
            if not conditions:
                print(f"  ! skip {ds}/{model}: no condition files found under {search_dir}")
                continue
            plain_cache = {}

            for cond, cue_path in sorted(conditions.items()):
                plain_name = plain_cond_for(cond)
                if cond == plain_name:
                    continue
                if plain_name not in plain_cache:
                    plain_path = conditions.get(plain_name)
                    plain_cache[plain_name] = load_calls(plain_path) if plain_path else None
                plain = plain_cache[plain_name]
                if plain is None:
                    print(f"  ! skip {ds}/{model}/{cond}: no matching plain baseline '{plain_name}'")
                    continue
                calls_cue = load_calls(cue_path)
                cue = cond
                mismatched_phrasing = cond.startswith(PHRASING_MISMATCH_PREFIXES)
                mock_offset = MOCK_HISTORY_OFFSET.get(strip_phrasing(cond), 0)

                common = sorted(set(entropy) & set(plain) & set(calls_cue), key=str)
                common = [e for e in common if entropy[e] is not None]
                n = len(common)
                if n < 20:
                    continue

                ent = np.array([entropy[e] for e in common])
                sp = (np.array([plain[e] for e in common], dtype=float) > 0).astype(int)
                cc_corrected = np.clip(np.array([calls_cue[e] for e in common], dtype=float) - mock_offset, 0, None)
                sc = (cc_corrected > 0).astype(int)
                n_pos_total = int(sp.sum() + sc.sum())

                auc_plain = safe_auc(sp, ent)
                auc_cue = safe_auc(sc, ent)

                # McNemar's exact test on the paired binary decision: did the
                # cue change *whether* the model searches at all (not by how much).
                table = [[int(((sp == 1) & (sc == 1)).sum()), int(((sp == 1) & (sc == 0)).sum())],
                         [int(((sp == 0) & (sc == 1)).sum()), int(((sp == 0) & (sc == 0)).sum())]]
                mc = mcnemar(table, exact=(n < 200))

                per_cue_rows.append(dict(
                    dataset=ds, model=model, cue=cue, n=n,
                    phrasing_mismatch=mismatched_phrasing,
                    pct_searched_plain=round(100 * sp.mean(), 1),
                    pct_searched_cue=round(100 * sc.mean(), 1),
                    pct_point_delta=round(100 * (sc.mean() - sp.mean()), 1),
                    auc_entropy_plain=round(auc_plain, 3) if auc_plain == auc_plain else "",
                    auc_entropy_cue=round(auc_cue, 3) if auc_cue == auc_cue else "",
                    n_switched_off=table[0][1], n_switched_on=table[1][0],
                    mcnemar_p=f"{mc.pvalue:.2g}",
                ))

                if n_pos_total < MIN_POSITIVES:
                    interaction_rows.append(dict(
                        dataset=ds, model=model, cue=cue, n=n, n_pos_total=n_pos_total,
                        phrasing_mismatch=mismatched_phrasing,
                        b_entropy="", p_entropy="", or_entropy="",
                        b_is_cue="", p_is_cue="", or_is_cue="",
                        b_interaction="", p_interaction="1.0", or_interaction="",
                        pseudo_r2="", skipped_low_positives=True,
                    ))
                    continue

                long_df = pd.DataFrame({
                    "example_id": list(common) * 2,
                    "entropy": np.concatenate([ent, ent]),
                    "is_cue": np.concatenate([np.zeros(n), np.ones(n)]),
                    "searched": np.concatenate([sp, sc]),
                })
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fit = smf.logit("searched ~ entropy * is_cue", data=long_df).fit(
                            cov_type="cluster", cov_kwds={"groups": long_df["example_id"]},
                            disp=False, maxiter=200,
                        )
                    if not fit.mle_retvals.get("converged", True):
                        raise RuntimeError("did not converge")
                    b = fit.params
                    p = fit.pvalues
                    interaction_rows.append(dict(
                        dataset=ds, model=model, cue=cue, n=n, n_pos_total=n_pos_total,
                        phrasing_mismatch=mismatched_phrasing,
                        b_entropy=round(b["entropy"], 4), p_entropy=f"{p['entropy']:.2g}",
                        or_entropy=round(float(np.exp(b["entropy"])), 4),
                        b_is_cue=round(b["is_cue"], 4), p_is_cue=f"{p['is_cue']:.2g}",
                        or_is_cue=round(float(np.exp(b["is_cue"])), 4),
                        b_interaction=round(b["entropy:is_cue"], 4),
                        p_interaction=f"{p['entropy:is_cue']:.2g}",
                        or_interaction=round(float(np.exp(b["entropy:is_cue"])), 4),
                        pseudo_r2=round(fit.prsquared, 4),
                        skipped_low_positives=False,
                    ))
                except Exception as e:
                    interaction_rows.append(dict(
                        dataset=ds, model=model, cue=cue, n=n, n_pos_total=n_pos_total,
                        phrasing_mismatch=mismatched_phrasing,
                        b_entropy="", p_entropy="", or_entropy="",
                        b_is_cue="", p_is_cue="", or_is_cue="",
                        b_interaction="", p_interaction="1.0", or_interaction="",
                        pseudo_r2="", skipped_low_positives=f"fit_failed: {e}",
                    ))

    fdr = bh_fdr([float(r["p_interaction"]) for r in interaction_rows])
    for r, q in zip(interaction_rows, fdr):
        r["p_interaction_fdr"] = round(float(q), 4)

    per_cue_path = os.path.join(OUT_DIR, "necessity_vs_template_logistic_per_cue.csv")
    interaction_path = os.path.join(OUT_DIR, "necessity_vs_template_logistic_interaction.csv")
    with open(per_cue_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_cue_rows[0].keys()))
        w.writeheader()
        w.writerows(per_cue_rows)
    with open(interaction_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(interaction_rows[0].keys()))
        w.writeheader()
        w.writerows(interaction_rows)
    print(f"wrote {per_cue_path}  ({len(per_cue_rows)} rows)")
    print(f"wrote {interaction_path}  ({len(interaction_rows)} rows, FDR-corrected)")

    fittable = [r for r in interaction_rows if r["b_interaction"] != ""]
    skipped = len(interaction_rows) - len(fittable)
    print(f"\n{skipped}/{len(interaction_rows)} cells skipped (too few positive "
          f"outcomes to trust a logistic fit, threshold={MIN_POSITIVES}).")

    print("\n=== FDR-significant necessity x cue interaction on the BINARY decision (q<0.05) ===")
    sig = [r for r in fittable if r["p_interaction_fdr"] < 0.05]
    sig.sort(key=lambda r: -abs(r["b_interaction"]))
    for r in sig:
        flag = " [terse: phrasing != entropy-probe phrasing]" if r["phrasing_mismatch"] else ""
        direction = ("ANTI-CALIBRATED (necessity matters LESS to the search-or-not "
                     "decision under the cue)" if r["b_interaction"] < 0 else
                     "SHARPENED (necessity matters MORE under the cue)")
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:26s} n={r['n']:4d}  "
              f"b_int={r['b_interaction']:+.4f} (OR={r['or_interaction']}) q={r['p_interaction_fdr']:.3g}  "
              f"{direction}{flag}")
    print(f"\n{len(sig)}/{len(fittable)} fittable cells FDR-significant at q<0.05.")

    print("\n=== confident_parametric cue specifically ===")
    for r in per_cue_rows:
        if r["cue"].endswith("confident_parametric"):
            print(f"  {r['dataset']:6s} {r['model']:20s} n={r['n']:4d}  "
                  f"%searched: plain={r['pct_searched_plain']:.1f} -> cue={r['pct_searched_cue']:.1f} "
                  f"(mcnemar p={r['mcnemar_p']})  AUC(entropy): plain={r['auc_entropy_plain']} cue={r['auc_entropy_cue']}")
    for r in interaction_rows:
        if r["cue"].endswith("confident_parametric") and r["b_interaction"] != "":
            print(f"  {r['dataset']:6s} {r['model']:20s}  "
                  f"b_interaction={r['b_interaction']:+.4f} (OR={r['or_interaction']}, "
                  f"p={r['p_interaction']}, q={r['p_interaction_fdr']:.3g})  pseudo_r2={r['pseudo_r2']}")


if __name__ == "__main__":
    main()
