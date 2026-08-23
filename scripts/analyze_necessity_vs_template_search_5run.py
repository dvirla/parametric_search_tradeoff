"""
Causal test: is a cue's effect on search-call volume driven by genuine epistemic
necessity, or is it a necessity-blind template override?

Design. We already have, per (dataset, model, example): an entropy-based necessity
proxy (`semantic_entropy`, from the cue-free 3-run no_search parametric probe --
computed once, independent of any cue) and search-call counts under `plain` plus
several cue conditions (frames_cues_full / medqa_grid, same example set, paired
within-subject). Because entropy is measured under a *separate*, cue-free probe,
it cannot be caused by the cue -- it is a valid pre-treatment covariate. The cue
itself is applied uniformly to every example in a run (not example-randomized),
so this is a paired (within-subject) quasi-experiment, not a fully randomized one:
we can identify the cue's *average* effect and whether that effect's *size varies
with necessity* (an interaction/moderation test), which is the operative question
here -- not the average effect alone (already reported in dual_metric_*).

Two complementary tests per (dataset, model, cue):
  1. rho(entropy, calls_plain) vs rho(entropy, calls_cue): does the cue attenuate
     the necessity-search coupling that exists at baseline?
  2. rho(entropy, delta_search) where delta_search = calls_cue - calls_plain: if
     the cue is a rational, necessity-aware adjustment, suppression should
     concentrate on LOW-necessity (already-confident) examples, giving a
     *positive* correlation between entropy and delta (less suppression, i.e.
     delta closer to 0 or positive, at high necessity). A necessity-blind
     template override predicts delta uncorrelated with entropy (uniform shift
     regardless of the model's actual uncertainty).
  3. OLS interaction model on the pooled long-format (plain+cue) data per
     (dataset, model, cue): calls ~ entropy + is_cue + entropy:is_cue, cluster-
     robust SEs by example_id (each example contributes one plain and one cue
     observation). A significant NEGATIVE entropy:is_cue coefficient means the
     cue's search-suppressing effect is *strongest exactly where necessity is
     lowest and weakest where necessity is highest* -- i.e. the override is at
     least partly necessity-sensitive, not purely templated. A near-zero,
     non-significant interaction means the cue shifts search by a roughly
     constant amount regardless of necessity -- consistent with a necessity-
     blind, template-triggered override.

5-run version of analyze_necessity_vs_template_search.py -- same design, but the
necessity proxy is the 5-run semantic entropy (up to 7 discrete levels instead of
3), giving finer resolution and a less coarse interaction-slope estimate.

Usage:
    uv run python scripts/analyze_necessity_vs_template_search_5run.py
"""
import csv
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "necessity_vs_template_5run")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]

# Condition-name parsing + plain-baseline pairing, mirroring
# scripts/dual_metric_analysis.py (frames_cond/medqa_cond/plain_cond_for) so the
# two analyses stay consistent about what counts as a "condition" and which
# plain run it's compared against.


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
        entropy_glob="frames-cues_no_search_*_llm_clusters_5run.json",
        search_dir="results/frames_cues_full",
        cond_fn=frames_cond,
    ),
    "medqa": dict(
        entropy_dir="results/medqa_parametric",
        entropy_glob="medqa-500_no_search_*_llm_clusters_5run.json",
        search_dir="results/medqa_grid",
        cond_fn=medqa_cond,
    ),
}

# Entropy (necessity) was measured once, cue-free, under the ORIGINAL/verbose
# question phrasing. Conditions built on `terse` phrasing pair against a
# `terse_plain` baseline whose question text differs from what entropy was
# measured on for the same example_id -- same underlying fact, different
# surface form. Flagged in the output so terse-paired rows aren't read with
# the same confidence as verbose/orig-paired rows.
PHRASING_MISMATCH_PREFIXES = ("terse_",)

# AgentAsSampler.acall() (src/services/agent_sampler.py) counts search calls over
# pydantic-ai's all_messages(), which includes the injected message_history as a
# literal prefix -- so raw sampler_search_calls for these history-injected cue
# conditions is inflated by exactly this many FAKE search calls from the mocked
# history itself (each round injects exactly one tool_call; already discovered once
# before in scripts/compare_searchmulti_rounds.py -- an exact correction, not an
# approximation). Keyed by the base cue name with any verbose_/orig_/terse_
# phrasing prefix stripped.
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
    """All (condition -> filepath) pairs in a model's search-results dir."""
    out = {}
    for path in glob.glob(os.path.join(REPO, search_dir, "*.json")):
        cond = cond_fn(os.path.basename(path))
        if cond is not None:
            out[cond] = path
    return out


def load_calls(path):
    data = json.load(open(path))
    return {row["example_id"]: row.get("sampler_search_calls") for row in data}


def fisher_dependent_z(r_xy, r_xz, r_yz, n):
    """Steiger's (1980) Z test for two correlations sharing variable x (entropy),
    on the SAME n subjects: compares rho(entropy,calls_plain) vs rho(entropy,calls_cue)
    where calls_plain and calls_cue are themselves correlated (r_yz)."""
    rm2 = (r_xy ** 2 + r_xz ** 2) / 2
    f = (1 - r_yz) / (2 * (1 - rm2)) if (1 - rm2) != 0 else np.nan
    f = min(max(f, 0), 1) if f == f else np.nan
    h = (1 - f * rm2) / (1 - rm2) if (1 - rm2) != 0 else np.nan
    zy = np.arctanh(r_xy)
    zz = np.arctanh(r_xz)
    if h != h or h <= 0:
        return np.nan, np.nan
    se = np.sqrt((1 - f * rm2) / ((n - 3) * h)) if (n - 3) > 0 else np.nan
    if se != se or se == 0:
        return np.nan, np.nan
    z = (zy - zz) / se
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def main():
    per_cue_rows = []
    interaction_rows = []

    for ds, cfg in DATASETS.items():
        for model in MODELS:
            ent_dir = os.path.join(cfg["entropy_dir"], model)
            entropy = load_one(ent_dir, cfg["entropy_glob"])
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
                    continue  # this IS a plain condition, not a cue
                if plain_name not in plain_cache:
                    plain_path = conditions.get(plain_name)
                    plain_cache[plain_name] = load_calls(plain_path) if plain_path else None
                plain = plain_cache[plain_name]
                if plain is None:
                    print(f"  ! skip {ds}/{model}/{cond}: no matching plain baseline '{plain_name}'")
                    continue
                calls_cue = load_calls(cue_path)
                cue = cond  # keep full condition name (e.g. terse_direct, epi_strong_boost)
                mismatched_phrasing = cond.startswith(PHRASING_MISMATCH_PREFIXES)
                mock_offset = MOCK_HISTORY_OFFSET.get(strip_phrasing(cond), 0)

                common = sorted(set(entropy) & set(plain) & set(calls_cue), key=str)
                common = [e for e in common if entropy[e] is not None]
                n = len(common)
                if n < 20:
                    continue

                ent = np.array([entropy[e] for e in common])
                cp = np.array([plain[e] for e in common], dtype=float)
                cc = np.clip(np.array([calls_cue[e] for e in common], dtype=float) - mock_offset, 0, None)
                delta = cc - cp

                rho_plain, p_plain = stats.spearmanr(ent, cp)
                rho_cue, p_cue = stats.spearmanr(ent, cc)
                rho_delta, p_delta = stats.spearmanr(ent, delta)
                r_plain_cue, _ = stats.pearsonr(cp, cc)  # for the dependent-correlation test
                # use Pearson versions of rho_plain/rho_cue (paired w/ same-x Steiger test)
                r_pear_plain, _ = stats.pearsonr(ent, cp)
                r_pear_cue, _ = stats.pearsonr(ent, cc)
                z_att, p_att = fisher_dependent_z(r_pear_plain, r_pear_cue, r_plain_cue, n)

                per_cue_rows.append(dict(
                    dataset=ds, model=model, cue=cue, n=n,
                    phrasing_mismatch=mismatched_phrasing,
                    rho_plain=round(rho_plain, 3), p_plain=f"{p_plain:.2g}",
                    rho_cue=round(rho_cue, 3), p_cue=f"{p_cue:.2g}",
                    rho_delta=round(rho_delta, 3), p_delta=f"{p_delta:.2g}",
                    mean_calls_plain=round(cp.mean(), 3), mean_calls_cue=round(cc.mean(), 3),
                    mean_delta=round(delta.mean(), 3),
                    attenuation_z=round(z_att, 3) if z_att == z_att else "",
                    attenuation_p=f"{p_att:.2g}" if p_att == p_att else "",
                ))

                # interaction regression: calls ~ entropy * is_cue, cluster by example
                long_df = pd.DataFrame({
                    "example_id": list(common) * 2,
                    "entropy": np.concatenate([ent, ent]),
                    "is_cue": np.concatenate([np.zeros(n), np.ones(n)]),
                    "calls": np.concatenate([cp, cc]),
                })
                model_fit = smf.ols("calls ~ entropy * is_cue", data=long_df).fit(
                    cov_type="cluster", cov_kwds={"groups": long_df["example_id"]}
                )
                interaction_rows.append(dict(
                    dataset=ds, model=model, cue=cue, n=n,
                    phrasing_mismatch=mismatched_phrasing,
                    b_entropy=round(model_fit.params["entropy"], 4),
                    p_entropy=f"{model_fit.pvalues['entropy']:.2g}",
                    b_is_cue=round(model_fit.params["is_cue"], 4),
                    p_is_cue=f"{model_fit.pvalues['is_cue']:.2g}",
                    b_interaction=round(model_fit.params["entropy:is_cue"], 4),
                    p_interaction=f"{model_fit.pvalues['entropy:is_cue']:.2g}",
                ))

    # Benjamini-Hochberg FDR correction across ALL interaction tests -- with
    # ~100+ (dataset, model, cue) cells now in play, raw p<0.05 is not a
    # meaningful bar on its own.
    pvals = np.array([float(r["p_interaction"]) for r in interaction_rows])
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(pvals) + 1)
    fdr = pvals * len(pvals) / ranks
    fdr_sorted = np.minimum.accumulate(fdr[order][::-1])[::-1]
    fdr_final = np.empty_like(fdr_sorted)
    fdr_final[order] = np.minimum(fdr_sorted, 1.0)
    for r, q in zip(interaction_rows, fdr_final):
        r["p_interaction_fdr"] = round(float(q), 4)

    per_cue_path = os.path.join(OUT_DIR, "necessity_vs_template_per_cue.csv")
    interaction_path = os.path.join(OUT_DIR, "necessity_vs_template_interaction.csv")
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

    print(f"\n=== all cells with FDR-significant necessity x cue interaction (q<0.05), sorted by |effect| ===")
    sig = [r for r in interaction_rows if r["p_interaction_fdr"] < 0.05]
    sig.sort(key=lambda r: -abs(r["b_interaction"]))
    for r in sig:
        flag = " [terse: phrasing != entropy-probe phrasing]" if r["phrasing_mismatch"] else ""
        direction = "ANTI-CALIBRATED (suppresses more where necessity is higher)" if r["b_interaction"] < 0 else "CALIBRATED (cue effect shrinks as necessity rises)"
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:26s} n={r['n']:4d}  "
              f"b_int={r['b_interaction']:+.4f} q={r['p_interaction_fdr']:.3g}  {direction}{flag}")
    print(f"\n{len(sig)}/{len(interaction_rows)} cells FDR-significant at q<0.05.")

    print("\n=== confident_parametric cue specifically (the explicit 'no need to search' instruction) ===")
    for r in per_cue_rows:
        if r["cue"].endswith("confident_parametric"):
            print(f"  {r['dataset']:6s} {r['model']:20s} n={r['n']:4d}  "
                  f"rho_plain={r['rho_plain']:+.3f} rho_cue={r['rho_cue']:+.3f}  "
                  f"rho_delta={r['rho_delta']:+.3f} (p={r['p_delta']})  "
                  f"atten_p={r['attenuation_p']}")
    print()
    for r in interaction_rows:
        if r["cue"].endswith("confident_parametric"):
            q = r["p_interaction_fdr"]
            print(f"  {r['dataset']:6s} {r['model']:20s}  "
                  f"b_interaction={r['b_interaction']:+.4f} (p={r['p_interaction']}, q={q:.3g})  "
                  f"b_is_cue={r['b_is_cue']:+.4f} (p={r['p_is_cue']})")


if __name__ == "__main__":
    main()
