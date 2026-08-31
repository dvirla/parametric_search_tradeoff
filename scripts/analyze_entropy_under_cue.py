"""
Does a cue change the model's own belief/consistency (semantic entropy measured
UNDER the cue, in a no-search rollout), or only its downstream search-triggering
behavior? Every entropy number used elsewhere in this thread
(analyze_necessity_vs_template_search*, analyze_baseline_calibration*) is
measured on a cue-FREE `plain` no-search probe and treated as a stable,
pre-treatment necessity covariate -- valid for licensing causal language about
the cue's effect on search behavior, but silent on whether the cue also moves
the thing entropy is supposed to measure. This script tests that assumption
directly, using no-search rollouts collected under cues by a sibling session
(gpt-oss:120b judge -- same clusterer as the baseline).

RESOLUTION HANDLING (important): plain and cue-condition entropy have each
been re-clustered at two different resolutions over time -- an original 3-run
pass and a newer 5-run pass -- and NOT every (dataset, model, cue) cell has
been re-clustered at 5-run yet. This script discovers ALL available cluster
files per cell (plain and cue independently) and prefers the 5-run version
when present, falling back to 3-run only when no 5-run file exists yet --
matching how every other 5-run-preferring script in this project already
behaves (e.g. analyze_baseline_calibration.py). The resolution actually used
on each side is recorded in the output (`plain_resolution`, `cue_resolution`)
because a mismatched pair (e.g. 5-run plain vs. 3-run cue) has coarser
statistical power on the 3-run side -- this is disclosed per row, not hidden.

Two readings the paired shift can support (see accuracy_revision.md and
docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md for the full discussion):
  - entropy_cue ~= entropy_plain, but search calls still shift under the cue
    (already established elsewhere) => PURE POLICY SHORTCUT: the trigger moved,
    the belief didn't.
  - entropy_cue != entropy_plain => the cue is doing something to the model's
    actual self-consistency (anchoring/suggestibility), not just its decision
    threshold -- a different, still-interesting, less clean claim.

Two diagnostics before trusting any entropy_cue != entropy_plain shift as real:
  1. Response-length covariate: cues can change verbosity/format enough to
     change how the LLM-judge clusters otherwise-equivalent answers, inflating
     or deflating entropy as a measurement artifact rather than a real belief
     shift (see [[project_cue_final_response_axes]] for precedent -- ELABORATE's
     dominant effect elsewhere in this project was de-formatting, not hedging).
     Computed here directly from the raw per-run rollout text, no LLM call
     needed. If |delta entropy| correlates tightly with |delta length| across
     examples, be skeptical of the shift as a belief-manipulation finding.
  2. Cross-reference against the existing level-shift/slope-change mechanism
     classification (cue_suppression_mechanism.csv): if a cue lands in a
     "level shift only" cell (necessity-tracking slope statistically intact)
     AND shows flat entropy here, that is the cleanest available evidence for
     the pure-policy-shortcut reading specifically (not just consistent with
     it -- direct, since it rules out the belief-manipulation alternative for
     that exact cell).

This does NOT do the modal-answer redirection check (joint semantic comparison
of the plain vs. cue canonical answer) -- see analyze_modal_answer_shift*.py.

Usage:
    uv run python scripts/analyze_entropy_under_cue.py
"""
import csv
import glob
import json
import os
import re

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "entropy_under_cue")
os.makedirs(OUT_DIR, exist_ok=True)

TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
        "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b",
        "nemotron-cascade-2_30b": "nemotron-cascade-2:30b", "qwen3.5_122b": "qwen3.5:122b"}

DATASETS = {
    "frames": dict(dir="results/frames_parametric", prefix="frames-cues",
                    mechanism_prefix="verbose_"),
    "medqa": dict(dir="results/medqa_parametric", prefix="medqa-500",
                   mechanism_prefix="orig_"),
}

MECHANISM_CSV = os.path.join(REPO, "results", "cue_suppression_mechanism", "cue_suppression_mechanism.csv")


def discover_cluster_files(model_dir, prefix, tag):
    """Glob every *_llm_clusters[_5run].json under model_dir for this
    dataset/model and group by cue (None = plain baseline), each holding
    whichever of {'3run', '5run'} actually exist as {resolution: path}."""
    pattern = re.compile(
        r"^" + re.escape(f"{prefix}_no_search_{tag}") + r"(?:_(?P<cue>[a-z0-9_]+?))?_llm_clusters(?P<five>_5run)?\.json$"
    )
    out = {}  # cue_or_None -> {"3run": path, "5run": path}
    for path in glob.glob(os.path.join(model_dir, f"{prefix}_no_search_{tag}*_llm_clusters*.json")):
        fname = os.path.basename(path)
        m = pattern.match(fname)
        if not m:
            continue
        cue = m.group("cue")
        res = "5run" if m.group("five") else "3run"
        out.setdefault(cue, {})[res] = path
    return out


def pick_best(entry):
    """entry: {"3run": path, "5run": path} (either key may be absent).
    STRICT 5-run only, no 3-run fallback: 5-run coverage is now complete
    enough (all 6 models, both datasets, per the latest reclustering pass)
    that mixing resolutions is no longer necessary -- and avoiding the mix
    sidesteps the exact low-n/mismatched-resolution issue found earlier for
    nemotron-cascade-2_30b (309/501 examples on the stale 3-run plain file)."""
    if "5run" in entry:
        return entry["5run"], "5run"
    return None, None


def load_entropy(path):
    data = json.load(open(path))
    return {row["example_id"]: row["semantic_entropy"] for row in data}


def load_mean_length(model_dir, prefix, tag, cue):
    """Mean sampler_response character length per example_id, averaged over
    every available run_N file for this condition (no fixed run count --
    resolution varies per cell, so just use whatever raw rollouts exist)."""
    pat = f"{prefix}_no_search_{tag}_{cue}_run_*.json" if cue else f"{prefix}_no_search_{tag}_run_*.json"
    paths = glob.glob(os.path.join(model_dir, pat))
    paths = [p for p in paths if re.search(r"_run_\d+\.json$", p)]
    if not paths:
        return None
    sums, counts = {}, {}
    for path in paths:
        for row in json.load(open(path)):
            eid = row["example_id"]
            resp = row.get("sampler_response") or ""
            sums[eid] = sums.get(eid, 0) + len(resp)
            counts[eid] = counts.get(eid, 0) + 1
    return {eid: sums[eid] / counts[eid] for eid in sums}


def main():
    mechanism = {}
    if os.path.exists(MECHANISM_CSV):
        for r in csv.DictReader(open(MECHANISM_CSV)):
            mechanism[(r["dataset"], r["model"], r["cue"])] = r

    rows = []
    skipped_no_plain = []
    for ds, cfg in DATASETS.items():
        for model, tag in TAGS.items():
            model_dir = os.path.join(REPO, cfg["dir"], model)
            if not os.path.isdir(model_dir):
                continue
            by_cue = discover_cluster_files(model_dir, cfg["prefix"], tag)

            plain_path, plain_res = pick_best(by_cue.get(None, {}))
            if plain_path is None:
                skipped_no_plain.append((ds, model))
                continue
            entropy_plain = load_entropy(plain_path)

            for cue, entry in sorted((k, v) for k, v in by_cue.items() if k is not None):
                if cue.startswith("searchmulti"):
                    continue  # dropped: the no-search searchmulti collection was noisy, per user
                cue_path, cue_res = pick_best(entry)
                if cue_path is None:
                    continue  # cue exists only at 3-run resolution, no 5-run yet
                entropy_cue = load_entropy(cue_path)

                common = sorted(set(entropy_plain) & set(entropy_cue), key=str)
                n = len(common)
                if n < 20:
                    continue
                ep = np.array([entropy_plain[e] for e in common])
                ec = np.array([entropy_cue[e] for e in common])
                delta = ec - ep

                n_up = int((delta > 0).sum())
                n_down = int((delta < 0).sum())
                n_flat = int((delta == 0).sum())
                sign_p = stats.binomtest(n_up, n_up + n_down, 0.5).pvalue if (n_up + n_down) > 0 else 1.0
                try:
                    wilcoxon_p = stats.wilcoxon(ep, ec).pvalue if n_up + n_down > 0 else 1.0
                except ValueError:
                    wilcoxon_p = float("nan")
                rho, rho_p = stats.spearmanr(ep, ec)

                len_plain = load_mean_length(model_dir, cfg["prefix"], tag, None)
                len_cue = load_mean_length(model_dir, cfg["prefix"], tag, cue)
                len_corr = ""
                mean_len_delta = ""
                if len_plain is not None and len_cue is not None:
                    common_len = [e for e in common if e in len_plain and e in len_cue]
                    lp = np.array([len_plain[e] for e in common_len])
                    lc = np.array([len_cue[e] for e in common_len])
                    dl = lc - lp
                    de = np.array([entropy_cue[e] - entropy_plain[e] for e in common_len])
                    mean_len_delta = round(float(dl.mean()), 1)
                    if len(set(np.abs(de))) > 1 and len(set(np.abs(dl))) > 1:
                        len_corr, _ = stats.spearmanr(np.abs(de), np.abs(dl))
                        len_corr = round(float(len_corr), 3)

                mech_key = (ds, model, cfg["mechanism_prefix"] + cue)
                mech = mechanism.get(mech_key, {})

                rows.append(dict(
                    dataset=ds, model=model, cue=cue, n=n,
                    plain_resolution=plain_res, cue_resolution=cue_res,
                    mean_entropy_plain=round(float(ep.mean()), 3),
                    mean_entropy_cue=round(float(ec.mean()), 3),
                    mean_delta=round(float(delta.mean()), 3),
                    n_up=n_up, n_down=n_down, n_flat=n_flat,
                    sign_test_p=f"{sign_p:.2g}",
                    wilcoxon_p=f"{wilcoxon_p:.2g}" if wilcoxon_p == wilcoxon_p else "",
                    rho_plain_vs_cue=round(float(rho), 3) if rho == rho else "",
                    mean_response_length_delta_chars=mean_len_delta,
                    abs_delta_entropy_vs_abs_delta_length_rho=len_corr,
                    mechanism=mech.get("mechanism", ""),
                    level_shift=mech.get("level_shift", ""),
                    slope_change=mech.get("slope_change", ""),
                    q_slope_change=mech.get("q_slope_change", ""),
                ))

    if skipped_no_plain:
        print(f"Skipped {len(skipped_no_plain)} (dataset,model) with cue cluster data but NO plain baseline "
              f"cluster file yet (clustering pending, not a data gap): {skipped_no_plain}\n")

    if not rows:
        print("No cells found -- check that the listed cue cluster files exist.")
        return

    out_path = os.path.join(OUT_DIR, "entropy_under_cue.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}  ({len(rows)} rows)\n")

    n_mixed_res = sum(1 for r in rows if r["plain_resolution"] != r["cue_resolution"])
    print(f"Resolution: {sum(1 for r in rows if r['plain_resolution']=='5run' and r['cue_resolution']=='5run')} "
          f"cells 5run/5run, {n_mixed_res} cells mixed resolution (plain != cue), "
          f"{sum(1 for r in rows if r['plain_resolution']=='3run' and r['cue_resolution']=='3run')} cells 3run/3run.\n")

    print("=== entropy shift under cue, vs. cue-free plain baseline ===")
    for r in sorted(rows, key=lambda r: (r["dataset"], r["model"], r["cue"])):
        sig = " *" if r["sign_test_p"] != "" and float(r["sign_test_p"]) < 0.05 else ""
        res_flag = f"[{r['plain_resolution']}/{r['cue_resolution']}]"
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:22s} {res_flag:14s} n={r['n']:4d}  "
              f"H: {r['mean_entropy_plain']:.3f} -> {r['mean_entropy_cue']:.3f} "
              f"(delta={r['mean_delta']:+.3f}, up={r['n_up']}/down={r['n_down']}/flat={r['n_flat']}, "
              f"sign_p={r['sign_test_p']}{sig})  rho(plain,cue)={r['rho_plain_vs_cue']}  "
              f"mech={r['mechanism'] or '?'}  len_delta={r['mean_response_length_delta_chars']}  "
              f"|dH|~|dLen| rho={r['abs_delta_entropy_vs_abs_delta_length_rho']}")

    print("\n=== cells with a significant entropy shift (sign test p<0.05) ===")
    sig_rows = [r for r in rows if r["sign_test_p"] != "" and float(r["sign_test_p"]) < 0.05]
    for r in sig_rows:
        reading = "BELIEF SHIFT (entropy itself moves under the cue)"
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:22s}  delta={r['mean_delta']:+.3f}  "
              f"mechanism={r['mechanism'] or 'not yet classified'}  -> {reading}")
    flat_rows = [r for r in rows if r not in sig_rows]
    print(f"\n{len(sig_rows)}/{len(rows)} cells show a significant entropy shift; "
          f"{len(flat_rows)}/{len(rows)} show flat entropy under the cue (candidate pure-policy-shortcut cells).")
    for r in flat_rows:
        print(f"  FLAT: {r['dataset']:6s} {r['model']:20s} {r['cue']:22s}  "
              f"mechanism={r['mechanism'] or 'not yet classified'}")


if __name__ == "__main__":
    main()
