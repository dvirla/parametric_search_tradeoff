"""
A much cleaner identification strategy for the search-volume-mediation question
than analyze_causal_mediation.py's observational path model, which broke because
realized search-call count is endogenous (generated during the same rollout as
the answer -- see accuracy_revision.md for the diagnostic: within-entropy-stratum
calls-vs-correctness correlation is strongly NEGATIVE, r=-0.32 to -0.66, the
signature of reverse causation: struggling causes more searching, not the
reverse).

This script instead uses an actual MANIPULATED mediator: `gemma4-frames-robust-*`
is gemma4:31b SFT'd on curated on-policy rollouts, filtered during data curation
to keep only training examples where |search-call delta vs. the plain-condition
rollout| <= 1 AND the answer was correct (see project_frames_cue_robustness_sft
memory). This directly intervenes on the mediator: it trains the model to keep
search volume roughly constant across cues, instead of relying on naturally-
occurring (confounded) variation in search-call count.

Design: compare the SAME 8 cues on the SAME 102 held-out FRAMES test examples,
BASE gemma4:31b (results/frames_cues_full/gemma4_31b) vs. the SFT'd
gemma4-frames-robust-q4km (results/frames_cue_eval_test/gemma4-frames-robust-q4km),
restricted to exactly the 102 example_ids in the SFT eval set. Both graded with
the SAME regex grader (heuristic_match OR relaxed_match from regrade_regex.py --
required since the SFT eval files were never LLM-judge graded, sampler_correct
is None throughout; grading BOTH models the same way keeps the comparison
apples-to-apples rather than introducing a grader-mismatch confound).

Per cue, four numbers side by side:
  d_calls_base = mean(calls_cue) - mean(calls_plain), BASE model
  d_calls_sft  = mean(calls_cue) - mean(calls_plain), SFT model
  d_acc_base   = acc_cue - acc_plain (regex), BASE model
  d_acc_sft    = acc_cue - acc_plain (regex), SFT model

If the SFT intervention successfully flattens d_calls (|d_calls_sft| << |d_calls_base|)
AND ALSO flattens d_acc (|d_acc_sft| << |d_acc_base|), that's direct evidence the
accuracy cost IS causally downstream of the search-volume change (removing the
mediator removes the effect). If d_calls flattens but d_acc does NOT, that's
evidence the accuracy cost is NOT mediated by search volume -- something else
about the cue (answer-generation/synthesis) is doing it, and search-call
robustness training didn't touch that channel.

Significance: paired McNemar's test (base vs cue correctness, per example) for
accuracy; Wilcoxon signed-rank for calls. n=102 -- small, treat p-values as
suggestive, not definitive.

Usage:
    uv run python scripts/analyze_sft_intervention_mediation.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT_DIR = os.path.join(REPO, "results", "sft_intervention_mediation")
os.makedirs(OUT_DIR, exist_ok=True)

from regrade_regex import heuristic_match, relaxed_match  # noqa: E402

BASE_DIR = os.path.join(REPO, "results", "frames_cues_full", "gemma4_31b")
SFT_DIR = os.path.join(REPO, "results", "frames_cue_eval_test", "gemma4-frames-robust-q4km")

CUES = ["direct", "elaborate", "multiturn", "natural", "polite", "query", "searchmulti", "confident_parametric"]


def regex_correct(gold, resp):
    return heuristic_match(gold, resp) or relaxed_match(gold, resp)


def load(path):
    rows = json.load(open(path))
    out = {}
    for r in rows:
        eid = r["example_id"]
        out[eid] = dict(
            calls=r.get("sampler_search_calls"),
            correct=regex_correct(r.get("correct_answer") or "", r.get("sampler_response") or ""),
        )
    return out


def mcnemar_p(plain_correct, cue_correct):
    """Exact/normal-approx McNemar on paired binary outcomes."""
    b = sum(1 for p, c in zip(plain_correct, cue_correct) if p and not c)  # plain right, cue wrong
    c_ = sum(1 for p, c in zip(plain_correct, cue_correct) if not p and c)  # plain wrong, cue right
    n_disc = b + c_
    if n_disc == 0:
        return 1.0, b, c_
    if n_disc <= 25:
        p = stats.binomtest(min(b, c_), n_disc, 0.5, alternative="two-sided").pvalue
    else:
        z = (abs(b - c_) - 1) / np.sqrt(n_disc)
        p = 2 * (1 - stats.norm.cdf(z))
    return p, b, c_


def main():
    sft_plain_path = os.path.join(SFT_DIR, "frames-cues_baseline_gemma4-frames-robust-q4km_verbose_plain.json")
    sft_plain = load(sft_plain_path)
    test_ids = sorted(sft_plain, key=str)
    n_test = len(test_ids)
    print(f"Test set: {n_test} examples (from SFT eval verbose_plain)\n")

    base_plain_path = os.path.join(BASE_DIR, "frames-cues_baseline_gemma4:31b_verbose_plain.json")
    base_plain = load(base_plain_path)
    missing = [e for e in test_ids if e not in base_plain]
    if missing:
        print(f"  ! {len(missing)} test-set example_ids missing from BASE plain file -- check id alignment")
    test_ids = [e for e in test_ids if e in base_plain]

    rows_out = []
    for cue in CUES:
        sft_cue_path = os.path.join(SFT_DIR, f"frames-cues_baseline_gemma4-frames-robust-q4km_verbose_{cue}.json")
        base_cue_path = os.path.join(BASE_DIR, f"frames-cues_baseline_gemma4:31b_verbose_{cue}.json")
        if not os.path.exists(sft_cue_path) or not os.path.exists(base_cue_path):
            print(f"  ! skip {cue}: missing file(s)")
            continue
        sft_cue = load(sft_cue_path)
        base_cue = load(base_cue_path)

        ids = [e for e in test_ids if e in sft_cue and e in base_cue]
        n = len(ids)
        if n < n_test:
            print(f"  ! {cue}: only {n}/{n_test} ids present in both cue files")

        calls_base_plain = np.array([base_plain[e]["calls"] for e in ids], dtype=float)
        calls_base_cue = np.array([base_cue[e]["calls"] for e in ids], dtype=float)
        calls_sft_plain = np.array([sft_plain[e]["calls"] for e in ids], dtype=float)
        calls_sft_cue = np.array([sft_cue[e]["calls"] for e in ids], dtype=float)

        corr_base_plain = [base_plain[e]["correct"] for e in ids]
        corr_base_cue = [base_cue[e]["correct"] for e in ids]
        corr_sft_plain = [sft_plain[e]["correct"] for e in ids]
        corr_sft_cue = [sft_cue[e]["correct"] for e in ids]

        d_calls_base = calls_base_cue.mean() - calls_base_plain.mean()
        d_calls_sft = calls_sft_cue.mean() - calls_sft_plain.mean()
        d_acc_base = np.mean(corr_base_cue) - np.mean(corr_base_plain)
        d_acc_sft = np.mean(corr_sft_cue) - np.mean(corr_sft_plain)

        w_calls_base_p = stats.wilcoxon(calls_base_cue - calls_base_plain).pvalue if np.any(calls_base_cue != calls_base_plain) else 1.0
        w_calls_sft_p = stats.wilcoxon(calls_sft_cue - calls_sft_plain).pvalue if np.any(calls_sft_cue != calls_sft_plain) else 1.0
        mc_base_p, b_base, c_base = mcnemar_p(corr_base_plain, corr_base_cue)
        mc_sft_p, b_sft, c_sft = mcnemar_p(corr_sft_plain, corr_sft_cue)

        rows_out.append(dict(
            cue=cue, n=n,
            d_calls_base=round(d_calls_base, 3), p_calls_base=round(w_calls_base_p, 4),
            d_calls_sft=round(d_calls_sft, 3), p_calls_sft=round(w_calls_sft_p, 4),
            d_acc_base=round(d_acc_base, 4), p_acc_base=round(mc_base_p, 4),
            d_acc_sft=round(d_acc_sft, 4), p_acc_sft=round(mc_sft_p, 4),
            calls_swing_ratio=round(abs(d_calls_sft) / abs(d_calls_base), 3) if d_calls_base != 0 else float("nan"),
            acc_swing_ratio=round(abs(d_acc_sft) / abs(d_acc_base), 3) if d_acc_base != 0 else float("nan"),
        ))

    out_path = os.path.join(OUT_DIR, "sft_intervention_mediation.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"wrote {out_path}\n")

    print(f"{'cue':20s} {'d_calls_base':>13s} {'d_calls_sft':>12s} {'ratio':>7s}   "
          f"{'d_acc_base':>11s}(p) {'d_acc_sft':>11s}(p) {'ratio':>7s}")
    for r in rows_out:
        print(f"{r['cue']:20s} {r['d_calls_base']:+13.2f} {r['d_calls_sft']:+12.2f} {r['calls_swing_ratio']:>7.2f}   "
              f"{100*r['d_acc_base']:+9.1f}pp({r['p_acc_base']:.2g}) {100*r['d_acc_sft']:+9.1f}pp({r['p_acc_sft']:.2g}) "
              f"{r['acc_swing_ratio']:>7.2f}")

    print("\nInterpretation: calls_swing_ratio << 1 means the SFT intervention successfully flattened the "
          "cue's effect on search volume.\nacc_swing_ratio << 1 (alongside calls_swing_ratio << 1) supports "
          "search-volume-MEDIATED accuracy cost.\nacc_swing_ratio ~ 1 despite calls_swing_ratio << 1 means the "
          "accuracy cost persists even with search volume held flat -- NOT mediated by search volume.")


if __name__ == "__main__":
    main()
