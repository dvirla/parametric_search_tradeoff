"""
Reframes the Stage 2 "level shift only" cells (calls ~ entropy + is_cue +
entropy:is_cue slope statistically unchanged from plain, per
analyze_cue_suppression_mechanism.py) as a headline finding in their own right,
not a benign residual category: if a content-free cue can move search VOLUME by
a lot while accuracy barely moves, that volume was arguably not tied to
information need to begin with, independent of whether the entropy-tracking
SLOPE survives.

Joins cue_suppression_mechanism.csv's level_shift/mechanism classification
against a fresh per-cell |delta accuracy| (regex-graded for consistency across
every cue, including ones never LLM-graded -- same convention as
dual_metric_analysis.py / analyze_search_mediation.py).

Usage:
    uv run python scripts/analyze_volume_accuracy_decoupling.py
"""
import csv
import json
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT_DIR = os.path.join(REPO, "results", "cue_suppression_mechanism")

from regrade_regex import heuristic_match, relaxed_match  # noqa: E402

TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
        "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b"}
FRAMES_DIR = os.path.join(REPO, "results", "frames_cues_full")
MEDQA_DIR = os.path.join(REPO, "results", "medqa_grid")


def rc(gold, resp):
    return heuristic_match(gold, resp) or relaxed_match(gold, resp)


def load(path):
    rows = json.load(open(path))
    return {r["example_id"]: dict(calls=r.get("sampler_search_calls"),
                                    correct=rc(r.get("correct_answer") or "", r.get("sampler_response") or ""))
            for r in rows}


def plain_path(ds, model, cue):
    tag = TAGS[model]
    if ds == "frames":
        phrasing = cue.split("_", 1)[0] if cue.startswith(("terse_", "verbose_")) else "verbose"
        return os.path.join(FRAMES_DIR, model, f"frames-cues_baseline_{tag}_{phrasing}_plain.json")
    phrasing = cue.split("_", 1)[0] if cue.startswith(("terse_", "orig_")) else "orig"
    return os.path.join(MEDQA_DIR, model, f"medqa-500_baseline_{tag}_{phrasing}_plain.json")


def cue_path(ds, model, cue):
    tag = TAGS[model]
    if ds == "frames":
        return os.path.join(FRAMES_DIR, model, f"frames-cues_baseline_{tag}_{cue}.json")
    return os.path.join(MEDQA_DIR, model, f"medqa-500_baseline_{tag}_{cue}.json")


def main():
    mech_rows = list(csv.DictReader(open(os.path.join(OUT_DIR, "cue_suppression_mechanism.csv"))))
    results = []
    for r in mech_rows:
        ds, model, cue, mech = r["dataset"], r["model"], r["cue"], r["mechanism"]
        try:
            plain = load(plain_path(ds, model, cue))
            cuef = load(cue_path(ds, model, cue))
        except FileNotFoundError:
            continue
        ids = sorted(set(plain) & set(cuef))
        if len(ids) < 50:
            continue
        calls_p = np.array([plain[e]["calls"] for e in ids], dtype=float)
        calls_c = np.array([cuef[e]["calls"] for e in ids], dtype=float)
        corr_p = np.array([plain[e]["correct"] for e in ids])
        corr_c = np.array([cuef[e]["correct"] for e in ids])
        results.append(dict(
            dataset=ds, model=model, cue=cue, mechanism=mech, n=len(ids),
            base_calls=round(float(calls_p.mean()), 3),
            d_calls=round(float(calls_c.mean() - calls_p.mean()), 3),
            d_acc=round(float(corr_c.mean() - corr_p.mean()), 4),
        ))

    out_path = os.path.join(OUT_DIR, "volume_vs_accuracy_delta.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"wrote {out_path} ({len(results)} rows)\n")

    levelshift = [r for r in results if "level shift only" in r["mechanism"]]
    erosion = [r for r in results if r["mechanism"] in
               ("calibration eroded", "calibration inverted", "calibration sharpened")]

    dc = np.array([abs(r["d_calls"]) for r in levelshift])
    da = np.array([abs(r["d_acc"]) * 100 for r in levelshift])
    rho, p = stats.spearmanr(dc, da)
    big_swing = dc > 1.0
    big_swing_small_acc = big_swing & (da < 3)
    print(f"Level-shift-only cells (n={len(levelshift)}): "
          f"corr(|d_calls|, |d_acc|) rho={rho:+.3f} (p={p:.3g})")
    print(f"  median |d_calls|={np.median(dc):.2f} calls, median |d_acc|={np.median(da):.1f}pp")
    print(f"  cells with |d_calls|>1.0: {big_swing.sum()}/{len(levelshift)}")
    print(f"  of those, |d_acc|<3pp: {big_swing_small_acc.sum()}/{big_swing.sum()} "
          f"({100*big_swing_small_acc.sum()/big_swing.sum():.0f}%)")

    dc2 = np.array([abs(r["d_calls"]) for r in erosion])
    da2 = np.array([abs(r["d_acc"]) * 100 for r in erosion])
    print(f"\nMechanism-breaking cells (n={len(erosion)}), for contrast: "
          f"median |d_calls|={np.median(dc2):.2f}, median |d_acc|={np.median(da2):.1f}pp")

    print("\nMost extreme volume-without-accuracy cases:")
    for r in sorted(levelshift, key=lambda r: -abs(r["d_calls"]))[:8]:
        mult = (r["base_calls"] + r["d_calls"]) / r["base_calls"] if r["base_calls"] > 0 else float("inf")
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:24s} "
              f"{r['base_calls']:.2f}->{r['base_calls']+r['d_calls']:.2f} calls ({mult:.1f}x)  "
              f"d_acc={100*r['d_acc']:+.1f}pp")


if __name__ == "__main__":
    main()
