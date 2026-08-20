"""
Does a cue's accuracy effect route THROUGH its search-volume change (the model
searches less, commits to an answer sooner, gets it wrong more often on exactly
the examples where search dropped) -- or does the cue change the ANSWER
directly, independent of whether search volume changed at all (a synthesis/
response-generation effect, e.g. length/hedging/format changes documented
separately in project_cue_final_response_axes)?

This matters specifically for the §10 "level shift only, calibration intact"
cues: those cues don't erode necessity-TRACKING (the decision of whether to
search still follows entropy about as well as under plain), but that alone
doesn't tell us whether the accuracy cost of searching less is mediated by the
reduced search itself, or is coming from somewhere else entirely.

Design: per (dataset, model, cue), split examples into two subgroups relative
to plain:
  - AFFECTED:   sampler_search_calls changed between plain and this cue
  - UNAFFECTED: sampler_search_calls identical between plain and this cue
and compare accuracy delta (LLM-judge sampler_correct) in each subgroup.
  - delta_affected >> delta_unaffected (in magnitude)  -> consistent with
    search-volume-MEDIATED accuracy cost (fewer searches -> worse answers,
    specifically on the examples where search actually dropped)
  - delta_affected ~ delta_unaffected                  -> accuracy shift is
    roughly uniform regardless of whether search changed -> NOT mediated by
    search volume; more consistent with a direct effect of the cue on answer
    synthesis/generation.
This is descriptive (subgroup means, not a formal mediation model with
significance testing of the indirect path) -- treat as suggestive evidence
about mechanism, not a proven causal decomposition.

Usage:
    uv run python scripts/analyze_search_mediation.py
"""
import csv
import glob
import json
import os
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT_DIR = os.path.join(REPO, "results", "search_mediation")
os.makedirs(OUT_DIR, exist_ok=True)

from analyze_necessity_vs_template_search_5run import (  # noqa: E402
    MODELS, frames_cond, medqa_cond, plain_cond_for, discover_conditions,
)

DATASETS = {
    "frames": dict(search_dir="results/frames_cues_full", cond_fn=frames_cond),
    "medqa": dict(search_dir="results/medqa_grid", cond_fn=medqa_cond),
}


def load_rows(path):
    data = json.load(open(path))
    out = {}
    for row in data:
        eid = row["example_id"]
        out[eid] = dict(calls=row.get("sampler_search_calls"),
                         correct=row.get("sampler_correct"))
    return out


def main():
    results = []

    for ds, cfg in DATASETS.items():
        for model in MODELS:
            search_dir = os.path.join(cfg["search_dir"], model)
            conditions = discover_conditions(search_dir, cfg["cond_fn"])
            if not conditions:
                continue

            for cond, cue_path in sorted(conditions.items()):
                plain_name = plain_cond_for(cond)
                if cond == plain_name or plain_name not in conditions:
                    continue
                plain = load_rows(conditions[plain_name])
                cue = load_rows(cue_path)

                common = sorted(set(plain) & set(cue), key=str)
                common = [e for e in common
                          if plain[e]["correct"] is not None and cue[e]["correct"] is not None
                          and plain[e]["calls"] is not None and cue[e]["calls"] is not None]
                n = len(common)
                if n < 30:
                    continue

                affected = [e for e in common if plain[e]["calls"] != cue[e]["calls"]]
                unaffected = [e for e in common if plain[e]["calls"] == cue[e]["calls"]]
                if len(affected) < 10 or len(unaffected) < 10:
                    continue

                def delta_acc(subset):
                    p = np.mean([float(plain[e]["correct"]) for e in subset])
                    c = np.mean([float(cue[e]["correct"]) for e in subset])
                    return c - p, p, c

                d_aff, p_aff, c_aff = delta_acc(affected)
                d_unaff, p_unaff, c_unaff = delta_acc(unaffected)

                if abs(d_aff) < 1e-9 and abs(d_unaff) < 1e-9:
                    pattern = "no accuracy change"
                elif abs(d_unaff) < 1e-9 or abs(d_aff) > 2.5 * abs(d_unaff):
                    pattern = "search-mediated (concentrated in affected subgroup)"
                elif abs(d_aff - d_unaff) < 0.03:
                    pattern = "direct/uniform (not search-mediated)"
                else:
                    pattern = "mixed"

                results.append(dict(
                    dataset=ds, model=model, cue=cond,
                    n_affected=len(affected), n_unaffected=len(unaffected),
                    acc_plain_affected=round(p_aff, 3), acc_cue_affected=round(c_aff, 3),
                    delta_affected=round(d_aff, 4),
                    acc_plain_unaffected=round(p_unaff, 3), acc_cue_unaffected=round(c_unaff, 3),
                    delta_unaffected=round(d_unaff, 4),
                    pattern=pattern,
                ))

    out_path = os.path.join(OUT_DIR, "search_mediation.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    print(f"wrote {out_path} ({len(results)} rows)\n")

    from collections import Counter
    print("=== pattern counts, FRAMES ===")
    print(Counter(r["pattern"] for r in results if r["dataset"] == "frames"))
    print("\n=== pattern counts, MedQA ===")
    print(Counter(r["pattern"] for r in results if r["dataset"] == "medqa"))

    print("\n=== search-mediated cells (accuracy cost concentrated where search actually changed) ===")
    for r in sorted(results, key=lambda r: r["delta_affected"]):
        if r["pattern"] == "search-mediated (concentrated in affected subgroup)":
            print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:26s}  "
                  f"affected(n={r['n_affected']:3d}): {r['acc_plain_affected']:.3f}->{r['acc_cue_affected']:.3f} "
                  f"(d={r['delta_affected']:+.3f})   "
                  f"unaffected(n={r['n_unaffected']:3d}): {r['acc_plain_unaffected']:.3f}->{r['acc_cue_unaffected']:.3f} "
                  f"(d={r['delta_unaffected']:+.3f})")

    print("\n=== direct/uniform cells (accuracy shift NOT concentrated where search changed) ===")
    for r in sorted(results, key=lambda r: r["delta_affected"]):
        if r["pattern"] == "direct/uniform (not search-mediated)" and abs(r["delta_affected"]) > 0.02:
            print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:26s}  "
                  f"affected(n={r['n_affected']:3d}): {r['acc_plain_affected']:.3f}->{r['acc_cue_affected']:.3f} "
                  f"(d={r['delta_affected']:+.3f})   "
                  f"unaffected(n={r['n_unaffected']:3d}): {r['acc_plain_unaffected']:.3f}->{r['acc_cue_unaffected']:.3f} "
                  f"(d={r['delta_unaffected']:+.3f})")


if __name__ == "__main__":
    main()
