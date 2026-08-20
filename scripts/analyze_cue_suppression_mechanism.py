"""
Mechanistic decomposition of HOW each cue changes search-call behavior relative
to the plain condition's necessity-calibrated line, per (dataset, model, cue).

Supersedes the pooled calls~entropy+C(condition) framing in
analyze_entropy_vs_cue_variance.py for this specific question: that pooled
model answers "how much of TOTAL variance is explained by cue-identity vs
necessity" (an omnibus accounting question) but throws away which mechanism
any individual cue uses, since every cue is folded into one nuisance factor.

This script instead reuses the PER-CUE regression already fit in
analyze_necessity_vs_template_search{,_5run}.py (calls ~ entropy + is_cue +
entropy:is_cue, fit separately for plain vs. each cue) and decomposes each
cue's effect on the entropy-search relationship into two orthogonal pieces:

  - LEVEL SHIFT   (b_is_cue): how much the cue changes calls at entropy=0,
    i.e. suppression/boost that has nothing to do with necessity.
  - SLOPE CHANGE  (b_interaction): how much the cue changes the model's
    necessity-SENSITIVITY itself. slope_under_cue = b_entropy + b_interaction
    vs. slope_under_plain = b_entropy.

Four qualitatively different mechanisms fall out of the (level_shift,
slope_change) pair:
  - level shift only, slope preserved      -> "blanket suppression, calibration
                                                intact" (cue subtracts K calls
                                                everywhere, model still tracks
                                                its own uncertainty just as well)
  - slope collapses toward/through zero     -> "calibration destroyed" (cue makes
                                                search calls stop tracking
                                                necessity at all)
  - slope grows more negative (flips sign)  -> "calibration inverted"
  - slope increases (more necessity-sensitive)-> "calibration sharpened"

Usage:
    uv run python scripts/analyze_cue_suppression_mechanism.py
"""
import csv
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN_PATH = os.path.join(REPO, "results", "necessity_vs_template_5run", "necessity_vs_template_interaction.csv")
OUT_DIR = os.path.join(REPO, "results", "cue_suppression_mechanism")
os.makedirs(OUT_DIR, exist_ok=True)


def classify(slope_plain, slope_cue, interaction_sig):
    if not interaction_sig:
        return "level shift only (calibration intact)"
    if slope_plain == 0:
        return "n/a (no baseline slope)"
    ratio = slope_cue / slope_plain
    if ratio <= 0:
        return "calibration inverted"
    if ratio < 0.7:
        return "calibration eroded"
    if ratio > 1.3:
        return "calibration sharpened"
    return "calibration marginally changed (small but detectable)"


def main():
    rows = list(csv.DictReader(open(IN_PATH)))
    out_rows = []
    for r in rows:
        b_entropy = float(r["b_entropy"])
        b_is_cue = float(r["b_is_cue"])
        b_interaction = float(r["b_interaction"])
        q_int = float(r["p_interaction_fdr"])
        q_level = None  # p_is_cue not FDR-corrected in source; report raw
        slope_cue = b_entropy + b_interaction
        mechanism = classify(b_entropy, slope_cue, interaction_sig=(q_int < 0.05))
        out_rows.append(dict(
            dataset=r["dataset"], model=r["model"], cue=r["cue"], n=r["n"],
            slope_plain=round(b_entropy, 4),
            level_shift=round(b_is_cue, 4), p_level_shift=r["p_is_cue"],
            slope_change=round(b_interaction, 4), q_slope_change=round(q_int, 4),
            slope_under_cue=round(slope_cue, 4),
            slope_ratio=round(slope_cue / b_entropy, 3) if b_entropy != 0 else float("nan"),
            mechanism=mechanism,
        ))

    out_path = os.path.join(OUT_DIR, "cue_suppression_mechanism.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {out_path} ({len(out_rows)} rows)\n")

    from collections import Counter
    print("=== mechanism counts (FRAMES) ===")
    print(Counter(r["mechanism"] for r in out_rows if r["dataset"] == "frames"))
    print("\n=== mechanism counts (MedQA) ===")
    print(Counter(r["mechanism"] for r in out_rows if r["dataset"] == "medqa"))

    print("\n=== cells with a significant slope_change (q<0.05), sorted by |slope_change| ===")
    sig = [r for r in out_rows if r["q_slope_change"] < 0.05]
    sig.sort(key=lambda r: -abs(r["slope_change"]))
    for r in sig:
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:26s}  "
              f"slope_plain={r['slope_plain']:+.2f}  level_shift={r['level_shift']:+.2f}  "
              f"slope_change={r['slope_change']:+.2f} (q={r['q_slope_change']:.3g})  "
              f"slope_under_cue={r['slope_under_cue']:+.2f}  ratio={r['slope_ratio']:+.2f}  "
              f"-> {r['mechanism']}")


if __name__ == "__main__":
    main()
