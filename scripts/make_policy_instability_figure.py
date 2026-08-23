"""
The headline figure for §1.3c: three panels making the same contrast at three
levels of granularity -- the model's own uncertainty (entropy) barely moves
under a cue, its canonical answer redirects at a small, cue-independent
background rate, yet the same cues cause large, cue-SPECIFIC swings in
search-triggering behavior. Read left-to-right: stable belief -> stable-ish
answer content -> unstable policy.

Panel A: entropy shift (entropy_cue - entropy_plain), signed mean per cue,
         pooled across available models/datasets, with the fraction of cells
         reaching significance (sign test) annotated.
Panel B: modal-answer redirection rate per cue (bar), with a dashed reference
         line at the all-cues-pooled rate to show how uniform it is.
Panel C: mean |level_shift| (search-call suppression magnitude) per cue, same
         cue order as A/B, colored by conversation-state vs. other, to show
         this is where the actual instability lives.

Usage:
    uv run python scripts/make_policy_instability_figure.py
"""
import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "policy_instability_summary")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#4575b4"
RED = "#d73027"
GRAY = "#999999"
PURPLE = "#762a83"

CUE_ORDER = ["confident_parametric", "direct", "elaborate", "multiturn", "searchmulti"]
CUE_LABELS = {"confident_parametric": "confident\nparametric", "direct": "direct",
              "elaborate": "elaborate", "multiturn": "multiturn", "searchmulti": "searchmulti*"}


def load_entropy():
    rows = list(csv.DictReader(open(os.path.join(REPO, "results", "entropy_under_cue", "entropy_under_cue.csv"))))
    by_cue = defaultdict(list)
    for r in rows:
        cue = r["cue"].split("2")[0].split("3")[0] if r["cue"].startswith("searchmulti") else r["cue"]
        by_cue[cue].append((float(r["mean_delta"]), float(r["sign_test_p"]) if r["sign_test_p"] else 1.0))
    out = {}
    for cue in CUE_ORDER:
        vals = by_cue.get(cue, [])
        if not vals:
            continue
        deltas = [v[0] for v in vals]
        n_sig = sum(1 for v in vals if v[1] < 0.05)
        out[cue] = dict(mean_delta=np.mean(deltas), n=len(vals), n_sig=n_sig)
    return out


def load_modal():
    rows = list(csv.DictReader(open(os.path.join(REPO, "results", "modal_answer_shift", "modal_answer_shift_judged.csv"))))
    by_cue = defaultdict(lambda: {"n": 0, "changed": 0})
    for r in rows:
        cue = r["cue"].split("2")[0].split("3")[0] if r["cue"].startswith("searchmulti") else r["cue"]
        by_cue[cue]["n"] += int(r["n"])
        by_cue[cue]["changed"] += int(r["n_changed"])
    total_n = sum(d["n"] for d in by_cue.values())
    total_changed = sum(d["changed"] for d in by_cue.values())
    pooled_pct = 100 * total_changed / total_n
    out = {cue: 100 * by_cue[cue]["changed"] / by_cue[cue]["n"] for cue in CUE_ORDER if by_cue[cue]["n"] > 0}
    return out, pooled_pct


def load_feature_axes():
    rows = list(csv.DictReader(open(os.path.join(REPO, "results", "cue_feature_axes", "cue_feature_axes_summary.csv"))))
    out = {}
    for r in rows:
        base = r["cue"]
        key = "searchmulti" if base.startswith("searchmulti") else base
        if key in CUE_ORDER:
            prev = out.get(key)
            val = abs(float(r["mean_level_shift"]))
            if prev is None or val > prev:  # take the max |shift| among searchmulti/2/3 for the pooled bar
                out[key] = val
    return out


def main():
    entropy = load_entropy()
    modal, pooled_pct = load_modal()
    feature = load_feature_axes()

    cues = [c for c in CUE_ORDER if c in entropy and c in modal and c in feature]
    conv_state = {"multiturn", "searchmulti"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

    ax = axes[0]
    x = np.arange(len(cues))
    deltas = [entropy[c]["mean_delta"] for c in cues]
    colors = [RED if entropy[c]["n_sig"] > 0 else BLUE for c in cues]
    ax.bar(x, deltas, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([CUE_LABELS[c] for c in cues], fontsize=8)
    ax.set_ylabel("mean entropy shift (bits)\ncue - plain")
    ax.set_title("A. Uncertainty: ~flat", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    for xi, c in zip(x, cues):
        ax.annotate(f"{entropy[c]['n_sig']}/{entropy[c]['n']} sig.", (xi, deltas[x.tolist().index(xi)]),
                    textcoords="offset points", xytext=(0, 5 if deltas[x.tolist().index(xi)] >= 0 else -12),
                    fontsize=6.5, ha="center", color="#555555")

    ax = axes[1]
    vals = [modal[c] for c in cues]
    ax.bar(x, vals, color=PURPLE)
    ax.axhline(pooled_pct, color="black", lw=1, ls="--")
    ax.annotate(f"pooled: {pooled_pct:.1f}%", (len(cues) - 0.5, pooled_pct), fontsize=7,
                ha="right", va="bottom", color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([CUE_LABELS[c] for c in cues], fontsize=8)
    ax.set_ylabel("% examples with a different\ncanonical (modal) answer")
    ax.set_title("B. Belief content: ~uniform, cue-independent", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    vals = [feature[c] for c in cues]
    colors = [RED if c in conv_state else GRAY for c in cues]
    ax.bar(x, vals, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels([CUE_LABELS[c] for c in cues], fontsize=8)
    ax.set_ylabel("mean |search-call level shift|")
    ax.set_title("C. Search-triggering POLICY: unstable", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=RED, label="conversation-state cue"),
                        Patch(color=GRAY, label="question-text cue")],
              fontsize=7, frameon=False, loc="upper left")

    fig.suptitle("The search-triggering policy shifts far more than the model's own uncertainty or belief content does",
                 fontsize=11.5, y=1.03)
    fig.text(0.5, -0.05,
              "Same cues, same questions, three outcome measures. A = paired entropy shift under the cue vs. the model's\n"
              "own cue-free baseline (red = reaches significance in >=1 cell, sign test p<0.05). B = rate the model's\n"
              "modal no-search answer changes to something judged non-equivalent. C = mean level-shift in live search\n"
              "calls (mechanism decomposition, §1.3), corrected for the mocked-history-call counting bug (§1.0).",
              ha="center", fontsize=7.8, color="#555555")
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"policy_instability_summary.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print("Wrote", path)


if __name__ == "__main__":
    main()
