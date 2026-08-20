"""
5-run version of make_necessity_vs_template_figure.py -- reads the 5-run entropy
causal-interaction results instead of the 3-run ones.

Heatmap of the necessity x cue interaction coefficient (calls ~ entropy * is_cue,
cluster-robust by example) from scripts/analyze_necessity_vs_template_search.py,
one panel per dataset, cues (rows) x models (columns).

Color = b_interaction (diverging, BLUE=positive/necessity-shrinking = cue effect
weakens as necessity rises = "calibrated"; RED=negative = cue effect strengthens
as necessity rises = "anti-calibrated"). Cells significant after Benjamini-Hochberg
FDR correction (q<0.05, across all 122 tests) are boxed and starred; all others are
shown at reduced alpha so the eye is drawn to what actually survived correction.

Usage:
    uv run python scripts/make_necessity_vs_template_figure.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "necessity_vs_template_5run")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#4575b4"  # calibrated (interaction > 0)
RED = "#d73027"   # anti-calibrated (interaction < 0)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_LABELS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
                 "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron3:30b"}

# canonical row order: primary phrasing (verbose/orig) cues first, then the
# terse phrasing-mismatched variants, each alphabetical within its group
BASE_ORDER = ["confident_parametric", "direct", "elaborate", "natural", "polite",
              "query", "multiturn", "searchmulti", "searchmulti2", "searchmulti3",
              "epi_strong_boost", "epi_strong_hedge"]


def sig_stars(q):
    if q is None or q != q:
        return ""
    if q < 0.001:
        return "***"
    if q < 0.01:
        return "**"
    if q < 0.05:
        return "*"
    return ""


def load_rows():
    return list(csv.DictReader(open(os.path.join(REPO, "results", "necessity_vs_template_5run",
                                                    "necessity_vs_template_interaction.csv"))))


def row_label_and_base(cue):
    is_terse = cue.startswith("terse_")
    base = cue.split("_", 1)[1] if cue.startswith(("verbose_", "orig_", "terse_")) else cue
    label = base + (" (terse)" if is_terse else "")
    return label, base, is_terse


def build_matrix(rows, dataset):
    sub = [r for r in rows if r["dataset"] == dataset]
    cues = sorted(set(r["cue"] for r in sub),
                  key=lambda c: (BASE_ORDER.index(row_label_and_base(c)[1])
                                 if row_label_and_base(c)[1] in BASE_ORDER else 99,
                                 row_label_and_base(c)[2], c))
    labels = [row_label_and_base(c)[0] for c in cues]
    mat = np.full((len(cues), len(MODELS)), np.nan)
    qmat = np.full((len(cues), len(MODELS)), np.nan)
    for r in sub:
        i = cues.index(r["cue"])
        j = MODELS.index(r["model"]) if r["model"] in MODELS else None
        if j is None:
            continue
        mat[i, j] = float(r["b_interaction"])
        qmat[i, j] = float(r["p_interaction_fdr"])
    return labels, mat, qmat


def plot_panel(ax, labels, mat, qmat, title):
    vmax = np.nanmax(np.abs(mat))
    cmap = plt.get_cmap("RdBu")  # RdBu: low=red, high=blue -- matches BLUE=+, RED=-
    im = ax.imshow(mat, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v, q = mat[i, j], qmat[i, j]
            if v != v:
                ax.text(j, i, "-", ha="center", va="center", fontsize=7, color="#999999")
                continue
            sig = q == q and q < 0.05
            stars = sig_stars(q)
            txt_color = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}{stars}", ha="center", va="center",
                     fontsize=7.5, color=txt_color, fontweight="bold" if sig else "normal")
            if sig:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                            edgecolor="black", lw=1.6))
            else:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                            facecolor="white", alpha=0.45, edgecolor="none"))

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="minor", length=0)
    return im


def main():
    rows = load_rows()
    fig = plt.figure(figsize=(11, 9.2))
    ax0 = fig.add_axes([0.16, 0.32, 0.335, 0.50])
    ax1 = fig.add_axes([0.615, 0.32, 0.335, 0.50])
    cax = fig.add_axes([0.20, 0.12, 0.62, 0.022])

    labels_f, mat_f, q_f = build_matrix(rows, "frames")
    im1 = plot_panel(ax0, labels_f, mat_f, q_f, "FRAMES")

    labels_m, mat_m, q_m = build_matrix(rows, "medqa")
    plot_panel(ax1, labels_m, mat_m, q_m, "MedQA")

    cbar = fig.colorbar(im1, cax=cax, orientation="horizontal")
    cbar.set_label(r"RED: cue suppresses MORE as necessity rises (anti-calibrated)   |   "
                    r"BLUE: cue effect shrinks as necessity rises (calibrated)",
                    fontsize=8.5)

    fig.suptitle("Does a cue's effect on search-call volume depend on the model's own\n"
                  "epistemic necessity (entropy, from a cue-free no-search probe)?",
                  fontsize=12, y=0.965)
    fig.text(0.5, 0.05,
              "Boxed + starred cells: FDR-significant (q<0.05, Benjamini-Hochberg across all "
              f"{len(rows)} model x cue x dataset tests). *q<.05 **q<.01 ***q<.001. "
              "Faded cells did not survive correction.",
              ha="center", fontsize=8, color="#555555")

    for ext in ("png", "pdf"):
        path = os.path.join(OUT_DIR, f"necessity_vs_template_heatmap.{ext}")
        fig.savefig(path, dpi=200)
        print("Wrote", path)


if __name__ == "__main__":
    main()
