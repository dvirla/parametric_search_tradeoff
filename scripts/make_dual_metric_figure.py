import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "dual_metric_appendix")
os.makedirs(OUT_DIR, exist_ok=True)

BLUE = "#4575b4"
RED = "#d73027"
GREY = "#777777"

cond_rows = list(csv.DictReader(open(os.path.join(REPO, "results", "dual_metric_condition_table.csv"))))
delta_rows = list(csv.DictReader(open(os.path.join(REPO, "results", "dual_metric_cue_deltas.csv"))))

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))

# --- Panel A: level agreement scatter, llm_acc vs regex_acc ---
ax = axes[0]
for ds, color, marker in [("frames", BLUE, "o"), ("medqa", RED, "^")]:
    xs = [float(r["regex_acc"]) for r in cond_rows if r["dataset"] == ds]
    ys = [float(r["llm_acc"]) for r in cond_rows if r["dataset"] == ds]
    ax.scatter(xs, ys, s=22, alpha=0.65, color=color, marker=marker, label=ds.upper(), edgecolors="none")
ax.plot([0, 1], [0, 1], color=GREY, lw=1, ls="--", zorder=0)
ax.set_xlabel("Regex (SQuAD-style EM) accuracy")
ax.set_ylabel("LLM-judge accuracy")
ax.set_title("A. Per-condition accuracy: EM vs. LLM judge\n(each point = one model x cue x phrasing cell)")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.legend(frameon=False, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)

# --- Panel B: per-cue delta-vs-plain, llm vs regex ---
ax = axes[1]
by_cue = defaultdict(lambda: {"d_llm": [], "d_regex": []})
for r in delta_rows:
    by_cue[r["cue"]]["d_llm"].append(float(r["d_llm"]))
    by_cue[r["cue"]]["d_regex"].append(float(r["d_regex"]))

cue_order = ["direct", "query", "polite", "natural", "elaborate", "hedge", "boost"]
cue_order = [c for c in cue_order if c in by_cue]
means_llm = [100 * np.mean(by_cue[c]["d_llm"]) for c in cue_order]
means_regex = [100 * np.mean(by_cue[c]["d_regex"]) for c in cue_order]
sem_llm = [100 * np.std(by_cue[c]["d_llm"], ddof=1) / np.sqrt(len(by_cue[c]["d_llm"])) for c in cue_order]
sem_regex = [100 * np.std(by_cue[c]["d_regex"], ddof=1) / np.sqrt(len(by_cue[c]["d_regex"])) for c in cue_order]

y = np.arange(len(cue_order))
h = 0.35
ax.barh(y + h / 2, means_llm, height=h, xerr=sem_llm, color=BLUE, label="LLM judge", capsize=2)
ax.barh(y - h / 2, means_regex, height=h, xerr=sem_regex, color=RED, label="Regex EM", capsize=2)
ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(cue_order)
ax.set_xlabel("Mean accuracy delta vs. plain (pp), pooled across models & datasets")
ax.set_title("B. Per-cue accuracy shift: same direction,\ndifferent magnitude, under both metrics", pad=38)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.invert_yaxis()
ax.set_xlim(-9.5, 2.5)

fig.tight_layout()
for ext in ("pdf", "png"):
    path = os.path.join(OUT_DIR, f"dual_metric_agreement.{ext}")
    fig.savefig(path, dpi=200)
    print("Wrote", path)
