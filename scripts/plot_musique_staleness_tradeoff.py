"""
MuSiQue staleness tradeoff plot (3 models).

Compares the NON-STALE subset of MuSiQue questions across two phrasing
conditions — formal (`musique-formal`) vs. natural2 (`musique-natural2`) — in
terms of search calls vs. accuracy, for three models (Gemini 3.1 Pro,
Nemotron Nano 30B, Qwen 3.5 122B).

Joins are by `example_id`:
  - formal reevaluated JSON carries `example_id` directly.
  - natural2 reevaluated JSON has `example_id=None`, so its paraphrase `problem`
    is mapped back to `example_id` via `data/musique_val_natural2.jsonl`
    (`text` -> `example_id`).
  - staleness verdicts (`results/musique_staleness_flash.csv`) are keyed on
    `example_id`.

By default only faithful natural2 paraphrases are kept (validation_status=pass
AND equivalent=True); pass --keep-all-paraphrases to include flagged ones.

Output: a three-panel figure
  (A) aggregate tradeoff scatter (mean search calls vs accuracy), formal vs
      natural2 per model, with an arrow showing each model's phrasing shift.
  (B) accuracy grouped bars (formal vs natural2) per model, Wilson CI + McNemar.
  (C) mean search-call grouped bars per model, 95% CI + Wilcoxon.
Plus a paired per-question CSV.

Usage:
  uv run python scripts/plot_musique_staleness_tradeoff.py
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy import stats as sp

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import viz

# (label, formal_slug-for-color/label, formal_json, natural2_json)
MODELS = [
    ("Gemini 3.1 Pro", "gemini-3-pro-preview",
     "results/musique-formal/musique-formal_baseline_gemini-3-pro-preview_run_1_reevaluated.json",
     "results/musique-natural2/musique-natural2_baseline_gemini-3.1-pro-preview_run_1_reevaluated.json"),
    ("Nemotron Nano 30B", "nemotron-3-nano_30b",
     "results/musique-formal/musique-formal_baseline_nemotron-3-nano_30b_run_1_reevaluated.json",
     "results/musique-natural2/musique-natural2_baseline_nemotron-3-nano:30b_run_1_reevaluated.json"),
    ("Qwen 3.5 122B", "qwen3.5_122b",
     "results/musique-formal/musique-formal_baseline_qwen3.5_122b_run_1_reevaluated.json",
     "results/musique-natural2/musique-natural2_baseline_qwen3.5:122b_run_1_reevaluated.json"),
]
MARKERS = ["o", "s", "^"]


def load_eval_by_problem(path):
    out = {}
    for e in json.load(open(path)):
        m = e["metrics"]
        out[e["problem"]] = {"correct": bool(m["correct"]), "search_calls": int(m["search_calls"])}
    return out


def load_nonstale(csv_path):
    keep = set()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["is_stale"] == "False":
                keep.add(r["example_id"])
    return keep


def build_paired(formal_json, natural2_json, nonstale, text2id, valid_ids):
    formal = json.load(open(formal_json))
    natural = json.load(open(natural2_json))
    f_by_id = {e["example_id"]: e["metrics"] for e in formal}
    n_by_id = {text2id[e["problem"]]: e["metrics"] for e in natural if e["problem"] in text2id}

    rows = []
    for eid in nonstale:
        if eid not in f_by_id or eid not in n_by_id:
            continue
        if valid_ids is not None and eid not in valid_ids:
            continue
        rows.append({
            "example_id": eid,
            "formal_correct": bool(f_by_id[eid]["correct"]),
            "formal_search": int(f_by_id[eid]["search_calls"]),
            "natural2_correct": bool(n_by_id[eid]["correct"]),
            "natural2_search": int(n_by_id[eid]["search_calls"]),
        })
    return rows


def summarize(rows, key_correct, key_search):
    k = sum(r[key_correct] for r in rows)
    n = len(rows)
    lo, hi = viz.wilson_ci(k, n)
    searches = np.array([r[key_search] for r in rows], dtype=float)
    mean_s, ci_s = viz.mean_ci95(searches)
    return {"n": n, "k": k, "acc": k / n, "acc_lo": lo, "acc_hi": hi,
            "mean_search": mean_s, "search_ci": ci_s}


def paired_stats(rows):
    b = sum(1 for r in rows if r["formal_correct"] and not r["natural2_correct"])
    c = sum(1 for r in rows if not r["formal_correct"] and r["natural2_correct"])
    mcnemar_p = sp.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else float("nan")
    fs = [r["formal_search"] for r in rows]
    ns = [r["natural2_search"] for r in rows]
    wilcoxon_p = sp.wilcoxon(fs, ns).pvalue if any(a != b for a, b in zip(fs, ns)) else float("nan")
    return mcnemar_p, wilcoxon_p, b, c


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--staleness-csv", default="results/musique_staleness_flash.csv")
    p.add_argument("--natural2-jsonl", default="data/musique_val_natural2.jsonl")
    p.add_argument("--output-dir", default="results/musique_staleness_tradeoff")
    p.add_argument("--keep-all-paraphrases", action="store_true",
                   help="Include natural2 paraphrases flagged as fail/non-equivalent.")
    args = p.parse_args()

    nonstale = load_nonstale(args.staleness_csv)
    n2 = [json.loads(l) for l in open(args.natural2_jsonl)]
    text2id = {b["text"]: b["example_id"] for b in n2}
    valid_ids = None
    if not args.keep_all_paraphrases:
        valid_ids = {b["example_id"] for b in n2
                     if b["validation_status"] == "pass" and b["equivalent"]}

    FORMAL, NAT2 = viz.BENCHMARK, viz.NATURAL  # blue / red
    per_model = []
    print(f"Non-stale MuSiQue, formal vs natural2 "
          f"({'all paraphrases' if args.keep_all_paraphrases else 'faithful paraphrases only'}):")
    all_rows = []
    for (label, slug, fj, nj), marker in zip(MODELS, MARKERS):
        rows = build_paired(fj, nj, nonstale, text2id, valid_ids)
        for r in rows:
            all_rows.append({"model": label, **r})
        f_sum = summarize(rows, "formal_correct", "formal_search")
        n_sum = summarize(rows, "natural2_correct", "natural2_search")
        mcp, wcp, b, c = paired_stats(rows)
        per_model.append((label, slug, marker, f_sum, n_sum, mcp, wcp))
        print(f"\n{label} (n={len(rows)}):")
        print(f"  formal   : acc={f_sum['acc']:.3f} [{f_sum['acc_lo']:.3f},{f_sum['acc_hi']:.3f}]  "
              f"search={f_sum['mean_search']:.2f} ±{f_sum['search_ci']:.2f}")
        print(f"  natural2 : acc={n_sum['acc']:.3f} [{n_sum['acc_lo']:.3f},{n_sum['acc_hi']:.3f}]  "
              f"search={n_sum['mean_search']:.2f} ±{n_sum['search_ci']:.2f}")
        print(f"  McNemar acc p={mcp:.3g} {viz.sig_stars(mcp)} (formal-only={b}, natural2-only={c}) | "
              f"Wilcoxon search p={wcp:.3g} {viz.sig_stars(wcp)}")

    # ── Figure ──────────────────────────────────────────────────────────────
    viz.apply_theme()
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(16, 4.8))

    # Panel A — tradeoff scatter with formal->natural2 arrows
    for label, slug, marker, f_sum, n_sum, *_ in per_model:
        for s, color in [(f_sum, FORMAL), (n_sum, NAT2)]:
            axA.errorbar(s["mean_search"], s["acc"], xerr=s["search_ci"],
                         yerr=[[s["acc"] - s["acc_lo"]], [s["acc_hi"] - s["acc"]]],
                         fmt=marker, color=color, markersize=10, capsize=3,
                         elinewidth=1.2, zorder=3)
        axA.add_patch(FancyArrowPatch(
            (f_sum["mean_search"], f_sum["acc"]), (n_sum["mean_search"], n_sum["acc"]),
            arrowstyle="-|>", mutation_scale=12, color=viz.ERR_MUTE, lw=1.0, zorder=2,
            shrinkA=6, shrinkB=6))
        axA.annotate(label, (n_sum["mean_search"], n_sum["acc"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=7)
    # legends: phrasing (color) + model (marker)
    from matplotlib.lines import Line2D
    phrasing_leg = [Line2D([0], [0], marker="o", color="w", markerfacecolor=FORMAL, markersize=9, label="formal"),
                    Line2D([0], [0], marker="o", color="w", markerfacecolor=NAT2, markersize=9, label="natural2")]
    model_leg = [Line2D([0], [0], marker=m, color="w", markerfacecolor=viz.ERR_DARK, markersize=9, label=lbl)
                 for (lbl, *_), m in zip(MODELS, MARKERS)]
    leg1 = axA.legend(handles=phrasing_leg, loc="lower left", fontsize=8, frameon=True, title="phrasing")
    axA.add_artist(leg1)
    axA.legend(handles=model_leg, loc="upper left", fontsize=8, frameon=True)
    axA.set_xlabel("Mean search calls per question")
    axA.set_ylabel("Accuracy")
    axA.set_title("(A) Search–accuracy tradeoff\n(arrow: formal → natural2)")
    axA.grid(True, alpha=0.3)

    # Panel B — accuracy grouped bars, Panel C — search grouped bars
    x = np.arange(len(per_model))
    w = 0.38
    for ax, kind in [(axB, "acc"), (axC, "search")]:
        for j, (s_key, color, lbl) in enumerate([("f", FORMAL, "formal"), ("n", NAT2, "natural2")]):
            vals, errs = [], []
            for label, slug, marker, f_sum, n_sum, mcp, wcp in per_model:
                s = f_sum if s_key == "f" else n_sum
                if kind == "acc":
                    vals.append(s["acc"]); errs.append([s["acc"] - s["acc_lo"], s["acc_hi"] - s["acc"]])
                else:
                    vals.append(s["mean_search"]); errs.append([s["search_ci"], s["search_ci"]])
            errs = np.array(errs).T
            ax.bar(x + (j - 0.5) * w, vals, w, yerr=errs, color=color, capsize=3,
                   label=lbl, edgecolor="white")
        # significance stars above each model group
        for i, (label, slug, marker, f_sum, n_sum, mcp, wcp) in enumerate(per_model):
            pval = mcp if kind == "acc" else wcp
            top = max(f_sum["acc_hi"], n_sum["acc_hi"]) if kind == "acc" else \
                  max(f_sum["mean_search"] + f_sum["search_ci"], n_sum["mean_search"] + n_sum["search_ci"])
            ax.text(i, top * 1.02 + (0.0 if kind == "acc" else 0.3),
                    viz.sig_stars(pval), ha="center", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([m[0] for m in MODELS], fontsize=8, rotation=12, ha="right")
        ax.legend(fontsize=8, frameon=True)
        ax.grid(True, axis="y", alpha=0.3)
    axB.set_ylabel("Accuracy"); axB.set_ylim(0, 1.0)
    axB.set_title(f"(B) Accuracy (n={per_model[0][3]['n']}/model)")
    axC.set_ylabel("Mean search calls per question")
    axC.set_title("(C) Search volume")

    fig.suptitle("MuSiQue non-stale: formal vs. natural2 phrasing (3 models)", fontsize=13, y=1.02)
    viz.savefig(fig, args.output_dir, "musique_staleness_tradeoff")

    out_csv = os.path.join(args.output_dir, "musique_nonstale_paired.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "example_id", "formal_correct", "formal_search",
                                          "natural2_correct", "natural2_search"])
        w.writeheader(); w.writerows(all_rows)
    print(f"\nSaved paired data to {out_csv}")


if __name__ == "__main__":
    main()
