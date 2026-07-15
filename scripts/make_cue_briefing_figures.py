#!/usr/bin/env python3
"""Consolidated briefing figures for the prompt-cue study (3 issues), FRAMES + MedQA.

Includes options for pooled/unpooled and prominent phrasing.
Outputs into results/cue_briefing/:
  fig1_search_shift.png
  fig2_suppression.png
  fig3a_gap_vs_length.png
  fig3b_accuracy_per_model.png
  fig4_phrasing_interaction.png
  fig4_phrasing_paired_interaction.png
"""
import os
import argparse
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, wilcoxon, binomtest
from itertools import combinations

# Setup CLI args
parser = argparse.ArgumentParser(description="Generate briefing figures.")
parser.add_argument("--pooling", choices=["pooled", "unpooled"], default="pooled", help="Pooling strategy (default: pooled)")
parser.add_argument("--prominent", action="store_true", help="Use prominent phrasing only")
args = parser.parse_args()

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = os.path.join(ROOT, "results", "cue_briefing")
os.makedirs(OUT, exist_ok=True)

MODEL_ORDER = ["gemini-3.1-pro-preview", "qwen3.5_122b", "qwen3.5_35b", "qwen3.5_4b", "gemma4_31b", "gemma4_e4b", "nemotron-3-nano_30b"]
MODEL_LABEL = {"gemini-3.1-pro-preview": "Gemini 3.1 Pro", "qwen3.5_122b": "Qwen3.5 122B",
               "qwen3.5_35b": "Qwen3.5 35B", "qwen3.5_4b": "Qwen3.5 4B",
               "gemma4_31b": "Gemma4 31B", "gemma4_e4b": "Gemma4 E4B", "nemotron-3-nano_30b": "Nemotron3 30B"}
MODEL_COLOR = {"gemini-3.1-pro-preview": "#0571b0", "qwen3.5_122b": "#ca0020",
               "qwen3.5_35b": "#f4a582", "qwen3.5_4b": "#92c5de",
               "gemma4_31b": "#5aae61", "gemma4_e4b": "#e66101", "nemotron-3-nano_30b": "#9970ab"}
CUES = ["natural", "elaborate", "polite", "query", "direct"]
CUE_LABEL = {c: c.upper() for c in CUES}
plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "figure.dpi": 130, "savefig.bbox": "tight"})

PHRASINGS = {"FRAMES": ["verbose", "terse", "epi_strong"], "MedQA": ["orig", "terse"]}
PHRASING_COLOR = {"verbose": "#1b9e77", "terse": "#d95f02", "epi_strong": "#7570b3", "orig": "#1b9e77"}
PHRASING_SHORT = {"verbose": "V", "terse": "T", "epi_strong": "E", "orig": "V"}

def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c

def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

def load_tokens(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls", "thinking_words"]]

TOK = {"FRAMES": load_tokens(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
       "MedQA": load_tokens(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))}

def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_RG = _load_module("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))

def load_graded(dataset, grid_dir):
    rows = []
    for path, slug, cond in _RG.find_grid_files(grid_dir, dataset):
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            rows.append({"model": slug, "condition": cond, "example_id": g["example_id"],
                         "llm": int(bool(g["llm_correct"])), "regex": int(bool(g["regex_strict"]))})
    df = pd.DataFrame(rows)
    df["cue"] = df["condition"].map(base_cue)
    def get_ph(c):
        if c.startswith("epi_strong_"): return "epi_strong"
        return c.split("_", 1)[0]
    df["phrasing"] = df["condition"].map(get_ph)
    return df

GRADED = {"FRAMES": load_graded("frames", os.path.join(ROOT, "results/frames_cues_full")),
          "MedQA": load_graded("medqa", os.path.join(ROOT, "results/medqa_grid"))}

def mcnemar_delta(gdf, m, cue, metric, ph=None):
    sub = gdf[gdf.model == m]
    if ph is not None:
        phrasings = [ph]
    else:
        phrasings = sub.phrasing.unique()

    n10 = n01 = N = 0
    for current_ph in phrasings:
        c = sub[(sub.phrasing == current_ph) & (sub.cue == cue)][["example_id", metric]]
        p = sub[(sub.phrasing == current_ph) & (sub.cue == "plain")][["example_id", metric]]
        if c.empty or p.empty:
            continue
        j = c.merge(p, on="example_id", suffixes=("_c", "_p"))
        cc, pp = j[metric + "_c"].values, j[metric + "_p"].values
        n10 += int(((cc == 1) & (pp == 0)).sum())
        n01 += int(((cc == 0) & (pp == 1)).sum())
        N += len(j)
    if N == 0:
        return np.nan, np.nan, 0
    disc = n10 + n01
    p_val = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / N, p_val, N

PROMINENT = {}
if args.prominent:
    for ds in ["FRAMES", "MedQA"]:
        gdf = GRADED[ds]
        for m in MODEL_ORDER:
            for cue in CUES:
                cands = []
                for ph in PHRASINGS[ds]:
                    d, p_val, _ = mcnemar_delta(gdf, m, cue, "regex", ph=ph)
                    if not np.isnan(d):
                        cands.append({"ph": ph, "d": d, "p": p_val, "abs_d": abs(d)})
                if cands:
                    sig_cands = [c for c in cands if c["p"] < 0.05]
                    if sig_cands:
                        best = max(sig_cands, key=lambda x: x["abs_d"])
                    else:
                        best = max(cands, key=lambda x: x["abs_d"])
                    PROMINENT[(ds, m, cue)] = best["ph"]

def wilcoxon_search(tok, m, cue, ph=None):
    if ph is not None:
        sub = tok[(tok.model == m) & (tok.phrasing == ph)]
    else:
        sub = tok[tok.model == m]
    piv = sub.pivot_table(index="example_id", columns="cue", values="search_calls", aggfunc="mean")
    if cue not in piv.columns or "plain" not in piv.columns:
        return np.nan, np.nan
    d = piv[[cue, "plain"]].dropna()
    if len(d) < 5:
        return np.nan, np.nan
    diff = (d[cue] - d["plain"]).values
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        p_val = wilcoxon(d[cue].values, d["plain"].values, zero_method="wilcox").pvalue
    except ValueError:
        p_val = 1.0
    return float(diff.mean()), p_val

def abs_change_ci(tok, m, cue, ph=None, nboot=2000, seed=0):
    if ph is not None:
        sub = tok[(tok.model == m) & (tok.phrasing == ph)]
    else:
        sub = tok[tok.model == m]
    ps = sub.pivot_table(index="example_id", columns="cue", values="search_calls", aggfunc="mean")
    pt = sub.pivot_table(index="example_id", columns="cue", values="thinking_words", aggfunc="mean")
    if cue not in ps.columns or "plain" not in ps.columns:
        return None
    idx = ps[[cue, "plain"]].dropna().index.intersection(pt[[cue, "plain"]].dropna().index)
    if len(idx) < 5:
        return None
    sc, sp = ps.loc[idx, cue].values, ps.loc[idx, "plain"].values
    tc, tp = pt.loc[idx, cue].values, pt.loc[idx, "plain"].values
    dS, dT = (sc - sp), (tc - tp)
    x, y = dS.mean(), dT.mean()
    rng = np.random.default_rng(seed)
    n = len(idx)
    bx = np.array([dS[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    by = np.array([dT[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    xlo, xhi = np.percentile(bx, [2.5, 97.5])
    ylo, yhi = np.percentile(by, [2.5, 97.5])
    return dict(cue=cue, x=x, y=y, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                xsig=(xlo > 0 or xhi < 0), ysig=(ylo > 0 or yhi < 0))

# ===========================================================================
# FIGURE 1
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(15 if (args.pooling == "unpooled" and not args.prominent) else 13.5, 6 if (args.pooling == "unpooled" and not args.prominent) else 4.3), constrained_layout=True)
for ax, ds in zip(axes, ["FRAMES", "MedQA"]):
    tok = TOK[ds]
    models = [m for m in MODEL_ORDER if m in set(tok.model)]
    
    if args.pooling == "unpooled" and not args.prominent:
        phrasings = PHRASINGS[ds]
        row_labels = []
        mat, pmat = [], []
        for m in models:
            for ph in phrasings:
                row_labels.append(f"{MODEL_LABEL[m]} ({ph})")
                r_mat, r_pmat = [], []
                for cue in CUES:
                    diff, p_val = wilcoxon_search(tok, m, cue, ph=ph)
                    r_mat.append(diff)
                    r_pmat.append(p_val)
                mat.append(r_mat)
                pmat.append(r_pmat)
        mat = np.array(mat)
        pmat = np.array(pmat)
        phmat = np.full(mat.shape, "", dtype=object)
    else:
        mat = np.full((len(models), len(CUES)), np.nan)
        pmat = np.full((len(models), len(CUES)), np.nan)
        phmat = np.full((len(models), len(CUES)), "", dtype=object)
        row_labels = [MODEL_LABEL[m] for m in models]
        for i, m in enumerate(models):
            for j, cue in enumerate(CUES):
                if args.prominent:
                    ph = PROMINENT.get((ds, m, cue))
                    if ph:
                        diff, p_val = wilcoxon_search(tok, m, cue, ph=ph)
                        mat[i, j], pmat[i, j] = diff, p_val
                        phmat[i, j] = PHRASING_SHORT[ph]
                else:
                    diff, p_val = wilcoxon_search(tok, m, cue, ph=None)
                    mat[i, j], pmat[i, j] = diff, p_val

    ntest = int(np.sum(~np.isnan(pmat)))
    padj = np.minimum(pmat * ntest, 1.0)
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CUES)))
    ax.set_xticklabels([CUE_LABEL[c] for c in CUES], fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8 if (args.pooling == "unpooled" and not args.prominent) else 10)
    for i in range(len(row_labels)):
        for j in range(len(CUES)):
            v = mat[i, j]
            if np.isnan(v): continue
            col = "white" if abs(v) > 0.55 * vmax else "black"
            if args.prominent:
                ax.text(j, i - 0.15, f"{v:+.2f}" if ds == "MedQA" else f"{v:+.1f}", ha="center", va="center", color=col, fontsize=10, fontweight="bold")
                s = stars(padj[i, j])
                ax.text(j, i + 0.15, s if s else "n.s.", ha="center", va="center", color=col, fontsize=8 if s else 6.5, fontstyle="normal" if s else "italic")
                ax.text(j, i + 0.35, f"({phmat[i, j]})", ha="center", va="center", color=col, fontsize=7)
            else:
                ax.text(j, i - 0.13, f"{v:+.2f}" if ds == "MedQA" else f"{v:+.1f}", ha="center", va="center", color=col, fontsize=10, fontweight="bold")
                s = stars(padj[i, j])
                ax.text(j, i + 0.26, s if s else "n.s.", ha="center", va="center", color=col, fontsize=8 if s else 6.5, fontstyle="normal" if s else "italic")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Δ search calls / question")
    title_suffix = " (Prominent Phrasing)" if args.prominent else (" (by Phrasing)" if args.pooling == "unpooled" else "")
    ax.set_title(f"{ds}: Δ mean search calls vs PLAIN{title_suffix}")

suptitle_text = "Issue 1 — "
if args.prominent:
    suptitle_text += "Heatmap using the most prominent phrasing per cue\n(V = verbose, T = terse, E = epistemic_strong)"
elif args.pooling == "unpooled":
    suptitle_text += "Search shifts separated by Phrasing"
else:
    suptitle_text += "Instruction wording moves search calls (identical questions)\nBlue = fewer searches than a bare question · Red = more · stars = paired Wilcoxon, Bonferroni  (* .05  ** .01  *** .001)"
fig.suptitle(suptitle_text, fontsize=11.5, fontweight="bold")
out_name = "fig1_search_shift_prominent.png" if args.prominent else "fig1_search_shift.png"
fig.savefig(os.path.join(OUT, out_name))
plt.close(fig)

# ===========================================================================
# FIGURE 2
# ===========================================================================
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(31.5, 8.6), constrained_layout=True)
for r, ds in enumerate(["FRAMES", "MedQA"]):
    tok = TOK[ds]
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        pts = []
        if args.prominent:
            for cue in CUES:
                ph = PROMINENT.get((ds, m, cue))
                if ph:
                    p = abs_change_ci(tok, m, cue, ph=ph)
                    if p:
                        p["ph"] = ph
                        pts.append(p)
        elif args.pooling == "unpooled":
            for ph in PHRASINGS[ds]:
                for cue in CUES:
                    p = abs_change_ci(tok, m, cue, ph=ph)
                    if p:
                        p["ph"] = ph
                        pts.append(p)
        else:
            for cue in CUES:
                p = abs_change_ci(tok, m, cue, ph=None)
                if p:
                    p["ph"] = None
                    pts.append(p)

        if not pts:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{MODEL_LABEL[m]}\n({ds})\nno data", ha="center", va="center", fontsize=9, color="#999")
            continue
            
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        xmarg = (max(xs) - min(xs) or 1) * 0.25 + max(abs(p["xhi"] - p["x"]) for p in pts)
        ymarg = (max(ys) - min(ys) or 1) * 0.25 + max(abs(p["yhi"] - p["y"]) for p in pts)
        xlo, xhi = min(0, min(xs)) - xmarg, max(0, max(xs)) + xmarg
        ylo, yhi = min(0, min(ys)) - ymarg, max(0, max(ys)) + ymarg
        fx = (0 - xlo) / (xhi - xlo)
        ax.axhspan(0, yhi, xmin=0, xmax=fx, color="#0571b0", alpha=0.06)
        ax.axhspan(ylo, 0, xmin=0, xmax=fx, color="#ca0020", alpha=0.06)
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#888", lw=0.8)
        
        for p in pts:
            sig_both = p["xsig"] and p["ysig"]
            col = PHRASING_COLOR[p["ph"]] if p["ph"] else MODEL_COLOR[m]
            ax.errorbar(p["x"], p["y"],
                        xerr=[[p["x"] - p["xlo"]], [p["xhi"] - p["x"]]],
                        yerr=[[p["y"] - p["ylo"]], [p["yhi"] - p["y"]]],
                        fmt="none", ecolor=col, elinewidth=1.1, capsize=2, alpha=0.6)
            ax.scatter([p["x"]], [p["y"]], s=85,
                       color=col if sig_both else "white",
                       edgecolors=col if not sig_both else "black",
                       linewidths=1.4 if not sig_both else 0.8, zorder=4)
            lbl = f"{CUE_LABEL[p['cue']][:4].title()} ({PHRASING_SHORT[p['ph']]})" if p["ph"] else CUE_LABEL[p['cue']][:4].title()
            ax.annotate(lbl, (p["x"], p["y"]), fontsize=7.5,
                        color="#333", xytext=(5, 3), textcoords="offset points")
        
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        if r == 0 and cidx == 0:
            ax.text(0.03, 0.97, "SUBST.\nsearch↓ think↑", transform=ax.transAxes, fontsize=8, color="#0571b0", fontweight="bold", va="top")
            ax.text(0.03, 0.03, "SUPPR.\nsearch↓ think↓", transform=ax.transAxes, fontsize=8, color="#ca0020", fontweight="bold", va="bottom")
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=10)
        if cidx == 0: ax.set_ylabel("Δ thinking words vs PLAIN")
        if r == 1: ax.set_xlabel("Δ search calls vs PLAIN")

if args.prominent:
    fig.suptitle("Issue 2 — Suppression with Prominent Phrasing per Cue", fontsize=12, fontweight="bold")
elif args.pooling == "unpooled":
    fig.suptitle("Issue 2 — Suppression separated by Phrasing (Colors denote phrasing)", fontsize=12, fontweight="bold")
else:
    fig.suptitle("Issue 2 — When a cue cuts search, is the effort redirected into thinking?  (absolute units, per model; 95% bootstrap CIs)\nFilled = both shifts significant. Upper-left = SUBSTITUTION (only Qwen); lower-left = SUPPRESSION (Gemini/Gemma). Nemotron mostly searches MORE.", fontsize=12, fontweight="bold")

out_name = "fig2_suppression_prominent.png" if args.prominent else "fig2_suppression.png"
fig.savefig(os.path.join(OUT, out_name))
plt.close(fig)

# ===========================================================================
# FIGURE 3a
# ===========================================================================
def condition_gap_table(gdf):
    rows = []
    for (m, ph, cue), g in gdf.groupby(["model", "phrasing", "cue"]):
        rows.append({"model": m, "phrasing": ph, "cue": cue,
                     "llm_acc": 100 * g["llm"].mean(), "regex_acc": 100 * g["regex"].mean(),
                     "gap": 100 * (g["llm"].mean() - g["regex"].mean())})
    return pd.DataFrame(rows)

def resp_words(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d.groupby(["model", "phrasing", "cue"])["response_words"].mean().reset_index()

RW = {"FRAMES": resp_words(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
      "MedQA": resp_words(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))}

fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)
DS_MARK = {"FRAMES": "o", "MedQA": "^"}
for ds in ["FRAMES", "MedQA"]:
    tab = condition_gap_table(GRADED[ds]).merge(RW[ds], on=["model", "phrasing", "cue"], how="left")
    tab = tab[tab["model"].isin(MODEL_ORDER)]
    if args.prominent:
        tab = tab[tab.apply(lambda r: PROMINENT.get((ds, r["model"], r["cue"])) == r["phrasing"], axis=1)]
        
    x = tab["response_words"].values
    y = tab["gap"].values
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    mcol = tab.loc[ok, "model"].map(MODEL_COLOR).values
    ax.scatter(x, y, marker=DS_MARK[ds], s=40, c=mcol, edgecolors="white", linewidths=0.4, alpha=0.85)
    r, p_val = pearsonr(x, y)
    b, a = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    rng = np.random.default_rng(0)
    n = len(x)
    preds = np.array([np.polyval(np.polyfit(x[bi], y[bi], 1), xs)
                      for bi in (rng.integers(0, n, n) for _ in range(600))])
    lo, hi = np.percentile(preds, [2.5, 97.5], axis=0)
    line_col = "#333" if ds == "FRAMES" else "#888"
    ax.fill_between(xs, lo, hi, color=line_col, alpha=0.10, lw=0)
    ax.plot(xs, a + b * xs, color=line_col, ls="--", lw=1.4,
            label=f"{ds}: r={r:+.2f}{stars(p_val)} (p={p_val:.1e})")

ax.set_xlabel("Mean response length (words)")
ax.set_ylabel("Judge − regex accuracy gap (pp)")
ax.set_title("Issue 3a — Judge−regex gap (Prominent Phrasing)" if args.prominent else "Issue 3a — Judge−regex gap is largest on the SHORTEST answers\npoint = one model×phrasing×cue · color = model · shape = dataset")
mh = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m], label=MODEL_LABEL[m])
      for m in MODEL_ORDER]
leg1 = ax.legend(handles=mh, loc="upper right", fontsize=8, title="model")
ax.add_artist(leg1)
ax.legend(loc="lower left", fontsize=8.5)
out_name = "fig3a_gap_vs_length_prominent.png" if args.prominent else "fig3a_gap_vs_length.png"
fig.savefig(os.path.join(OUT, out_name))
plt.close(fig)

# ===========================================================================
# FIGURE 3b
# ===========================================================================
if args.prominent:
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(31.5, 8.4), constrained_layout=True, sharey="row")
    for r, ds in enumerate(["FRAMES", "MedQA"]):
        gdf = GRADED[ds]
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            present = [c for c in CUES if PROMINENT.get((ds, m, c))]
            if not present:
                ax.set_axis_off()
                if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=11)
                continue
            xb = np.arange(len(present))
            w = 0.6
            dr = []
            for c in present:
                ph = PROMINENT.get((ds, m, c))
                dr.append((mcnemar_delta(gdf, m, c, "regex", ph=ph), ph))
                
            ax.bar(xb, [d[0][0] for d in dr], w, color=[PHRASING_COLOR[d[1]] for d in dr], label="Regex")
            for xi, (d, ph) in zip(xb, dr):
                val = d[0]
                offset = 0.4 if val >= 0 else -0.4
                ax.text(xi, val + offset, f"{val:+.0f}{stars(d[1])}\n({PHRASING_SHORT[ph]})",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=7.5, color="#123")
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xticks(xb)
            ax.set_xticklabels([CUE_LABEL[c] for c in present], rotation=25, ha="right", fontsize=9)
            if r == 0:
                ax.set_title(MODEL_LABEL[m], fontsize=11)
            if cidx == 0:
                ax.set_ylabel(f"Δ regex acc vs PLAIN (pp)\n{ds}")
    fig.suptitle("Issue 3b — Regex Accuracy cost of each cue, using Prominent Phrasing per cue", fontsize=14, fontweight="bold")
    out_name = "fig3b_accuracy_per_model_prominent.png"

elif args.pooling == "unpooled":
    row_specs = [("FRAMES", "verbose"), ("FRAMES", "terse"), ("FRAMES", "epi_strong"),
                 ("MedQA", "orig"), ("MedQA", "terse")]
    fig, axes = plt.subplots(len(row_specs), len(MODEL_ORDER), figsize=(31.5, 3.5 * len(row_specs)), constrained_layout=True, sharey="row")
    for r, (ds, ph) in enumerate(row_specs):
        gdf = GRADED[ds]
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            present = [c for c in CUES if not gdf[(gdf.model == m) & (gdf.cue == c) & (gdf.phrasing == ph)].empty]
            if not present:
                ax.set_axis_off()
                if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=11)
                continue
            xb = np.arange(len(present))
            w = 0.38
            dj = [mcnemar_delta(gdf, m, c, "llm", ph=ph) for c in present]
            dr = [mcnemar_delta(gdf, m, c, "regex", ph=ph) for c in present]
            ax.bar(xb - w/2, [d[0] for d in dj], w, color="#2166ac", label="LLM judge")
            ax.bar(xb + w/2, [d[0] for d in dr], w, color="#f4a582", label="Regex")
            for xi, d in zip(xb - w/2, dj):
                ax.text(xi, d[0] + (0.4 if d[0] >= 0 else -0.4), f"{d[0]:+.0f}{stars(d[1])}",
                        ha="center", va="bottom" if d[0] >= 0 else "top", fontsize=7, color="#123")
            for xi, d in zip(xb + w/2, dr):
                ax.text(xi, d[0] + (0.4 if d[0] >= 0 else -0.4), f"{d[0]:+.0f}{stars(d[1])}",
                        ha="center", va="bottom" if d[0] >= 0 else "top", fontsize=7, color="#b3541a")
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xticks(xb)
            ax.set_xticklabels([CUE_LABEL[c] for c in present], rotation=25, ha="right", fontsize=8)
            if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=11)
            if cidx == 0: ax.set_ylabel(f"Δ acc vs PLAIN (pp)\n{ds} - {ph.upper()}")
            if r == 0 and cidx == 0: ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Issue 3b — Accuracy cost of each cue, un-pooled by Phrasing", fontsize=14, fontweight="bold")
    out_name = "fig3b_accuracy_per_model.png"

else: # pooled
    fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(31.5, 8.4), constrained_layout=True, sharey="row")
    for r, ds in enumerate(["FRAMES", "MedQA"]):
        gdf = GRADED[ds]
        for cidx, m in enumerate(MODEL_ORDER):
            ax = axes[r][cidx]
            present = [c for c in CUES if not gdf[(gdf.model == m) & (gdf.cue == c)].empty]
            if not present:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{MODEL_LABEL[m]}\n({ds})\nno data", ha="center", va="center", color="#999", fontsize=9)
                continue
            xb = np.arange(len(present))
            w = 0.38
            dj = [mcnemar_delta(gdf, m, c, "llm", ph=None) for c in present]
            dr = [mcnemar_delta(gdf, m, c, "regex", ph=None) for c in present]
            ax.bar(xb - w/2, [d[0] for d in dj], w, color="#2166ac", label="LLM judge")
            ax.bar(xb + w/2, [d[0] for d in dr], w, color="#f4a582", label="Regex")
            for xi, d in zip(xb - w/2, dj):
                ax.text(xi, d[0] + (0.4 if d[0] >= 0 else -0.4), f"{d[0]:+.0f}{stars(d[1])}",
                        ha="center", va="bottom" if d[0] >= 0 else "top", fontsize=7, color="#123")
            for xi, d in zip(xb + w/2, dr):
                ax.text(xi, d[0] + (0.4 if d[0] >= 0 else -0.4), f"{d[0]:+.0f}{stars(d[1])}",
                        ha="center", va="bottom" if d[0] >= 0 else "top", fontsize=7, color="#b3541a")
            ax.axhline(0, color="#333", lw=0.8)
            ax.set_xticks(xb)
            ax.set_xticklabels([CUE_LABEL[c] for c in present], rotation=25, ha="right", fontsize=8)
            ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=10)
            if cidx == 0: ax.set_ylabel("Δ accuracy vs PLAIN (pp)")
            if r == 0 and cidx == 0: ax.legend(fontsize=8, loc="lower left")
    fig.suptitle("Issue 3b — Real accuracy cost of each cue, per model: LLM judge vs regex  (exact McNemar vs PLAIN; * .05 ** .01 *** .001)\nRegex over-penalizes terse cues (DIRECT/NATURAL); judge-confirmed drops are smaller and mostly on DIRECT/QUERY. Nemotron gains under cues.", fontsize=12, fontweight="bold")
    out_name = "fig3b_accuracy_per_model.png"

fig.savefig(os.path.join(OUT, out_name))
plt.close(fig)

# ===========================================================================
# FIGURE 4 (Interaction & Difference)
# ===========================================================================
ALL_CUES = ["plain", "natural", "elaborate", "polite", "query", "direct"]
ALL_CUE_LABEL = {c: c.upper() for c in ALL_CUES}

def load_tokens_fig4(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls"]]

TOK4 = {
    "FRAMES": load_tokens_fig4(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
    "MedQA": load_tokens_fig4(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))
}

sns.set_theme(style="whitegrid")

# Interaction
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(28, 8), constrained_layout=True)
for r, ds in enumerate(["FRAMES", "MedQA"]):
    df = TOK4[ds]
    df = df[df["cue"].isin(ALL_CUES)]
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        m_df = df[df["model"] == m]
        
        if m_df.empty or len(m_df["phrasing"].unique()) < 2:
            ax.set_axis_off()
            if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
            continue
            
        # Using err_kws instead of errwidth/capsize to support both seaborn versions
        sns.pointplot(data=m_df, x="cue", y="search_calls", hue="phrasing", ax=ax, 
                      order=ALL_CUES, palette=PHRASING_COLOR, dodge=True, markers=["o", "s", "D"],
                      err_kws={'linewidth': 1.5}, capsize=0.1)
        
        ax.set_xticklabels([ALL_CUE_LABEL[c][:4] for c in ALL_CUES], rotation=30)
        ax.set_xlabel("")
        if cidx == 0:
            ax.set_ylabel(f"{ds}\nMean Search Calls")
        else:
            ax.set_ylabel("")
            
        if r == 0:
            ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
            
        if r == 0 and cidx == 0:
            ax.legend(title="Phrasing", fontsize=9)
        else:
            if ax.get_legend(): ax.get_legend().remove()

fig.suptitle("Interaction between Cue and Phrasing (ANOVA Visualized)\nParallel lines = no interaction (phrasing acts independently). Crossing/diverging lines = phrasing is coupled with specific cues.", fontsize=16, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig4_phrasing_interaction.png"))
plt.close(fig)

# Difference
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(28, 8), constrained_layout=True)
for r, (ds, ph1, ph2) in enumerate([("FRAMES", "verbose", "terse"), ("MedQA", "orig", "terse")]):
    df = TOK4[ds]
    df = df[df["cue"].isin(ALL_CUES)]
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        m_df = df[df["model"] == m]
        
        if m_df.empty or ph1 not in m_df["phrasing"].values or ph2 not in m_df["phrasing"].values:
            ax.set_axis_off()
            if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
            continue
            
        piv = m_df.pivot_table(index=["example_id", "cue"], columns="phrasing", values="search_calls").reset_index()
        piv = piv.dropna(subset=[ph1, ph2])
        piv["diff"] = piv[ph1] - piv[ph2]
        
        if piv.empty:
            ax.set_axis_off()
            if r == 0: ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")
            continue

        sns.pointplot(data=piv, x="cue", y="diff", ax=ax, order=ALL_CUES, 
                      color="#d95f02", markers="o", err_kws={'linewidth': 1.5}, capsize=0.1)
        
        ax.axhline(0, color="black", ls="--", lw=1)
        ax.set_xticklabels([ALL_CUE_LABEL[c][:4] for c in ALL_CUES], rotation=30)
        ax.set_xlabel("")
        if cidx == 0:
            ax.set_ylabel(f"{ds}\nΔ Search Calls\n({ph1.capitalize()} − {ph2.capitalize()})")
        else:
            ax.set_ylabel("")
            
        if r == 0:
            ax.set_title(MODEL_LABEL[m], fontsize=12, fontweight="bold")

fig.suptitle("Repeated-Measures Interaction Plot: Paired Differences (Verbose/Orig − Terse)\nError bars show the 95% CI of the WITHIN-SUBJECT difference (correctly reflecting the MixedLM ANOVA significance).\nFlat line = No Interaction. Line not at 0 = Main Effect. Zig-zag line = Significant Interaction.", fontsize=15, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig4_phrasing_paired_interaction.png"))
plt.close(fig)

# ---------------------------------------------------------------------------
# Console summaries (similar to unpooled and pooled)
# ---------------------------------------------------------------------------
print("=== PHRASING PAIRWISE COMPARISONS ===")
for ds in ["FRAMES", "MedQA"]:
    print(f"\n--- {ds} ---")
    tok = TOK[ds]
    phrasings = PHRASINGS[ds]
    if len(phrasings) < 2: continue
    
    for m in MODEL_ORDER:
        mtok = tok[tok.model == m]
        if mtok.empty: continue
        print(f"\n{MODEL_LABEL[m]}:")
        
        # Pooled by conditions (overall effect of phrasing)
        print("  Pooled across all cues (including plain):")
        piv_pool = mtok.pivot_table(index=["example_id", "cue"], columns="phrasing", values="search_calls", aggfunc="mean").dropna()
        if len(piv_pool) > 5:
            for ph1, ph2 in combinations(phrasings, 2):
                if ph1 in piv_pool.columns and ph2 in piv_pool.columns:
                    diff = (piv_pool[ph1] - piv_pool[ph2]).mean()
                    try:
                        p_val = wilcoxon(piv_pool[ph1], piv_pool[ph2], zero_method="wilcox").pvalue
                    except ValueError:
                        p_val = 1.0
                    p_adj = min(p_val * len(list(combinations(phrasings, 2))), 1.0)
                    print(f"    {ph1} vs {ph2}: Δsearch = {diff:+.2f} (p_adj={p_adj:.3f}) {stars(p_adj)}")
                    
        # Within-cue comparisons
        for cue in ["plain"] + CUES:
            cue_tok = mtok[mtok.cue == cue]
            piv = cue_tok.pivot_table(index="example_id", columns="phrasing", values="search_calls", aggfunc="mean").dropna()
            if len(piv) > 5:
                pairs = list(combinations(phrasings, 2))
                for ph1, ph2 in pairs:
                    if ph1 in piv.columns and ph2 in piv.columns:
                        diff = (piv[ph1] - piv[ph2]).mean()
                        try:
                            p_val = wilcoxon(piv[ph1], piv[ph2], zero_method="wilcox").pvalue
                        except ValueError:
                            p_val = 1.0
                        p_adj = min(p_val * len(pairs), 1.0)
                        if p_adj < 0.05 or abs(diff) > 0.5:
                            print(f"    [{cue.upper()}] {ph1} vs {ph2}: Δsearch = {diff:+.2f} (p_adj={p_adj:.3f}) {stars(p_adj)}")

print("\nSAVED all figures to", OUT)
