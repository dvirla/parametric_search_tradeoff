#!/usr/bin/env python3
import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon, binomtest

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = os.path.join(ROOT, "results", "slides")
os.makedirs(OUT, exist_ok=True)

# Only 4 representative models for slides
MODEL_ORDER = ["gemini-3.1-pro-preview", "qwen3.5_122b", "gemma4_31b", "nemotron-3-nano_30b"]
MODEL_LABEL = {"gemini-3.1-pro-preview": "Gemini 3.1 Pro", "qwen3.5_122b": "Qwen3.5 122B", "gemma4_31b": "Gemma4 31B", "nemotron-3-nano_30b": "Nemotron3 30B"}
MODEL_COLOR = {"gemini-3.1-pro-preview": "#0571b0", "qwen3.5_122b": "#ca0020", "gemma4_31b": "#5aae61", "nemotron-3-nano_30b": "#9970ab"}

CUES = ["natural", "elaborate", "polite", "query", "direct"]

plt.rcParams.update({"font.size": 12, "axes.titlesize": 13, "axes.titleweight": "bold",
                     "figure.dpi": 150, "savefig.bbox": "tight"})

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

_spec = importlib.util.spec_from_file_location("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))
_RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RG)

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

def get_conditions(ds):
    base_ph = "verbose" if ds == "FRAMES" else "orig"
    # Style group: polite, terse(plain)
    # Instructions group: natural(short), elaborate, query, direct
    conds = [
        (base_ph, "polite"),
        ("terse", "plain"),
        (base_ph, "natural"),
        (base_ph, "elaborate"),
        (base_ph, "query"),
        (base_ph, "direct")
    ]
    return base_ph, conds

def get_label(ph, cue):
    if cue == "plain":
        return f"{ph.upper()} (PLAIN)"
    if cue == "natural":
        return "SHORT"
    return cue.upper()

def mcnemar_delta(gdf, m, target_ph, target_cue, base_ph, base_cue="plain", metric="regex"):
    sub = gdf[gdf.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", metric]].rename(columns={metric: "c"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", metric]].rename(columns={metric: "p"})
    if t.empty or b.empty: return np.nan, np.nan, 0
    j = t.merge(b, on="example_id")
    cc, pp = j["c"].values, j["p"].values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    N = len(j)
    if N == 0: return np.nan, np.nan, 0
    disc = n10 + n01
    p_val = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / N, p_val, N

def abs_change_ci(tok, m, target_ph, target_cue, base_ph, base_cue="plain", nboot=2000, seed=0):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls", "thinking_words"]].rename(columns={"search_calls": "sc_t", "thinking_words": "tc_t"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls", "thinking_words"]].rename(columns={"search_calls": "sc_b", "thinking_words": "tc_b"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5: return None
    sc_t, sc_b = j["sc_t"].values, j["sc_b"].values
    tc_t, tc_b = j["tc_t"].values, j["tc_b"].values
    dS, dT = (sc_t - sc_b), (tc_t - tc_b)
    x, y = dS.mean(), dT.mean()
    rng = np.random.default_rng(seed)
    n = len(j)
    bx = np.array([dS[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    by = np.array([dT[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    xlo, xhi = np.percentile(bx, [2.5, 97.5])
    ylo, yhi = np.percentile(by, [2.5, 97.5])
    return dict(ph=target_ph, cue=target_cue, x=x, y=y, xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
                xsig=(xlo > 0 or xhi < 0), ysig=(ylo > 0 or yhi < 0))

def wilcoxon_search(tok, m, target_ph, target_cue, base_ph, base_cue="plain"):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5:
        return np.nan, np.nan
    diff = (j["target"] - j["base"]).values
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        p = wilcoxon(j["target"].values, j["base"].values, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
        
    mean_base = j["base"].mean()
    mean_target = j["target"].mean()
    if mean_base > 0:
        pct_change = 100.0 * (mean_target - mean_base) / mean_base
    else:
        pct_change = np.nan
        
    return float(pct_change), p

# SLIDE FIG 1: Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14.5, 4.3), constrained_layout=True)
for ax, ds in zip(axes, ["FRAMES", "MedQA"]):
    tok = TOK[ds]
    models = [m for m in MODEL_ORDER if m in set(tok.model)]
    base_ph, conds = get_conditions(ds)
    mat = np.full((len(models), len(conds)), np.nan)
    pmat = np.full((len(models), len(conds)), np.nan)
    
    for i, m in enumerate(models):
        for j, (ph, cue) in enumerate(conds):
            diff, p = wilcoxon_search(tok, m, ph, cue, base_ph)
            mat[i, j], pmat[i, j] = diff, p
            
    ntest = int(np.sum(~np.isnan(pmat)))
    padj = np.minimum(pmat * ntest, 1.0)
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.axvline(1.5, color='black', lw=2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([get_label(ph, cue) for ph, cue in conds], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in models])
    for i in range(len(models)):
        for j in range(len(conds)):
            v = mat[i, j]
            if np.isnan(v): continue
            col = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i - 0.15, f"{v:+.0f}%",
                    ha="center", va="center", color=col, fontsize=10, fontweight="bold")
            s = stars(padj[i, j])
            ax.text(j, i + 0.15, s if s else "n.s.", ha="center", va="center", color=col,
                    fontsize=8 if s else 6.5, fontstyle="normal" if s else "italic")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(f"Δ search calls vs PLAIN {base_ph.upper()} (%)")
    ax.set_title(f"{ds}: Search Shift")
fig.suptitle("Search Frequency Shift Across Cues", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "slide_heatmap.png"))
plt.close(fig)

# SLIDE FIG 1b: Search Shift Bars (2x4 Grid)
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(18, 8), constrained_layout=True)
for r, ds in enumerate(["FRAMES", "MedQA"]):
    tok = TOK[ds]
    base_ph, conds = get_conditions(ds)
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        xb = np.arange(len(conds))
        w = 0.6
        
        dr = []
        for (ph, cue) in conds:
            pct_change, p_val = wilcoxon_search(tok, m, ph, cue, base_ph)
            dr.append((pct_change, p_val, ph, cue))
            
        vals = [d[0] for d in dr]
        bars = ax.bar(xb, vals, w, color=MODEL_COLOR[m])
        for xi, d in zip(xb, dr):
            val = d[0]
            if np.isnan(val): continue
            
            val_no_nan = [v for v in vals if not np.isnan(v)]
            vmax, vmin = max(val_no_nan) if val_no_nan else 10, min(val_no_nan) if val_no_nan else 0
            offset = (vmax - vmin or 10) * 0.05
            offset = offset if val >= 0 else -offset
            ax.text(xi, val + offset, f"{val:+.0f}%\n{stars(d[1])}",
                    ha="center", va="bottom" if val >= 0 else "top", fontsize=9, color="#123")
        ax.axhline(0, color="#333", lw=0.8)
        ax.axvline(1.5, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(xb)
        ax.set_xticklabels([get_label(ph, cue) for (ph, cue) in conds], rotation=25, ha="right", fontsize=10)
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=12)
        if cidx == 0: ax.set_ylabel(f"Δ search calls vs PLAIN {base_ph.upper()} (%)\n{ds}")
fig.suptitle("Search Shift Trade-offs by Cue (Independent scales)", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "slide_search_bars.png"))
plt.close(fig)

# SLIDE FIG 2: Suppression (2x4 Grid)
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(18, 8), constrained_layout=True)
for r, ds in enumerate(["FRAMES", "MedQA"]):
    tok = TOK[ds]
    base_ph, conds = get_conditions(ds)
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        pts = []
        for (ph, cue) in conds:
            p = abs_change_ci(tok, m, ph, cue, base_ph)
            if p: pts.append(p)
        if not pts:
            ax.set_axis_off()
            continue
        xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
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
            col = MODEL_COLOR[m]
            ax.errorbar(p["x"], p["y"], xerr=[[p["x"] - p["xlo"]], [p["xhi"] - p["x"]]],
                        yerr=[[p["y"] - p["ylo"]], [p["yhi"] - p["y"]]], fmt="none", ecolor=col, elinewidth=1.1, alpha=0.6)
            ax.scatter([p["x"]], [p["y"]], s=120, color=col if sig_both else "white", edgecolors=col if not sig_both else "black", zorder=4)
            lbl = get_label(p["ph"], p["cue"])
            ax.annotate(lbl, (p["x"], p["y"]), fontsize=9, color="#333", xytext=(5, 3), textcoords="offset points")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        if r == 0 and cidx == 0:
            ax.text(0.03, 0.97, "SUBSTITUTION\nsearch↓ think↑", transform=ax.transAxes, fontsize=9, color="#0571b0", fontweight="bold", va="top")
            ax.text(0.03, 0.03, "SUPPRESSION\nsearch↓ think↓", transform=ax.transAxes, fontsize=9, color="#ca0020", fontweight="bold", va="bottom")
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=12)
        if cidx == 0: ax.set_ylabel(f"Δ thinking words vs PLAIN {base_ph.upper()}")
        if r == 1: ax.set_xlabel(f"Δ search calls vs PLAIN {base_ph.upper()}")
fig.suptitle("Effort Reallocation: Substitution vs Suppression", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "slide_suppression.png"))
plt.close(fig)

# SLIDE FIG 3b: Accuracy (2x4 Grid)
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(18, 8), constrained_layout=True, sharey="row")
for r, ds in enumerate(["FRAMES", "MedQA"]):
    gdf = GRADED[ds]
    base_ph, conds = get_conditions(ds)
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        xb = np.arange(len(conds))
        w = 0.6
        dr = [(mcnemar_delta(gdf, m, ph, cue, base_ph), ph, cue) for (ph, cue) in conds]
        ax.bar(xb, [d[0][0] for d in dr], w, color=MODEL_COLOR[m], label="Regex")
        for xi, (d, ph, cue) in zip(xb, dr):
            val = d[0]
            if np.isnan(val): continue
            offset = 0.8 if val >= 0 else -0.8
            ax.text(xi, val + offset, f"{val:+.0f}{stars(d[1])}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, color="#123")
        ax.axhline(0, color="#333", lw=0.8)
        ax.axvline(1.5, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(xb)
        ax.set_xticklabels([get_label(ph, cue) for ph, cue in conds], rotation=25, ha="right", fontsize=10)
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=12)
        if cidx == 0: ax.set_ylabel(f"Δ regex acc vs PLAIN {base_ph.upper()} (pp)\n{ds}")
fig.suptitle("Accuracy Trade-offs by Cue", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "slide_accuracy.png"))
plt.close(fig)

# SLIDE FIG 4: Combined Search and Accuracy Bars
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(18, 8), constrained_layout=True)
for r, ds in enumerate(["FRAMES", "MedQA"]):
    gdf = GRADED[ds]
    tok = TOK[ds]
    base_ph, conds = get_conditions(ds)
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        xb = np.arange(len(conds))
        w = 0.35
        
        dr_acc = []
        dr_search = []
        for (ph, cue) in conds:
            acc_d, acc_p, _ = mcnemar_delta(gdf, m, ph, cue, base_ph)
            search_d, search_p = wilcoxon_search(tok, m, ph, cue, base_ph)
            dr_acc.append((acc_d, acc_p, ph, cue))
            dr_search.append((search_d, search_p, ph, cue))
            
        search_vals = [d[0] for d in dr_search]
        acc_vals = [d[0] for d in dr_acc]
        
        # Plot bars
        ax.bar(xb - w/2, search_vals, w, color="#4daf4a", label="Δ Search (%)")
        ax.bar(xb + w/2, acc_vals, w, color="#377eb8", label="Δ Regex Acc (pp)")
        
        # Add values and stars
        for xi, s_val, a_val, s_p, a_p in zip(xb, search_vals, acc_vals, [d[1] for d in dr_search], [d[1] for d in dr_acc]):
            if not np.isnan(s_val):
                s_off = 2 if s_val >= 0 else -2
                ax.text(xi - w/2, s_val + s_off, f"{s_val:+.0f}{stars(s_p)}",
                        ha="center", va="bottom" if s_val >= 0 else "top", fontsize=8, color="#123")
            if not np.isnan(a_val):
                a_off = 2 if a_val >= 0 else -2
                ax.text(xi + w/2, a_val + a_off, f"{a_val:+.0f}{stars(a_p)}",
                        ha="center", va="bottom" if a_val >= 0 else "top", fontsize=8, color="#123")
                        
        ax.axhline(0, color="#333", lw=0.8)
        ax.axvline(1.5, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(xb)
        tick_labels = [get_label(ph, cue) for (ph, cue) in conds]
        ax.set_xticklabels(tick_labels, rotation=25, ha="right", fontsize=10)
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=12)
        
        if cidx == 0:
            ax.set_ylabel(f"Δ vs PLAIN {base_ph.upper()}\n{ds}")
            
        if r == 0 and cidx == 0:
            ax.legend(loc='best', fontsize=9)

fig.suptitle("Combined Search Shift and Accuracy Trade-offs by Cue\n(Green = Search %, Blue = Accuracy pp)", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "slide_combined_search_acc.png"))
plt.close(fig)

def zero_search_delta(tok, m, target_ph, target_cue, base_ph, base_cue="plain"):
    sub = tok[tok.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "t_search"})
    b = sub[(sub.phrasing == base_ph) & (sub.cue == base_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "b_search"})
    if t.empty or b.empty: return np.nan, np.nan, 0
    j = t.merge(b, on="example_id").dropna()
    cc = (j["t_search"] == 0).astype(int).values
    pp = (j["b_search"] == 0).astype(int).values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    N = len(j)
    if N == 0: return np.nan, np.nan, 0
    disc = n10 + n01
    p_val = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / N, p_val, N

# SLIDE FIG 5: Zero Searches
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(18, 8), constrained_layout=True, sharey="row")
for r, ds in enumerate(["FRAMES", "MedQA"]):
    tok = TOK[ds]
    base_ph, conds = get_conditions(ds)
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        xb = np.arange(len(conds))
        w = 0.6
        dr = [(zero_search_delta(tok, m, ph, cue, base_ph), ph, cue) for (ph, cue) in conds]
        ax.bar(xb, [d[0][0] for d in dr], w, color=MODEL_COLOR[m])
        for xi, (d, ph, cue) in zip(xb, dr):
            val = d[0]
            if np.isnan(val): continue
            offset = 1.0 if val >= 0 else -1.0
            ax.text(xi, val + offset, f"{val:+.0f}{stars(d[1])}", ha="center", va="bottom" if val >= 0 else "top", fontsize=9, color="#123")
        ax.axhline(0, color="#333", lw=0.8)
        ax.axvline(1.5, color="gray", linestyle="--", lw=1.2)
        ax.set_xticks(xb)
        ax.set_xticklabels([get_label(ph, cue) for ph, cue in conds], rotation=25, ha="right", fontsize=10)
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=12)
        if cidx == 0: ax.set_ylabel(f"Δ zero-search vs PLAIN {base_ph.upper()} (pp)\n{ds}")
fig.suptitle("Change in Zero-Search Frequency by Cue", fontsize=14, fontweight="bold")
fig.savefig(os.path.join(OUT, "slide_zero_search.png"))
plt.close(fig)

print("Saved slide figures.")
