#!/usr/bin/env python3
"""Compact briefing figures for the prompt-cue study (3 issues), FRAMES + MedQA.

All effect sizes are ABSOLUTE (search calls; thinking words) -- no percentages,
no symlog. Significance is computed per model:
  * Issue 1 (search):   paired Wilcoxon, cue vs plain, Bonferroni per dataset.
  * Issue 2 (tradeoff): 95% bootstrap CIs on both axes (per model x cue).
  * Issue 3 (accuracy): exact McNemar per model x cue vs plain, for BOTH the
                        LLM judge and the regex grader (per-example verdicts
                        recomputed from the grid JSONs via the regrade modules).

Outputs into results/cue_briefing/:
  fig1_search_shift.png          (FRAMES | MedQA heatmaps)
  fig2_suppression.png           (2 datasets x 4 models, absolute Δsearch vs Δthink)
  fig3a_gap_vs_length.png        (judge-regex gap vs response length, both datasets)
  fig3b_accuracy_per_model.png   (2 datasets x 4 models, judge vs regex Δacc + McNemar)
"""
import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, wilcoxon, binomtest

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


# ---------------------------------------------------------------------------
# Load token data (search_calls, thinking_words) -- schema differs per dataset:
#   frames: dataset in {verbose,terse,epi_strong}, condition="verbose_plain"
#   medqa : dataset in {orig,terse},               condition="direct"/"plain"
# Unify to columns: model, phrasing(=dataset), cue, example_id, search_calls, thinking_words
# ---------------------------------------------------------------------------
def load_tokens(path):
    d = pd.read_csv(path)
    d["cue"] = d["condition"].map(base_cue)
    d["phrasing"] = d["dataset"]
    return d[["model", "phrasing", "cue", "example_id", "search_calls", "thinking_words"]]

TOK = {"FRAMES": load_tokens(os.path.join(ROOT, "results/frames_token_analysis/joined_tokens.csv")),
       "MedQA": load_tokens(os.path.join(ROOT, "results/medqa_token_analysis/joined_tokens.csv"))}


# ---------------------------------------------------------------------------
# Load per-example graded verdicts (llm + regex) from the grid JSONs, using the
# offline regrade modules (no LLM calls, pure string matching).
# ---------------------------------------------------------------------------
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
    df["phrasing"] = df["condition"].map(lambda c: c.split("_", 1)[0])
    return df

GRADED = {"FRAMES": load_graded("frames", os.path.join(ROOT, "results/frames_cues_full")),
          "MedQA": load_graded("medqa", os.path.join(ROOT, "results/medqa_grid"))}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilcoxon_search(tok, m, cue):
    """Paired Wilcoxon of search_calls, cue vs plain, phrasings pooled per example."""
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
        p = wilcoxon(d[cue].values, d["plain"].values, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    return float(diff.mean()), p


def abs_change_ci(tok, m, cue, nboot=2000, seed=0):
    """Absolute mean Δ (cue-plain) for search & thinking with 95% bootstrap CIs."""
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


def mcnemar_delta(gdf, m, cue, metric):
    """Exact McNemar for cue vs plain on `metric` (llm/regex), phrasings pooled.
    Returns (Δacc pp, p, N)."""
    sub = gdf[gdf.model == m]
    n10 = n01 = N = 0
    for ph in sub.phrasing.unique():
        c = sub[(sub.phrasing == ph) & (sub.cue == cue)][["example_id", metric]]
        p = sub[(sub.phrasing == ph) & (sub.cue == "plain")][["example_id", metric]]
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
    p = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / N, p, N


# ===========================================================================
# FIGURE 1 -- Issue 1: Δ mean search calls vs PLAIN (absolute), FRAMES | MedQA
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.3), constrained_layout=True)
for ax, ds in zip(axes, ["FRAMES", "MedQA"]):
    tok = TOK[ds]
    models = [m for m in MODEL_ORDER if m in set(tok.model)]
    mat = np.full((len(models), len(CUES)), np.nan)
    pmat = np.full((len(models), len(CUES)), np.nan)
    for i, m in enumerate(models):
        for j, cue in enumerate(CUES):
            diff, p = wilcoxon_search(tok, m, cue)
            mat[i, j], pmat[i, j] = diff, p
    ntest = int(np.sum(~np.isnan(pmat)))
    padj = np.minimum(pmat * ntest, 1.0)
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(CUES)))
    ax.set_xticklabels([CUE_LABEL[c] for c in CUES], fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([MODEL_LABEL[m] for m in models])
    for i in range(len(models)):
        for j in range(len(CUES)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            col = "white" if abs(v) > 0.55 * vmax else "black"
            ax.text(j, i - 0.13, f"{v:+.2f}" if ds == "MedQA" else f"{v:+.1f}",
                    ha="center", va="center", color=col, fontsize=10, fontweight="bold")
            s = stars(padj[i, j])
            ax.text(j, i + 0.26, s if s else "n.s.", ha="center", va="center", color=col,
                    fontsize=8 if s else 6.5, fontstyle="normal" if s else "italic")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Δ search calls / question")
    ax.set_title(f"{ds}: Δ mean search calls vs PLAIN")
fig.suptitle("Issue 1 — Instruction wording moves search calls (identical questions)\n"
             "Blue = fewer searches than a bare question · Red = more · "
             "stars = paired Wilcoxon, Bonferroni  (* .05  ** .01  *** .001)",
             fontsize=11.5, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig1_search_shift.png"))
plt.close(fig)


# ===========================================================================
# FIGURE 2 -- Issue 2: absolute Δsearch vs Δthinking, per model, per dataset.
# 2 rows (dataset) x 7 cols (model). Each panel self-scaled (units differ a lot).
# ===========================================================================
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(31.5, 8.6), constrained_layout=True)
for r, ds in enumerate(["FRAMES", "MedQA"]):
    tok = TOK[ds]
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        pts = [p for cue in CUES if (p := abs_change_ci(tok, m, cue)) is not None]
        if not pts:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{MODEL_LABEL[m]}\n({ds})\nno data", ha="center", va="center",
                    fontsize=9, color="#999")
            continue
        xs = [p["x"] for p in pts]
        ys = [p["y"] for p in pts]
        xmarg = (max(xs) - min(xs) or 1) * 0.25 + max(abs(p["xhi"] - p["x"]) for p in pts)
        ymarg = (max(ys) - min(ys) or 1) * 0.25 + max(abs(p["yhi"] - p["y"]) for p in pts)
        xlo = min(0, min(xs)) - xmarg
        xhi = max(0, max(xs)) + xmarg
        ylo = min(0, min(ys)) - ymarg
        yhi = max(0, max(ys)) + ymarg
        fx = (0 - xlo) / (xhi - xlo)  # axes-fraction position of x=0
        ax.axhspan(0, yhi, xmin=0, xmax=fx, color="#0571b0", alpha=0.06)   # substitution
        ax.axhspan(ylo, 0, xmin=0, xmax=fx, color="#ca0020", alpha=0.06)   # suppression
        ax.axhline(0, color="#888", lw=0.8)
        ax.axvline(0, color="#888", lw=0.8)
        for p in pts:
            sig_both = p["xsig"] and p["ysig"]
            ax.errorbar(p["x"], p["y"],
                        xerr=[[p["x"] - p["xlo"]], [p["xhi"] - p["x"]]],
                        yerr=[[p["y"] - p["ylo"]], [p["yhi"] - p["y"]]],
                        fmt="none", ecolor=MODEL_COLOR[m], elinewidth=1.1, capsize=2, alpha=0.6)
            ax.scatter([p["x"]], [p["y"]], s=85,
                       color=MODEL_COLOR[m] if sig_both else "white",
                       edgecolors=MODEL_COLOR[m] if not sig_both else "black",
                       linewidths=1.4 if not sig_both else 0.8, zorder=4)
            ax.annotate(CUE_LABEL[p["cue"]][:4].title(), (p["x"], p["y"]), fontsize=7.5,
                        color="#333", xytext=(5, 3), textcoords="offset points")
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)
        if r == 0 and cidx == 0:
            ax.text(0.03, 0.97, "SUBST.\nsearch↓ think↑", transform=ax.transAxes, fontsize=8,
                    color="#0571b0", fontweight="bold", va="top")
            ax.text(0.03, 0.03, "SUPPR.\nsearch↓ think↓", transform=ax.transAxes, fontsize=8,
                    color="#ca0020", fontweight="bold", va="bottom")
        ax.set_title(f"{MODEL_LABEL[m]} · {ds}", fontsize=10)
        if cidx == 0:
            ax.set_ylabel("Δ thinking words vs PLAIN")
        if r == 1:
            ax.set_xlabel("Δ search calls vs PLAIN")
fig.suptitle("Issue 2 — When a cue cuts search, is the effort redirected into thinking?  "
             "(absolute units, per model; 95% bootstrap CIs)\n"
             "Filled = both shifts significant. Upper-left = SUBSTITUTION (only Qwen); "
             "lower-left = SUPPRESSION (Gemini/Gemma). Nemotron mostly searches MORE.",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig2_suppression.png"))
plt.close(fig)


# ===========================================================================
# FIGURE 3a -- Issue 3: judge-regex gap vs response length (both datasets)
# Computed per (model, condition) from the same graded verdicts; pooled points.
# ===========================================================================
def condition_gap_table(gdf, tok):
    """Per (model, phrasing, cue): llm_acc, regex_acc, gap, mean response words."""
    rows = []
    words = tok  # tok has response? no -> use token df's thinking? need response words
    for (m, ph, cue), g in gdf.groupby(["model", "phrasing", "cue"]):
        rows.append({"model": m, "phrasing": ph, "cue": cue,
                     "llm_acc": 100 * g["llm"].mean(), "regex_acc": 100 * g["regex"].mean(),
                     "gap": 100 * (g["llm"].mean() - g["regex"].mean())})
    return pd.DataFrame(rows)

# response-length per (model,phrasing,cue) from the token CSVs (response_words)
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
    tab = condition_gap_table(GRADED[ds], None).merge(RW[ds], on=["model", "phrasing", "cue"], how="left")
    tab = tab[tab["model"].isin(MODEL_ORDER)]
    x = tab["response_words"].values
    y = tab["gap"].values
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    mcol = tab.loc[ok, "model"].map(MODEL_COLOR).values
    ax.scatter(x, y, marker=DS_MARK[ds], s=40, c=mcol, edgecolors="white", linewidths=0.4,
               alpha=0.85)
    r, p = pearsonr(x, y)
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
            label=f"{ds}: r={r:+.2f}{stars(p)} (p={p:.1e})")
ax.set_xlabel("Mean response length (words)")
ax.set_ylabel("Judge − regex accuracy gap (pp)")
ax.set_title("Issue 3a — Judge−regex gap is largest on the SHORTEST answers\n"
             "point = one model×phrasing×cue · color = model · shape = dataset")
mh = [plt.Line2D([], [], marker="o", ls="", color=MODEL_COLOR[m], label=MODEL_LABEL[m])
      for m in MODEL_ORDER]
leg1 = ax.legend(handles=mh, loc="upper right", fontsize=8, title="model")
ax.add_artist(leg1)
ax.legend(loc="lower left", fontsize=8.5)
fig.savefig(os.path.join(OUT, "fig3a_gap_vs_length.png"))
plt.close(fig)


# ===========================================================================
# FIGURE 3b -- Issue 3: per-model Δaccuracy vs PLAIN, JUDGE vs REGEX, McNemar.
# 2 rows (dataset) x 7 cols (model).
# ===========================================================================
fig, axes = plt.subplots(2, len(MODEL_ORDER), figsize=(31.5, 8.4), constrained_layout=True, sharey="row")
for r, ds in enumerate(["FRAMES", "MedQA"]):
    gdf = GRADED[ds]
    for cidx, m in enumerate(MODEL_ORDER):
        ax = axes[r][cidx]
        present = [c for c in CUES if not gdf[(gdf.model == m) & (gdf.cue == c)].empty]
        if not present:
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{MODEL_LABEL[m]}\n({ds})\nno data", ha="center", va="center",
                    color="#999", fontsize=9)
            continue
        xb = np.arange(len(present))
        w = 0.38
        dj = [mcnemar_delta(gdf, m, c, "llm") for c in present]
        dr = [mcnemar_delta(gdf, m, c, "regex") for c in present]
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
        if cidx == 0:
            ax.set_ylabel("Δ accuracy vs PLAIN (pp)")
        if r == 0 and cidx == 0:
            ax.legend(fontsize=8, loc="lower left")
fig.suptitle("Issue 3b — Real accuracy cost of each cue, per model: LLM judge vs regex  "
             "(exact McNemar vs PLAIN; * .05 ** .01 *** .001)\n"
             "Regex over-penalizes terse cues (DIRECT/NATURAL); judge-confirmed drops are smaller "
             "and mostly on DIRECT/QUERY. Nemotron gains under cues.",
             fontsize=12, fontweight="bold")
fig.savefig(os.path.join(OUT, "fig3b_accuracy_per_model.png"))
plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary of the per-model significance driving fig3b
# ---------------------------------------------------------------------------
print("=== fig3b: Δacc vs PLAIN (pp) [judge / regex] with McNemar p, per model ===")
for ds in ["FRAMES", "MedQA"]:
    print(f"\n--- {ds} ---")
    gdf = GRADED[ds]
    for m in MODEL_ORDER:
        parts = []
        for c in CUES:
            if gdf[(gdf.model == m) & (gdf.cue == c)].empty:
                continue
            dj = mcnemar_delta(gdf, m, c, "llm")
            dr = mcnemar_delta(gdf, m, c, "regex")
            parts.append(f"{c[:4]}: J{dj[0]:+.0f}{stars(dj[1]) or ''}/R{dr[0]:+.0f}{stars(dr[1]) or ''}")
        if parts:
            print(f"  {MODEL_LABEL[m]:14s} " + "  ".join(parts))
print("\nSAVED fig1, fig2, fig3a, fig3b to", OUT)
