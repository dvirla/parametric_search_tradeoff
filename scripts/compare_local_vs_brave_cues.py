#!/usr/bin/env python3
"""Compare FRAMES cue-grid findings between the local BM25 index and the live Brave
search backend, for the models run under both: gemini-3.5-flash, gemma4:31b,
gpt-oss:120b, qwen3.5:122b.

Reuses the same regex grading (scripts/regrade_regex.py) and delta-vs-PLAIN
statistics (McNemar for accuracy, Wilcoxon for search calls) as
scripts/make_cue_briefing_figures.py, applied per-backend instead of per-dataset.
search_calls comes directly from each row's sampler_search_calls (no separate
token-analysis CSV / thinking-word data available for the brave runs, so the
suppression-scatter figure from the original script is out of scope here).

Inputs:
  results/frames_cues_full/<model>/*.json        (local BM25 index)
  results/frames_cues_full_brave/<model>/*.json  (live Brave API)

Outputs into results/cue_briefing_local_vs_brave/:
  compare_accuracy.png, compare_search.png, compare_table.csv
"""
import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = os.path.join(ROOT, "results", "cue_briefing_local_vs_brave")
os.makedirs(OUT, exist_ok=True)

MODELS = ["gemini-3.5-flash", "gemma4_31b", "gpt-oss_120b", "qwen3.5_122b"]
MODEL_LABEL = {
    "gemini-3.5-flash": "Gemini 3.5 Flash", "qwen3.5_122b": "Qwen3.5 122B",
    "gemma4_31b": "Gemma4 31B", "gpt-oss_120b": "GPT-OSS 120B",
}
MIN_N = 30  # below this, still compute but flag as low-n in the table

_spec = importlib.util.spec_from_file_location("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))
_RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RG)


def base_cue(cond):
    c = cond
    for p in ("verbose_", "terse_", "orig_", "epi_strong_"):
        if c.startswith(p):
            c = c[len(p):]
    return c


SEARCHMULTI_ROUND_OFFSET = {"searchmulti": 1, "searchmulti2": 2, "searchmulti3": 3}


def stars(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def load_source(grid_dir):
    """Per-row table: model, condition, phrasing, cue, example_id, regex, search_calls."""
    rows = []
    for path, slug, cond in _RG.find_grid_files(grid_dir, "frames"):
        if slug not in MODELS:
            continue
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            rows.append({
                "model": slug, "condition": cond, "example_id": g["example_id"],
                "regex": int(bool(g["regex_strict"])),
                "search_calls": r.get("sampler_search_calls", 0) or 0,
            })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["cue"] = df["condition"].map(base_cue)
    df["phrasing"] = df["condition"].map(lambda c: c.split("_", 1)[0])
    # AgentAsSampler.acall() counts search calls over pydantic-ai's all_messages(), which includes
    # the injected message_history as a literal prefix -- so multiturn/searchmulti rows are
    # inflated by exactly this many FAKE search calls from the mocked history itself (each round
    # always injects exactly one tool_call, so this is an exact correction, not an approximation).
    df["search_calls"] = df["search_calls"] - df["cue"].map(SEARCHMULTI_ROUND_OFFSET).fillna(0)
    return df


LOCAL = load_source(os.path.join(ROOT, "results/frames_cues_full"))
BRAVE = load_source(os.path.join(ROOT, "results/frames_cues_full_brave"))
SOURCES = {"local": LOCAL, "brave": BRAVE}

RERUN_LOCAL = load_source(os.path.join(ROOT, "results/frames_cues_rerun"))
RERUN_BRAVE = load_source(os.path.join(ROOT, "results/frames_cues_rerun_brave"))
RERUNS = {"local": RERUN_LOCAL, "brave": RERUN_BRAVE}
RERUN_MIN_N = 100
RERUN_LABEL = "PLAIN↔PLAIN"

CONDITIONS = [("verbose", "polite"), ("terse", "plain"), ("verbose", "natural"),
              ("verbose", "elaborate"), ("verbose", "query"), ("verbose", "direct"),
              ("verbose", "multiturn"), ("verbose", "searchmulti")]
BASE_PH, BASE_CUE = "verbose", "plain"


def get_label(ph, cue):
    if cue == "plain":
        return f"{ph.upper()} (PLAIN)"
    if cue == "natural":
        return "SHORT"
    return cue.upper()


def mcnemar_delta(df, m, target_ph, target_cue):
    sub = df[df.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "regex"]].rename(columns={"regex": "c"})
    b = sub[(sub.phrasing == BASE_PH) & (sub.cue == BASE_CUE)][["example_id", "regex"]].rename(columns={"regex": "p"})
    if t.empty or b.empty:
        return np.nan, np.nan, 0, np.nan, np.nan
    j = t.merge(b, on="example_id")
    Nn = len(j)
    if Nn == 0:
        return np.nan, np.nan, 0, np.nan, np.nan
    cc, pp = j["c"].values, j["p"].values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    disc = n10 + n01
    p_val = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / Nn, p_val, Nn, cc.mean(), pp.mean()


def wilcoxon_search(df, m, target_ph, target_cue):
    sub = df[df.model == m]
    t = sub[(sub.phrasing == target_ph) & (sub.cue == target_cue)][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    b = sub[(sub.phrasing == BASE_PH) & (sub.cue == BASE_CUE)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    j = t.merge(b, on="example_id").dropna()
    if len(j) < 5:
        return np.nan, np.nan, 0
    diff = (j["target"] - j["base"]).values
    if np.allclose(diff, 0):
        pct_change, p = 0.0, 1.0
    else:
        try:
            p = wilcoxon(j["target"].values, j["base"].values, zero_method="wilcox").pvalue
        except ValueError:
            p = 1.0
        mean_base = j["base"].mean()
        pct_change = 100.0 * (j["target"].mean() - mean_base) / mean_base if mean_base > 0 else np.nan
    return float(pct_change), p, len(j)


def rerun_acc_delta(df, rerun_df, m):
    """pp change in regex accuracy: 2nd PLAIN run vs 1st (McNemar, paired). Noise floor."""
    if df.empty or rerun_df.empty or m not in set(df.model) or m not in set(rerun_df.model):
        return np.nan, np.nan, 0
    b = df[(df.model == m) & (df.phrasing == BASE_PH) & (df.cue == BASE_CUE)][["example_id", "regex"]].rename(columns={"regex": "p"})
    t = rerun_df[rerun_df.model == m][["example_id", "regex"]].rename(columns={"regex": "c"})
    j = t.merge(b, on="example_id")
    if len(j) < RERUN_MIN_N:
        return np.nan, np.nan, 0
    cc, pp = j["c"].values, j["p"].values
    n10, n01 = int(((cc == 1) & (pp == 0)).sum()), int(((cc == 0) & (pp == 1)).sum())
    Nn = len(j)
    disc = n10 + n01
    pv = binomtest(min(n10, n01), disc, 0.5).pvalue if disc > 0 else 1.0
    return 100 * (n10 - n01) / Nn, pv, Nn


def rerun_search_delta(df, rerun_df, m):
    """% change in search calls: 2nd PLAIN run vs 1st (Wilcoxon, paired). Noise floor."""
    if df.empty or rerun_df.empty or m not in set(df.model) or m not in set(rerun_df.model):
        return np.nan, np.nan
    b = df[(df.model == m) & (df.phrasing == BASE_PH) & (df.cue == BASE_CUE)][["example_id", "search_calls"]].rename(columns={"search_calls": "base"})
    t = rerun_df[rerun_df.model == m][["example_id", "search_calls"]].rename(columns={"search_calls": "target"})
    j = b.merge(t, on="example_id").dropna()
    if len(j) < RERUN_MIN_N:
        return np.nan, np.nan
    diff = (j["target"] - j["base"]).values
    if np.allclose(diff, 0):
        return 0.0, 1.0
    try:
        p = wilcoxon(j["target"].values, j["base"].values, zero_method="wilcox").pvalue
    except ValueError:
        p = 1.0
    mb = j["base"].mean()
    return (100.0 * (j["target"].mean() - mb) / mb if mb > 0 else np.nan), p


# ---------------------------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------------------------
rows = []
for m in MODELS:
    for ph, cue in CONDITIONS:
        row = {"model": m, "condition": get_label(ph, cue)}
        for src_name, df in SOURCES.items():
            if df.empty or m not in set(df.model):
                row[f"{src_name}_n"] = 0
                continue
            acc_d, acc_p, n_acc, acc_cond, acc_plain = mcnemar_delta(df, m, ph, cue)
            sc_d, sc_p, n_sc = wilcoxon_search(df, m, ph, cue)
            row[f"{src_name}_n"] = n_acc
            row[f"{src_name}_acc_plain"] = acc_plain
            row[f"{src_name}_acc_cond"] = acc_cond
            row[f"{src_name}_delta_acc_pp"] = acc_d
            row[f"{src_name}_delta_acc_p"] = acc_p
            row[f"{src_name}_delta_search_pct"] = sc_d
            row[f"{src_name}_delta_search_p"] = sc_p
        rows.append(row)
    # PLAIN<->PLAIN reference row: run-to-run noise floor, not a cue effect
    row = {"model": m, "condition": RERUN_LABEL}
    for src_name, df in SOURCES.items():
        rerun_df = RERUNS[src_name]
        acc_d, acc_p, n_acc = rerun_acc_delta(df, rerun_df, m)
        sc_d, sc_p = rerun_search_delta(df, rerun_df, m)
        row[f"{src_name}_n"] = n_acc
        row[f"{src_name}_delta_acc_pp"] = acc_d
        row[f"{src_name}_delta_acc_p"] = acc_p
        row[f"{src_name}_delta_search_pct"] = sc_d
        row[f"{src_name}_delta_search_p"] = sc_p
    rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "compare_table.csv"), index=False)
print(f"Wrote {os.path.join(OUT, 'compare_table.csv')}")
print(table.to_string(index=False))

# ---------------------------------------------------------------------------
# Grouped bar figures: local vs brave, per model, per condition
# ---------------------------------------------------------------------------
def grouped_bars(value_col, p_col, ylabel, title, fname, pct_fmt="{:+.0f}"):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(4.2 * len(MODELS), 4.2), constrained_layout=True, sharey=False)
    for ax, m in zip(axes, MODELS):
        sub = table[table.model == m]
        labels = sub["condition"].tolist()
        is_rr = sub["condition"].eq(RERUN_LABEL).values
        xb = np.arange(len(labels))
        w = 0.35
        lv = sub[f"local_{value_col}"].values
        bv = sub[f"brave_{value_col}"].values
        lp = sub[f"local_{p_col}"].values
        bp = sub[f"brave_{p_col}"].values
        lcolor = ["#9e9e9e" if rr else "#377eb8" for rr in is_rr]
        bcolor = ["#c2c2c2" if rr else "#e6550d" for rr in is_rr]
        lbars = ax.bar(xb - w / 2, lv, w, color=lcolor, label="Local index")
        bbars = ax.bar(xb + w / 2, bv, w, color=bcolor, label="Brave")
        for bar, rr in zip(lbars, is_rr):
            if rr:
                bar.set_hatch("//"); bar.set_edgecolor("#444")
        for bar, rr in zip(bbars, is_rr):
            if rr:
                bar.set_hatch("//"); bar.set_edgecolor("#444")
        if is_rr.any():
            ax.axvspan(np.where(is_rr)[0][0] - 0.5, np.where(is_rr)[0][0] + 0.5, color="#9e9e9e", alpha=0.12)
        for xi, v, p in zip(xb - w / 2, lv, lp):
            if not np.isnan(v):
                ax.text(xi, v + (1 if v >= 0 else -1), f"{pct_fmt.format(v)}{stars(p)}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7, rotation=90)
        for xi, v, p in zip(xb + w / 2, bv, bp):
            if not np.isnan(v):
                ax.text(xi, v + (1 if v >= 0 else -1), f"{pct_fmt.format(v)}{stars(p)}",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=7, rotation=90)
        ax.axhline(0, color="#333", lw=0.8)
        # Style group (POLITE, TERSE-PLAIN) vs instructions group (SHORT, ELABORATE, QUERY, DIRECT)
        ax.axvline(1.5, color="gray", linestyle="--", lw=1.2)
        # Cue conditions vs PLAIN<->PLAIN noise-floor reference
        ax.axvline(len(labels) - 1.5, color="gray", linestyle="--", lw=1.0)
        ax.set_xticks(xb)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(MODEL_LABEL[m], fontsize=11)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title + "\nhatched grey = PLAIN↔PLAIN (2nd plain run vs 1st): run-to-run noise floor", fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"Wrote {os.path.join(OUT, fname)}")


grouped_bars("delta_acc_pp", "delta_acc_p", "Δ regex acc vs PLAIN (pp)",
             "FRAMES cue effect on accuracy: Local index vs Brave (McNemar vs PLAIN, * p<.05)",
             "compare_accuracy.png")
grouped_bars("delta_search_pct", "delta_search_p", "Δ search calls vs PLAIN (%)",
             "FRAMES cue effect on search volume: Local index vs Brave (Wilcoxon vs PLAIN, * p<.05)",
             "compare_search.png")

print(f"\nSaved comparison outputs to {OUT}")
