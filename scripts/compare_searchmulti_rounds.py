"""Round-count ablation for the mocked-search history cue: does the search_calls effect
(vs PLAIN) grow, plateau, or reverse as the prior conversation gets more search rounds?

Scoped to exactly the 8 (dataset, model) cases where searchmulti (1 round) already showed
an INCREASE in mean search_calls vs PLAIN -- everywhere else searchmulti suppressed search,
so a round-count ablation isn't meaningful there.

Conditions compared per case: PLAIN, SEARCHMULTI (1 round), SEARCHMULTI2 (2 rounds),
SEARCHMULTI3 (3 rounds) -- local-search-backend only (--search-backend local).

Inputs:
  results/frames_cues_full/<model>/*.json  (FRAMES, phrasing="verbose")
  results/medqa_grid/<model>/*.json        (MedQA, phrasing="orig")

Outputs into results/cue_briefing_searchmulti_rounds/:
  rounds_search_calls.png, rounds_accuracy.png, rounds_table.csv
"""
import os
import importlib.util
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/home/dvirla/projects/parametric_search_tradeoff"
OUT = os.path.join(ROOT, "results", "cue_briefing_searchmulti_rounds")
os.makedirs(OUT, exist_ok=True)

# (dataset_label, phrasing_prefix, grid_dir, model_slug) -- the 8 cases where searchmulti
# (1 round) increased mean search_calls vs PLAIN.
CASES = [
    ("FRAMES", "verbose", "results/frames_cues_full", "nemotron-3-nano_30b"),
    ("FRAMES", "verbose", "results/frames_cues_full", "gemma4_e4b"),
    ("MedQA", "orig", "results/medqa_grid", "gemma4_31b"),
    ("MedQA", "orig", "results/medqa_grid", "gpt-oss_20b"),
    ("MedQA", "orig", "results/medqa_grid", "gemma4_e4b"),
    ("MedQA", "orig", "results/medqa_grid", "gpt-oss_120b"),
    ("MedQA", "orig", "results/medqa_grid", "nemotron-3-nano_30b"),
    ("MedQA", "orig", "results/medqa_grid", "qwen3.5_35b"),
]

ROUND_CUES = ["plain", "searchmulti", "searchmulti2", "searchmulti3"]
ROUND_LABELS = {"plain": "PLAIN", "searchmulti": "1 ROUND", "searchmulti2": "2 ROUNDS", "searchmulti3": "3 ROUNDS"}

MODEL_LABEL = {
    "nemotron-3-nano_30b": "Nemotron3 30B", "gemma4_e4b": "Gemma4 E4B",
    "gemma4_31b": "Gemma4 31B", "gpt-oss_20b": "GPT-OSS 20B", "gpt-oss_120b": "GPT-OSS 120B",
    "qwen3.5_35b": "Qwen3.5 35B",
}
MODEL_COLOR = {
    "nemotron-3-nano_30b": "#9970ab", "gemma4_e4b": "#e66101",
    "gemma4_31b": "#5aae61", "gpt-oss_20b": "#80cdc1", "gpt-oss_120b": "#01665e",
    "qwen3.5_35b": "#f4a582",
}

_spec = importlib.util.spec_from_file_location("regrade_regex", os.path.join(ROOT, "scripts/regrade_regex.py"))
_RG = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_RG)


def load_case(grid_dir, dataset_name, model_slug, phrasing):
    """Per-row table for one model: condition (bare cue), example_id, regex, search_calls."""
    rows = []
    for path, slug, cond in _RG.find_grid_files(os.path.join(ROOT, grid_dir), dataset_name):
        if slug != model_slug:
            continue
        # condition_from_filename returns "phrasing_cue" for both datasets (e.g. "verbose_plain",
        # "orig_searchmulti2") -- strip the phrasing prefix to get the bare round-cue.
        cue = cond[len(f"{phrasing}_"):] if cond.startswith(f"{phrasing}_") else None
        if cue is None or cue not in ROUND_CUES:
            continue
        for r in _RG.load_rows(path):
            g = _RG.grade_row(r)
            rows.append({
                "cue": cue, "example_id": g["example_id"],
                "regex": int(bool(g["regex_strict"])),
                "search_calls": r.get("sampler_search_calls", 0) or 0,
            })
    return pd.DataFrame(rows)


def mean_ci(vals, nboot=2000, seed=0):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 5:
        return None
    rng = np.random.default_rng(seed)
    n = len(vals)
    boot = np.array([vals[rng.integers(0, n, n)].mean() for _ in range(nboot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(mean=vals.mean(), lo=lo, hi=hi, n=n)


# ---------------------------------------------------------------------------
# Build the per-case, per-round table
# ---------------------------------------------------------------------------
rows = []
case_data = {}
for ds, ph, grid_dir, model in CASES:
    df = load_case(grid_dir, ds.lower(), model, ph)
    case_data[(ds, model)] = df
    for cue in ROUND_CUES:
        sub = df[df.cue == cue]
        sc = mean_ci(sub["search_calls"].values) if not sub.empty else None
        acc = mean_ci(sub["regex"].values) if not sub.empty else None
        rows.append({
            "dataset": ds, "model": model, "condition": ROUND_LABELS[cue],
            "n": len(sub),
            "search_calls_mean": sc["mean"] if sc else np.nan,
            "search_calls_lo": sc["lo"] if sc else np.nan,
            "search_calls_hi": sc["hi"] if sc else np.nan,
            "regex_acc_mean": acc["mean"] if acc else np.nan,
            "regex_acc_lo": acc["lo"] if acc else np.nan,
            "regex_acc_hi": acc["hi"] if acc else np.nan,
        })

table = pd.DataFrame(rows)
table.to_csv(os.path.join(OUT, "rounds_table.csv"), index=False)
print(f"Wrote {os.path.join(OUT, 'rounds_table.csv')}")
print(table.to_string(index=False))


# ---------------------------------------------------------------------------
# Figures: one panel per case, x-axis = round count
# ---------------------------------------------------------------------------
def plot_metric(mean_col, lo_col, hi_col, ylabel, title, fname, pct_fmt="{:.2f}"):
    n_cases = len(CASES)
    ncols = 4
    nrows = -(-n_cases // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)
    for i, (ds, ph, grid_dir, model) in enumerate(CASES):
        ax = axes[i // ncols][i % ncols]
        sub = table[(table.dataset == ds) & (table.model == model)]
        labels = [ROUND_LABELS[c] for c in ROUND_CUES]
        means = [sub[sub.condition == lbl][mean_col].values[0] if lbl in sub.condition.values else np.nan for lbl in labels]
        los = [sub[sub.condition == lbl][lo_col].values[0] if lbl in sub.condition.values else np.nan for lbl in labels]
        his = [sub[sub.condition == lbl][hi_col].values[0] if lbl in sub.condition.values else np.nan for lbl in labels]
        xb = np.arange(len(labels))
        err_lo = [max(0.0, m - l) if not (np.isnan(m) or np.isnan(l)) else 0.0 for m, l in zip(means, los)]
        err_hi = [max(0.0, h - m) if not (np.isnan(m) or np.isnan(h)) else 0.0 for m, h in zip(means, his)]
        color = MODEL_COLOR.get(model, "#377eb8")
        ax.bar(xb, means, 0.6, color=color, yerr=[err_lo, err_hi], capsize=3, ecolor="#333",
               error_kw={"elinewidth": 1.1})
        for xi, val in zip(xb, means):
            if not np.isnan(val):
                ax.text(xi, val, pct_fmt.format(val), ha="center", va="bottom", fontsize=8, color="#123")
        ax.set_xticks(xb)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8.5)
        ax.set_title(f"{MODEL_LABEL.get(model, model)} · {ds}", fontsize=10.5)
        if i % ncols == 0:
            ax.set_ylabel(ylabel)
    for j in range(n_cases, nrows * ncols):
        axes[j // ncols][j % ncols].set_axis_off()
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print(f"Wrote {os.path.join(OUT, fname)}")


plot_metric("search_calls_mean", "search_calls_lo", "search_calls_hi",
            "Mean search calls (95% CI)",
            "Round-Count Ablation: Search-Call Volume by # Mocked-Search Rounds\n"
            "(only the 8 cases where 1-round SEARCHMULTI increased search vs PLAIN)",
            "rounds_search_calls.png")
plot_metric("regex_acc_mean", "regex_acc_lo", "regex_acc_hi",
            "Mean regex accuracy (95% CI)",
            "Round-Count Ablation: Accuracy by # Mocked-Search Rounds",
            "rounds_accuracy.png", pct_fmt="{:.2f}")

print(f"\nSaved round-count ablation outputs to {OUT}")
