"""Three-way phrasing comparison: formal vs natural vs natural2.

Nemotron and Qwen only — natural2 has no reevaluated Gemini result yet.
Uses the same reevaluated JSON files and pairing JSONLs as
analyze_phrasing_corrected.py.  Produces accuracy and search-volume
bar charts in the same style.

Usage:
  uv run python scripts/analyze_phrasing_natural2.py
  uv run python scripts/analyze_phrasing_natural2.py --output-dir results/phrasing_effect_corrected
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, binomtest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import viz

SLUGS = ["nemotron-3-nano_30b", "qwen3.5_122b"]

FORMAL_TMPL   = "results/musique-formal/musique-formal_baseline_{slug}_run_1_reevaluated.json"
NATURAL_TMPL  = "results/musique-natural/musique-natural_baseline_{slug}_run_1_reevaluated.json"
# natural2 filenames use colons in the model name
NATURAL2_SLUG = {
    "nemotron-3-nano_30b": "nemotron-3-nano:30b",
    "qwen3.5_122b":        "qwen3.5:122b",
}
NATURAL2_TMPL = "results/musique-natural2/musique-natural2_baseline_{slug}_run_1_reevaluated.json"


def load_pairing(path):
    text2eid, hops = {}, {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            text2eid[r["text"]] = r["example_id"]
            hops[r["example_id"]] = r.get("reasoning_hops")
    return text2eid, hops


def load_by_eid(path, text2eid=None):
    """Load {eid: (correct, searches)}.  If text2eid provided, maps problem→eid."""
    d = json.load(open(path))
    out = {}
    for e in d:
        if "example_id" in e:
            eid = e["example_id"]
        elif text2eid is not None:
            eid = text2eid.get(e.get("problem"))
        else:
            eid = None
        if eid is not None:
            out[eid] = (bool(e["sampler_correct"]), e.get("sampler_search_calls") or 0)
    return out


def mcnemar_p(a, b, ids):
    disc = [(a[k][0], b[k][0]) for k in ids]
    bc = sum(1 for x, y in disc if x and not y)
    cb = sum(1 for x, y in disc if not x and y)
    return binomtest(min(bc, cb), bc + cb, 0.5).pvalue if (bc + cb) else 1.0


def wilcoxon_p(a, b, ids):
    as_ = [a[k][1] for k in ids]
    bs_ = [b[k][1] for k in ids]
    if any(x != y for x, y in zip(as_, bs_)):
        try:
            return wilcoxon(as_, bs_).pvalue
        except ValueError:
            pass
    return 1.0


COND_COLORS = {
    "formal":   viz.BENCHMARK,
    "natural":  viz.NATURAL,
    "natural2": "#2ca02c",   # green — third phrasing
}
COND_LABELS = {
    "formal":   "Formal (benchmark)",
    "natural":  "Natural (paraphrase 1)",
    "natural2": "Natural2 (paraphrase 2)",
}


def make_figure(rows, metric, ylabel, title, slugs, output_dir):
    viz.apply_theme()
    df = pd.DataFrame(rows)
    sub = df[df.metric == metric]

    conditions = ["formal", "natural", "natural2"]
    n_conds = len(conditions)
    w = 0.22
    offsets = np.linspace(-(n_conds - 1) * w / 2, (n_conds - 1) * w / 2, n_conds)
    x = np.arange(len(slugs))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, off in zip(conditions, offsets):
        vals, los, his = [], [], []
        for slug in slugs:
            row = sub[(sub.model == slug) & (sub.cond == cond)]
            if row.empty:
                vals.append(np.nan); los.append(0); his.append(0)
            else:
                r = row.iloc[0]
                vals.append(r["val"])
                los.append(r["val"] - r["lo"])
                his.append(r["hi"] - r["val"])
        bars = ax.bar(x + off, vals, w, yerr=[los, his], capsize=3,
                      color=COND_COLORS[cond], label=COND_LABELS[cond],
                      error_kw=dict(ecolor=viz.ERR_DARK, lw=1), zorder=3)
        for bar, v, hi in zip(bars, vals, his):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        v + hi + (0.015 if metric == "accuracy" else 0.08),
                        f"{v:.1%}" if metric == "accuracy" else f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    # significance stars between formal and natural2
    for i, slug in enumerate(slugs):
        f_row = sub[(sub.model == slug) & (sub.cond == "formal")]
        n2_row = sub[(sub.model == slug) & (sub.cond == "natural2")]
        if not f_row.empty and not n2_row.empty:
            p = f_row.iloc[0]["p_vs_formal"]
            top = max(f_row.iloc[0]["hi"], n2_row.iloc[0]["hi"])
            stars = viz.sig_stars(p)
            if stars:
                pad = 0.04 if metric == "accuracy" else 0.2
                ax.text(x[i], top + pad, f"F↔N2: {stars}", ha="center", fontsize=8,
                        color=COND_COLORS["natural2"], fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([viz.display_name(s) for s in slugs], fontsize=10, fontweight="bold")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if metric == "accuracy":
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fname = f"phrasing_{metric}_natural2"
    viz.savefig(fig, output_dir, fname)
    plt.close(fig)
    print(f"  -> {fname}.png/pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/phrasing_effect_corrected")
    ap.add_argument("--pairing1", default="data/musique_val_natural.jsonl")
    ap.add_argument("--pairing2", default="data/musique_val_natural2.jsonl")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    text2eid1, hops1 = load_pairing(args.pairing1)
    text2eid2, hops2 = load_pairing(args.pairing2)
    print(f"Pairing1: {len(text2eid1)} texts.  Pairing2: {len(text2eid2)} texts.")

    rows = []
    for slug in SLUGS:
        f  = load_by_eid(FORMAL_TMPL.format(slug=slug))
        n1 = load_by_eid(NATURAL_TMPL.format(slug=slug), text2eid1)
        n2_slug = NATURAL2_SLUG[slug]
        n2 = load_by_eid(NATURAL2_TMPL.format(slug=n2_slug), text2eid2)

        common = sorted(set(f) & set(n1) & set(n2))
        print(f"{viz.display_name(slug)}: formal={len(f)}  natural={len(n1)}  "
              f"natural2={len(n2)}  common={len(common)}")

        for cond, data, ref in [("formal", f, None), ("natural", n1, f), ("natural2", n2, f)]:
            ids = common
            N = len(ids)
            for metric in ("accuracy", "searches"):
                if metric == "accuracy":
                    k = sum(data[k][0] for k in ids)
                    val = k / N
                    lo, hi = viz.wilson_ci(k, N)
                    p = mcnemar_p(ref, data, ids) if ref else 1.0
                else:
                    arr = np.array([data[k][1] for k in ids], float)
                    val, hw = viz.mean_ci95(arr)
                    lo, hi = val - hw, val + hw
                    if ref is not None:
                        ref_arr = np.array([ref[k][1] for k in ids], float)
                        if any(a != b for a, b in zip(ref_arr, arr)):
                            try:
                                p = wilcoxon(ref_arr, arr).pvalue
                            except ValueError:
                                p = 1.0
                        else:
                            p = 1.0
                    else:
                        p = 1.0
                rows.append(dict(model=slug, cond=cond, metric=metric,
                                 val=val, lo=lo, hi=hi, p_vs_formal=p, n=N))

    # console summary
    df = pd.DataFrame(rows)
    print("\n=== SUMMARY ===")
    for metric in ("accuracy", "searches"):
        print(f"\n-- {metric}")
        for slug in SLUGS:
            sub = df[(df.model == slug) & (df.metric == metric)]
            parts = []
            for _, r in sub.iterrows():
                parts.append(f"{r.cond}={r.val:.3f}(p={r.p_vs_formal:.3f}{viz.sig_stars(r.p_vs_formal)})")
            print(f"  {viz.display_name(slug)}: " + "  ".join(parts))

    df.to_csv(os.path.join(args.output_dir, "phrasing_natural2_stats.csv"), index=False)

    make_figure(rows, "accuracy", "Aggregate accuracy",
                "Accuracy: formal vs natural vs natural2\n(all graded by gpt-oss:120b, paired n≈590)",
                SLUGS, args.output_dir)
    make_figure(rows, "searches", "Mean searches / example",
                "Search volume: formal vs natural vs natural2",
                SLUGS, args.output_dir)


if __name__ == "__main__":
    main()
