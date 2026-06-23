"""
Analyze the FRAMES prompt-cue sensitivity experiment.

For each model we have one eval-result JSON per cue condition (produced by
run_frames_cue_experiment.sh). The dependent variables are per-problem `sampler_search_calls`
and `sampler_correct`. We pair every cue condition against the NEUTRAL anchor on example_id and
test whether the cue shifts the agent's retrieval policy.

Hypothesis:
  - need-irrelevant cues (epistemic, framing, mitigation) should NOT change search behavior;
    a significant change indicates miscalibration / tool-use sycophancy.
  - need-relevant cues (structural explicitness, output constraints) may legitimately change it.

Per condition vs neutral (paired by example_id):
  - search rate (fraction with search_calls > 0) and mean search_calls
  - accuracy (expected ~flat — confirms behavior-not-accuracy story)
  - McNemar exact test on searched-or-not; Wilcoxon signed-rank on search_calls

Usage:
  uv run python scripts/analyze_frames_cues.py \
      --results-dir results/frames_cues/gpt-oss_20b \
      --cues-dir data/frames_cues \
      --model-label "gpt-oss:20b" \
      --output-dir results/frames_cues/analysis
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Cue-class grouping mirrors CUE_SPECS in scripts/paraphrase_frames_cues.py.
CONDITION_ORDER = [
    "neutral",
    "epi_strong_boost", "epi_mild_boost", "epi_mild_hedge", "epi_strong_hedge",
    "framing_first_person", "mitigation_polite",
    "structural_implicit", "output_final_only", "output_detailed",
]
CUE_CLASS = {
    "neutral": "control",
    "epi_strong_boost": "need_irrelevant", "epi_mild_boost": "need_irrelevant",
    "epi_mild_hedge": "need_irrelevant", "epi_strong_hedge": "need_irrelevant",
    "framing_first_person": "need_irrelevant", "mitigation_polite": "need_irrelevant",
    "structural_implicit": "need_relevant",
    "output_final_only": "need_relevant", "output_detailed": "need_relevant",
}


def _problem_to_example_id(cues_dir: str, condition: str) -> dict[str, str]:
    """Map question text (the result-JSON key) -> example_id, via the condition JSONL."""
    path = os.path.join(cues_dir, f"{condition}.jsonl")
    mapping = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            mapping[r["text"].strip()] = str(r["example_id"])
    return mapping


def _find_result_json(results_dir: str, condition: str) -> str | None:
    hits = glob.glob(os.path.join(results_dir, f"frames-cues_baseline_*_{condition}.json"))
    return hits[0] if hits else None


def load_condition(results_dir: str, cues_dir: str, condition: str) -> pd.DataFrame | None:
    """Return per-example DataFrame: example_id, search_calls, searched(bool), correct(bool)."""
    res_path = _find_result_json(results_dir, condition)
    if not res_path:
        return None
    with open(res_path) as f:
        results = json.load(f)
    prob2id = _problem_to_example_id(cues_dir, condition)

    rows = []
    for r in results:
        ex = prob2id.get(str(r["problem"]).strip())
        if ex is None:
            continue
        sc = int(r.get("sampler_search_calls", 0) or 0)
        rows.append({
            "example_id": ex,
            "search_calls": sc,
            "searched": sc > 0,
            "correct": bool(r.get("sampler_correct")),
        })
    if not rows:
        return None
    df = pd.DataFrame(rows).drop_duplicates("example_id").set_index("example_id")
    return df


def mcnemar_exact(b: int, c: int) -> float:
    """Exact (binomial) McNemar p-value for discordant pairs b, c."""
    n = b + c
    if n == 0:
        return 1.0
    return binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue


def main(args):
    conditions = [c for c in CONDITION_ORDER
                  if _find_result_json(args.results_dir, c) is not None]
    if "neutral" not in conditions:
        raise SystemExit(f"No neutral result JSON found in {args.results_dir}; cannot pair.")

    per_cond = {c: load_condition(args.results_dir, args.cues_dir, c) for c in conditions}
    per_cond = {c: d for c, d in per_cond.items() if d is not None}
    neutral = per_cond["neutral"]

    summary = []
    for cond in conditions:
        if cond not in per_cond:
            continue
        df = per_cond[cond]
        if cond == "neutral":
            common = df.index
            n = len(common)
            summary.append(dict(
                condition=cond, cue_class=CUE_CLASS[cond], n=n,
                search_rate=df["searched"].mean(), mean_search_calls=df["search_calls"].mean(),
                accuracy=df["correct"].mean(),
                d_search_rate=0.0, d_mean_search=0.0, d_accuracy=0.0,
                mcnemar_p=np.nan, wilcoxon_p=np.nan, b_neutralYES_condNO=np.nan,
                c_neutralNO_condYES=np.nan,
            ))
            continue

        common = neutral.index.intersection(df.index)
        n = len(common)
        nb = neutral.loc[common]
        cb = df.loc[common]

        # McNemar on searched-or-not (paired)
        b = int(((nb["searched"]) & (~cb["searched"])).sum())   # neutral searched, cond didn't
        c = int(((~nb["searched"]) & (cb["searched"])).sum())   # cond searched, neutral didn't
        mp = mcnemar_exact(b, c)

        # Wilcoxon signed-rank on search_calls (paired); guard all-zero differences
        diff = cb["search_calls"].to_numpy() - nb["search_calls"].to_numpy()
        if np.any(diff != 0):
            wp = wilcoxon(cb["search_calls"], nb["search_calls"], zero_method="wilcox").pvalue
        else:
            wp = 1.0

        summary.append(dict(
            condition=cond, cue_class=CUE_CLASS[cond], n=n,
            search_rate=cb["searched"].mean(), mean_search_calls=cb["search_calls"].mean(),
            accuracy=cb["correct"].mean(),
            d_search_rate=cb["searched"].mean() - nb["searched"].mean(),
            d_mean_search=cb["search_calls"].mean() - nb["search_calls"].mean(),
            d_accuracy=cb["correct"].mean() - nb["correct"].mean(),
            mcnemar_p=mp, wilcoxon_p=wp,
            b_neutralYES_condNO=b, c_neutralNO_condYES=c,
        ))

    sdf = pd.DataFrame(summary)
    os.makedirs(args.output_dir, exist_ok=True)
    label = args.model_label or os.path.basename(args.results_dir.rstrip("/"))
    slug = label.replace(":", "_").replace("/", "_")
    csv_path = os.path.join(args.output_dir, f"frames_cues_summary_{slug}.csv")
    sdf.to_csv(csv_path, index=False)

    pd.set_option("display.width", 160, "display.max_columns", 20)
    print(f"\n=== FRAMES cue sensitivity — {label} ===")
    print(sdf.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nWrote {csv_path}")

    _plot(sdf, args.output_dir, slug, label)


def _plot(sdf: pd.DataFrame, output_dir: str, slug: str, label: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src import viz
        viz.apply_theme()
    except Exception as e:  # plotting is best-effort
        print(f"[plot skipped] {e}")
        return

    plot_df = sdf[sdf["condition"] != "neutral"].copy()
    class_color = {"need_irrelevant": viz.LIGHT_BLUE, "need_relevant": viz.PR}
    colors = [class_color.get(cc, "#999999") for cc in plot_df["cue_class"]]
    neutral_rate = float(sdf.loc[sdf["condition"] == "neutral", "search_rate"].iloc[0])

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(plot_df))
    ax.bar(x, plot_df["search_rate"], color=colors)
    ax.axhline(neutral_rate, ls="--", color="#444444", label=f"neutral = {neutral_rate:.2f}")
    for i, (_, row) in enumerate(plot_df.iterrows()):
        stars = viz.sig_stars(row["mcnemar_p"]) if pd.notna(row["mcnemar_p"]) else ""
        if stars:
            ax.text(i, row["search_rate"], stars, ha="center", va="bottom", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["condition"], rotation=40, ha="right")
    ax.set_ylabel("search rate (fraction searching ≥1)")
    ax.set_title(f"FRAMES cue sensitivity — {label}\n(blue = need-irrelevant, orange = need-relevant; "
                 f"stars = McNemar vs neutral)")
    ax.legend()
    viz.savefig(fig, output_dir, f"frames_cues_search_rate_{slug}")
    print(f"Wrote plot frames_cues_search_rate_{slug}.png/.pdf in {output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Paired analysis of FRAMES prompt-cue experiment")
    p.add_argument("--results-dir", required=True, help="results/frames_cues/<model_slug>/")
    p.add_argument("--cues-dir", default="data/frames_cues", help="Condition JSONL dir (for problem->id map)")
    p.add_argument("--model-label", default=None, help="Display label (e.g. 'gpt-oss:20b')")
    p.add_argument("--output-dir", default="results/frames_cues/analysis", help="Where to write CSV + plots")
    main(p.parse_args())
