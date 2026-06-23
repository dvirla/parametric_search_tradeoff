"""
FRAMES staleness tradeoff plot.

Compares the NON-STALE subset of FRAMES questions across two phrasing
conditions — natural (`frames`) vs. benchmark/formal (`frames-benchmark`) — in
terms of search calls vs. accuracy, for a single model (default Gemini 3.1 Pro).

Staleness verdicts come from `results/frames_staleness_flash.csv` (keyed on the
ORIGINAL natural question text == staleness `aggregate_question`). The benchmark
eval is phrased as paraphrases, so it is mapped back to the original question via
`data/frames_benchmark.jsonl` (`text` -> `original_question`).

The paired set can additionally be SCREENED by the formal-rewrite auditor verdicts
in `results/frames_benchmark_audit.csv` (`--audit-screen leak|flagged`) so the plot
reflects only non-stale AND non-leaked (or fully non-flagged) questions — i.e. cases
where the formal rewrite faithfully preserves the natural question's information need.

Output: a two-panel figure
  (A) aggregate tradeoff scatter: mean search calls (x) vs accuracy (y), with
      95% CIs in both dimensions, one point per phrasing condition.
  (B) accuracy as a function of search-call volume (binned), per condition.
Plus a paired per-question CSV for reproducibility.

Usage:
  uv run python scripts/plot_frames_staleness_tradeoff.py
"""

import os
import sys
import csv
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src import viz


def load_eval(path: str) -> dict[str, dict]:
    """Map problem-text -> {'correct': bool, 'search_calls': int} from an eval JSON."""
    out = {}
    for e in json.load(open(path)):
        m = e.get("metrics", {})
        out[e["problem"]] = {
            "correct": bool(m["correct"]),
            "search_calls": int(m["search_calls"]),
        }
    return out


def load_nonstale(csv_path: str) -> set[str]:
    """Return the set of original questions classified NOT_STALE."""
    keep = set()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            if r["is_stale"] == "False":
                keep.add(r["aggregate_question"])
    return keep


def load_audit_flagged(csv_path: str, level: str) -> set[str]:
    """Original questions to SCREEN OUT per the formal-rewrite auditor verdict.

    level='leak'    -> drop where the rewrite leaks an intermediate entity.
    level='flagged' -> also drop non-equivalent / non-neutral / drifted / failed rewrites.
    Returns both the raw and whitespace-stripped question (benchmark `original_question`
    carries trailing whitespace) so membership tests are robust to either form.
    """
    flagged = set()
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            leak = r["leaks_intermediate"] == "True"
            if level == "leak":
                bad = leak
            else:  # flagged
                bad = (leak or r["equivalent"] == "False"
                       or r["is_neutral_register"] == "False"
                       or r["drift"] not in ("none", "")
                       or r["validation_status"] == "fail")
            if bad:
                oq = r["original_question"]
                flagged.add(oq)
                flagged.add(oq.strip())
    return flagged


def build_paired(args) -> list[dict]:
    """Join natural + benchmark eval results on the original question, restricted
    to the non-stale subset. Returns one record per paired question."""
    nonstale = load_nonstale(args.staleness_csv)
    text2orig = {b["text"]: b["original_question"]
                 for b in (json.loads(l) for l in open(args.benchmark_jsonl))}

    nat = load_eval(args.natural_json)       # keyed by original question
    ben = load_eval(args.benchmark_json)     # keyed by paraphrase text

    # benchmark: re-key paraphrase -> original question
    ben_by_orig = {text2orig[p]: v for p, v in ben.items() if p in text2orig}

    # optional audit screen: drop questions whose formal rewrite is leaked/flagged
    screened = set()
    if args.audit_screen != "none":
        if not os.path.exists(args.audit_csv):
            raise SystemExit(f"--audit-screen={args.audit_screen} but audit CSV not found: {args.audit_csv}")
        screened = load_audit_flagged(args.audit_csv, args.audit_screen)

    rows = []
    n_screened = 0
    for orig in nonstale:
        if orig in nat and orig in ben_by_orig:
            if orig in screened or orig.strip() in screened:
                n_screened += 1
                continue
            rows.append({
                "question": orig,
                "nat_correct": nat[orig]["correct"],
                "nat_search": nat[orig]["search_calls"],
                "ben_correct": ben_by_orig[orig]["correct"],
                "ben_search": ben_by_orig[orig]["search_calls"],
            })
    if args.audit_screen != "none":
        print(f"Audit screen ('{args.audit_screen}'): dropped {n_screened} paired question(s) "
              f"using {args.audit_csv}")
    return rows


def summarize(rows, key_correct, key_search):
    k = sum(r[key_correct] for r in rows)
    n = len(rows)
    acc = k / n
    lo, hi = viz.wilson_ci(k, n)
    searches = np.array([r[key_search] for r in rows], dtype=float)
    mean_s, ci_s = viz.mean_ci95(searches)
    return {
        "n": n, "k": k, "acc": acc, "acc_lo": lo, "acc_hi": hi,
        "mean_search": mean_s, "search_ci": ci_s, "searches": searches,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--natural-json", default="results/frames_baseline_gemini-3.1-pro-preview_run_1.json")
    p.add_argument("--benchmark-json", default="results/frames-benchmark_baseline_gemini-3.1-pro-preview_run_1.json")
    p.add_argument("--benchmark-jsonl", default="data/frames_benchmark.jsonl")
    p.add_argument("--staleness-csv", default="results/frames_staleness_flash.csv")
    p.add_argument("--audit-csv", default="results/frames_benchmark_audit.csv",
                   help="Formal-rewrite auditor verdicts (used when --audit-screen != none)")
    p.add_argument("--audit-screen", choices=["none", "leak", "flagged"], default="none",
                   help="Restrict to non-stale AND: 'leak' drops leaked rewrites; "
                        "'flagged' also drops non-equivalent/non-neutral/drifted/failed rewrites")
    p.add_argument("--model-label", default="Gemini 3.1 Pro")
    p.add_argument("--output-dir", default="results/frames_staleness_tradeoff")
    args = p.parse_args()

    rows = build_paired(args)
    nat = summarize(rows, "nat_correct", "nat_search")
    ben = summarize(rows, "ben_correct", "ben_search")

    # Stats: McNemar (paired accuracy) + Wilcoxon (paired search calls)
    from scipy import stats as sp
    b = sum(1 for r in rows if r["nat_correct"] and not r["ben_correct"])  # nat right, ben wrong
    c = sum(1 for r in rows if not r["nat_correct"] and r["ben_correct"])  # ben right, nat wrong
    mcnemar_p = sp.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else float("nan")
    wilcoxon_p = sp.wilcoxon([r["nat_search"] for r in rows],
                             [r["ben_search"] for r in rows]).pvalue

    screen_tag = "" if args.audit_screen == "none" else f", non-{args.audit_screen}"
    print(f"\nNon-stale{screen_tag} paired FRAMES questions ({args.model_label}): n={len(rows)}")
    print(f"  Natural   : acc={nat['acc']:.3f} [{nat['acc_lo']:.3f},{nat['acc_hi']:.3f}]  "
          f"mean search={nat['mean_search']:.2f} ±{nat['search_ci']:.2f}")
    print(f"  Benchmark : acc={ben['acc']:.3f} [{ben['acc_lo']:.3f},{ben['acc_hi']:.3f}]  "
          f"mean search={ben['mean_search']:.2f} ±{ben['search_ci']:.2f}")
    print(f"  McNemar (acc) p={mcnemar_p:.3g} {viz.sig_stars(mcnemar_p)}  "
          f"(nat-only-correct={b}, ben-only-correct={c})")
    print(f"  Wilcoxon (search) p={wilcoxon_p:.3g} {viz.sig_stars(wilcoxon_p)}")

    # ── Figure ──────────────────────────────────────────────────────────────
    viz.apply_theme()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))
    conds = [("Natural", nat, viz.NATURAL, "o"), ("Benchmark (formal)", ben, viz.BENCHMARK, "s")]

    # Panel A — aggregate tradeoff scatter
    for label, s, color, marker in conds:
        axA.errorbar(
            s["mean_search"], s["acc"],
            xerr=s["search_ci"],
            yerr=[[s["acc"] - s["acc_lo"]], [s["acc_hi"] - s["acc"]]],
            fmt=marker, color=color, markersize=11, capsize=4, elinewidth=1.5,
            label=f"{label} (acc={s['acc']:.2f}, search={s['mean_search']:.1f})",
        )
    axA.set_xlabel("Mean search calls per question")
    axA.set_ylabel("Accuracy")
    axA.set_title(f"(A) Search–accuracy tradeoff\nnon-stale{screen_tag} FRAMES, n={len(rows)}")
    axA.legend(loc="best", fontsize=8, frameon=True)
    axA.grid(True, alpha=0.3)

    # Panel B — accuracy vs search-call volume (binned)
    bins = [0, 1, 6, 11, 16, 21, np.inf]
    labels = ["0", "1–5", "6–10", "11–15", "16–20", "21+"]
    centers = np.arange(len(labels))
    for cond_label, key_c, key_s, color, marker in [
        ("Natural", "nat_correct", "nat_search", viz.NATURAL, "o"),
        ("Benchmark (formal)", "ben_correct", "ben_search", viz.BENCHMARK, "s"),
    ]:
        accs, los, his, ns = [], [], [], []
        for i in range(len(labels)):
            sub = [r for r in rows if bins[i] <= r[key_s] < bins[i + 1]]
            n = len(sub)
            ns.append(n)
            if n == 0:
                accs.append(np.nan); los.append(np.nan); his.append(np.nan); continue
            k = sum(r[key_c] for r in sub)
            lo, hi = viz.wilson_ci(k, n)
            accs.append(k / n); los.append(k / n - lo); his.append(hi - k / n)
        accs = np.array(accs)
        axB.errorbar(centers, accs, yerr=[los, his], fmt=f"-{marker}", color=color,
                     capsize=3, markersize=7, label=cond_label)
        for x, a, n in zip(centers, accs, ns):
            if not np.isnan(a):
                axB.annotate(f"n={n}", (x, a), textcoords="offset points",
                             xytext=(0, 8), ha="center", fontsize=6, color=color)
    axB.set_xticks(centers); axB.set_xticklabels(labels)
    axB.set_xlabel("Search calls (binned)")
    axB.set_ylabel("Accuracy")
    axB.set_title("(B) Accuracy by search-call volume")
    axB.legend(loc="best", fontsize=8, frameon=True)
    axB.grid(True, alpha=0.3)

    fig.suptitle(f"FRAMES non-stale{screen_tag}: natural vs. benchmark phrasing — {args.model_label}",
                 fontsize=12, y=1.02)
    suffix = "" if args.audit_screen == "none" else f"_non_{args.audit_screen}"
    viz.savefig(fig, args.output_dir, f"frames_staleness_tradeoff{suffix}")

    # paired CSV for reproducibility
    out_csv = os.path.join(args.output_dir, f"frames_nonstale{suffix}_paired.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "nat_correct", "nat_search",
                                          "ben_correct", "ben_search"])
        w.writeheader()
        w.writerows(rows)
    print(f"Saved paired data to {out_csv}")


if __name__ == "__main__":
    main()
