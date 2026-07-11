#!/usr/bin/env python3
"""Compare PLAIN vs ELABORATE final responses (sampler_response) paired within model×dataset.

Reads the cue-experiment eval JSONs, pairs plain/elaborate on example_id, drops runaway
(UsageLimitExceeded) trajectories, and quantifies differences on several text axes with
paired (within-question) Wilcoxon signed-rank tests.
"""
import json
import re
import argparse
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent

# (label, dataset, model, plain_path, elaborate_path)
CELLS = [
    ("Gemini 3.1 / facts_open", "facts_open", "gemini-3.1-pro",
     "results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_plain.json",
     "results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_elaborate.json"),
    ("Qwen 122b / facts_open", "facts_open", "qwen3.5:122b",
     "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_plain.json",
     "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_elaborate.json"),
    ("Gemma4 31b / facts_open", "facts_open", "gemma4:31b",
     "results/facts_gemma100/facts-open_baseline_gemma4:31b_plain.json",
     "results/facts_gemma100/facts-open_baseline_gemma4:31b_elaborate.json"),
    ("Gemini 3.1 / MedQA", "medqa", "gemini-3.1-pro",
     "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_plain.json",
     "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_elaborate.json"),
    ("Qwen 122b / MedQA", "medqa", "qwen3.5:122b",
     "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_plain.json",
     "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_elaborate.json"),
    ("Gemma4 31b / MedQA", "medqa", "gemma4:31b",
     "results/cue_smoke_medqa/medqa_baseline_gemma4:31b_plain.json",
     "results/cue_smoke_medqa/medqa_baseline_gemma4:31b_elaborate.json"),
]

HEDGE_RE = re.compile(
    r"\b(may|might|could|possibly|perhaps|likely|unlikely|uncertain|unclear|"
    r"appears?|seems?|suggests?|approximately|around|roughly|probably|"
    r"i'm not sure|not certain|it is possible|potentially)\b", re.I)
BULLET_RE = re.compile(r"^\s*([-*+]|\d+[.)])\s+", re.M)
HEADER_RE = re.compile(r"^\s*#{1,6}\s+", re.M)
BOLD_RE = re.compile(r"\*\*[^*\n]+\*\*")
SENT_RE = re.compile(r"[.!?]+(?:\s|$)")


def metrics(text):
    text = text or ""
    words = text.split()
    return {
        "chars": len(text),
        "words": len(words),
        "sentences": max(1, len(SENT_RE.findall(text))),
        "lines": text.count("\n") + 1,
        "bullets": len(BULLET_RE.findall(text)),
        "headers": len(HEADER_RE.findall(text)),
        "bold": len(BOLD_RE.findall(text)),
        "hedges": len(HEDGE_RE.findall(text)),
        "hedge_per_100w": 100.0 * len(HEDGE_RE.findall(text)) / max(1, len(words)),
    }


AXES = ["chars", "words", "sentences", "lines", "bullets", "headers", "bold",
        "hedges", "hedge_per_100w"]


def load(path):
    return json.load(open(ROOT / path))


def usable(item):
    return item.get("stop_reason") in (None, "None")


def wilcoxon_safe(p, e):
    diffs = np.array(e) - np.array(p)
    if np.count_nonzero(diffs) == 0:
        return float("nan"), 0
    try:
        stat, pval = stats.wilcoxon(p, e)
    except ValueError:
        return float("nan"), int(np.count_nonzero(diffs))
    return pval, int(np.count_nonzero(diffs))


def analyze():
    all_rows = []
    for label, dataset, model, pp, ep in CELLS:
        plain = {x["example_id"]: x for x in load(pp)}
        elab = {x["example_id"]: x for x in load(ep)}
        # keep only ids present + usable on BOTH sides
        ids = [i for i in plain if i in elab and usable(plain[i]) and usable(elab[i])]
        n = len(ids)
        pm = {a: [] for a in AXES}
        em = {a: [] for a in AXES}
        p_correct = e_correct = 0
        p_search = []
        e_search = []
        for i in ids:
            mp = metrics(plain[i]["sampler_response"])
            me = metrics(elab[i]["sampler_response"])
            for a in AXES:
                pm[a].append(mp[a])
                em[a].append(me[a])
            p_correct += int(plain[i]["sampler_correct"])
            e_correct += int(elab[i]["sampler_correct"])
            p_search.append(plain[i]["sampler_search_calls"])
            e_search.append(elab[i]["sampler_search_calls"])

        print(f"\n{'='*78}\n{label}   (n={n} paired, usable both sides)\n{'='*78}")
        print(f"  accuracy   PLAIN {p_correct/n:.1%}  ->  ELAB {e_correct/n:.1%}   "
              f"(Δ {(e_correct-p_correct)/n:+.1%})")
        print(f"  search     PLAIN med {np.median(p_search):.1f} mean {np.mean(p_search):.2f}"
              f"  ->  ELAB med {np.median(e_search):.1f} mean {np.mean(e_search):.2f}")
        print(f"  {'axis':<14}{'PLAIN med':>11}{'ELAB med':>11}{'Δmed':>9}"
              f"{'PLAIN mean':>12}{'ELAB mean':>11}{'ratio':>8}{'Wilcoxon p':>12}")
        for a in AXES:
            pv, nnz = wilcoxon_safe(pm[a], em[a])
            pmed, emed = np.median(pm[a]), np.median(em[a])
            pmean, emean = np.mean(pm[a]), np.mean(em[a])
            ratio = emean / pmean if pmean else float("nan")
            sig = "***" if pv < 0.001 else "**" if pv < 0.01 else "*" if pv < 0.05 else ""
            pstr = "n/a" if np.isnan(pv) else f"{pv:.4g}"
            print(f"  {a:<14}{pmed:>11.1f}{emed:>11.1f}{emed-pmed:>+9.1f}"
                  f"{pmean:>12.1f}{emean:>11.1f}{ratio:>8.2f}{pstr:>12} {sig}")
            all_rows.append({
                "cell": label, "dataset": dataset, "model": model, "axis": a, "n": n,
                "plain_median": pmed, "elab_median": emed,
                "plain_mean": pmean, "elab_mean": emean,
                "ratio": ratio, "wilcoxon_p": None if np.isnan(pv) else pv,
                "delta_median": emed - pmed,
            })
    return all_rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="optional path to write tidy CSV")
    args = ap.parse_args()
    rows = analyze()
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {len(rows)} rows -> {args.csv}")
