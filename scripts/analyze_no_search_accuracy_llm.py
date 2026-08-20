"""
Final, LLM-graded version of analyze_no_search_accuracy.py, using the completed
regrade in results/no_search_llm_grades/ (see scripts/regrade_no_search_llm.py)
instead of regex grading. Regex was shown to undercount MedQA accuracy by
26-36pp relative to the LLM judge (see accuracy_revision.md) -- this script
produces the numbers that should be cited in the paper.

Reports, per (dataset, model):
  - mean per-run no-search accuracy, pass@5, majority@5
  - plain (search-enabled) accuracy from the EXISTING live LLM-judge grades
    already present on results/{frames_cues_full,medqa_grid}/*_plain.json
  - search_adds = plain - no_search (both LLM-graded, directly comparable)

Usage:
    uv run python scripts/analyze_no_search_accuracy_llm.py
"""
import csv
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRADES_DIR = os.path.join(REPO, "results", "no_search_llm_grades")
OUT_DIR = os.path.join(REPO, "results", "no_search_accuracy")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
MODEL_TAGS = {"gemma4_31b": "gemma4:31b", "gpt-oss_120b": "gpt-oss:120b",
              "gpt-oss_20b": "gpt-oss:20b", "nemotron-3-nano_30b": "nemotron-3-nano:30b"}
DATASETS = {
    "frames": "results/frames_cues_full/{model}/frames-cues_baseline_{tag}_verbose_plain.json",
    "medqa": "results/medqa_grid/{model}/medqa-500_baseline_{tag}_orig_plain.json",
}


def load_llm_grades(ds, model, n):
    path = os.path.join(GRADES_DIR, f"{ds}_{model}_run{n}.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["example_id"]] = row["correct"]
    return out


def main():
    rows = []
    for ds, plain_fmt in DATASETS.items():
        for model in MODELS:
            run_correct = {n: load_llm_grades(ds, model, n) for n in range(1, 6)}
            ids = sorted(set.intersection(*(set(d) for d in run_correct.values())), key=str)
            mat = np.array([[run_correct[n][e] for n in range(1, 6)] for e in ids])
            per_run_acc = mat.mean(axis=0)
            any_correct = mat.any(axis=1).mean()
            majority_correct = (mat.sum(axis=1) >= 3).mean()
            mean_per_run = float(per_run_acc.mean())

            plain_path = os.path.join(REPO, plain_fmt.format(model=model, tag=MODEL_TAGS[model]))
            plain = json.load(open(plain_path))
            plain_correct_map = {r["example_id"]: bool(r["sampler_correct"]) for r in plain}
            plain_ids = [e for e in ids if e in plain_correct_map]
            plain_acc = float(np.mean([plain_correct_map[e] for e in plain_ids]))

            rows.append(dict(
                dataset=ds, model=model, n=len(ids),
                per_run_acc_mean=round(mean_per_run, 4),
                per_run_acc_min=round(float(per_run_acc.min()), 4),
                per_run_acc_max=round(float(per_run_acc.max()), 4),
                pass_at_5=round(float(any_correct), 4),
                majority_at_5=round(float(majority_correct), 4),
                plain_search_acc=round(plain_acc, 4),
                search_adds=round(plain_acc - mean_per_run, 4),
                grading="llm_judge",
            ))

    out_path = os.path.join(OUT_DIR, "no_search_accuracy_llm.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}\n")

    print(f"{'dataset':7s} {'model':20s} {'per_run':>8s} {'pass@5':>7s} {'maj@5':>7s} {'plain_acc':>10s} {'search_adds':>12s}")
    for r in rows:
        print(f"{r['dataset']:7s} {r['model']:20s} {100*r['per_run_acc_mean']:7.1f}% "
              f"{100*r['pass_at_5']:6.1f}% {100*r['majority_at_5']:6.1f}% "
              f"{100*r['plain_search_acc']:9.1f}% {100*r['search_adds']:+11.1f}pp")


if __name__ == "__main__":
    main()
