"""
Which comes first on MedQA: models not searching because they're already
confident, or search genuinely not helping? Stage -1's aggregate "search adds
~0pp" number conflates two very different populations: the ~80-96% of examples
where the model never searches under `plain` (where plain=no-search trivially),
and the small minority where it does search. This conditions on that split.

Usage:
    uv run python scripts/analyze_medqa_search_conditional.py
"""
import csv
import json
import os

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRADES_DIR = os.path.join(REPO, "results", "no_search_llm_grades")
OUT_DIR = os.path.join(REPO, "results", "no_search_accuracy")

MODELS = [("gemma4_31b", "gemma4:31b"), ("gpt-oss_120b", "gpt-oss:120b"),
          ("gpt-oss_20b", "gpt-oss:20b"), ("nemotron-3-nano_30b", "nemotron-3-nano:30b")]


def load_llm_grades(model, n):
    out = {}
    with open(os.path.join(GRADES_DIR, f"medqa_{model}_run{n}.jsonl")) as f:
        for line in f:
            row = json.loads(line)
            out[row["example_id"]] = row["correct"]
    return out


def main():
    rows = []
    for model, tag in MODELS:
        plain_path = os.path.join(REPO, "results", "medqa_grid", model,
                                   f"medqa-500_baseline_{tag}_orig_plain.json")
        plain = json.load(open(plain_path))
        plain_map = {r["example_id"]: (r["sampler_search_calls"], bool(r["sampler_correct"])) for r in plain}
        run_correct = {n: load_llm_grades(model, n) for n in range(1, 6)}
        ids = sorted(set.intersection(*(set(d) for d in run_correct.values())) & set(plain_map), key=str)
        ns_acc = {e: np.mean([run_correct[n][e] for n in range(1, 6)]) for e in ids}

        searched = [e for e in ids if plain_map[e][0] and plain_map[e][0] > 0]
        not_searched = [e for e in ids if not plain_map[e][0]]

        for subset, label in [(searched, "searched"), (not_searched, "not_searched")]:
            if not subset:
                continue
            plain_acc = float(np.mean([plain_map[e][1] for e in subset]))
            nsacc = float(np.mean([ns_acc[e] for e in subset]))
            rows.append(dict(model=model, subset=label, n=len(subset),
                              pct_of_total=round(100 * len(subset) / len(ids), 1),
                              no_search_acc=round(nsacc, 4), plain_acc=round(plain_acc, 4),
                              delta=round(plain_acc - nsacc, 4)))

    out_path = os.path.join(OUT_DIR, "medqa_search_conditional.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}\n")
    for r in rows:
        print(f"{r['model']:20s} {r['subset']:13s} n={r['n']:4d} ({r['pct_of_total']:5.1f}%)  "
              f"no_search={r['no_search_acc']:.3f}  plain={r['plain_acc']:.3f}  delta={r['delta']:+.3f}")


if __name__ == "__main__":
    main()
