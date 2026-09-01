"""
Does semantic entropy (self-consistency over 5 no-search rollouts) validly
predict correctness -- the Stage 0 instrument-validity check for the whole
epistemic-alignment framework (docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md).

Uses the completed LLM-judge regrade (results/no_search_llm_grades/, see
scripts/regrade_no_search_llm.py) -- the TRUE no-search accuracy per example,
not the plain-condition proxy used before the regrade landed. Regex grading was
shown to undercount MedQA accuracy by 26-36pp; this is the number that should be
cited in the paper (accuracy_revision.md).

Usage:
    uv run python scripts/analyze_entropy_vs_correctness.py
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
GRADES_DIR = os.path.join(REPO, "results", "no_search_llm_grades")
OUT_DIR = os.path.join(REPO, "results", "entropy_vs_correctness")
os.makedirs(OUT_DIR, exist_ok=True)

from analyze_necessity_vs_template_search_5run import DATASETS, TAGS, load_one  # noqa: E402
from regrade_regex import heuristic_match, normalize  # noqa: E402
from src.services.common import normalize_response  # noqa: E402

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b", "nemotron-cascade-2_30b"]

# nemotron-cascade-2_30b was added to this analysis after the original LLM-judge regrade
# (scripts/regrade_no_search_llm.py, results/no_search_llm_grades/) had already run over just
# the original 4 models. It has since been extended to cover this model too (5,005 new
# gemini-3-flash-preview gradings), so every model now uses llm_judge -- this set is kept
# (empty) rather than removed so a future model added without its own regrade run falls back
# to free, local regex/EM grading instead of crashing, with the grading method marked
# explicitly per row so the two are never silently conflated: EM is known to undercount MedQA
# accuracy by 26-36pp vs. the LLM judge (accuracy_revision.md S1.1).
REGEX_FALLBACK_MODELS = set()


def load_llm_grades(ds, model, n):
    path = os.path.join(GRADES_DIR, f"{ds}_{model}_run{n}.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["example_id"]] = row["correct"]
    return out


def load_regex_grades(ds, model, tag, n):
    prefix = "frames-cues" if ds == "frames" else "medqa-500"
    pattern = os.path.join(REPO, "results", f"{ds}_parametric", model, f"{prefix}_no_search_{tag}_run_{n}.json")
    files = glob.glob(pattern)
    if len(files) != 1:
        return None
    data = json.load(open(files[0]))
    out = {}
    for row in data:
        gold = row.get("correct_answer") or ""
        response = normalize_response(row.get("sampler_response") or "")
        out[row["example_id"]] = heuristic_match(gold, response)
    return out


def main():
    rows = []
    for ds, cfg in DATASETS.items():
        for model in MODELS:
            entropy = load_one(os.path.join(cfg["entropy_dir"], model), cfg["entropy_glob"].format(tag=TAGS[model]))
            if model in REGEX_FALLBACK_MODELS:
                grading = "regex_em"
                run_correct = {n: load_regex_grades(ds, model, TAGS[model], n) for n in range(1, 6)}
            else:
                grading = "llm_judge"
                run_correct = {n: load_llm_grades(ds, model, n) for n in range(1, 6)}
            ids = sorted(set.intersection(*(set(d) for d in run_correct.values())) & set(entropy), key=str)
            ids = [e for e in ids if entropy[e] is not None]

            ent = np.array([entropy[e] for e in ids])
            frac_correct = np.array([np.mean([run_correct[n][e] for n in range(1, 6)]) for e in ids])
            rho, p = stats.spearmanr(ent, frac_correct)

            zero_mask = ent == 0.0
            rows.append(dict(
                dataset=ds, model=model, grading=grading, n=len(ids),
                rho=round(rho, 4), p=f"{p:.3g}",
                acc_at_entropy0=round(float(frac_correct[zero_mask].mean()), 4),
                n_entropy0=int(zero_mask.sum()),
                acc_at_entropy_gt0=round(float(frac_correct[~zero_mask].mean()), 4),
                n_entropy_gt0=int((~zero_mask).sum()),
            ))

    out_path = os.path.join(OUT_DIR, "entropy_vs_correctness.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}\n")

    for r in rows:
        print(f"{r['dataset']:7s} {r['model']:20s} [{r['grading']:9s}] rho={r['rho']:+.3f} (p={r['p']})  "
              f"acc@entropy=0: {r['acc_at_entropy0']:.3f} (n={r['n_entropy0']})  "
              f"acc@entropy>0: {r['acc_at_entropy_gt0']:.3f} (n={r['n_entropy_gt0']})")


if __name__ == "__main__":
    main()
