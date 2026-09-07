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

# HotpotQA is added here as a LOCAL extension of the shared DATASETS rather than by editing
# analyze_necessity_vs_template_search_5run.py, whose own output would otherwise change.
# Its entropy_glob is plain-specific: HotpotQA has four cluster files per model (one per cue),
# so a bare wildcard would match a cue file instead of the cue-free baseline.
DATASETS = dict(DATASETS)
DATASETS["hotpotqa"] = dict(
    entropy_dir="results/hotpotqa_parametric",
    entropy_glob="hotpotqa-300_no_search_{tag}_plain_llm_clusters_5run.json",
)
# EVERY HotpotQA row was collected with --no_grader, so no LLM-judge grades exist for it at all
# (results/no_search_llm_grades/ covers frames+medqa only). This dataset is therefore regex/EM
# throughout. NOT like-for-like with the frames/medqa rows: EM undercounts accuracy relative to
# the judge (7-11pp on FRAMES, 26-36pp on MedQA), so compare HotpotQA's rho -- a rank statistic,
# far less sensitive to a uniform level shift -- rather than its acc@ levels.
REGEX_ONLY_DATASETS = {"hotpotqa"}
# Per-dataset file naming for the raw no_search rollouts.
_PREFIX = {"frames": "frames-cues", "medqa": "medqa-500", "hotpotqa": "hotpotqa-300"}
# HotpotQA's driver names every run "<cond>_run_<r>", so its plain rollouts carry a _plain infix
# that frames/medqa do not have.
_PLAIN_INFIX = {"hotpotqa": "_plain"}


def load_llm_grades(ds, model, n):
    path = os.path.join(GRADES_DIR, f"{ds}_{model}_run{n}.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["example_id"]] = row["correct"]
    return out


def load_regex_grades(ds, model, tag, n):
    prefix = _PREFIX[ds]
    infix = _PLAIN_INFIX.get(ds, "")
    pattern = os.path.join(REPO, "results", f"{ds}_parametric", model,
                           f"{prefix}_no_search_{tag}{infix}_run_{n}.json")
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
            if ds in REGEX_ONLY_DATASETS or model in REGEX_FALLBACK_MODELS:
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
