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

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b",
          "nemotron-cascade-2_30b", "qwen3.5_122b"]

# nemotron-cascade-2_30b was added to this analysis after the original LLM-judge regrade
# (scripts/regrade_no_search_llm.py, results/no_search_llm_grades/) had already run over just
# the original 4 models. It has since been extended to cover this model too (5,005 new
# gemini-3-flash-preview gradings), so every model now uses llm_judge -- this set is kept
# (empty) rather than removed so a future model added without its own regrade run falls back
# to free, local regex/EM grading instead of crashing, with the grading method marked
# explicitly per row so the two are never silently conflated: EM is known to undercount MedQA
# accuracy by 26-36pp vs. the LLM judge (accuracy_revision.md S1.1).
# qwen3.5_122b has no entries in results/no_search_llm_grades/ (the regrade pass covered the
# other five models only), but its raw 5-run no_search rollouts are on disk for all three
# datasets, so it is graded by EM. Read it in the rho_em column, which every row now carries.
REGEX_FALLBACK_MODELS = {"qwen3.5_122b"}

# HotpotQA is added here as a LOCAL extension of the shared DATASETS rather than by editing
# analyze_necessity_vs_template_search_5run.py, whose own output would otherwise change.
# Its entropy_glob is plain-specific: HotpotQA has four cluster files per model (one per cue),
# so a bare wildcard would match a cue file instead of the cue-free baseline.
# TAGS is imported from analyze_necessity_vs_template_search_5run and stops at 5 models, so
# adding a 6th to MODELS without extending it raises KeyError mid-run -- which leaves the
# PREVIOUS csv in place and makes the failure look like a silently missing row.
TAGS = dict(TAGS)
TAGS.setdefault("qwen3.5_122b", "qwen3.5:122b")
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
            # ALSO compute the EM/regex verdict whenever the raw rollouts are on disk, so both
            # graders appear side by side. Necessary because the grader choice is NOT neutral for
            # this statistic: EM's undercount is correlated with entropy (high-entropy examples
            # give hedged/verbose answers EM misses disproportionately), so it ATTENUATES rho --
            # by ~0.15-0.20 on FRAMES and ~0.30-0.49 on MedQA, where the relationship nearly
            # vanishes. A single `grading` tag hid that; with rho_em present, an EM-only dataset
            # such as HotpotQA can be compared against the other datasets' EM column instead of
            # being read against their judge column.
            em_correct = {n: load_regex_grades(ds, model, TAGS[model], n) for n in range(1, 6)}
            has_em = all(v is not None for v in em_correct.values())
            ids = sorted(set.intersection(*(set(d) for d in run_correct.values())) & set(entropy), key=str)
            ids = [e for e in ids if entropy[e] is not None]

            ent = np.array([entropy[e] for e in ids])
            frac_correct = np.array([np.mean([run_correct[n][e] for n in range(1, 6)]) for e in ids])
            rho, p = stats.spearmanr(ent, frac_correct)
            if has_em:
                frac_em = np.array([np.mean([em_correct[n][e] for n in range(1, 6)]) for e in ids])
                rho_em, p_em = stats.spearmanr(ent, frac_em)
            else:
                frac_em, rho_em, p_em = None, float("nan"), float("nan")

            zero_mask = ent == 0.0
            rows.append(dict(
                dataset=ds, model=model, grading=grading, n=len(ids),
                rho=round(rho, 4), p=f"{p:.3g}",
                acc_at_entropy0=round(float(frac_correct[zero_mask].mean()), 4),
                n_entropy0=int(zero_mask.sum()),
                acc_at_entropy_gt0=round(float(frac_correct[~zero_mask].mean()), 4),
                n_entropy_gt0=int((~zero_mask).sum()),
                rho_em=round(rho_em, 4) if rho_em == rho_em else "",
                p_em=f"{p_em:.3g}" if p_em == p_em else "",
                acc_em_at_entropy0=round(float(frac_em[zero_mask].mean()), 4) if frac_em is not None else "",
                acc_em_at_entropy_gt0=round(float(frac_em[~zero_mask].mean()), 4) if frac_em is not None else "",
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
