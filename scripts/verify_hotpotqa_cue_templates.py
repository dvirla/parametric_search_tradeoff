"""
Verify that every cue condition's prompt template was actually applied, and that NOTHING fell
through to the dataset's default template.

Why this check exists: `hotpotqa*` is in neither PLAIN_QUERY_DATASETS nor NATURAL_QUERY_DATASETS,
so with no `--query_template` the dataset routing falls through to the STRUCTURED template
(Explanation / Exact Answer / Confidence). That is NOT a plain passthrough -- an earlier
hotpotqa/bioasq launch mislabeled exactly this as "plain" and the runs had to be renamed
*_query.json. Every condition in the cue grid therefore passes --query_template EXPLICITLY, and
this script confirms it held, both by construction and in the emitted responses.

Two independent checks:

  1. CONSTRUCTION -- rebuild the query string for each condition exactly as
     EvaluationService._process_single does, and assert `plain` is a VERBATIM passthrough of the
     raw question (and differs from the structured default).

  2. EVIDENCE IN THE RESPONSES -- the structured template's distinctive signature is the
     CO-OCCURRENCE of "Exact Answer:" and "Confidence:", which no model emits spontaneously.
     (A lone "Explanation:" is NOT evidence -- models write that heading on their own; matching
     on it alone produces false positives.) Any non-`query` condition showing the pair means the
     structured default leaked. The `query` condition is the POSITIVE CONTROL: it SHOULD show the
     pair, and if it doesn't, --query_template isn't reaching the model at all.

Usage:
    uv run python scripts/verify_hotpotqa_cue_templates.py --results-root results/hotpotqa_cue_grid
    uv run python scripts/verify_hotpotqa_cue_templates.py --dataset hotpotqa-300 --construction-only
"""

import os
import re
import sys
import json
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

CONDITIONS = ["plain", "natural", "elaborate", "polite", "direct",
              "confident_parametric", "query", "multiturn", "searchmulti"]
# Conditions whose template is the plain passthrough (the cue lives in the conversation history).
HISTORY_CONDITIONS = {"multiturn", "searchmulti"}

STRUCT_EXACT = re.compile(r"Exact Answer:", re.I)
STRUCT_CONF = re.compile(r"Confidence:\s*\d", re.I)


def setup_args():
    p = argparse.ArgumentParser(description="Verify cue-template application for the HotpotQA grid.")
    p.add_argument("--results-root", default="results/hotpotqa_cue_grid")
    p.add_argument("--dataset", default="hotpotqa-300")
    p.add_argument("--construction-only", action="store_true",
                   help="Skip the response scan (useful before any run has produced output).")
    return p.parse_args()


def check_construction(dataset: str) -> int:
    from src.services.qa_eval import (
        EvaluationService, PLAIN_QUERY_TEMPLATE, NATURAL_QUERY_TEMPLATE, ELABORATE_QUERY_TEMPLATE,
        POLITE_QUERY_TEMPLATE, DIRECT_QUERY_TEMPLATE, CONFIDENT_PARAMETRIC_QUERY_TEMPLATE,
        QUERY_TEMPLATE, PLAIN_QUERY_DATASETS, NATURAL_QUERY_DATASETS)
    from src.services.entity_questions import ENTITY_QUESTIONS_QUERY_TEMPLATE, ENTITY_STYLE_DATASETS

    tmpl = {"plain": PLAIN_QUERY_TEMPLATE, "natural": NATURAL_QUERY_TEMPLATE,
            "elaborate": ELABORATE_QUERY_TEMPLATE, "polite": POLITE_QUERY_TEMPLATE,
            "direct": DIRECT_QUERY_TEMPLATE,
            "confident_parametric": CONFIDENT_PARAMETRIC_QUERY_TEMPLATE,
            "query": QUERY_TEMPLATE, "entity": ENTITY_QUESTIONS_QUERY_TEMPLATE}

    svc = object.__new__(EvaluationService)
    problem = svc._load_dataset(dataset, None)[0]["problem"]

    print("=" * 78)
    print("1. CONSTRUCTION")
    print("=" * 78)
    print(f"  dataset={dataset}")
    print(f"    in ENTITY_STYLE_DATASETS : {dataset in ENTITY_STYLE_DATASETS}")
    print(f"    in PLAIN_QUERY_DATASETS  : {dataset in PLAIN_QUERY_DATASETS}")
    print(f"    in NATURAL_QUERY_DATASETS: {dataset in NATURAL_QUERY_DATASETS}")
    print("    -> with NO --query_template this dataset falls through to the STRUCTURED default,")
    print("       which is why every condition must pass one explicitly.\n")
    print(f"  raw question: {problem!r}\n")

    failures = 0
    for cond in CONDITIONS:
        name = "plain" if cond in HISTORY_CONDITIONS else cond
        q = tmpl[name].format(Question=problem)
        verbatim = q == problem
        suffix = q[len(problem):].strip().replace("\n", " ")[:52]
        print(f"    {cond:22s} template={name:22s} verbatim={str(verbatim):5s} +{suffix!r}")
        if cond in HISTORY_CONDITIONS or cond == "plain":
            if not verbatim:
                print(f"      FAIL: {cond} must be a verbatim passthrough")
                failures += 1
        elif verbatim:
            print(f"      FAIL: {cond} added nothing to the question")
            failures += 1
    if tmpl["plain"].format(Question=problem) == QUERY_TEMPLATE.format(Question=problem):
        print("      FAIL: plain collapsed into the structured default")
        failures += 1
    print(f"\n  construction failures: {failures}")
    return failures


def condition_of(stem: str) -> str | None:
    for c in sorted(CONDITIONS, key=len, reverse=True):
        if stem.endswith("_" + c):
            return c
    return None


def check_responses(results_root: str, dataset: str) -> int:
    print("\n" + "=" * 78)
    print("2. EVIDENCE IN THE RESPONSES")
    print("=" * 78)
    print("  Structured-default signature = 'Exact Answer:' AND 'Confidence: <digit>' together.")
    print("  A lone 'Explanation:' is the model's own markdown and is NOT counted.\n")
    paths = sorted(glob.glob(os.path.join(results_root, "*", f"{dataset}_baseline_*_*.json")))
    if not paths:
        print(f"  (no result files under {results_root})")
        return 0
    failures = 0
    print(f"  {'model':22s} {'condition':22s} {'n':>5s} {'struct':>7s} {'medwords':>9s}")
    seen_query = False
    for p in paths:
        model = os.path.basename(os.path.dirname(p))
        cond = condition_of(os.path.basename(p)[:-len(".json")])
        if cond is None:
            print(f"  [skip] unparseable: {os.path.basename(p)}")
            continue
        rows = json.load(open(p))
        texts = [str(r.get("sampler_response", "")) for r in rows]
        struct = sum(1 for t in texts if STRUCT_EXACT.search(t) and STRUCT_CONF.search(t))
        lens = sorted(len(t.split()) for t in texts) or [0]
        print(f"  {model:22s} {cond:22s} {len(rows):5d} {struct:7d} {lens[len(lens)//2]:9d}")
        if cond == "query":
            seen_query = True
            if rows and struct == 0:
                print("      FAIL: 'query' produced NO structured output -- "
                      "--query_template is not reaching the model.")
                failures += 1
        elif struct > 0:
            print(f"      FAIL: structured default leaked into '{cond}' ({struct} responses)")
            failures += 1
    if not seen_query:
        print("\n  NOTE: no 'query' runs yet -- the positive control is still unverified.")
    print(f"\n  response failures: {failures}")
    return failures


def main():
    args = setup_args()
    failures = check_construction(args.dataset)
    if not args.construction_only:
        failures += check_responses(args.results_root, args.dataset)
    print("\n" + ("ALL TEMPLATE CHECKS PASSED" if failures == 0
                  else f"{failures} TEMPLATE CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
