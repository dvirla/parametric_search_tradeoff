"""
Batch LLM-judge regrade of all no-search rollouts (5 runs x 4 models x 2 datasets,
~20,015 rows) that were only ever regex-graded (--no_grader was used at collection
time). Reuses the EXACT grading templates/logic the production pipeline uses for
the search-enabled runs (src/services/qa_eval.py: STANDARD_GRADER_TEMPLATE for
FRAMES, MEDQA_GRADER_TEMPLATE + options-aware grading for MedQA, same grader model
default as scripts/run_medqa_grid_experiment.sh: gemini-3-flash-preview / Google)
so the new grades are directly comparable to the existing `plain` LLM-judge grades.

Motivation: regex grading was shown to undercount MedQA accuracy by 26-32pp
relative to the LLM judge on the `plain` condition (see accuracy_revision.md), and
spot-checking no-search "regex-wrong" rows found the same false-negative pattern.
This regrades the no-search side properly instead of inferring it indirectly.

Resumable: writes one JSONL cache file per (dataset, model, run) as it goes
(`results/no_search_llm_grades/<dataset>_<model>_run<n>.jsonl`, one line per graded
example_id), so a crash/interrupt loses at most the in-flight batch.

Usage:
    uv run python scripts/regrade_no_search_llm.py --concurrency 20
    uv run python scripts/regrade_no_search_llm.py --dataset medqa   # scope to one dataset
    uv run python scripts/regrade_no_search_llm.py --dry-run         # just print the plan
"""
import argparse
import asyncio
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from src.services.base_agent import BaseAgent  # noqa: E402
from src.services.agent_sampler import AgentAsSampler  # noqa: E402
from src.services.qa_eval import STANDARD_GRADER_TEMPLATE, MEDQA_GRADER_TEMPLATE, _format_medqa_options  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "no_search_llm_grades")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_120b", "gpt-oss_20b", "nemotron-3-nano_30b"]
GRADER_MODEL = os.environ.get("GRADER_MODEL", "gemini-3-flash-preview")
GRADER_PROVIDER = os.environ.get("GRADER_PROVIDER", "Google")

DATASETS = {
    "frames": dict(dir="results/frames_parametric", glob="frames-cues_no_search_*_run_{n}.json"),
    "medqa": dict(dir="results/medqa_parametric", glob="medqa-500_no_search_*_run_{n}.json"),
}


def load_medqa_options():
    path = os.path.join(REPO, "data", "medqa_500.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["example_id"]] = (row.get("options"), row.get("answer_idx"))
    return out


def cache_path(ds, model, n):
    return os.path.join(OUT_DIR, f"{ds}_{model}_run{n}.jsonl")


def load_cache(ds, model, n):
    path = cache_path(ds, model, n)
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["example_id"]] = row["correct"]
    return out


async def grade_one(grader, question, correct_answer, response, options_block, sem, retries=3):
    if options_block is not None:
        prompt = MEDQA_GRADER_TEMPLATE.format(question=question, options=options_block,
                                               correct_answer=correct_answer, response=response)
    else:
        prompt = STANDARD_GRADER_TEMPLATE.format(question=question, correct_answer=correct_answer, response=response)
    messages = [grader._pack_message(content=prompt, role="user")]
    async with sem:
        for attempt in range(retries):
            try:
                resp = await grader.acall(messages)
                text = resp.response_text.output
                m = re.search(r"correct:\s*(yes|no)", text, re.IGNORECASE)
                return m.group(1).lower() == "yes" if m else False
            except Exception as e:
                if attempt == retries - 1:
                    print(f"    ! grading failed after {retries} attempts: {e}")
                    return None
                await asyncio.sleep(2 * (attempt + 1))


async def grade_file(ds, model, n, path, medqa_options, grader, sem, dry_run):
    rows = json.load(open(path))
    cache = load_cache(ds, model, n)
    pending = [r for r in rows if r["example_id"] not in cache]
    print(f"  {ds}/{model}/run_{n}: {len(rows)} total, {len(cache)} cached, {len(pending)} to grade")
    if dry_run or not pending:
        return

    out_f = open(cache_path(ds, model, n), "a")

    async def _one(row):
        eid = row["example_id"]
        options_block = None
        if medqa_options is not None:
            opts, ans_idx = medqa_options.get(eid, (None, None))
            options_block = _format_medqa_options(opts, ans_idx)
        correct = await grade_one(grader, row.get("problem", ""), row.get("correct_answer", ""),
                                   row.get("sampler_response") or "", options_block, sem)
        if correct is not None:
            out_f.write(json.dumps({"example_id": eid, "correct": correct}) + "\n")
            out_f.flush()
        return correct

    BATCH = 200
    for i in range(0, len(pending), BATCH):
        batch = pending[i:i + BATCH]
        await asyncio.gather(*(_one(r) for r in batch))
        print(f"    ...{min(i + BATCH, len(pending))}/{len(pending)} graded")
    out_f.close()


async def main_async(args):
    grader_raw = BaseAgent(provider_name=GRADER_PROVIDER, model_name=GRADER_MODEL, agent_name="no_search_regrader")
    grader = AgentAsSampler(grader_raw)
    sem = asyncio.Semaphore(args.concurrency)
    medqa_options = load_medqa_options()

    datasets = [args.dataset] if args.dataset else list(DATASETS.keys())
    total_pending = 0
    for ds in datasets:
        cfg = DATASETS[ds]
        for model in MODELS:
            for n in range(1, 6):
                files = glob.glob(os.path.join(REPO, cfg["dir"], model, cfg["glob"].format(n=n)))
                if len(files) != 1:
                    print(f"  ! {ds}/{model}/run_{n}: expected 1 file, got {len(files)}")
                    continue
                await grade_file(ds, model, n, files[0], medqa_options if ds == "medqa" else None,
                                  grader, sem, args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run complete, no grading performed.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--dataset", choices=list(DATASETS), default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"Grader: {GRADER_MODEL} via {GRADER_PROVIDER}, concurrency={args.concurrency}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
