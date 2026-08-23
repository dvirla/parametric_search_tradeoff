"""
Backfills LLM-judge grading for MedQA's `multiturn`/`searchmulti`/`confident_parametric`
conditions, which were discovered (while switching Figure 1's MedQA panel to
LLM-judge grading, accuracy_revision.md Sec 1.0) to have `sampler_correct == None`
on 100% of rows, for ALL 11 models -- these three conditions were apparently never
graded at collection time (`--no_grader` was used), unlike the other 6 MedQA
conditions (plain/polite/natural/elaborate/direct/query), which are ~92% graded.

Scoped to the 6 models this session's parametric-uncertainty/entropy work already
centers on (confirmed with the user before launching): gemma4_31b, gpt-oss_20b,
gpt-oss_120b, nemotron-3-nano_30b, nemotron-cascade-2_30b, qwen3.5_122b. NOT the
full 11-model roster -- the other 5 models (gemini-3.1-pro-preview, gemini-3.5-flash,
gemma4_e4b, qwen3.5_35b, qwen3.5_4b) are left ungraded for these 3 conditions.

Reuses the EXACT grading templates/logic as scripts/regrade_no_search_llm.py
(src/services/qa_eval.py: MEDQA_GRADER_TEMPLATE, options-aware, gemini-3-flash-preview
via Google) so these grades are directly comparable to the existing `plain` LLM-judge
grades already present in the same files.

Writes to a resumable JSONL cache first (results/medqa_conversation_cue_llm_grades/),
NOT directly into the production results/medqa_grid/ files -- run
scripts/apply_medqa_conversation_cue_grades.py afterwards to merge the cache back
into the actual grid JSON files (a separate, explicit, reviewable step, since those
files are read directly by make_aggregate_cue_tradeoff_figure.py and everything else).

Usage:
    uv run python scripts/regrade_medqa_conversation_cues_llm.py --concurrency 20
    uv run python scripts/regrade_medqa_conversation_cues_llm.py --dry-run
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
from src.services.qa_eval import MEDQA_GRADER_TEMPLATE, _format_medqa_options  # noqa: E402

OUT_DIR = os.path.join(REPO, "results", "medqa_conversation_cue_llm_grades")
os.makedirs(OUT_DIR, exist_ok=True)

MODELS = ["gemma4_31b", "gpt-oss_20b", "gpt-oss_120b", "nemotron-3-nano_30b",
          "nemotron-cascade-2_30b", "qwen3.5_122b"]
CONDITIONS = ["multiturn", "searchmulti", "confident_parametric"]
GRADER_MODEL = os.environ.get("GRADER_MODEL", "gemini-3-flash-preview")
GRADER_PROVIDER = os.environ.get("GRADER_PROVIDER", "Google")
GRID_DIR = os.path.join(REPO, "results", "medqa_grid")


def load_medqa_options():
    path = os.path.join(REPO, "data", "medqa_500.jsonl")
    out = {}
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            out[row["example_id"]] = (row.get("options"), row.get("answer_idx"))
    return out


def find_file(model, cond):
    matches = glob.glob(os.path.join(GRID_DIR, model, f"*orig_{cond}.json"))
    return matches[0] if len(matches) == 1 else None


def cache_path(model, cond):
    return os.path.join(OUT_DIR, f"{model}_{cond}.jsonl")


def load_cache(model, cond):
    path = cache_path(model, cond)
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
    prompt = MEDQA_GRADER_TEMPLATE.format(question=question, options=options_block,
                                           correct_answer=correct_answer, response=response)
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


async def grade_file(model, cond, path, medqa_options, grader, sem, dry_run):
    rows = json.load(open(path))
    cache = load_cache(model, cond)
    pending = [r for r in rows if r["example_id"] not in cache and r.get("sampler_correct") is None]
    already_graded = [r for r in rows if r.get("sampler_correct") is not None]
    print(f"  {model}/{cond}: {len(rows)} total, {len(already_graded)} already graded, "
          f"{len(cache)} cached from this job, {len(pending)} to grade")
    if dry_run or not pending:
        return

    out_f = open(cache_path(model, cond), "a")

    async def _one(row):
        eid = row["example_id"]
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
    grader_raw = BaseAgent(provider_name=GRADER_PROVIDER, model_name=GRADER_MODEL, agent_name="medqa_conv_cue_regrader")
    grader = AgentAsSampler(grader_raw)
    sem = asyncio.Semaphore(args.concurrency)
    medqa_options = load_medqa_options()

    for model in MODELS:
        for cond in CONDITIONS:
            path = find_file(model, cond)
            if path is None:
                print(f"  ! {model}/{cond}: expected exactly 1 file, skipping")
                continue
            await grade_file(model, cond, path, medqa_options, grader, sem, args.dry_run)

    print("\nDone." if not args.dry_run else "\nDry run complete, no grading performed.")
    if not args.dry_run:
        print(f"\nNext step: uv run python scripts/apply_medqa_conversation_cue_grades.py "
              f"to merge {OUT_DIR}/ back into results/medqa_grid/.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    print(f"Grader: {GRADER_MODEL} via {GRADER_PROVIDER}, concurrency={args.concurrency}")
    print(f"Models: {MODELS}")
    print(f"Conditions: {CONDITIONS}")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
