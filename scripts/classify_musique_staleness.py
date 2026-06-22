"""
Multi-hop Staleness Classifier (MuSiQue / FRAMES)

Uses an LLM judge to classify whether each question depends on time-sensitive
information (i.e., could have a different answer at a different point in time).

The judge is forced to DECOMPOSE every question into its atomic reasoning steps
and assess each step independently, so that a stale intermediate hop in an
otherwise innocuous-looking multi-hop question is not missed. A question is
STALE if the answer to ANY step (intermediate or final) could change over time —
even in principle and even if such a change would be rare ("any theoretical
change" calibration).

Examples of stale questions:
  - "Who is the minister for defence in X?" — current officeholder may change
  - "What is the population of X?" — changes over time
  - "What is the capital of the county where Fort Deposit is located?" —
    a county seat is an administrative designation that can be reassigned

Examples of non-stale questions:
  - "Where was X born?" — historical fact, won't change
  - "What year did X win the Nobel Prize?" — past event, fixed

Supports two input schemas in --result_files JSON (auto-detected per entry):
  - aggregate schema: {"aggregate_question", "aggregate_answer", "example_id", ...}
  - eval schema:      {"problem", "correct_answer", "example_id"(optional), ...}
FRAMES eval files have no example_id and no hop decomposition; a stable id is
derived from the question text and hop_count is left blank.
"""

import os
import sys
import json
import re
import time
import asyncio
import argparse
import csv
import hashlib
from collections import defaultdict

from pydantic import BaseModel, Field

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.base_agent import BaseAgent


STALENESS_JUDGE_SYSTEM_PROMPT = """You are an expert at determining whether answering a factual question requires any time-sensitive knowledge.

Many questions are MULTI-HOP: answering them requires chaining several intermediate facts together. You must reason about EVERY intermediate step, not just the surface phrasing of the question. A question is STALE if the answer to ANY step — intermediate OR final — could change over time, even in principle and even if such a change would be rare.

Procedure:
1. Decompose the question into its atomic reasoning steps (sub-questions). Make every implicit hop explicit, including hops hidden inside a relative clause (e.g. "the county where X was born", "the band that Y joined").
2. For each step, identify what kind of fact its answer is, and decide whether that answer could ever change over time.

Treat a step as TIME-SENSITIVE (could change) if its answer is any of:
- A current officeholder or title holder (president, minister, CEO, manager, champion, captain...)
- A current statistic or measurement (population, GDP, price, count, ranking, attendance...)
- A current record or superlative (tallest, largest, richest, fastest, most recent...)
- An ongoing or current relationship/affiliation (current spouse, current employer, current team, current member, current owner...)
- An administrative designation that an authority can reassign (county seat, capital city, official name, jurisdiction, electoral district, borders...)
- Anything qualified by "current", "present", "now", "today", "latest", "as of", or an open-ended present tense

Treat a step as FIXED (cannot change) only if its answer is a settled historical fact:
- A dated past event, birth, death, or biographical fact about a person
- Authorship, invention, discovery, or creation of a work or thing
- A past award, title, or achievement tied to a specific point in time
- A physical/geographic fact that is not administratively reassigned (a mountain's elevation, a river's source, where a historical event occurred)
- A scientific or mathematical constant

The overall question is STALE if ANY step is time-sensitive; otherwise NOT_STALE. When a step is genuinely ambiguous, err toward time-sensitive (STALE)."""


STALENESS_JUDGE_TEMPLATE = """Classify the following question as STALE or NOT_STALE by decomposing it into atomic reasoning steps and assessing each step.

Question: {question}

Decompose the question into every atomic reasoning step, decide for each step whether its answer could ever change over time, then set is_stale to true if ANY step is time-sensitive."""


class HopAssessment(BaseModel):
    """Per-hop staleness assessment of one atomic reasoning step."""
    step: str = Field(description="The atomic sub-question / reasoning step.")
    answer_type: str = Field(description="What kind of fact the answer to this step is.")
    is_time_sensitive: bool = Field(description="True if this step's answer could ever change over time.")
    reason: str = Field(description="One short clause explaining the time-sensitivity decision.")


class StalenessClassification(BaseModel):
    """Structured staleness verdict over a (possibly multi-hop) question."""
    hops: list[HopAssessment] = Field(description="The atomic reasoning steps and their per-step assessments.")
    is_stale: bool = Field(description="True if ANY hop is time-sensitive, otherwise False.")
    reason: str = Field(description="One sentence summarizing why the question is or is not stale.")


def build_judge(provider_name: str, model_name: str) -> BaseAgent:
    """Build the staleness judge agent with structured (per-hop) output."""
    print(f"Initializing staleness judge ({model_name} via {provider_name})...")
    return BaseAgent(
        provider_name=provider_name,
        model_name=model_name,
        agent_name="staleness_judge",
        system_prompt=STALENESS_JUDGE_SYSTEM_PROMPT,
        output_type=StalenessClassification,
        use_thinking=True,
    )


def classify_question(judge: BaseAgent, question: str, max_retries: int = 3) -> dict:
    """Ask the judge to decompose and classify a question as stale or not stale."""
    prompt = STALENESS_JUDGE_TEMPLATE.format(question=question)

    for attempt in range(max_retries):
        try:
            result: StalenessClassification = judge.run(prompt).output
            return {
                "is_stale": bool(result.is_stale),
                "classification": "STALE" if result.is_stale else "NOT_STALE",
                "reason": result.reason.strip(),
                "hops": [h.model_dump() for h in result.hops],
            }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {"is_stale": None, "classification": "ERROR", "reason": str(e), "hops": []}


async def classify_question_async(judge: BaseAgent, question: str, max_retries: int = 3) -> dict:
    """Async variant of classify_question for concurrent API calls."""
    prompt = STALENESS_JUDGE_TEMPLATE.format(question=question)

    for attempt in range(max_retries):
        try:
            response = await judge.arun(prompt)
            result: StalenessClassification = response.output
            return {
                "is_stale": bool(result.is_stale),
                "classification": "STALE" if result.is_stale else "NOT_STALE",
                "reason": result.reason.strip(),
                "hops": [h.model_dump() for h in result.hops],
            }
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {"is_stale": None, "classification": "ERROR", "reason": str(e), "hops": []}


async def classify_parallel(judge: BaseAgent, pending: list[dict], writer, f,
                            workers: int, total: int, processed: int, stale_count: int) -> tuple[int, int]:
    """Classify a list of questions concurrently, writing rows as each completes.

    Used for the --result_files path (no per-hop bucketing). Rows are written in
    completion order — fine since the CSV is keyed by example_id and resume reads
    by example_id. Returns (processed, stale_count)."""
    sem = asyncio.Semaphore(workers)

    async def work(q: dict) -> tuple[dict, dict]:
        async with sem:
            result = await classify_question_async(judge, q["aggregate_question"])
            return q, result

    tasks = [asyncio.create_task(work(q)) for q in pending]
    done = 0
    for fut in asyncio.as_completed(tasks):
        q, result = await fut
        done += 1
        processed += 1
        if result["is_stale"]:
            stale_count += 1
        row = {
            "example_id": q["example_id"],
            "hop_count": q.get("hop_count"),
            "aggregate_question": q["aggregate_question"],
            "aggregate_answer": q["aggregate_answer"],
            "classification": result["classification"],
            "is_stale": result["is_stale"],
            "reason": result["reason"],
            "hops": json.dumps(result.get("hops", [])),
        }
        writer.writerow(row)
        f.flush()
        print(f"[{done}/{len(pending)}] {q['example_id']} -> {result['classification']}: {result['reason'][:70]}")
    return processed, stale_count


def _hop_count_from_example_id(example_id) -> int | None:
    """Parse leading hop count from a MuSiQue example id like '4hop1__...' or '2hop__...'."""
    if not example_id:
        return None
    m = re.match(r"^(\d+)hop", str(example_id))
    return int(m.group(1)) if m else None


def _entry_question_answer(entry: dict) -> tuple[str, str]:
    """Extract (question, answer) from either the aggregate or the eval result schema."""
    question = entry.get("aggregate_question") or entry.get("problem") or ""
    answer = entry.get("aggregate_answer")
    if answer is None:
        answer = entry.get("correct_answer", "")
    return question, (answer or "")


def load_questions_from_results(result_paths: list[str]) -> list[dict]:
    """Load questions from existing result JSON files (deduplicated by example_id/question).

    Auto-detects the aggregate schema (``aggregate_question``/``aggregate_answer``)
    and the eval schema (``problem``/``correct_answer``). FRAMES eval entries have
    no ``example_id``, so a stable id is derived from the question text.
    """
    seen_ids = set()
    questions = []
    for path in result_paths:
        with open(path, "r") as f:
            results = json.load(f)
        for entry in results:
            question, answer = _entry_question_answer(entry)
            if not question:
                continue
            example_id = entry.get("example_id") or "q_" + hashlib.md5(question.encode("utf-8")).hexdigest()[:12]
            if example_id in seen_ids:
                continue
            seen_ids.add(example_id)
            questions.append({
                "example_id": example_id,
                "aggregate_question": question,
                "aggregate_answer": answer,
                "hop_count": _hop_count_from_example_id(entry.get("example_id")),
                "sub_questions": [
                    sq.get("question", "") for sq in entry.get("sub_questions_results", [])
                ],
            })
    print(f"Loaded {len(questions)} unique questions from {len(result_paths)} result file(s).")
    return questions


def load_questions_from_dataset(seed: int) -> list[dict]:
    """Load questions directly from HuggingFace MuSiQue dataset (validation split only, shuffled)."""
    from datasets import load_dataset
    import random

    print("Loading MuSiQue dataset from HuggingFace (validation split only)...")
    ds = load_dataset("dgslibisey/MuSiQue")

    rows = list(ds["validation"]) if "validation" in ds else []

    filtered = [r for r in rows if r.get("answerable") is True]
    print(f"Filtered to {len(filtered)} answerable examples from validation split.")

    rng = random.Random(seed)
    rng.shuffle(filtered)

    questions = []
    for row in filtered:
        sub_qs = row.get("question_decomposition", [])
        questions.append({
            "example_id": row["id"],
            "aggregate_question": row["question"],
            "aggregate_answer": row.get("answer", ""),
            "sub_questions": [sq.get("question", "") for sq in sub_qs],
            "hop_count": len(sub_qs),
        })
    print(f"Loaded {len(questions)} examples (shuffled with seed={seed}).")
    return questions


def setup_args():
    parser = argparse.ArgumentParser(description="Classify MuSiQue questions as stale or not stale.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--result_files",
        nargs="+",
        help="Path(s) to musique result JSON files to classify questions from.",
    )
    source.add_argument(
        "--from_dataset",
        action="store_true",
        help="Load questions directly from HuggingFace MuSiQue dataset.",
    )
    parser.add_argument("--per_hop_limit", type=int, default=200, help="Target number of non-stale questions per hop-count bucket (only with --from_dataset).")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (only with --from_dataset).")
    parser.add_argument("--output", type=str, default="results/musique_staleness.csv", help="Output CSV path.")
    parser.add_argument("--provider", type=str, default="Google", help="Judge provider (Google, OpenAI, Anthropic, ollama).")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview", help="Judge model name.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API calls (only with --result_files).")
    parser.add_argument("--resume", action="store_true", help="Skip already-classified examples.")
    return parser.parse_args()


def main():
    args = setup_args()

    # Load questions
    if args.result_files:
        questions = load_questions_from_results(args.result_files)
        per_hop_limit = None  # no limit when loading from result files
    else:
        questions = load_questions_from_dataset(args.seed)
        per_hop_limit = args.per_hop_limit

    # Load existing results for resume
    completed_ids = set()
    existing_rows = []
    hop_count_done: dict[int, int] = defaultdict(int)
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                completed_ids.add(row["example_id"])
                if per_hop_limit is not None and row.get("hop_count") and row.get("is_stale") == "False":
                    hop_count_done[int(row["hop_count"])] += 1
        print(f"Resuming: {len(completed_ids)} examples already classified.")
        if per_hop_limit is not None:
            for hc, cnt in sorted(hop_count_done.items()):
                print(f"  hop_count={hc}: {cnt}/{per_hop_limit}")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    judge = build_judge(args.provider, args.model)

    fieldnames = ["example_id", "hop_count", "aggregate_question", "aggregate_answer", "classification", "is_stale", "reason", "hops"]

    # Open in append mode if resuming, else write mode
    write_mode = "a" if args.resume and existing_rows else "w"
    with open(args.output, write_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_mode == "w":
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)

        total = len(questions)
        stale_count = sum(1 for r in existing_rows if r.get("is_stale") == "True")
        processed = len(completed_ids)

        # Parallel path: result_files mode (no per-hop bucketing) with workers > 1.
        if per_hop_limit is None and args.workers > 1:
            pending = [q for q in questions if q["example_id"] not in completed_ids]
            print(f"Classifying {len(pending)} pending questions with {args.workers} concurrent workers...")
            if pending:
                processed, stale_count = asyncio.run(
                    classify_parallel(judge, pending, writer, f, args.workers, total, processed, stale_count)
                )
            _report(processed, stale_count, per_hop_limit, hop_count_done, args.output)
            return

        for i, q in enumerate(questions):
            example_id = q["example_id"]
            hop_count = q.get("hop_count", len(q.get("sub_questions", [])))

            if example_id in completed_ids:
                print(f"[{i+1}/{total}] Skipping {example_id} (already classified)")
                continue

            if per_hop_limit is not None and hop_count_done[hop_count] >= per_hop_limit:
                continue  # non-stale bucket full, skip silently

            print(f"[{i+1}/{total}] Classifying {example_id} (hop_count={hop_count}, non-stale={hop_count_done[hop_count]}/{per_hop_limit})...")
            print(f"  Q: {q['aggregate_question'][:100]}...")

            result = classify_question(judge, q["aggregate_question"])
            processed += 1
            if result["is_stale"]:
                stale_count += 1
            if per_hop_limit is not None and result["is_stale"] is False:
                hop_count_done[hop_count] += 1

            row = {
                "example_id": example_id,
                "hop_count": hop_count,
                "aggregate_question": q["aggregate_question"],
                "aggregate_answer": q["aggregate_answer"],
                "classification": result["classification"],
                "is_stale": result["is_stale"],
                "reason": result["reason"],
                "hops": json.dumps(result.get("hops", [])),
            }
            writer.writerow(row)
            f.flush()

            print(f"  -> {result['classification']}: {result['reason'][:80]}")

            # Early exit if all seen hop-count buckets are full
            if per_hop_limit is not None and all(v >= per_hop_limit for v in hop_count_done.values()):
                # Peek ahead: check if any remaining questions have an unseen or unfull hop-count
                remaining_hop_counts = {q2.get("hop_count", len(q2.get("sub_questions", []))) for q2 in questions[i+1:]}
                if not remaining_hop_counts - hop_count_done.keys() and all(
                    hop_count_done[hc] >= per_hop_limit for hc in remaining_hop_counts
                ):
                    print("\nAll hop-count buckets full. Stopping early.")
                    break

    _report(processed, stale_count, per_hop_limit, hop_count_done, args.output)


def _report(processed, stale_count, per_hop_limit, hop_count_done, output):
    if processed > 0:
        print(f"\n--- Done. {processed} questions classified, {stale_count} stale ({stale_count/processed*100:.1f}%) ---")
    else:
        print("\n--- Done. No new questions classified. ---")
    if per_hop_limit is not None:
        for hc, cnt in sorted(hop_count_done.items()):
            print(f"  hop_count={hc}: {cnt}")
    print(f"Results saved to {output}")


if __name__ == "__main__":
    main()
