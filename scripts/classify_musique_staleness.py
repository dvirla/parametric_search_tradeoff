"""
MuSiQue Staleness Classifier

Uses an LLM judge (gpt-oss:20b via ollama) to classify whether each MuSiQue
question depends on time-sensitive information (i.e., could have a different
answer at a different point in time).

Examples of stale questions:
  - "Who is the minister for defence in X?" — current officeholder may change
  - "What is the population of X?" — changes over time
  - "Who is the current CEO of X?" — current role holder may change

Examples of non-stale questions:
  - "Where was X born?" — historical fact, won't change
  - "What year did X win the Nobel Prize?" — past event, fixed
"""

import os
import sys
import json
import re
import time
import argparse
import csv

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler


STALENESS_JUDGE_SYSTEM_PROMPT = """You are an expert at identifying whether a factual question requires time-sensitive knowledge.

A question is STALE if its correct answer could plausibly change over time. This includes:
- Current officeholders (presidents, ministers, CEOs, etc.)
- Current statistics (population, GDP, rankings, prices)
- Current records (tallest building, richest person, etc.)
- Ongoing relationships (current spouse, current employer, current team)
- Any fact described as "current", "present", "now", "today", or "latest"

A question is NOT STALE if its correct answer is a fixed historical fact:
- Historical events and their dates
- Births, deaths, and biographical facts about historical figures
- Geographic facts (where a place is located, what country something is in)
- Authorship, invention, or creation of things
- Past achievements and awards
- Scientific or mathematical constants

For multi-hop questions, mark as STALE if ANY hop requires time-sensitive knowledge."""


STALENESS_JUDGE_TEMPLATE = """Classify the following question as STALE or NOT_STALE.

Question: {question}

Think through whether any part of answering this question requires knowing time-sensitive information (facts that could change over time, like current officeholders, current statistics, or ongoing relationships).

Respond in exactly this format:
Classification: STALE or NOT_STALE
Reason: <one sentence explaining why>"""


def build_judge() -> AgentAsSampler:
    """Build the staleness judge agent using gpt-oss:20b via ollama."""
    print("Initializing staleness judge (gpt-oss:20b via ollama)...")
    raw = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:20b",
        agent_name="musique_staleness_judge",
        system_prompt=STALENESS_JUDGE_SYSTEM_PROMPT,
        use_thinking=False,
    )
    return AgentAsSampler(raw)


def classify_question(judge: AgentAsSampler, question: str, max_retries: int = 3) -> dict:
    """Ask the judge to classify a question as stale or not stale."""
    prompt = STALENESS_JUDGE_TEMPLATE.format(question=question)
    messages = [judge._pack_message(content=prompt, role="user")]

    for attempt in range(max_retries):
        try:
            response = judge(messages)
            text = response.response_text.output

            classification_match = re.search(r"Classification:\s*(STALE|NOT_STALE)", text, re.IGNORECASE)
            reason_match = re.search(r"Reason:\s*(.+)", text, re.IGNORECASE)

            if classification_match:
                is_stale = classification_match.group(1).upper() == "STALE"
                reason = reason_match.group(1).strip() if reason_match else ""
                return {
                    "is_stale": is_stale,
                    "classification": "STALE" if is_stale else "NOT_STALE",
                    "reason": reason,
                    "raw_response": text,
                }
            else:
                # Fallback: look for stale/not_stale anywhere in the text
                if re.search(r"\bNOT[_\s]STALE\b", text, re.IGNORECASE):
                    return {"is_stale": False, "classification": "NOT_STALE", "reason": "", "raw_response": text}
                elif re.search(r"\bSTALE\b", text, re.IGNORECASE):
                    return {"is_stale": True, "classification": "STALE", "reason": "", "raw_response": text}
                else:
                    print(f"  Warning: could not parse classification from response: {text[:100]}")
                    return {"is_stale": None, "classification": "UNKNOWN", "reason": "", "raw_response": text}

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return {"is_stale": None, "classification": "ERROR", "reason": str(e), "raw_response": ""}


def load_questions_from_results(result_paths: list[str]) -> list[dict]:
    """Load questions from existing musique result JSON files (deduplicated by example_id)."""
    seen_ids = set()
    questions = []
    for path in result_paths:
        with open(path, "r") as f:
            results = json.load(f)
        for entry in results:
            example_id = entry["example_id"]
            if example_id in seen_ids:
                continue
            seen_ids.add(example_id)
            questions.append({
                "example_id": example_id,
                "aggregate_question": entry["aggregate_question"],
                "aggregate_answer": entry.get("aggregate_answer", ""),
                "sub_questions": [
                    sq.get("question", "") for sq in entry.get("sub_questions_results", [])
                ],
            })
    print(f"Loaded {len(questions)} unique questions from {len(result_paths)} result file(s).")
    return questions


def load_questions_from_dataset(num_examples: int, seed: int) -> list[dict]:
    """Load questions directly from HuggingFace MuSiQue dataset."""
    from datasets import load_dataset
    import random

    print("Loading MuSiQue dataset from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue")

    rows = []
    for split_name in ["train", "validation"]:
        if split_name in ds:
            rows.extend(ds[split_name])

    filtered = [
        r for r in rows
        if r.get("answerable") is True
    ]
    print(f"Filtered to {len(filtered)} answerable 4-hop examples.")

    rng = random.Random(seed)
    sample_size = min(num_examples, len(filtered))
    sampled = rng.sample(filtered, sample_size)

    questions = []
    for row in sampled:
        sub_qs = row.get("question_decomposition", [])
        questions.append({
            "example_id": row["id"],
            "aggregate_question": row["question"],
            "aggregate_answer": row.get("answer", ""),
            "sub_questions": [sq.get("question", "") for sq in sub_qs],
        })
    print(f"Sampled {len(questions)} examples.")
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
    parser.add_argument("--num_examples", type=int, default=100, help="Number of examples (only with --from_dataset).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (only with --from_dataset).")
    parser.add_argument("--output", type=str, default="results/musique_staleness.csv", help="Output CSV path.")
    parser.add_argument("--resume", action="store_true", help="Skip already-classified examples.")
    return parser.parse_args()


def main():
    args = setup_args()

    # Load questions
    if args.result_files:
        questions = load_questions_from_results(args.result_files)
    else:
        questions = load_questions_from_dataset(args.num_examples, args.seed)

    # Load existing results for resume
    completed_ids = set()
    existing_rows = []
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_rows.append(row)
                completed_ids.add(row["example_id"])
        print(f"Resuming: {len(completed_ids)} examples already classified.")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)

    judge = build_judge()

    fieldnames = ["example_id", "aggregate_question", "aggregate_answer", "classification", "is_stale", "reason"]

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

        for i, q in enumerate(questions):
            example_id = q["example_id"]
            if example_id in completed_ids:
                print(f"[{i+1}/{total}] Skipping {example_id} (already classified)")
                continue

            print(f"[{i+1}/{total}] Classifying {example_id}...")
            print(f"  Q: {q['aggregate_question'][:100]}...")

            result = classify_question(judge, q["aggregate_question"])
            processed += 1
            if result["is_stale"]:
                stale_count += 1

            row = {
                "example_id": example_id,
                "aggregate_question": q["aggregate_question"],
                "aggregate_answer": q["aggregate_answer"],
                "classification": result["classification"],
                "is_stale": result["is_stale"],
                "reason": result["reason"],
            }
            writer.writerow(row)
            f.flush()

            print(f"  -> {result['classification']}: {result['reason'][:80]}")

    print(f"\n--- Done. {processed} questions classified, {stale_count} stale ({stale_count/processed*100:.1f}%) ---")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
