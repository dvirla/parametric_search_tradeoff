"""
MuSiQue Parametric Uncertainty Experiment

For each MuSiQue example:
  - Runs the model N times (no search) per sub-question to measure parametric uncertainty
  - Runs the model once with Brave search on the aggregate question

Output JSON captures per-hop runs, agreement rates, and aggregate search results.
"""

import os
import sys
import json
import time
import argparse
import csv
from collections import Counter

import httpx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from pydantic import BaseModel, Field
from pydantic_ai import Tool
from pydantic_ai.messages import ModelResponse, ToolCallPart

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from src.services.brave_search import BraveSearchService

# Reuse helpers from run_musique_experiment
from scripts.run_musique_experiment import (
    load_musique_dataset,
    resolve_subquestion_text,
    grade_response,
    build_grader,
)


class HopAnswer(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to arrive at the answer.")
    final_answer: str = Field(description="Concise, direct answer to the question.")


def setup_args():
    parser = argparse.ArgumentParser(description="MuSiQue parametric uncertainty experiment.")
    parser.add_argument("--model_name", type=str, required=True, help="Ollama model name (e.g. gpt-oss:20b).")
    parser.add_argument("--num_runs", type=int, default=5, help="Runs per hop (default: 5).")
    parser.add_argument("--staleness_csv", type=str, default=None, help="CSV from classify_musique_staleness.py; restricts and annotates examples.")
    parser.add_argument("--num_examples", type=int, default=50, help="Examples to sample when no staleness CSV (default: 50).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--output_dir", type=str, default="results/musique_parametric", help="Output directory.")
    parser.add_argument("--resume", action="store_true", default=False, help="Skip already-completed example IDs.")
    return parser.parse_args()


def load_staleness_csv(csv_path: str) -> tuple[set[str], dict[str, bool | None]]:
    """Return (non_stale_id_set, is_stale_map) from a staleness CSV, excluding stale examples."""
    non_stale_ids = set()
    is_stale_map: dict[str, bool | None] = {}
    total = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            eid = row["example_id"]
            raw = row.get("is_stale", "")
            if raw == "True":
                is_stale_map[eid] = True
            elif raw == "False":
                is_stale_map[eid] = False
                non_stale_ids.add(eid)
            else:
                is_stale_map[eid] = None
    print(f"Staleness CSV: {total} total, {len(non_stale_ids)} non-stale (stale excluded).")
    return non_stale_ids, is_stale_map


def load_examples_from_staleness(staleness_csv: str) -> tuple[list[dict], dict[str, bool | None]]:
    """Load HF dataset filtered to non-stale examples from the staleness CSV."""
    non_stale_ids, is_stale_map = load_staleness_csv(staleness_csv)

    print("Loading MuSiQue dataset from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue")
    rows = []
    for split_name in ["train", "validation"]:
        if split_name in ds:
            for row in ds[split_name]:
                rows.append(row)

    examples = [
        r for r in rows
        if r.get("answerable") is True
        and r["id"] in non_stale_ids
    ]
    print(f"Matched {len(examples)} non-stale examples from HF dataset.")
    return examples, is_stale_map


def count_search_calls(response) -> int:
    """Count search tool calls in a pydantic-ai response."""
    count = 0
    for msg in response.all_messages():
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == "search":
                    count += 1
    return count


def run_single_hop(
    parametric_agent: BaseAgent,
    grader: AgentAsSampler,
    question: str,
    gold_answer: str,
    max_retries: int = 5,
) -> dict:
    """Run one parametric hop attempt. Returns {reasoning, final_answer, is_correct}."""
    for attempt in range(max_retries):
        try:
            response = parametric_agent.run(question)
            hop_answer: HopAnswer = response.output
            is_correct = grade_response(grader, question, gold_answer, hop_answer.final_answer)
            return {
                "reasoning": hop_answer.reasoning,
                "final_answer": hop_answer.final_answer,
                "is_correct": is_correct,
            }
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    Network error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} attempts.")
                return {"reasoning": "", "final_answer": f"ERROR: {e}", "is_correct": False}
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return {"reasoning": "", "final_answer": f"ERROR: {e}", "is_correct": False}


def run_aggregate(
    search_agent: BaseAgent,
    grader: AgentAsSampler,
    question: str,
    gold_answer: str,
    max_retries: int = 5,
) -> dict:
    """Run the aggregate question with search. Returns result dict including search_calls."""
    for attempt in range(max_retries):
        try:
            response = search_agent.run(question)
            hop_answer: HopAnswer = response.output
            search_calls = count_search_calls(response)
            is_correct = grade_response(grader, question, gold_answer, hop_answer.final_answer)
            return {
                "question": question,
                "gold_answer": gold_answer,
                "reasoning": hop_answer.reasoning,
                "final_answer": hop_answer.final_answer,
                "is_correct": is_correct,
                "search_calls": search_calls,
            }
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    Network error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} attempts.")
                return {
                    "question": question,
                    "gold_answer": gold_answer,
                    "reasoning": "",
                    "final_answer": f"ERROR: {e}",
                    "is_correct": False,
                    "search_calls": 0,
                }
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"    Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    Failed after {max_retries} attempts: {e}")
                return {
                    "question": question,
                    "gold_answer": gold_answer,
                    "reasoning": "",
                    "final_answer": f"ERROR: {e}",
                    "is_correct": False,
                    "search_calls": 0,
                }


def compute_agreement_rate(runs: list[dict]) -> float:
    """Fraction of runs sharing the plurality final_answer."""
    if not runs:
        return 0.0
    answers = [r["final_answer"] for r in runs]
    most_common_count = Counter(answers).most_common(1)[0][1]
    return most_common_count / len(answers)


def main():
    args = setup_args()

    model_slug = args.model_name.replace("/", "_").replace(":", "_")
    output_filename = f"musique_parametric_uncertainty_{model_slug}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)

    # Resume support
    existing_results = []
    completed_ids: set[str] = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            existing_results = json.load(f)
        completed_ids = {r["example_id"] for r in existing_results}
        print(f"Resuming: {len(completed_ids)} examples already completed.")

    # Load examples
    is_stale_map: dict[str, bool | None] = {}
    if args.staleness_csv:
        examples, is_stale_map = load_examples_from_staleness(args.staleness_csv)
    else:
        examples = load_musique_dataset(args.num_examples, args.seed)

    # Build agents
    print(f"Initializing parametric agent ({args.model_name} via ollama)...")
    parametric_agent = BaseAgent(
        provider_name="ollama",
        model_name=args.model_name,
        output_type=HopAnswer,
        agent_name=f"musique_parametric_{model_slug}",
    )

    print(f"Initializing search agent ({args.model_name} via ollama)...")
    search_service = BraveSearchService()
    search_agent = BaseAgent(
        provider_name="ollama",
        model_name=args.model_name,
        output_type=HopAnswer,
        tools=[Tool(search_service.search)],
        agent_name=f"musique_search_{model_slug}",
    )

    grader = build_grader()

    print(f"\n--- MuSiQue Parametric Uncertainty: {args.model_name}, {args.num_runs} runs/hop ---")
    print(f"Examples: {len(examples)}, Output: {output_path}\n")

    results = list(existing_results)

    for i, example in enumerate(examples):
        example_id = example["id"]
        if example_id in completed_ids:
            print(f"[{i+1}/{len(examples)}] Skipping {example_id} (already done)")
            continue

        print(f"[{i+1}/{len(examples)}] Processing {example_id}...")
        sub_questions = example["question_decomposition"]
        is_stale = is_stale_map.get(example_id)

        # Per-hop parametric runs
        sub_questions_results = []
        for hop_idx in range(len(sub_questions)):
            resolved_q = resolve_subquestion_text(sub_questions, hop_idx)
            gold_a = sub_questions[hop_idx]["answer"]
            print(f"  Hop {hop_idx}: {resolved_q[:80]}...")

            runs = []
            for run_i in range(args.num_runs):
                print(f"    Run {run_i+1}/{args.num_runs}...")
                run_result = run_single_hop(parametric_agent, grader, resolved_q, gold_a)
                runs.append(run_result)

            num_correct = sum(1 for r in runs if r["is_correct"])
            agreement_rate = compute_agreement_rate(runs)

            sub_questions_results.append({
                "hop_index": hop_idx,
                "question": resolved_q,
                "gold_answer": gold_a,
                "runs": runs,
                "num_correct": num_correct,
                "agreement_rate": agreement_rate,
            })

        # Aggregate run with search
        agg_question = example["question"]
        agg_answer = example["answer"]
        print(f"  Aggregate: {agg_question[:80]}...")
        agg_result = run_aggregate(search_agent, grader, agg_question, agg_answer)

        entry = {
            "example_id": example_id,
            "is_stale": is_stale,
            "aggregate_question": agg_question,
            "aggregate_answer": agg_answer,
            "sub_questions_results": sub_questions_results,
            "aggregate_result": agg_result,
        }
        results.append(entry)

        # Save after each example
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        hop_corrects = [r["num_correct"] / args.num_runs for r in sub_questions_results]
        avg_hop_acc = sum(hop_corrects) / len(hop_corrects) if hop_corrects else 0.0
        avg_agreement = sum(r["agreement_rate"] for r in sub_questions_results) / len(sub_questions_results) if sub_questions_results else 0.0
        print(
            f"  -> avg_hop_acc={avg_hop_acc:.2f}, avg_agreement={avg_agreement:.2f}, "
            f"agg_correct={agg_result['is_correct']}, search_calls={agg_result['search_calls']}"
        )

    print(f"\n--- Done. {len(results)} results saved to {output_path} ---")


if __name__ == "__main__":
    main()
