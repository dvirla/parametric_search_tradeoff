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
import math
import time
import argparse
import csv
import random
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


class AnswerClustering(BaseModel):
    cluster_ids: list[int] = Field(
        description=(
            "Cluster ID for each answer in the same order as provided. "
            "Answers that are semantically equivalent get the same integer ID. "
            "Use consecutive integers starting from 0."
        )
    )


def setup_args():
    parser = argparse.ArgumentParser(description="MuSiQue parametric uncertainty experiment.")
    parser.add_argument("--model_name", type=str, required=True, help="Ollama model name (e.g. gpt-oss:20b).")
    parser.add_argument("--provider", type=str, required=True, help="Ollama provider name (e.g. ollama).")
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


def load_examples_from_staleness(
    staleness_csv: str, num_examples: int, seed: int
) -> tuple[list[dict], dict[str, bool | None]]:
    """Load HF dataset filtered to non-stale examples, stratified by hop count (2/3/4).

    Samples num_examples total split as equally as possible across the three hop counts.
    """
    non_stale_ids, is_stale_map = load_staleness_csv(staleness_csv)

    print("Loading MuSiQue dataset from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue")
    rows = []
    for split_name in ["train", "validation"]:
        if split_name in ds:
            for row in ds[split_name]:
                rows.append(row)

    # Group non-stale answerable examples by hop count
    by_hops: dict[int, list[dict]] = {}
    for r in rows:
        if r.get("answerable") is True and r["id"] in non_stale_ids:
            n_hops = len(r.get("question_decomposition", []))
            by_hops.setdefault(n_hops, []).append(r)

    hop_counts = sorted(by_hops.keys())
    print(f"Non-stale examples by hop count: { {h: len(by_hops[h]) for h in hop_counts} }")

    # Stratified sample: split num_examples equally across available hop counts
    rng = random.Random(seed)
    n_groups = len(hop_counts)
    base = num_examples // n_groups
    remainder = num_examples % n_groups
    # Give extra samples to the largest groups first
    groups_by_size = sorted(hop_counts, key=lambda h: len(by_hops[h]), reverse=True)

    sampled: list[dict] = []
    for i, h in enumerate(hop_counts):
        quota = base + (1 if groups_by_size.index(h) < remainder else 0)
        pool = by_hops[h]
        take = min(quota, len(pool))
        sampled.extend(rng.sample(pool, take))
        print(f"  Hop {h}: sampled {take}/{len(pool)} (quota {quota})")

    rng.shuffle(sampled)
    print(f"Total sampled: {len(sampled)} non-stale examples across hops {hop_counts}.")
    return sampled, is_stale_map


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


_CLUSTER_PROMPT = """\
Question: {question}

The following {n} answers were given independently to this question:
{answers}

Group them by semantic equivalence. Two answers belong to the same cluster if they \
express the same fact (minor wording differences are fine; "United States" and "USA" are the same). \
Use consecutive integer cluster IDs starting from 0.\
"""


def build_clusterer() -> BaseAgent:
    """Build a semantic-clustering agent using the local gpt-oss:120b model."""
    print("Initializing clusterer agent (gpt-oss:120b via ollama)...")
    return BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=AnswerClustering,
        agent_name="musique_clusterer",
    )


def compute_semantic_entropy(
    clusterer: BaseAgent, question: str, runs: list[dict], max_retries: int = 3
) -> tuple[float, list[int]]:
    """Cluster the N final answers with an LLM and return (semantic_entropy, cluster_ids).

    Entropy is Shannon entropy (bits) over the cluster distribution.
    Falls back to treating every answer as its own cluster on failure.
    """
    answers = [r["final_answer"] for r in runs]
    n = len(answers)
    if n == 0:
        return 0.0, []

    answer_lines = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
    prompt = _CLUSTER_PROMPT.format(question=question, n=n, answers=answer_lines)

    for attempt in range(max_retries):
        try:
            response = clusterer.run(prompt)
            cluster_ids: list[int] = response.output.cluster_ids
            if len(cluster_ids) != n:
                raise ValueError(f"Expected {n} cluster IDs, got {len(cluster_ids)}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Clusterer error (attempt {attempt+1}): {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"    Clusterer failed; falling back to all-distinct clusters.")
                cluster_ids = list(range(n))

    counts = list(Counter(cluster_ids).values())
    entropy = -sum((c / n) * math.log2(c / n) for c in counts)
    return entropy, cluster_ids


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
        examples, is_stale_map = load_examples_from_staleness(
            args.staleness_csv, args.num_examples, args.seed
        )
    else:
        examples = load_musique_dataset(args.num_examples, args.seed)

    # Build agents
    print(f"Initializing parametric agent ({args.model_name} via {args.provider})...")
    parametric_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model_name,
        output_type=HopAnswer,
        agent_name=f"musique_parametric_{model_slug}",
    )

    print(f"Initializing search agent ({args.model_name} via {args.provider})...")
    search_service = BraveSearchService()
    search_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model_name,
        output_type=HopAnswer,
        tools=[Tool(search_service.search)],
        agent_name=f"musique_search_{model_slug}",
    )

    grader = build_grader()
    clusterer = build_clusterer()

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
            semantic_entropy, cluster_ids = compute_semantic_entropy(clusterer, resolved_q, runs)

            sub_questions_results.append({
                "hop_index": hop_idx,
                "question": resolved_q,
                "gold_answer": gold_a,
                "runs": runs,
                "num_correct": num_correct,
                "semantic_entropy": semantic_entropy,
                "cluster_ids": cluster_ids,
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
        avg_entropy = sum(r["semantic_entropy"] for r in sub_questions_results) / len(sub_questions_results) if sub_questions_results else 0.0
        print(
            f"  -> avg_hop_acc={avg_hop_acc:.2f}, avg_entropy={avg_entropy:.3f} bits, "
            f"agg_correct={agg_result['is_correct']}, search_calls={agg_result['search_calls']}"
        )

    print(f"\n--- Done. {len(results)} results saved to {output_path} ---")


if __name__ == "__main__":
    main()
