"""
MuSiQue Multi-Hop QA Experiment Runner

Runs models with/without search on 4-hop MuSiQue questions,
evaluating both individual sub-questions and the aggregate question.
"""

import os
import sys
import json
import re
import time
import argparse

import httpx

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset
from pydantic_ai import Tool

from src.services.base_agent import BaseAgent
from src.services.brave_search import BraveSearchService
from src.services.agent_sampler import AgentAsSampler
from src.services.qa_eval import QUERY_TEMPLATE, STANDARD_GRADER_TEMPLATE


def setup_args():
    parser = argparse.ArgumentParser(description="Run MuSiQue multi-hop QA experiment.")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. gemini-3-pro-preview).")
    parser.add_argument("--provider_name", type=str, default="Google", help="Provider: Google, OpenAI, Anthropic, ollama.")
    parser.add_argument("--mode", type=str, required=True, choices=["with_search", "no_search"], help="with_search or no_search.")
    parser.add_argument("--num_examples", type=int, default=15, help="Number of examples to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    parser.add_argument("--output_dir", type=str, default="results/musique", help="Output directory.")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume from existing results.")
    return parser.parse_args()


def load_musique_dataset(num_examples: int, seed: int) -> list[dict]:
    """Load MuSiQue from HuggingFace, filter answerable 4-hop questions, sample deterministically."""
    print("Loading MuSiQue dataset from HuggingFace...")
    ds = load_dataset("dgslibisey/MuSiQue")

    # Combine train and validation for a larger pool
    rows = []
    for split_name in ["train", "validation"]:
        if split_name in ds:
            for row in ds[split_name]:
                rows.append(row)

    # Filter: answerable=True and exactly 4 sub-questions
    filtered = [
        r for r in rows
        if r.get("answerable") is True
        and len(r.get("question_decomposition", [])) == 4
    ]
    print(f"Filtered to {len(filtered)} answerable 4-hop examples (from {len(rows)} total).")

    # Deterministic sample
    import random
    rng = random.Random(seed)
    sample_size = min(num_examples, len(filtered))
    sampled = rng.sample(filtered, sample_size)
    print(f"Sampled {len(sampled)} examples with seed={seed}.")
    return sampled


def resolve_subquestion_text(sub_questions: list[dict], hop_index: int) -> str:
    """Replace #N references in sub-question text with gold answers from prior hops.

    E.g. "Who is president of #1?" with prior answer "Indonesia" -> "Who is president of Indonesia?"
    """
    text = sub_questions[hop_index]["question"]
    for prev_idx in range(hop_index):
        placeholder = f"#{prev_idx + 1}"
        prev_answer = sub_questions[prev_idx]["answer"]
        text = text.replace(placeholder, prev_answer)
    return text


def build_agent(model_name: str, provider_name: str, mode: str, search_service: BraveSearchService | None) -> AgentAsSampler:
    """Build a baseline or no-search agent reusing project patterns."""
    if mode == "with_search":
        raw_agent = BaseAgent(
            provider_name=provider_name,
            model_name=model_name,
            tools=[Tool(search_service.search)],
            agent_name=f"musique_search_{model_name}",
        )
    else:
        raw_agent = BaseAgent(
            provider_name=provider_name,
            model_name=model_name,
            agent_name=f"musique_nosearch_{model_name}",
        )
    return AgentAsSampler(raw_agent)


def build_grader() -> AgentAsSampler:
    """Build a grader agent using local ollama model."""
    print("Initializing grader agent (gpt-oss:20b via ollama)...")
    raw = BaseAgent(provider_name="ollama", model_name="gpt-oss:20b", agent_name="musique_grader")
    return AgentAsSampler(raw)


def grade_response(grader: AgentAsSampler, question: str, gold_answer: str, response: str) -> bool:
    """Grade a response against the gold answer using STANDARD_GRADER_TEMPLATE."""
    grader_prompt = STANDARD_GRADER_TEMPLATE.format(
        question=question,
        correct_answer=gold_answer,
        response=response,
    )
    prompt_messages = [grader._pack_message(content=grader_prompt, role="user")]
    sampler_response = grader(prompt_messages)
    grading_text = sampler_response.response_text.output
    match = re.search(r"correct:\s*(yes|no)", grading_text, re.IGNORECASE)
    return match.group(1).lower() == "yes" if match else False


def run_single_question(agent: AgentAsSampler, grader: AgentAsSampler, question_text: str, gold_answer: str, max_retries: int = 5) -> dict:
    """Send a question through the agent, grade the response, return result dict."""
    query = QUERY_TEMPLATE.format(Question=question_text)
    prompt_messages = [agent._pack_message(content=query, role="user")]

    for attempt in range(max_retries):
        try:
            sampler_response = agent(prompt_messages)
            model_response = sampler_response.response_text.output
            search_calls = sampler_response.response_metadata.get("search_calls", 0)
            is_correct = grade_response(grader, question_text, gold_answer, str(model_response))
            return {
                "question": question_text,
                "gold_answer": gold_answer,
                "model_response": str(model_response),
                "is_correct": is_correct,
                "search_calls": search_calls,
            }
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Network error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts, returning incorrect.")
                return {
                    "question": question_text,
                    "gold_answer": gold_answer,
                    "model_response": f"ERROR: {e}",
                    "is_correct": False,
                    "search_calls": 0,
                }


def main():
    args = setup_args()

    model_slug = args.model_name.replace("/", "_").replace(":", "_")
    output_filename = f"musique_{model_slug}_{args.mode}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)

    # Load existing results for resume
    existing_results = []
    completed_ids = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path, "r") as f:
            existing_results = json.load(f)
        completed_ids = {r["example_id"] for r in existing_results}
        print(f"Resuming: {len(completed_ids)} examples already completed.")

    # Load dataset
    examples = load_musique_dataset(args.num_examples, args.seed)

    # Build agents
    search_service = BraveSearchService() if args.mode == "with_search" else None
    agent = build_agent(args.model_name, args.provider_name, args.mode, search_service)
    grader = build_grader()

    print(f"\n--- Running MuSiQue experiment: {args.model_name} / {args.mode} ---")
    print(f"Examples: {len(examples)}, Output: {output_path}\n")

    results = list(existing_results)

    for i, example in enumerate(examples):
        example_id = example["id"]
        if example_id in completed_ids:
            print(f"[{i+1}/{len(examples)}] Skipping {example_id} (already done)")
            continue

        print(f"[{i+1}/{len(examples)}] Processing {example_id}...")
        sub_questions = example["question_decomposition"]

        # Run each resolved sub-question
        sub_results = []
        for hop_idx in range(len(sub_questions)):
            resolved_q = resolve_subquestion_text(sub_questions, hop_idx)
            gold_a = sub_questions[hop_idx]["answer"]
            print(f"  Hop {hop_idx}: {resolved_q[:80]}...")
            result = run_single_question(agent, grader, resolved_q, gold_a)
            result["hop_index"] = hop_idx
            sub_results.append(result)

        # Run aggregate question
        agg_question = example["question"]
        agg_answer = example["answer"]
        print(f"  Aggregate: {agg_question[:80]}...")
        agg_result = run_single_question(agent, grader, agg_question, agg_answer)

        factoid_correct = [r["is_correct"] for r in sub_results]
        factoid_accuracy = sum(factoid_correct) / len(factoid_correct) if factoid_correct else 0.0
        total_search = sum(r["search_calls"] for r in sub_results) + agg_result["search_calls"]

        entry = {
            "example_id": example_id,
            "model_name": args.model_name,
            "mode": args.mode,
            "aggregate_question": agg_question,
            "aggregate_answer": agg_answer,
            "sub_questions_results": sub_results,
            "aggregate_result": agg_result,
            "factoid_accuracy": factoid_accuracy,
            "all_factoids_correct": all(factoid_correct),
            "aggregate_correct": agg_result["is_correct"],
            "total_search_calls": total_search,
        }
        results.append(entry)

        # Save after each example for resume support
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> factoid_acc={factoid_accuracy:.2f}, agg_correct={agg_result['is_correct']}, searches={total_search}")

    print(f"\n--- Done. {len(results)} results saved to {output_path} ---")


if __name__ == "__main__":
    main()
