"""
MuSiQue Aggregate No-Search Parametric Evaluation

Reads existing musique_parametric_uncertainty_<model>.json files and runs
the model N times (without search) on the aggregate question for each example.
This gives the actual parametric accuracy on the multi-hop aggregate question,
replacing the independence-composition proxy used in the interplay analysis.

Output: results/musique_parametric/musique_aggregate_nosearch_<model>.json
"""

import os
import sys
import json
import re
import time
import argparse

import httpx

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from pydantic import BaseModel, Field

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from src.services.qa_eval import STANDARD_GRADER_TEMPLATE, QUERY_TEMPLATE


class HopAnswer(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning to arrive at the answer.")
    final_answer: str = Field(description="Concise, direct answer to the question.")


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run aggregate no-search evaluation on existing MuSiQue parametric files."
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g. gemini-3-pro-preview).")
    parser.add_argument("--provider", type=str, required=True, help="Provider name (e.g. Google, OpenAI, ollama).")
    parser.add_argument("--input_file", type=str, default=None,
                        help="Path to musique_parametric_uncertainty_<model>.json. "
                             "Defaults to results/musique_parametric/musique_parametric_uncertainty_<model_slug>.json")
    parser.add_argument("--num_runs", type=int, default=5, help="Runs per aggregate question (default: 5).")
    parser.add_argument("--output_dir", type=str, default="results/musique_parametric",
                        help="Output directory (default: results/musique_parametric).")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip already-completed example IDs.")
    parser.add_argument("--max_retries", type=int, default=5, help="Max retries per LLM call (default: 5).")
    return parser.parse_args()


def build_grader() -> AgentAsSampler:
    print("Initializing grader agent (gpt-oss:120b via ollama)...")
    raw = BaseAgent(provider_name="ollama", model_name="gpt-oss:120b", agent_name="musique_agg_grader")
    return AgentAsSampler(raw)


def grade_response(grader: AgentAsSampler, question: str, gold_answer: str, response: str) -> bool:
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


def run_aggregate_nosearch(
    agent: BaseAgent,
    grader: AgentAsSampler,
    question: str,
    gold_answer: str,
    max_retries: int = 5,
) -> dict:
    """Run the aggregate question WITHOUT search. Returns result dict."""
    for attempt in range(max_retries):
        try:
            response = agent.run(question)
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


def main():
    args = setup_args()

    model_slug = args.model_name.replace("/", "_").replace(":", "_")

    # Resolve input file
    if args.input_file:
        input_path = args.input_file
    else:
        input_path = os.path.join(
            args.output_dir, f"musique_parametric_uncertainty_{model_slug}.json"
        )

    if not os.path.exists(input_path):
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    print(f"Loading parametric uncertainty file: {input_path}")
    with open(input_path) as f:
        source_data = json.load(f)
    print(f"  Loaded {len(source_data)} examples.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"musique_aggregate_nosearch_{model_slug}.json")

    # Resume support
    existing_results: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            existing_results = json.load(f)
        completed_ids = {r["example_id"] for r in existing_results}
        print(f"Resuming: {len(completed_ids)} examples already completed.")

    # Build agents
    print(f"Initializing parametric (no-search) agent: {args.model_name} via {args.provider}...")
    agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model_name,
        output_type=HopAnswer,
        agent_name=f"musique_agg_nosearch_{model_slug}",
    )
    grader = build_grader()

    print(f"\n--- MuSiQue Aggregate No-Search: {args.model_name}, {args.num_runs} runs ---")
    print(f"Examples: {len(source_data)}, Output: {output_path}\n")

    results = list(existing_results)

    for i, entry in enumerate(source_data):
        example_id = entry["example_id"]
        if example_id in completed_ids:
            print(f"[{i+1}/{len(source_data)}] Skipping {example_id} (already done)")
            continue

        agg_question = entry["aggregate_question"]
        agg_gold = entry["aggregate_answer"]
        print(f"[{i+1}/{len(source_data)}] {example_id}: {agg_question[:80]}...")

        runs = []
        for run_i in range(args.num_runs):
            print(f"  Run {run_i+1}/{args.num_runs}...")
            run_result = run_aggregate_nosearch(agent, grader, agg_question, agg_gold, args.max_retries)
            runs.append(run_result)
            print(f"    {'✓' if run_result['is_correct'] else '✗'} {run_result['final_answer'][:60]}")

        num_correct = sum(1 for r in runs if r["is_correct"])
        accuracy = num_correct / len(runs)
        print(f"  Accuracy: {num_correct}/{len(runs)} = {accuracy:.2f}")

        result_entry = {
            "example_id": example_id,
            "aggregate_question": agg_question,
            "aggregate_answer": agg_gold,
            "runs": runs,
            "num_correct": num_correct,
            "accuracy": accuracy,
        }
        results.append(result_entry)

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. Results saved to {output_path}")
    overall_acc = sum(r["num_correct"] for r in results) / (len(results) * args.num_runs) if results else 0.0
    print(f"Overall aggregate no-search accuracy: {overall_acc:.3f} across {len(results)} examples")


if __name__ == "__main__":
    main()
