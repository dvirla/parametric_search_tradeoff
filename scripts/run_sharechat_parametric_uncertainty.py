"""
ShareChat Parametric Uncertainty Experiment

For each annotated ShareChat example (from data/curated_sharechat_hops.json):
  - Runs the model N times (no search) per sub-question to measure parametric uncertainty
  - Outputs per-hop semantic entropy in the same format as MusiQue

Output is compatible with analyze_parametric_search_interplay.py.

Usage:
    uv run python scripts/run_sharechat_parametric_uncertainty.py \\
        --hops-json data/curated_sharechat_hops.json \\
        --model-name qwen3.5:122b \\
        --provider ollama \\
        --num-runs 5 \\
        --output-dir results/sharechat_parametric \\
        --resume
"""

import argparse
import json
import math
import os
import sys
import time
from collections import Counter

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from scripts.run_musique_experiment import grade_response

load_dotenv()


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────

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


# ─── CLI ──────────────────────────────────────────────────────────────────────

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ShareChat parametric uncertainty experiment.")
    parser.add_argument("--hops-json", type=str, default="data/curated_sharechat_hops.json",
                        help="Path to curated_sharechat_hops.json from label_sharechat_hops.py")
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--provider", type=str, required=True)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="results/sharechat_parametric")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--grader-ollama-url", default=None)
    parser.add_argument("--clusterer-ollama-url", default=None)
    return parser.parse_args()


# ─── HELPERS ──────────────────────────────────────────────────────────────────

_CLUSTER_PROMPT = """\
Question: {question}

The following {n} answers were given independently to this question:
{answers}

Group them by semantic equivalence. Two answers belong to the same cluster if they \
express the same fact (minor wording differences are fine; "United States" and "USA" are the same). \
Use consecutive integer cluster IDs starting from 0.\
"""


def build_clusterer(ollama_base_url: str | None = None) -> BaseAgent:
    print(f"  Initializing clusterer (gpt-oss:120b via ollama @ {ollama_base_url or 'default'})...")
    return BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=AnswerClustering,
        agent_name="sharechat_clusterer",
        ollama_base_url=ollama_base_url,
    )


def compute_semantic_entropy(
    clusterer: BaseAgent, question: str, runs: list[dict], max_retries: int = 3
) -> tuple[float, list[int]]:
    answers = [r["final_answer"] for r in runs]
    n = len(answers)
    if n == 0:
        return 0.0, []

    answer_lines = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
    prompt = _CLUSTER_PROMPT.format(question=question, n=n, answers=answer_lines)

    cluster_ids: list[int] = []
    for attempt in range(max_retries):
        try:
            response = clusterer.run(prompt)
            cluster_ids = response.output.cluster_ids
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



def run_single_hop(
    agent: BaseAgent,
    grader: AgentAsSampler,
    question: str,
    gold_answer: str,
    max_retries: int = 5,
) -> dict:
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
                wait = 2 ** attempt
                print(f"    Network error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {"reasoning": "", "final_answer": f"ERROR: {e}", "is_correct": False}
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"    Error (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return {"reasoning": "", "final_answer": f"ERROR: {e}", "is_correct": False}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = setup_args()

    # Load annotated hops
    with open(args.hops_json) as f:
        hops_data = json.load(f)
    annotations = hops_data.get("annotations", {})
    labeled = [ann for ann in annotations.values() if ann.get("labeled", False)]
    labeled.sort(key=lambda a: int(a["example_id"].split("_")[-1]))
    print(f"Loaded {len(labeled)} labeled examples from {args.hops_json}")

    # Output path
    model_slug = args.model_name.replace("/", "_").replace(":", "_")
    output_filename = f"curated_sharechat_parametric_uncertainty_{model_slug}.json"
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, output_filename)

    # Resume support
    existing_results: list[dict] = []
    completed_ids: set[str] = set()
    if args.resume and os.path.exists(output_path):
        with open(output_path) as f:
            existing_results = json.load(f)
        completed_ids = {r["example_id"] for r in existing_results}
        print(f"Resuming: {len(completed_ids)} examples already completed.")

    # Build agents
    print(f"Initializing parametric agent ({args.model_name} via {args.provider} @ {args.ollama_url or 'default'})...")
    parametric_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model_name,
        output_type=HopAnswer,
        agent_name=f"sharechat_parametric_{model_slug}",
        ollama_base_url=args.ollama_url,
    )

    grader = AgentAsSampler(BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        agent_name="sharechat_grader",
        ollama_base_url=args.grader_ollama_url,
    ))
    clusterer = build_clusterer(ollama_base_url=args.clusterer_ollama_url)

    print(f"\n--- ShareChat Parametric Uncertainty: {args.model_name}, {args.num_runs} runs/hop ---")
    print(f"Examples: {len(labeled)}, Output: {output_path}\n")

    results = list(existing_results)

    for i, ann in enumerate(labeled):
        example_id = ann["example_id"]
        if example_id in completed_ids:
            print(f"[{i+1}/{len(labeled)}] Skipping {example_id} (already done)")
            continue

        print(f"[{i+1}/{len(labeled)}] Processing {example_id}...")
        sub_questions = ann.get("sub_questions", [])

        sub_questions_results = []
        for hop in sub_questions:
            hop_idx = hop["hop_index"]
            question = hop["question"].strip()
            gold_answer = hop["gold_answer"].strip()

            if not question:
                print(f"  Hop {hop_idx}: skipped (empty question)")
                continue

            print(f"  Hop {hop_idx}: {question[:80]}...")

            runs = []
            for run_i in range(args.num_runs):
                print(f"    Run {run_i+1}/{args.num_runs}...")
                run_result = run_single_hop(parametric_agent, grader, question, gold_answer)
                runs.append(run_result)

            num_correct = sum(1 for r in runs if r["is_correct"])
            semantic_entropy, cluster_ids = compute_semantic_entropy(clusterer, question, runs)

            sub_questions_results.append({
                "hop_index": hop_idx,
                "question": question,
                "gold_answer": gold_answer,
                "runs": runs,
                "num_correct": num_correct,
                "num_runs": args.num_runs,
                "semantic_entropy": semantic_entropy,
                "cluster_ids": cluster_ids,
            })

        entry = {
            "example_id": example_id,
            "aggregate_question": ann.get("benchmark_question", ann.get("problem", "")),
            "aggregate_answer": ann.get("aggregate_answer", ""),
            "sub_questions_results": sub_questions_results,
        }
        results.append(entry)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        if sub_questions_results:
            avg_acc = sum(r["num_correct"] / args.num_runs for r in sub_questions_results) / len(sub_questions_results)
            avg_ent = sum(r["semantic_entropy"] for r in sub_questions_results) / len(sub_questions_results)
            print(f"  -> avg_hop_acc={avg_acc:.2f}, avg_entropy={avg_ent:.3f} bits")
        else:
            print("  -> no hops processed")

    print(f"\nDone. {len(results)} examples written to {output_path}")


if __name__ == "__main__":
    main()
