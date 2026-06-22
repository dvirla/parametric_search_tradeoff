"""
Re-evaluate MuSiQue parametric uncertainty results with updated grader and/or clusterer.

Loads an existing results JSON, re-grades is_correct fields and/or re-clusters answers,
then saves to a new file (or overwrites in-place with --overwrite).

Usage:
    uv run python scripts/re_evaluate_musique_parametric.py \
        --input results/musique_parametric_test/musique_parametric_uncertainty_gemini-3-pro-preview.json \
        --grader_model gemini-2.5-pro \
        --grader_provider google \
        --clusterer_model gemini-2.5-pro \
        --clusterer_provider google \
        --output results/musique_parametric_test/musique_parametric_uncertainty_gemini-3-pro-preview_v2.json

Flags:
    --skip_grading       Skip re-grading (only re-cluster)
    --skip_clustering    Skip re-clustering (only re-grade)
    --overwrite          Overwrite the input file instead of writing to --output
"""

import os
import sys
import json
import math
import time
import argparse
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.services.base_agent import BaseAgent
from src.services.agent_sampler import AgentAsSampler
from scripts.archive.run_musique_experiment import grade_response
from scripts.run_musique_parametric_uncertainty import (
    AnswerClustering,
    _CLUSTER_PROMPT,
)
from src.services.qa_eval import STANDARD_GRADER_TEMPLATE


def setup_args():
    parser = argparse.ArgumentParser(description="Re-evaluate MuSiQue parametric results.")
    parser.add_argument("--input", type=str, required=True, help="Path to existing results JSON.")
    parser.add_argument("--output", type=str, default=None, help="Output path (default: <input>_reevaluated.json).")
    parser.add_argument("--overwrite", action="store_true", default=False, help="Overwrite input file.")

    parser.add_argument("--grader_model", type=str, default="gpt-oss:20b", help="Grader model name.")
    parser.add_argument("--grader_provider", type=str, default="ollama", help="Grader provider.")
    parser.add_argument("--skip_grading", action="store_true", default=False, help="Skip re-grading.")

    parser.add_argument("--clusterer_model", type=str, default="gpt-oss:20b", help="Clusterer model name.")
    parser.add_argument("--clusterer_provider", type=str, default="ollama", help="Clusterer provider.")
    parser.add_argument("--skip_clustering", action="store_true", default=False, help="Skip re-clustering.")

    parser.add_argument("--max_retries", type=int, default=3, help="Retries per call.")
    return parser.parse_args()


def build_grader(provider: str, model: str) -> AgentAsSampler:
    print(f"Initializing grader ({model} via {provider})...")
    raw = BaseAgent(provider_name=provider, model_name=model, agent_name="musique_regrader")
    return AgentAsSampler(raw)


def build_clusterer(provider: str, model: str) -> BaseAgent:
    print(f"Initializing clusterer ({model} via {provider})...")
    return BaseAgent(
        provider_name=provider,
        model_name=model,
        output_type=AnswerClustering,
        agent_name="musique_reclustered",
    )


def recluster(clusterer: BaseAgent, question: str, runs: list[dict], max_retries: int = 3) -> tuple[float, list[int]]:
    answers = [r["final_answer"] for r in runs]
    n = len(answers)
    if n == 0:
        return 0.0, []

    answer_lines = "\n".join(f"{i+1}. {a}" for i, a in enumerate(answers))
    prompt = _CLUSTER_PROMPT.format(question=question, n=n, answers=answer_lines)

    cluster_ids = list(range(n))  # fallback: all distinct
    for attempt in range(max_retries):
        try:
            response = clusterer.run(prompt)
            ids = response.output.cluster_ids
            if len(ids) != n:
                raise ValueError(f"Expected {n} cluster IDs, got {len(ids)}")
            cluster_ids = ids
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"    Clusterer error (attempt {attempt+1}): {e}. Retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"    Clusterer failed; falling back to all-distinct clusters.")

    counts = list(Counter(cluster_ids).values())
    entropy = -sum((c / n) * math.log2(c / n) for c in counts)
    return entropy, cluster_ids


def main():
    args = setup_args()

    if args.skip_grading and args.skip_clustering:
        print("Both --skip_grading and --skip_clustering set — nothing to do.")
        return

    # Determine output path
    if args.overwrite:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = f"{base}_reevaluated{ext}"

    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")

    with open(args.input) as f:
        results = json.load(f)
    print(f"Loaded {len(results)} examples.")

    grader = None if args.skip_grading else build_grader(args.grader_provider, args.grader_model)
    clusterer = None if args.skip_clustering else build_clusterer(args.clusterer_provider, args.clusterer_model)

    for i, entry in enumerate(results):
        example_id = entry["example_id"]
        print(f"\n[{i+1}/{len(results)}] {example_id}")

        # Re-evaluate sub-question hop runs
        for hop in entry.get("sub_questions_results", []):
            question = hop["question"]
            gold_answer = hop["gold_answer"]
            runs = hop["runs"]

            if grader is not None:
                corrected = 0
                for run_idx, run in enumerate(runs):
                    old = run["is_correct"]
                    new = grade_response(grader, question, gold_answer, run["final_answer"])
                    run["is_correct"] = new
                    if old != new:
                        corrected += 1
                        print(f"  hop {hop['hop_index']} run {run_idx}: {old} -> {new}  [{run['final_answer'][:60]}]")
                hop["num_correct"] = sum(1 for r in runs if r["is_correct"])

            if clusterer is not None:
                old_entropy = hop.get("semantic_entropy")
                old_clusters = hop.get("cluster_ids")
                new_entropy, new_clusters = recluster(clusterer, question, runs, args.max_retries)
                hop["semantic_entropy"] = new_entropy
                hop["cluster_ids"] = new_clusters
                if old_clusters != new_clusters:
                    print(f"  hop {hop['hop_index']} clusters: {old_clusters} -> {new_clusters}  entropy: {old_entropy:.3f} -> {new_entropy:.3f}")

        # Re-grade aggregate result
        agg = entry.get("aggregate_result", {})
        if grader is not None and agg:
            old = agg.get("is_correct")
            new = grade_response(
                grader,
                agg["question"],
                agg["gold_answer"],
                agg["final_answer"],
            )
            agg["is_correct"] = new
            if old != new:
                print(f"  aggregate: {old} -> {new}  [{agg['final_answer'][:60]}]")

        # Save after each example (for crash-safety)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nDone. {len(results)} examples saved to {output_path}")


if __name__ == "__main__":
    main()
