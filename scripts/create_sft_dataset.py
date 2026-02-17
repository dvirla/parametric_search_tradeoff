"""Create an SFT dataset from no-search traces filtered by correctness and entropy.

Selects traces where:
  1. The problem has semantic entropy < threshold (model is confident/consistent)
  2. The individual trace was answered correctly

Output is ChatML-style JSONL suitable for SFTTrainer, with thinking wrapped in
<think>...</think> tags (GLM-4 convention).
"""

import argparse
import csv
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.entity_questions import ENTITY_QUESTIONS_QUERY_TEMPLATE


def extract_thinking_and_answer(message_trace: list[dict]) -> tuple[str, str]:
    """Extract thinking and final answer text from a message trace."""
    thinking = ""
    answer = ""
    for msg in reversed(message_trace):
        if msg.get("role") in ("model-text-response", "assistant"):
            for part in msg.get("parts", []):
                if part.get("type") == "thinking" and not thinking:
                    thinking = part.get("content", "")
                elif part.get("type") == "text" and not answer:
                    answer = part.get("content", "")
            if thinking or answer:
                break
    return thinking, answer


def load_entropy_data(entropy_csv: str) -> dict[str, float]:
    """Load semantic entropy CSV and return {problem: entropy}."""
    problem_entropy = {}
    with open(entropy_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_entropy[row["problem"]] = float(row["entropy"])
    print(f"Loaded entropy for {len(problem_entropy)} problems.")
    return problem_entropy


def load_traces(traces_path: str) -> list[dict]:
    """Load the no-search traces JSON."""
    with open(traces_path, "r") as f:
        traces = json.load(f)
    print(f"Loaded {len(traces)} traces.")
    return traces


def build_sft_examples(
    traces: list[dict],
    problem_entropy: dict[str, float],
    entropy_threshold: float = 0.2,
) -> list[dict]:
    """Filter and format traces into SFT training examples.

    Returns a list of ChatML message dicts.
    """
    examples = []
    problems_used = set()
    skipped_no_entropy = 0
    skipped_high_entropy = 0
    skipped_incorrect = 0

    for trace in traces:
        problem_clean = trace["problem_clean"]

        # Check entropy
        if problem_clean not in problem_entropy:
            skipped_no_entropy += 1
            continue

        entropy = problem_entropy[problem_clean]
        if entropy >= entropy_threshold:
            skipped_high_entropy += 1
            continue

        # Check correctness
        if not trace.get("correct"):
            skipped_incorrect += 1
            continue

        # Build the user message using the same template as evaluation
        # Extract the question from the full prompt, or use problem_clean directly
        user_content = ENTITY_QUESTIONS_QUERY_TEMPLATE.format(Question=problem_clean)

        # Extract thinking and answer from the message trace
        thinking, answer = extract_thinking_and_answer(trace.get("message_trace", []))

        if thinking:
            assistant_content = f"<think>{thinking}</think>\n{answer}"
        else:
            assistant_content = answer

        example = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ]
        }

        examples.append(example)
        problems_used.add(problem_clean)

    print(f"\nFiltering summary:")
    print(f"  Skipped (no entropy data): {skipped_no_entropy}")
    print(f"  Skipped (entropy >= {entropy_threshold}): {skipped_high_entropy}")
    print(f"  Skipped (incorrect answer): {skipped_incorrect}")
    print(f"  Kept: {len(examples)} examples from {len(problems_used)} unique problems")

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Create SFT dataset from no-search traces + semantic entropy"
    )
    parser.add_argument(
        "--traces",
        type=str,
        default="logs/popqa/glm_4_7/no_search_all_traces.json",
        help="Path to no-search traces JSON from download_no_search_traces.py",
    )
    parser.add_argument(
        "--entropy-csv",
        type=str,
        default="logs/semantic_entropy/semantic_entropy_popqa__glm_4_7.csv",
        help="Path to semantic entropy CSV",
    )
    parser.add_argument(
        "--entropy-threshold",
        type=float,
        default=0.2,
        help="Maximum entropy for inclusion (exclusive). Default: 0.2",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sft/glm_4_7_popqa_sft.jsonl",
        help="Output JSONL path",
    )
    args = parser.parse_args()

    # Load data
    traces = load_traces(args.traces)
    problem_entropy = load_entropy_data(args.entropy_csv)

    # Build SFT examples
    examples = build_sft_examples(traces, problem_entropy, args.entropy_threshold)

    if not examples:
        print("No examples passed filtering. Check entropy threshold and trace correctness.")
        return

    # Save JSONL
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(examples)} SFT examples to {args.output}")

    # Print a few samples for inspection
    print("\n--- Sample examples ---")
    for ex in examples[:3]:
        user_msg = ex["messages"][0]["content"][:100]
        asst_msg = ex["messages"][1]["content"][:150]
        print(f"  User: {user_msg}...")
        print(f"  Asst: {asst_msg}...")
        print()


if __name__ == "__main__":
    main()
