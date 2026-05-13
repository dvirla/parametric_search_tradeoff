"""Download all no-search agent traces from Logfire for a given model/dataset.

Unlike download_traces.py (which deduplicates by problem), this script keeps
ALL traces — typically 5 per problem from 5 no-search runs.  Traces are grouped
by problem text and sorted chronologically within each group.  Each trace is
labelled with a correctness flag by comparing its extracted answer to the ground
truth using the same matching logic from entity_questions.py.
"""

import argparse
import json
import os
import sys

import requests
import tqdm
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.download_traces import clean_problem, extract_message_trace
from src.services.entity_questions import any_match_grade, exact_match_grade

load_dotenv()

LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
LOGFIRE_API_URL = "https://logfire-us.pydantic.dev"


def fetch_all_traces(agent_name: str, model_name: str, days: int = 7) -> list[dict]:
    """Query Logfire for all traces matching agent/model within the time window."""
    if not LOGFIRE_API_KEY:
        raise ValueError("LOGFIRE_API_KEY environment variable not set.")

    headers = {
        "Authorization": f"Bearer {LOGFIRE_API_KEY}",
        "Content-Type": "application/json",
    }

    query = f"""
SELECT
    attributes->>'agent_name' as agent_name,
    attributes->'pydantic_ai.all_messages' as all_messages,
    attributes->'pydantic_ai.result' as result,
    start_timestamp,
    end_timestamp
FROM records
WHERE attributes->>'agent_name' LIKE '%{agent_name}%'
AND attributes->>'model_name' LIKE '%{model_name}%'
AND start_timestamp > NOW() - INTERVAL '{days} days'
ORDER BY start_timestamp ASC
"""

    api_limit = 10000
    params = {"sql": query, "limit": api_limit}

    print(f"Fetching traces for agent '{agent_name}', model '{model_name}' (last {days} days)...")
    response = requests.get(f"{LOGFIRE_API_URL}/v1/query", headers=headers, params=params)
    response.raise_for_status()

    data = response.json()
    columns = data.get("columns", [])

    def col(name: str):
        return next((c for c in columns if c["name"] == name), None)

    all_messages_col = col("all_messages")
    result_col = col("result")
    start_ts_col = col("start_timestamp")
    end_ts_col = col("end_timestamp")

    if not all_messages_col:
        print("WARNING: No all_messages column in response")
        return []

    num_records = len(all_messages_col["values"])
    print(f"Processing {num_records} raw records...")

    traces = []
    for idx in tqdm.tqdm(range(num_records), desc="Extracting traces"):
        all_messages = all_messages_col["values"][idx]
        if not all_messages or len(all_messages) < 2:
            continue

        try:
            # Extract problem from the first user message
            problem_content = ""
            for msg in all_messages:
                if msg.get("role") == "user":
                    for part in msg.get("parts", []):
                        if part.get("type") == "text":
                            problem_content = part.get("content", "")
                            break
                    if problem_content:
                        break

            if not problem_content:
                continue

            message_trace = extract_message_trace(all_messages)

            traces.append({
                "problem": problem_content.strip(),
                "problem_clean": clean_problem(problem_content.strip()),
                "agent_name": "no_search_agent",
                "start_timestamp": start_ts_col["values"][idx] if start_ts_col else None,
                "end_timestamp": end_ts_col["values"][idx] if end_ts_col else None,
                "result": result_col["values"][idx] if result_col else None,
                "message_trace": message_trace,
                "metadata": {
                    "total_messages": len(message_trace),
                    "thinking_messages": sum(1 for msg in message_trace for part in msg.get("parts", []) if part.get("type") == "thinking"),
                },
            })

        except (KeyError, IndexError, TypeError):
            continue

    print(f"Extracted {len(traces)} traces total.")
    return traces


def load_ground_truth(eval_dir: str, dataset: str, model_name: str) -> dict[str, dict]:
    """Load ground-truth answers from the no-search evaluation JSONs.

    Returns a dict mapping cleaned problem text -> {correct_answer, answer_aliases}.
    Uses the first run file found since all runs share the same questions/answers.
    """
    import glob as globmod

    # Find any no-search run JSON
    patterns = [
        os.path.join(eval_dir, f"{dataset}_no_search_{model_name}_run_*.json"),
        os.path.join(eval_dir, f"{dataset}_no_search_{model_name.replace(':', '-')}_run_*.json"),
    ]

    files = []
    for pat in patterns:
        files = sorted(globmod.glob(pat))
        if files:
            break

    if not files:
        raise FileNotFoundError(
            f"No no-search eval JSONs found in {eval_dir} for {dataset}/{model_name}. "
            f"Tried patterns: {patterns}"
        )

    print(f"Loading ground truth from {files[0]}...")
    with open(files[0], "r") as f:
        data = json.load(f)

    gt = {}
    for entry in data:
        problem_clean = clean_problem(entry["problem"])
        gt[problem_clean] = {
            "correct_answer": entry["correct_answer"],
            "answer_aliases": entry.get("answer_aliases", []),
        }

    print(f"Loaded ground truth for {len(gt)} problems.")
    return gt


def extract_answer_from_trace(message_trace: list[dict]) -> str:
    """Extract the final answer text from a message trace."""
    for msg in reversed(message_trace):
        if msg.get("role") in ("model-text-response", "assistant"):
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    return part.get("content", "")
    return ""


def label_correctness(traces: list[dict], ground_truth: dict[str, dict]) -> list[dict]:
    """Add a 'correct' field to each trace by comparing its answer to ground truth."""
    labelled = 0
    for trace in traces:
        problem_clean = trace["problem_clean"]
        gt = ground_truth.get(problem_clean)

        if gt is None:
            trace["correct"] = None
            continue

        answer_text = extract_answer_from_trace(trace.get("message_trace", []))
        # Also try extracting from result if answer_text is empty
        if not answer_text and trace.get("result"):
            result = trace["result"]
            if isinstance(result, dict):
                answer_text = result.get("answer", str(result))
            else:
                answer_text = str(result)

        gold = gt["correct_answer"]
        aliases = gt.get("answer_aliases", [])

        if isinstance(gold, list):
            if aliases:
                all_acceptable = list(gold) + aliases
                trace["correct"] = any_match_grade(answer_text, all_acceptable)
            else:
                trace["correct"] = exact_match_grade(answer_text, gold)
        else:
            acceptable = [gold] + aliases
            trace["correct"] = any_match_grade(answer_text, acceptable)

        trace["correct_answer"] = gold
        labelled += 1

    correct_count = sum(1 for t in traces if t.get("correct") is True)
    print(f"Labelled {labelled} traces: {correct_count} correct, {labelled - correct_count} incorrect.")
    return traces


def main():
    parser = argparse.ArgumentParser(
        description="Download ALL no-search traces from Logfire and label correctness."
    )
    parser.add_argument("--agent-name", type=str, default="no_search_agent",
                        help="Agent name pattern in Logfire")
    parser.add_argument("--model-name", type=str, default="glm-4.7-flash",
                        help="Model name pattern in Logfire")
    parser.add_argument("--dataset", type=str, default="popqa",
                        help="Dataset name (for finding eval JSONs)")
    parser.add_argument("--eval-dir", type=str, default="logs/popqa/glm_4_7",
                        help="Directory containing no-search eval JSONs")
    parser.add_argument("--output", type=str, default="logs/popqa/glm_4_7/no_search_all_traces.json",
                        help="Output path for traces JSON")
    parser.add_argument("--days", type=int, default=7,
                        help="Look back N days in Logfire")
    args = parser.parse_args()

    # 1. Fetch all traces from Logfire
    traces = fetch_all_traces(args.agent_name, args.model_name, days=args.days)

    if not traces:
        print("No traces found. Exiting.")
        return

    # 2. Group by problem for summary stats
    from collections import Counter
    problem_counts = Counter(t["problem_clean"] for t in traces)
    print(f"\nUnique problems: {len(problem_counts)}")
    print(f"Traces per problem distribution: {Counter(problem_counts.values())}")

    # 3. Load ground truth and label correctness
    ground_truth = load_ground_truth(args.eval_dir, args.dataset, args.model_name)
    traces = label_correctness(traces, ground_truth)

    # 4. Save full traces (including message_trace) matching baseline format
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(traces, f, indent=2)

    print(f"\nSaved {len(traces)} traces to {args.output}")


if __name__ == "__main__":
    main()
