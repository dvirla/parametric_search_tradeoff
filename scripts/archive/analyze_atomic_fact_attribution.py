"""
Decomposes each sampler_response into atomic facts, then determines whether
each fact can be traced back to the search results in the corresponding trace.

For each model × dataset pair:
  1. Load eval log  → sampler_response per problem
  2. Load traces    → search results per problem
  3. Use gemini-3-flash-preview to decompose responses into atomic facts
  4. Use gemini-3-flash-preview to attribute each fact to search results (or not)

Output: CSV at results/curated_sharechat/atomic_fact_attribution.csv

Usage:
  # Run all models and datasets
  uv run python scripts/archive/analyze_atomic_fact_attribution.py

  # Run a single model (can be distributed across machines)
  uv run python scripts/archive/analyze_atomic_fact_attribution.py --model gemini-3-pro-preview
  uv run python scripts/archive/analyze_atomic_fact_attribution.py --model nemotron-3-nano:30b --dataset curated-sharechat
"""

import argparse
import json
import os
import csv
import sys
from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PAIRS = [
    {
        "model_label": "qwen3.5:122b",
        "dataset": "curated-sharechat",
        "eval_log": "results/curated_sharechat/curated-sharechat_baseline_qwen3.5:122b_run_1.json",
        "traces":   "results/curated_sharechat/curated-sharechat_qwen3.5:122b_baseline_agent_run_1_traces.json",
    },
    {
        "model_label": "qwen3.5:122b",
        "dataset": "curated-sharechat-benchmark",
        "eval_log": "results/curated_sharechat/curated-sharechat-benchmark_baseline_qwen3.5:122b_run_1.json",
        "traces":   "results/curated_sharechat/curated-sharechat-benchmark_qwen3.5:122b_baseline_agent_run_1_traces.json",
    },
    {
        "model_label": "gemini-3-pro-preview",
        "dataset": "curated-sharechat",
        "eval_log": "results/curated_sharechat/curated-sharechat_baseline_gemini-3-pro-preview_run_1.json",
        "traces":   "results/curated_sharechat/curated-sharechat_gemini-3-pro-preview_baseline_agent_run_1_traces.json",
    },
    {
        "model_label": "gemini-3-pro-preview",
        "dataset": "curated-sharechat-benchmark",
        "eval_log": "results/curated_sharechat/curated-sharechat-benchmark_baseline_gemini-3-pro-preview_run_1.json",
        "traces":   "results/curated_sharechat/curated-sharechat-benchmark_gemini-3-pro-preview_baseline_agent_run_1_traces.json",
    },
    {
        "model_label": "nemotron-3-nano:30b",
        "dataset": "curated-sharechat",
        "eval_log": "results/curated_sharechat/curated-sharechat_baseline_nemotron-3-nano:30b_run_1.json",
        "traces":   "results/curated_sharechat/curated-sharechat_nemotron-3-nano:30b_baseline_agent_run_1_traces.json",
    },
    {
        "model_label": "nemotron-3-nano:30b",
        "dataset": "curated-sharechat-benchmark",
        "eval_log": "results/curated_sharechat/curated-sharechat-benchmark_baseline_nemotron-3-nano:30b_run_1.json",
        "traces":   "results/curated_sharechat/curated-sharechat-benchmark_nemotron-3-nano:30b_baseline_agent_run_1_traces.json",
    },
]

OUTPUT_CSV = "results/curated_sharechat/atomic_fact_attribution.csv"
FLASH_MODEL = "gemini-3-flash-preview"

FIELDNAMES = [
    "model", "dataset", "problem", "has_search", "search_calls",
    "atomic_fact", "attributed_to_search", "reasoning", "sampler_response",
]

# ---------------------------------------------------------------------------
# Pydantic output types
# ---------------------------------------------------------------------------

class AtomicFactList(BaseModel):
    facts: list[str] = Field(
        description="The minimal set of atomic, self-contained factual claims extracted from the response."
    )

class FactAttributionResult(BaseModel):
    attributed_to_search: bool = Field(
        description="True if the atomic fact is directly supported by or derivable from the provided search results."
    )
    reasoning: str = Field(
        description="One-sentence justification for the attribution decision."
    )

class BatchFactAttribution(BaseModel):
    attributions: list[FactAttributionResult] = Field(
        description="Attribution result for each fact, in the same order as the input list."
    )

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are an expert at extracting atomic facts from text.
Given a response to a question, extract the MINIMAL set of atomic, self-contained
factual claims needed to convey the response's informational content.

Rules:
- Each fact must be a single, indivisible statement independently verifiable on its own.
- Merge any two facts that overlap or are redundant — keep only one.
- Omit hedges, meta-commentary, uncertainty markers, and structural phrases.
- Omit facts that merely restate the question or introduce context without new information.
- Prefer fewer, denser facts over many shallow ones.
Return only concrete, substantive factual claims."""

DECOMPOSE_USER_TEMPLATE = """Question: {question}

Response:
{response}

Extract the minimal set of atomic factual claims from the response above."""

ATTRIBUTE_SYSTEM = """You are a fact-checking assistant. Given a numbered list of atomic
factual claims and a set of web search results, determine for EACH claim whether it is
directly supported by or can be derived from the search results.
Only mark a claim as attributed if the evidence is explicit in the search results —
do not infer beyond what is stated. Return one attribution per claim, in the same order."""

ATTRIBUTE_USER_TEMPLATE = """Facts to attribute:
{facts_numbered}

Search results:
{search_results}

For each fact above, determine whether it is traceable to the search results."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def extract_search_results(message_trace: list) -> list[dict]:
    """Extract all search result items from a message trace."""
    results = []
    for msg in message_trace:
        if msg.get("role") != "user":
            continue
        for part in msg.get("parts", []):
            if part.get("type") == "tool_call_response" and part.get("tool_name") == "search":
                raw = part.get("result", [])
                if isinstance(raw, list):
                    results.extend(raw)
    return results


def format_search_results(results: list[dict]) -> str:
    if not results:
        return "(no search results)"
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("snippet", "")
        lines.append(f"[{i}] {title}\n    {snippet}")
    return "\n".join(lines)


def build_trace_index(traces: list) -> dict[str, list[dict]]:
    """Map problem → list of search results."""
    index = {}
    for trace in traces:
        problem = trace["problem"]
        search_results = extract_search_results(trace.get("message_trace", []))
        index[problem] = search_results
    return index


def load_processed_keys(output_csv: str) -> set:
    """Return set of (model, dataset, problem) already written to the output CSV."""
    if not os.path.exists(output_csv):
        return set()
    processed = set()
    with open(output_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            processed.add((row["model"], row.get("dataset", ""), row["problem"]))
    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Decompose model responses into atomic facts and attribute each to search results.")
    parser.add_argument("--model", default=None,
                        help="Filter to a specific model label (e.g. 'gemini-3-pro-preview'). Default: all models.")
    parser.add_argument("--dataset", default=None,
                        choices=["curated-sharechat", "curated-sharechat-benchmark"],
                        help="Filter to a specific dataset. Default: all datasets.")
    args = parser.parse_args()

    pairs = MODEL_PAIRS
    if args.model:
        pairs = [p for p in pairs if p["model_label"] == args.model]
        if not pairs:
            print(f"No MODEL_PAIRS match --model '{args.model}'. Available: {sorted({p['model_label'] for p in MODEL_PAIRS})}")
            return
    if args.dataset:
        pairs = [p for p in pairs if p["dataset"] == args.dataset]
        if not pairs:
            print(f"No MODEL_PAIRS match --dataset '{args.dataset}'.")
            return

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    processed_keys = load_processed_keys(OUTPUT_CSV)
    if processed_keys:
        print(f"Resuming: {len(processed_keys)} (model, dataset, problem) triples already in output.")

    decompose_agent = BaseAgent(
        provider_name="Google",
        model_name=FLASH_MODEL,
        output_type=AtomicFactList,
        system_prompt=DECOMPOSE_SYSTEM,
        agent_name="decompose_agent",
    )
    attribute_agent = BaseAgent(
        provider_name="Google",
        model_name=FLASH_MODEL,
        output_type=BatchFactAttribution,
        system_prompt=ATTRIBUTE_SYSTEM,
        agent_name="attribute_agent",
    )

    # Open CSV for streaming writes — write header once if file is new.
    file_exists = os.path.exists(OUTPUT_CSV)
    csv_file = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()
        csv_file.flush()

    total_new_rows = 0

    try:
        for pair in pairs:
            model_label = pair["model_label"]
            dataset = pair["dataset"]
            print(f"\n{'='*60}")
            print(f"Processing model: {model_label}  dataset: {dataset}")
            print(f"{'='*60}")

            eval_data = load_json(pair["eval_log"])
            traces = load_json(pair["traces"])
            trace_index = build_trace_index(traces)

            for entry in tqdm(eval_data, desc=f"{model_label} / {dataset}"):
                problem = entry["problem"]

                if (model_label, dataset, problem) in processed_keys:
                    continue

                sampler_response = entry.get("sampler_response", "")
                search_calls = entry.get("sampler_search_calls", 0)

                if not sampler_response or not sampler_response.strip():
                    continue

                search_results = trace_index.get(problem, [])
                search_results_text = format_search_results(search_results)
                has_search = len(search_results) > 0

                # Step 1: decompose into atomic facts
                decompose_prompt = DECOMPOSE_USER_TEMPLATE.format(
                    question=problem,
                    response=sampler_response,
                )
                try:
                    decompose_result = decompose_agent.run(decompose_prompt)
                    facts = decompose_result.output.facts
                except Exception as e:
                    print(f"  [WARN] Decomposition failed for '{problem}': {e}")
                    facts = []

                problem_rows = []

                if not facts:
                    problem_rows.append({
                        "model": model_label,
                        "dataset": dataset,
                        "problem": problem,
                        "has_search": has_search,
                        "search_calls": search_calls,
                        "atomic_fact": "",
                        "attributed_to_search": None,
                        "reasoning": "Decomposition returned no facts.",
                        "sampler_response": sampler_response,
                    })
                elif not has_search:
                    for fact in facts:
                        problem_rows.append({
                            "model": model_label,
                            "dataset": dataset,
                            "problem": problem,
                            "has_search": False,
                            "search_calls": search_calls,
                            "atomic_fact": fact,
                            "attributed_to_search": False,
                            "reasoning": "No search results available for this query.",
                            "sampler_response": sampler_response,
                        })
                else:
                    # Step 2: attribute all facts in a single batch call
                    facts_numbered = "\n".join(f"{i+1}. {f}" for i, f in enumerate(facts))
                    attribute_prompt = ATTRIBUTE_USER_TEMPLATE.format(
                        facts_numbered=facts_numbered,
                        search_results=search_results_text,
                    )
                    attributions = None
                    try:
                        attr_result = attribute_agent.run(attribute_prompt)
                        attributions = attr_result.output.attributions
                        # Guard against length mismatch (model skipped or duplicated items)
                        if len(attributions) != len(facts):
                            print(f"  [WARN] Attribution count mismatch ({len(attributions)} vs {len(facts)} facts) for '{problem[:60]}' — falling back to None")
                            attributions = None
                    except Exception as e:
                        print(f"  [WARN] Batch attribution failed for '{problem[:60]}': {e}")

                    for i, fact in enumerate(facts):
                        if attributions is not None:
                            attributed = attributions[i].attributed_to_search
                            reasoning = attributions[i].reasoning
                        else:
                            attributed = None
                            reasoning = "Batch attribution failed or mismatched."
                        problem_rows.append({
                            "model": model_label,
                            "dataset": dataset,
                            "problem": problem,
                            "has_search": has_search,
                            "search_calls": search_calls,
                            "atomic_fact": fact,
                            "attributed_to_search": attributed,
                            "reasoning": reasoning,
                            "sampler_response": sampler_response,
                        })

                # Write this problem's rows immediately so progress survives crashes
                writer.writerows(problem_rows)
                csv_file.flush()
                total_new_rows += len(problem_rows)
    finally:
        csv_file.close()

    if total_new_rows == 0:
        print("\nNothing new to write.")
    else:
        print(f"\nDone! {total_new_rows} new rows written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
