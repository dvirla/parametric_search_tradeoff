"""
Decomposes each sampler_response into atomic facts, then determines whether
each fact can be traced back to the search results in the corresponding trace.

For each model pair (gemini-3-pro-preview, nemotron-3-nano):
  1. Load eval log  → sampler_response per problem
  2. Load traces    → search results per problem
  3. Use gemini-3-flash-preview to decompose responses into atomic facts
  4. Use gemini-3-flash-preview to attribute each fact to search results (or not)

Output: CSV at results/sharechat/atomic_fact_attribution.csv
"""

import json
import os
import csv
import sys
from typing import Optional
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
        "model_label": "gemini-3-pro-preview",
        "eval_log": "logs/sharechat/sharechat_baseline_gemini-3-pro-preview_run_1.json",
        "traces": "logs/sharechat/gemini-3-pro-baseline_agent_run_1_traces_20260315_215200.json",
    },
    {
        "model_label": "nemotron-3-nano",
        "eval_log": "logs/sharechat/sharechat_baseline_nemotron-3-nano:30b_run_1.json",
        "traces": "logs/sharechat/nemotron-3-nano-baseline_agent_run_1_traces_20260315_215232.json",
    },
]

OUTPUT_CSV = "results/sharechat/atomic_fact_attribution.csv"
FLASH_MODEL = "gemini-3-flash-preview"

# ---------------------------------------------------------------------------
# Pydantic output types
# ---------------------------------------------------------------------------

class AtomicFactList(BaseModel):
    facts: list[str] = Field(
        description="The minimal set of atomic, self-contained factual claims extracted from the response."
    )

class FactAttribution(BaseModel):
    attributed_to_search: bool = Field(
        description="True if the atomic fact is directly supported by or derivable from the provided search results."
    )
    reasoning: str = Field(
        description="One-sentence justification for the attribution decision."
    )

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are an expert at extracting atomic facts from text.
Given a response to a question, decompose it into the minimal set of atomic,
self-contained factual claims. Each fact must be a single, indivisible
statement that can be independently verified. Do not include meta-commentary,
hedges, or structural markers — only concrete factual claims.
Do not decompose into overlapping or redundant facts. Focus on distilling the core factual content of the response."""

DECOMPOSE_USER_TEMPLATE = """Question: {question}

Response:
{response}

Extract every atomic factual claim from the response above."""

ATTRIBUTE_SYSTEM = """You are a fact-checking assistant. Given an atomic factual claim
and a set of web search results, determine whether the claim is directly supported by
or can be derived from the search results. Only mark it as attributed if the evidence
is explicit in the search results — do not infer beyond what is stated."""

ATTRIBUTE_USER_TEMPLATE = """Atomic fact: {fact}

Search results:
{search_results}

Can this atomic fact be traced to the search results above?"""

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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # Instantiate agents once and reuse
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
        output_type=FactAttribution,
        system_prompt=ATTRIBUTE_SYSTEM,
        agent_name="attribute_agent",
    )

    rows = []

    for pair in MODEL_PAIRS:
        model_label = pair["model_label"]
        print(f"\n{'='*60}")
        print(f"Processing model: {model_label}")
        print(f"{'='*60}")

        eval_data = load_json(pair["eval_log"])
        traces = load_json(pair["traces"])
        trace_index = build_trace_index(traces)

        for entry in tqdm(eval_data, desc=model_label):
            problem = entry["problem"]
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

            if not facts:
                rows.append({
                    "model": model_label,
                    "problem": problem,
                    "has_search": has_search,
                    "search_calls": search_calls,
                    "atomic_fact": "",
                    "attributed_to_search": None,
                    "reasoning": "Decomposition returned no facts.",
                    "sampler_response": sampler_response,
                })
                continue

            # Step 2: attribute each fact to search results
            for fact in facts:
                if not has_search:
                    # No search was used — attribution is trivially False
                    rows.append({
                        "model": model_label,
                        "problem": problem,
                        "has_search": False,
                        "search_calls": search_calls,
                        "atomic_fact": fact,
                        "attributed_to_search": False,
                        "reasoning": "No search results available for this query.",
                        "sampler_response": sampler_response,
                    })
                    continue

                attribute_prompt = ATTRIBUTE_USER_TEMPLATE.format(
                    fact=fact,
                    search_results=search_results_text,
                )
                try:
                    attr_result = attribute_agent.run(attribute_prompt)
                    attributed = attr_result.output.attributed_to_search
                    reasoning = attr_result.output.reasoning
                except Exception as e:
                    print(f"  [WARN] Attribution failed for fact '{fact[:60]}...': {e}")
                    attributed = None
                    reasoning = f"Error: {e}"

                rows.append({
                    "model": model_label,
                    "problem": problem,
                    "has_search": has_search,
                    "search_calls": search_calls,
                    "atomic_fact": fact,
                    "attributed_to_search": attributed,
                    "reasoning": reasoning,
                    "sampler_response": sampler_response,
                })

    # Write CSV
    fieldnames = [
        "model", "problem", "has_search", "search_calls",
        "atomic_fact", "attributed_to_search", "reasoning", "sampler_response",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {len(rows)} rows written to {OUTPUT_CSV}")

    # Print a quick summary
    from collections import Counter
    for model_label in [p["model_label"] for p in MODEL_PAIRS]:
        model_rows = [r for r in rows if r["model"] == model_label]
        search_rows = [r for r in model_rows if r["has_search"]]
        attributed = sum(1 for r in search_rows if r["attributed_to_search"] is True)
        not_attributed = sum(1 for r in search_rows if r["attributed_to_search"] is False)
        no_search_rows = [r for r in model_rows if not r["has_search"]]
        print(f"\n{model_label}:")
        print(f"  Total facts:              {len(model_rows)}")
        print(f"  Facts with search ctx:    {len(search_rows)}")
        print(f"    - Attributed to search: {attributed}")
        print(f"    - Not attributed:       {not_attributed}")
        print(f"  Facts without search:     {len(no_search_rows)}")


if __name__ == "__main__":
    main()
