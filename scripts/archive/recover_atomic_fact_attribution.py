"""
Recovery script: reconstructs results/sharechat/atomic_fact_attribution.csv from Logfire.

The analyze_atomic_fact_attribution.py script finished processing all 169 gemini-3-pro-preview
problems but crashed before writing the CSV. This script fetches the decompose_agent and
attribute_agent traces from Logfire and reconstructs the CSV in the exact same format.

CSV fieldnames:
  model, problem, has_search, search_calls, atomic_fact, attributed_to_search, reasoning,
  sampler_response
"""

import json
import os
import csv
import sys
import requests
from collections import defaultdict
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
load_dotenv()

LOGFIRE_API_KEY = os.getenv("LOGFIRE_API_KEY")
LOGFIRE_API_URL = "https://logfire-us.pydantic.dev"

OUTPUT_CSV = "results/sharechat/atomic_fact_attribution.csv"
EVAL_LOG = "logs/sharechat/sharechat_baseline_gemini-3-pro-preview_run_1.json"
MODEL_LABEL = "gemini-3-pro-preview"
NUM_DECOMPOSE = 169  # number of problems analyzed


# ---------------------------------------------------------------------------
# Logfire query helpers
# ---------------------------------------------------------------------------

def logfire_query(sql: str, limit: int = 10000) -> dict:
    headers = {
        "Authorization": f"Bearer {LOGFIRE_API_KEY}",
        "Content-Type": "application/json",
    }
    params = {"sql": sql, "limit": limit}
    resp = requests.get(f"{LOGFIRE_API_URL}/v1/query", headers=headers, params=params)
    resp.raise_for_status()
    return resp.json()


def columns_to_rows(data: dict) -> list[dict]:
    """Convert column-oriented Logfire response to list of row dicts."""
    columns = data.get("columns", [])
    if not columns:
        return []
    n = len(columns[0]["values"])
    return [{col["name"]: col["values"][i] for col in columns} for i in range(n)]


def get_first_user_text(all_messages: list) -> str:
    for msg in all_messages:
        if msg.get("role") == "user":
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    return part.get("content", "")
    return ""


def get_final_result_args(all_messages: list) -> dict:
    """Extract the structured output from the final_result tool_call in the assistant message."""
    for msg in all_messages:
        if msg.get("role") == "model" or msg.get("role") == "assistant":
            for part in msg.get("parts", []):
                if part.get("type") == "tool_call" and part.get("name") == "final_result":
                    args = part.get("arguments", {})
                    if isinstance(args, dict):
                        return args
    return {}


# ---------------------------------------------------------------------------
# Prompt parsers
# ---------------------------------------------------------------------------

def parse_decompose_prompt(text: str) -> tuple[str, str]:
    """
    Parses 'Question: {q}\\n\\nResponse:\\n{r}\\n\\nExtract every atomic...'
    Returns (question, sampler_response).
    """
    question = ""
    response = ""
    marker = "Response:\n"
    extract_suffix = "\n\nExtract every atomic factual claim from the response above."

    if marker in text:
        before, after = text.split(marker, 1)
        question = before.replace("Question: ", "", 1).strip()
        if after.endswith(extract_suffix):
            response = after[: -len(extract_suffix)].strip()
        elif extract_suffix.strip() in after:
            response = after.rsplit(extract_suffix.strip(), 1)[0].strip()
        else:
            response = after.strip()

    return question, response


def parse_attribute_prompt(text: str) -> str:
    """
    Parses 'Atomic fact: {fact}\\n\\nSearch results:...'
    Returns the atomic fact.
    """
    if "Atomic fact: " in text:
        after = text.split("Atomic fact: ", 1)[1]
        return after.split("\n\nSearch results:")[0].strip()
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not LOGFIRE_API_KEY:
        raise ValueError("LOGFIRE_API_KEY not set in .env")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    # ------------------------------------------------------------------
    # Load eval log to get search_calls per problem
    # ------------------------------------------------------------------
    search_calls_by_problem: dict[str, int] = {}
    try:
        with open(EVAL_LOG) as f:
            eval_data = json.load(f)
        for entry in eval_data:
            raw_problem = entry.get("problem", "").strip()
            # Strip the instruction suffix the same way qa_eval does
            separator = "\n\nYour response should be in the following format:"
            if separator in raw_problem:
                raw_problem = raw_problem.split(separator)[0].strip()
            search_calls_by_problem[raw_problem] = entry.get("sampler_search_calls", 0)
        print(f"Loaded {len(search_calls_by_problem)} problems from eval log")
    except Exception as e:
        print(f"Warning: could not load eval log ({e}); search_calls will default to 0")

    # ------------------------------------------------------------------
    # Fetch the latest 169 decompose_agent records from Logfire
    # ------------------------------------------------------------------
    print(f"\nFetching {NUM_DECOMPOSE} decompose_agent records from Logfire...")
    decompose_sql = f"""
SELECT
    attributes->>'agent_name'          AS agent_name,
    attributes->'pydantic_ai.all_messages' AS all_messages,
    attributes->'pydantic_ai.result'   AS result,
    start_timestamp
FROM records
WHERE attributes->>'agent_name' = 'decompose_agent'
AND start_timestamp > NOW() - INTERVAL '14 days'
ORDER BY start_timestamp DESC
LIMIT {NUM_DECOMPOSE}
"""
    decompose_rows = columns_to_rows(logfire_query(decompose_sql, limit=NUM_DECOMPOSE))
    print(f"Got {len(decompose_rows)} decompose records")

    if not decompose_rows:
        print("ERROR: no decompose_agent records found. Check agent_name and time range.")
        sys.exit(1)

    # Find the time range covered by these decompose records so we can
    # limit the attribute query to roughly the same window.
    timestamps = [r["start_timestamp"] for r in decompose_rows if r.get("start_timestamp")]
    earliest_ts = min(timestamps) if timestamps else None
    print(f"Decompose records span: {min(timestamps)} → {max(timestamps)}" if timestamps else "")

    # ------------------------------------------------------------------
    # Fetch attribute_agent records in the same window
    # ------------------------------------------------------------------
    print("\nFetching attribute_agent records from Logfire...")
    if earliest_ts:
        # Use the earliest decompose timestamp as lower bound (with small buffer)
        attr_where_extra = f"AND start_timestamp >= '{earliest_ts}'::timestamptz - INTERVAL '5 minutes'"
    else:
        attr_where_extra = ""

    attribute_sql = f"""
SELECT
    attributes->>'agent_name'          AS agent_name,
    attributes->'pydantic_ai.all_messages' AS all_messages,
    attributes->'pydantic_ai.result'   AS result,
    start_timestamp
FROM records
WHERE attributes->>'agent_name' = 'attribute_agent'
AND start_timestamp > NOW() - INTERVAL '14 days'
{attr_where_extra}
ORDER BY start_timestamp ASC
LIMIT 10000
"""
    attribute_rows = columns_to_rows(logfire_query(attribute_sql, limit=10000))
    print(f"Got {len(attribute_rows)} attribute records")

    # ------------------------------------------------------------------
    # Build attribution lookup: fact_text → list of attribution dicts
    # (ordered oldest→newest so we consume them in call order)
    # ------------------------------------------------------------------
    # attribute_rows are already ASC ordered
    fact_attribution_queue: dict[str, list[dict]] = defaultdict(list)
    skipped_attr = 0
    for row in attribute_rows:
        all_messages = row.get("all_messages") or []
        if not all_messages:
            skipped_attr += 1
            continue
        result = get_final_result_args(all_messages)
        if not result:
            skipped_attr += 1
            continue
        prompt_text = get_first_user_text(all_messages)
        fact = parse_attribute_prompt(prompt_text)
        if not fact:
            skipped_attr += 1
            continue
        fact_attribution_queue[fact].append({
            "attributed_to_search": result.get("attributed_to_search"),
            "reasoning": result.get("reasoning", ""),
            "timestamp": row.get("start_timestamp"),
        })
    print(f"Built attribution queue for {len(fact_attribution_queue)} unique fact texts "
          f"(skipped {skipped_attr} malformed records)")

    # ------------------------------------------------------------------
    # Process decompose records (reverse to chronological order)
    # ------------------------------------------------------------------
    decompose_rows_asc = list(reversed(decompose_rows))  # oldest first
    rows = []
    unmatched_facts = 0

    for rec in decompose_rows_asc:
        all_messages = rec.get("all_messages") or []
        if not all_messages:
            continue

        prompt_text = get_first_user_text(all_messages)
        question, sampler_response = parse_decompose_prompt(prompt_text)
        if not question:
            print(f"  [WARN] Could not parse question from decompose record, skipping")
            continue

        result = get_final_result_args(all_messages)
        facts: list[str] = result.get("facts") or []

        # search_calls from eval log (match on cleaned question)
        search_calls = search_calls_by_problem.get(question, 0)

        if not facts:
            rows.append({
                "model": MODEL_LABEL,
                "problem": question,
                "has_search": False,
                "search_calls": search_calls,
                "atomic_fact": "",
                "attributed_to_search": None,
                "reasoning": "Decomposition returned no facts.",
                "sampler_response": sampler_response,
            })
            continue

        for fact in facts:
            queue = fact_attribution_queue.get(fact)
            if queue:
                attr = queue.pop(0)  # consume oldest matching attribution
                rows.append({
                    "model": MODEL_LABEL,
                    "problem": question,
                    "has_search": True,
                    "search_calls": search_calls,
                    "atomic_fact": fact,
                    "attributed_to_search": attr["attributed_to_search"],
                    "reasoning": attr["reasoning"],
                    "sampler_response": sampler_response,
                })
            else:
                # No attribution call → problem had no search results
                unmatched_facts += 1
                rows.append({
                    "model": MODEL_LABEL,
                    "problem": question,
                    "has_search": False,
                    "search_calls": search_calls,
                    "atomic_fact": fact,
                    "attributed_to_search": False,
                    "reasoning": "No search results available for this query.",
                    "sampler_response": sampler_response,
                })

    if unmatched_facts:
        print(f"  [INFO] {unmatched_facts} facts had no matching attribute_agent record "
              "(inferred as no-search problems)")

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    fieldnames = [
        "model", "problem", "has_search", "search_calls",
        "atomic_fact", "attributed_to_search", "reasoning", "sampler_response",
    ]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone! {len(rows)} rows written to {OUTPUT_CSV}")

    # Summary
    search_rows = [r for r in rows if r["has_search"]]
    attributed = sum(1 for r in search_rows if r["attributed_to_search"] is True)
    not_attributed = sum(1 for r in search_rows if r["attributed_to_search"] is False)
    no_search_rows = [r for r in rows if not r["has_search"]]
    unique_problems = len({r["problem"] for r in rows})
    print(f"\n{MODEL_LABEL}:")
    print(f"  Unique problems:          {unique_problems}")
    print(f"  Total fact rows:          {len(rows)}")
    print(f"  Facts with search ctx:    {len(search_rows)}")
    print(f"    - Attributed to search: {attributed}")
    print(f"    - Not attributed:       {not_attributed}")
    print(f"  Facts without search:     {len(no_search_rows)}")


if __name__ == "__main__":
    main()
