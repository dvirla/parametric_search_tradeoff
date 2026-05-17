"""
Enrich MusiQue SFT data with no-search traces from Logfire.

For each training example (after the first --skip-first validation items), collects
hops where the model was certain (semantic_entropy == 0) and correct (num_correct > 0),
then retrieves the full Logfire traces and converts them to ChatML for SFT.

Usage:
    # With pre-downloaded traces:
    uv run python scripts/enrich_sft_nosearch.py \\
        --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \\
        --traces-json path/to/traces.json \\
        --output-dir data/sft/musique_nosearch

    # Download from Logfire directly (requires LOGFIRE_API_KEY in .env):
    uv run python scripts/enrich_sft_nosearch.py \\
        --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \\
        --output-dir data/sft/musique_nosearch \\
        --lookback-days 90 \\
        --output-traces data/sft/musique_nosearch/raw_parametric_traces.json
"""

import argparse
import json
import os
import sys

import requests
import tqdm
from dotenv import load_dotenv

load_dotenv()

LOGFIRE_API_URL = "https://logfire-us.pydantic.dev"


# ---------------------------------------------------------------------------
# Logfire download (no deduplication — we want all 5 runs per question)
# ---------------------------------------------------------------------------

def _extract_problem(all_messages: list[dict]) -> str:
    """Return the first user text part, or empty string."""
    for msg in all_messages:
        if msg.get("role") == "user":
            for part in msg.get("parts", []):
                if part.get("type") == "text":
                    return part.get("content", "").strip()
    return ""


def _extract_message_trace(all_messages: list[dict]) -> list[dict]:
    """Convert pydantic-ai all_messages to the standard trace format used by download_traces.py."""
    trace = []
    for msg in all_messages:
        message_obj = {
            "role": msg.get("role", "unknown"),
            "timestamp": msg.get("timestamp"),
            "finish_reason": msg.get("finish_reason"),
            "parts": [],
        }
        for part in msg.get("parts", []):
            ptype = part.get("type")
            if ptype == "text":
                message_obj["parts"].append({"type": "text", "content": part.get("content", "")})
            elif ptype == "thinking":
                message_obj["parts"].append({"type": "thinking", "content": part.get("content", "")})
            elif ptype == "tool_call":
                message_obj["parts"].append({
                    "type": "tool_call",
                    "tool_call_id": part.get("id"),
                    "tool_name": part.get("name"),
                    "arguments": part.get("arguments", {}),
                })
            elif ptype == "tool_call_response":
                message_obj["parts"].append({
                    "type": "tool_call_response",
                    "tool_call_id": part.get("id"),
                    "tool_name": part.get("name"),
                    "result": part.get("result"),
                })
            else:
                message_obj["parts"].append(part)
        trace.append(message_obj)
    return trace


_PAGE_SIZE = 10000  # Logfire API hard cap per request


def _fetch_page(headers: dict, agent_name: str, model_name: str, lookback_days: int, offset: int) -> list:
    """Fetch one page of raw Logfire records (up to _PAGE_SIZE rows)."""
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
AND start_timestamp > NOW() - INTERVAL '{lookback_days} days'
ORDER BY start_timestamp DESC
LIMIT {_PAGE_SIZE} OFFSET {offset}
"""
    params = {"sql": query, "limit": _PAGE_SIZE}
    response = requests.get(f"{LOGFIRE_API_URL}/v1/query", headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    columns = {col["name"]: col for col in data.get("columns", [])}
    if "all_messages" not in columns:
        return []
    n = len(columns["all_messages"]["values"])
    rows = []
    for i in range(n):
        rows.append({
            "agent_name": columns["agent_name"]["values"][i],
            "all_messages": columns["all_messages"]["values"][i],
            "result": columns.get("result", {}).get("values", [None] * n)[i],
            "start_timestamp": columns.get("start_timestamp", {}).get("values", [None] * n)[i],
            "end_timestamp": columns.get("end_timestamp", {}).get("values", [None] * n)[i],
        })
    return rows


def download_parametric_traces(
    agent_name: str,
    model_name: str,
    lookback_days: int,
    limit: int | None,
) -> list[dict]:
    """Download all parametric traces from Logfire using OFFSET pagination.

    The API caps each response at 10,000 rows, so we page until we receive
    fewer than _PAGE_SIZE rows (last page) or until `limit` is reached.
    """
    api_key = os.getenv("LOGFIRE_API_KEY")
    if not api_key:
        raise ValueError("LOGFIRE_API_KEY environment variable not set.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Querying Logfire for agent '{agent_name}', model '{model_name}', {lookback_days}d lookback...")

    all_rows: list[dict] = []
    offset = 0
    page = 0
    while True:
        page += 1
        print(f"  Fetching page {page} (offset {offset})...")
        rows = _fetch_page(headers, agent_name, model_name, lookback_days, offset)
        all_rows.extend(rows)
        print(f"    Got {len(rows)} rows (total so far: {len(all_rows)})")
        if len(rows) < _PAGE_SIZE:
            break  # last page
        if limit and len(all_rows) >= limit:
            all_rows = all_rows[:limit]
            break
        offset += _PAGE_SIZE

    print(f"Total raw rows from Logfire: {len(all_rows)}")

    traces = []
    for row in tqdm.tqdm(all_rows, desc="Extracting traces"):
        all_messages = row["all_messages"]
        if not all_messages or len(all_messages) < 2:
            continue
        try:
            problem = _extract_problem(all_messages)
            if not problem:
                continue
            message_trace = _extract_message_trace(all_messages)
            traces.append({
                "problem": problem,
                "agent_name": row["agent_name"],
                "start_timestamp": row["start_timestamp"],
                "end_timestamp": row["end_timestamp"],
                "result": row["result"],
                "message_trace": message_trace,
            })
        except Exception as e:
            print(f"  [skip] {e}")
            continue

    print(f"Extracted {len(traces)} valid traces.")
    return traces


# ---------------------------------------------------------------------------
# ChatML conversion from logfire trace format
# ---------------------------------------------------------------------------

_REAL_SEARCH_TOOLS = {"search", "search_searchterm", "brave_search"}
_FINAL_RESULT_TOOL = "final_result"


def _parse_final_result_args(args) -> dict | None:
    """Return the final_result args dict, or None if unparseable."""
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


def parametric_trace_to_chatml(message_trace: list[dict]) -> list[dict] | None:
    """Convert a parametric (no-search) logfire trace to a clean 3-message ChatML.

    Strips pydantic-ai's internal final_result tool call machinery and
    pydantic validation-error retry messages. Produces:
        [system, user, assistant(<think>...</think>\\nanswer)]

    Returns None if:
    - The trace contains real search tool calls (not a no-search trace).
    - No valid final_result call with a parseable final_answer is found.
    - The user question cannot be identified.
    """
    system_content = ""
    user_question = None

    # Collect all assistant messages that have a valid final_result call.
    # There may be multiple if pydantic retried due to JSON validation errors;
    # we want the last successful one.
    successful_calls: list[tuple[str, str]] = []  # (thinking_text, final_answer)

    for msg in message_trace:
        role = msg.get("role", "")
        parts = msg.get("parts", [])

        if role == "system":
            system_content = next((p["content"] for p in parts if p.get("type") == "text"), "")

        elif role == "user":
            text_parts = [p for p in parts if p.get("type") == "text"]
            tool_returns = [p for p in parts if p.get("type") == "tool_call_response"]
            # Skip tool_call_response messages and pydantic validation-error messages.
            if text_parts and not tool_returns:
                content = text_parts[0].get("content", "")
                # Pydantic validation errors start with a digit + "validation error"
                if "validation error" not in content.lower():
                    user_question = content

        elif role == "assistant":
            call_parts = [p for p in parts if p.get("type") == "tool_call"]
            text_parts = [p for p in parts if p.get("type") == "text"]

            # Reject the entire trace if any real search call is present.
            if any(c.get("tool_name", "") in _REAL_SEARCH_TOOLS for c in call_parts):
                return None

            thinking_parts = [p for p in parts if p.get("type") == "thinking"]
            thinking_text = "\n".join(p["content"] for p in thinking_parts if p.get("content"))

            # Path 1: model used the final_result tool call (normal structured output).
            final_calls = [c for c in call_parts if c.get("tool_name") == _FINAL_RESULT_TOOL]
            if final_calls:
                args = _parse_final_result_args(final_calls[0].get("arguments"))
                if args is not None and "final_answer" in args:
                    # Prefer the structured `reasoning` field — it contains domain reasoning
                    # even in retry cases where the thinking block discusses tool format.
                    # Fall back to the raw thinking block only if reasoning is absent.
                    reasoning = args.get("reasoning", "").strip() or thinking_text
                    successful_calls.append((reasoning, str(args["final_answer"])))
                continue  # move to next message regardless

            # Path 2: model output the HopAnswer as plain-text JSON (pydantic fallback).
            # This happens when the model ignores the tool and writes raw JSON instead.
            for tp in text_parts:
                content = (tp.get("content") or "").strip()
                # Fast-reject non-JSON text
                if not content.startswith("{"):
                    continue
                args = _parse_final_result_args(content)
                if args is not None and "final_answer" in args:
                    reasoning = args.get("reasoning", "").strip() or thinking_text
                    successful_calls.append((reasoning, str(args["final_answer"])))
                    break

    if user_question is None or not successful_calls:
        return None

    # Take the last successful call (most recent retry is the valid one).
    thinking_text, final_answer = successful_calls[-1]

    sections = []
    if thinking_text:
        sections.append(f"<think>\n{thinking_text}\n</think>")
    sections.append(final_answer)

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_question},
        {"role": "assistant", "content": "\n".join(sections)},
    ]


# ---------------------------------------------------------------------------
# Correctness check (lightweight, no LLM grader needed)
# ---------------------------------------------------------------------------

def _norm(s: str) -> str:
    return s.strip().lower()


def _answer_matches(final_answer: str, gold_answer: str) -> bool:
    fa = _norm(final_answer)
    ga = _norm(gold_answer)
    return ga in fa or fa in ga


def _extract_final_answer(result) -> str:
    """Pull final_answer out of whatever the Logfire result field contains."""
    if isinstance(result, dict):
        return result.get("final_answer", "") or ""
    if isinstance(result, str):
        return result
    return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def setup_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Logfire parametric traces to no-search SFT data."
    )
    p.add_argument("--uncertainty-json", required=True,
                   help="Output of run_musique_parametric_uncertainty.py")
    p.add_argument("--traces-json", default=None,
                   help="Pre-downloaded traces JSON (skips Logfire download)")
    p.add_argument("--output-dir", default="data/sft/musique_nosearch",
                   help="Output directory (default: data/sft/musique_nosearch)")
    p.add_argument("--skip-first", type=int, default=600,
                   help="Number of leading examples to skip (validation set, default: 600)")
    p.add_argument("--agent-name", default="musique_parametric_nemotron-3-nano_30b",
                   help="Logfire agent name LIKE pattern")
    p.add_argument("--model-name", default="nemotron-3-nano",
                   help="Logfire model name LIKE pattern")
    p.add_argument("--lookback-days", type=int, default=90,
                   help="Logfire lookback window in days (default: 90)")
    p.add_argument("--limit", type=int, default=0,
                   help="Max traces to download per Logfire query (0 = no limit)")
    p.add_argument("--output-traces", default=None,
                   help="If set, save raw downloaded traces to this path")
    return p.parse_args()


def main() -> None:
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Load uncertainty JSON, collect target hops
    # ------------------------------------------------------------------
    print(f"Loading uncertainty JSON: {args.uncertainty_json}")
    with open(args.uncertainty_json) as f:
        uncertainty_data: list[dict] = json.load(f)
    print(f"  Total examples: {len(uncertainty_data)}, skipping first {args.skip_first}")

    train_data = uncertainty_data[args.skip_first:]
    print(f"  Train examples: {len(train_data)}")

    # target_questions maps normalized question text → {gold_answer, example_id, hop_index}
    target_questions: dict[str, dict] = {}
    for ex in train_data:
        for sq in ex.get("sub_questions_results", []):
            se = sq.get("semantic_entropy", 1.0) or 0.0
            if se <= 0 and sq.get("num_correct", 0) > 0:
                key = _norm(sq["question"])
                target_questions[key] = {
                    "question": sq["question"],
                    "gold_answer": sq["gold_answer"],
                    "example_id": ex["example_id"],
                    "hop_index": sq["hop_index"],
                }

    print(f"  Target hops (SE=0, correct>0): {len(target_questions)}")

    # ------------------------------------------------------------------
    # Step 2: Load or download traces
    # ------------------------------------------------------------------
    if args.traces_json:
        print(f"\nLoading traces from: {args.traces_json}")
        with open(args.traces_json) as f:
            all_traces: list[dict] = json.load(f)
        print(f"  Loaded {len(all_traces)} traces.")
    else:
        limit = args.limit if args.limit > 0 else None
        all_traces = download_parametric_traces(
            agent_name=args.agent_name,
            model_name=args.model_name,
            lookback_days=args.lookback_days,
            limit=limit,
        )
        if args.output_traces:
            os.makedirs(os.path.dirname(os.path.abspath(args.output_traces)), exist_ok=True)
            with open(args.output_traces, "w") as f:
                json.dump(all_traces, f, indent=2)
            print(f"  Saved raw traces to: {args.output_traces}")

    # ------------------------------------------------------------------
    # Step 3: Build question → best trace lookup
    # Group all traces by normalized question, then pick the best (correct) one.
    # ------------------------------------------------------------------
    grouped: dict[str, list[dict]] = {}
    for trace in all_traces:
        key = _norm(trace.get("problem", ""))
        if key:
            grouped.setdefault(key, []).append(trace)

    print(f"\nUnique questions in traces: {len(grouped)}")

    trace_lookup: dict[str, dict] = {}
    for key, traces in grouped.items():
        if key not in target_questions:
            continue
        gold = target_questions[key]["gold_answer"]
        # Only accept a trace whose result field confirms a correct answer.
        # Since SE=0 and num_correct>0 imply all 5 runs were correct, we expect to
        # find a match among the downloaded traces. If none match (e.g. malformed
        # result field), we skip the question rather than save an unverified trace.
        for t in traces:
            fa = _extract_final_answer(t.get("result"))
            if _answer_matches(fa, gold):
                trace_lookup[key] = t
                break

    matched = len(trace_lookup)
    print(f"Matched traces for target hops: {matched}/{len(target_questions)}")

    # ------------------------------------------------------------------
    # Step 4: Convert and write
    # ------------------------------------------------------------------
    out_path = os.path.join(args.output_dir, "nosearch_sft.jsonl")
    written = 0
    skipped = 0

    with open(out_path, "w") as f_out:
        for key, hop_info in target_questions.items():
            trace = trace_lookup.get(key)
            if trace is None:
                print(f"[WARN] No trace found for: {hop_info['question'][:80]}")
                skipped += 1
                continue
            chatml = parametric_trace_to_chatml(trace["message_trace"])
            if not chatml:
                print(f"[WARN] Empty ChatML for: {hop_info['question'][:80]}")
                skipped += 1
                continue
            f_out.write(json.dumps({"messages": chatml}, ensure_ascii=False) + "\n")
            written += 1

    print(f"\n--- Done ---")
    print(f"Target hops : {len(target_questions)}")
    print(f"Written     : {written}")
    print(f"Skipped     : {skipped}")
    print(f"Output      : {out_path}")


if __name__ == "__main__":
    main()
