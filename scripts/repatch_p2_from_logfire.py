"""Re-patch bad Procedure-2 entries by fetching the original natural rollout
traces from Logfire and applying the corrected splice logic.

Background
----------
Some P2 entries in procedure2_m_patching.jsonl were generated with a splice
point at message index 2 (immediately after system+user), producing traces
with no natural reasoning prefix before the canonical search.  The correct
behaviour — when a later-hop search is the model's very first action — is to
splice the missed-hop search at the END of the existing reasoning instead.

This script:
  1. Finds every "bad" P2 entry (patch_idx <= 2 → no assistant prefix).
  2. Downloads the original sft_rollout_agent traces from Logfire, filtered
     to the affected natural questions.
  3. Reconstructs pydantic-ai ModelMessage objects from Logfire's
     pydantic_ai.all_messages attribute.
  4. For each bad entry, finds the matching natural rollout trace and
     computes the corrected splice index (after the last tool return).
  5. Re-generates the bridge thinking (mediator LLM call).
  6. Re-runs k_continuation rollouts from the corrected patched history.
  7. Removes bad entries from the JSONL and appends good replacements.

Usage
-----
    .venv/bin/python scripts/repatch_p2_from_logfire.py \\
        --p2-jsonl data/sft/musique/procedure2_m_patching.jsonl \\
        --model qwen3.5:122b \\
        --provider ollama \\
        --mediator-model gpt-oss:20b \\
        --mediator-provider ollama \\
        --k-continuation 2 \\
        --lookback-days 7 \\
        --dry-run          # preview without writing
"""

import argparse
import json
import os
import re
import sys
import time

import requests
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic import TypeAdapter
from pydantic_ai.messages import ModelMessage

from src.services.base_agent import BaseAgent
from src.services.brave_search import BraveSearchService
from scripts.run_musique_experiment import build_grader
from scripts.create_musique_sft_data import (
    MAX_RETRIES,
    messages_to_chatml,
    synthesize_arm3_bridge,
    check_patch_consistency,
    make_patch_messages,
    run_one_rollout,
    compute_rollout_attribution,
    get_missed_uncertain_hops,
    _last_tool_return_idx,
)

load_dotenv()

LOGFIRE_API_URL = "https://logfire-us.pydantic.dev"
MESSAGE_ADAPTER = TypeAdapter(list[ModelMessage])


# ---------------------------------------------------------------------------
# Identify bad entries
# ---------------------------------------------------------------------------

def is_bad_entry(msgs: list[dict]) -> bool:
    """True if the patch call has no assistant message before it."""
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            if any(tc.get("id", "").startswith("patch_") for tc in m["tool_calls"]):
                prefix = msgs[:i]
                return not any(p["role"] == "assistant" for p in prefix)
    return False


def get_patch_canonical(msgs: list[dict]) -> str:
    for m in msgs:
        if m.get("role") == "assistant" and m.get("tool_calls"):
            for tc in m["tool_calls"]:
                if tc.get("id", "").startswith("patch_"):
                    return json.loads(tc["function"]["arguments"]).get("query", "")
    return ""


def get_user_question(msgs: list[dict]) -> str:
    for m in msgs:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


# ---------------------------------------------------------------------------
# Logfire download
# ---------------------------------------------------------------------------

def fetch_rollout_traces(
    natural_questions: set[str],
    model_name: str,
    lookback_days: int,
) -> list[dict]:
    """Download sft_rollout_agent traces from Logfire for the given questions."""
    api_key = os.getenv("LOGFIRE_API_KEY")
    if not api_key:
        raise ValueError("LOGFIRE_API_KEY not set in environment")

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # Fetch a generous slice; we'll filter client-side by question text.
    query = f"""
SELECT
    attributes->'pydantic_ai.all_messages' as all_messages,
    start_timestamp
FROM records
WHERE attributes->>'agent_name' = 'sft_rollout_agent'
AND attributes->>'model_name' LIKE '%{model_name.split(":")[0]}%'
AND start_timestamp > NOW() - INTERVAL '{lookback_days} days'
ORDER BY start_timestamp DESC
LIMIT 5000
"""
    print(f"Querying Logfire for sft_rollout_agent traces (last {lookback_days} days)…")
    resp = requests.get(
        f"{LOGFIRE_API_URL}/v1/query",
        headers=headers,
        params={"sql": query, "limit": 5000},
    )
    resp.raise_for_status()
    data = resp.json()

    cols = {c["name"]: c["values"] for c in data.get("columns", [])}
    all_messages_vals = cols.get("all_messages", [])
    print(f"  Retrieved {len(all_messages_vals)} raw trace records")

    matched: list[dict] = []
    for raw_msgs in all_messages_vals:
        if not raw_msgs:
            continue
        # Find the user question in this trace
        user_q = ""
        for msg in raw_msgs:
            if msg.get("role") == "user":
                for part in msg.get("parts", []):
                    if part.get("type") == "text":
                        user_q = part.get("content", "").strip()
                        break
                if user_q:
                    break
        if not user_q:
            continue
        # Match against any of the affected questions (prefix match is fine)
        for target_q in natural_questions:
            if user_q.startswith(target_q[:60]):
                matched.append({"question": user_q, "raw_messages": raw_msgs})
                break

    print(f"  Matched {len(matched)} traces for the affected questions")
    return matched


# ---------------------------------------------------------------------------
# Reconstruct pydantic-ai messages from Logfire raw format
# ---------------------------------------------------------------------------

def reconstruct_messages(raw_messages: list[dict]) -> list[ModelMessage]:
    """Convert Logfire pydantic_ai.all_messages back into ModelMessage objects.

    Logfire stores the result of result.all_messages() serialised as JSON via
    pydantic.  We use pydantic-ai's TypeAdapter to round-trip it back.
    """
    try:
        return MESSAGE_ADAPTER.validate_python(raw_messages)
    except Exception as e:
        raise ValueError(f"Could not reconstruct ModelMessages from Logfire data: {e}") from e


# ---------------------------------------------------------------------------
# Corrected splice index
# ---------------------------------------------------------------------------

def corrected_splice_index(messages: list[ModelMessage]) -> int:
    """Place the patch after the last tool return (end-of-trace fallback)."""
    idx = _last_tool_return_idx(messages)
    return idx + 1 if idx >= 0 else len(messages)


# ---------------------------------------------------------------------------
# Main repatch logic
# ---------------------------------------------------------------------------

def repatch(args: argparse.Namespace) -> None:
    # --- Load current JSONL ---
    with open(args.p2_jsonl) as f:
        lines = f.readlines()

    good_lines: list[str] = []
    bad_entries: list[dict] = []   # {line_idx, msgs, question, canonical}

    for i, line in enumerate(lines):
        msgs = json.loads(line)["messages"]
        if is_bad_entry(msgs):
            bad_entries.append({
                "line_idx": i,
                "msgs": msgs,
                "question": get_user_question(msgs),
                "canonical": get_patch_canonical(msgs),
            })
        else:
            good_lines.append(line)

    if not bad_entries:
        print("No bad entries found — nothing to do.")
        return

    print(f"Found {len(bad_entries)} bad entries out of {len(lines)} total.")
    for e in bad_entries:
        print(f"  entry {e['line_idx']}: canonical={e['canonical']!r}  Q={e['question'][:70]!r}")

    # --- Fetch Logfire traces ---
    unique_questions = {e["question"] for e in bad_entries}
    logfire_traces = fetch_rollout_traces(unique_questions, args.model, args.lookback_days)

    if not logfire_traces:
        print("ERROR: No matching traces found in Logfire. Check --lookback-days or LOGFIRE_API_KEY.")
        sys.exit(1)

    # Group traces by question prefix for fast lookup
    traces_by_question: dict[str, list[list[ModelMessage]]] = {}
    for t in logfire_traces:
        q = t["question"]
        try:
            reconstructed = reconstruct_messages(t["raw_messages"])
        except ValueError as e:
            print(f"  Skipping trace (reconstruction failed): {e}")
            continue
        traces_by_question.setdefault(q, []).append(reconstructed)

    if args.dry_run:
        print("\n[dry-run] Would re-patch the following:")
        for q, traces in traces_by_question.items():
            print(f"  {q[:80]!r}: {len(traces)} traces available")
        print("No files written.")
        return

    # --- Initialise agents ---
    print("\nInitialising agents…")
    rollout_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[],
        agent_name="sft_repatch_rollout_agent",
    )
    search_service = BraveSearchService()
    search_tool = lambda query: search_service.search(query)
    rollout_agent_with_search = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[search_tool],
        agent_name="sft_repatch_rollout_agent",
    )
    mediator_agent = BaseAgent(
        provider_name=args.mediator_provider,
        model_name=args.mediator_model,
        tools=[],
        agent_name="sft_mediator_agent",
    )

    grader = build_grader()

    # --- Process each unique bad question ---
    new_examples: list[str] = []

    # Group bad entries by question + canonical so we process each (q, canonical) pair once
    groups: dict[tuple[str, str], list[dict]] = {}
    for e in bad_entries:
        key = (e["question"], e["canonical"])
        groups.setdefault(key, []).append(e)

    for (question, canonical), entries in groups.items():
        print(f"\nRe-patching: {question[:70]!r}")
        print(f"  Canonical query: {canonical!r}")
        print(f"  {len(entries)} bad rollout(s) to fix")

        natural_traces = traces_by_question.get(question, [])
        if not natural_traces:
            # Try prefix match (question in Logfire may have prompt suffix)
            for q, traces in traces_by_question.items():
                if q.startswith(question[:60]):
                    natural_traces = traces
                    break

        if not natural_traces:
            print(f"  WARNING: No Logfire traces found for this question — skipping")
            continue

        print(f"  Found {len(natural_traces)} natural rollout trace(s) in Logfire")

        for rollout_idx, natural_msgs in enumerate(natural_traces):
            print(f"  Rollout {rollout_idx + 1}/{len(natural_traces)}…", end=" ", flush=True)

            splice_idx = corrected_splice_index(natural_msgs)
            prior_messages = natural_msgs[:splice_idx]

            # Sanity check: prior_messages must now have at least one assistant turn
            if not any(isinstance(m, ModelResponse) for m in prior_messages):
                print("no assistant prefix even after fallback — skip")
                continue

            if not check_patch_consistency(prior_messages, canonical, question, mediator_agent):
                print("inconsistent — skip")
                continue

            bridge = synthesize_arm3_bridge(question, canonical, prior_messages, mediator_agent)
            call_msg, return_msg = make_patch_messages(canonical, bridge, search_service)
            patched_history = prior_messages + [call_msg, return_msg]

            for ci in range(args.k_continuation):
                print(f"    continuation {ci + 1}/{args.k_continuation}…", end=" ", flush=True)
                cont = run_one_rollout(
                    rollout_agent_with_search, question, "",  # gold answer not needed for continuation grading here
                    grader, message_history=patched_history,
                )
                if cont is None:
                    print("error")
                    continue
                print("correct" if cont["is_correct"] else "wrong")
                if not cont["is_correct"]:
                    continue
                new_examples.append(json.dumps({"messages": messages_to_chatml(cont["messages"])}) + "\n")
                break  # one good continuation per rollout is enough

    # --- Write results ---
    print(f"\n{'='*60}")
    print(f"Good entries kept:    {len(good_lines)}")
    print(f"Bad entries removed:  {len(bad_entries)}")
    print(f"New entries produced: {len(new_examples)}")
    print(f"Final P2 count:       {len(good_lines) + len(new_examples)}")

    if not new_examples:
        print("WARNING: No new examples produced. JSONL will only contain the existing good entries.")

    final_lines = good_lines + new_examples
    with open(args.p2_jsonl, "w") as f:
        f.writelines(final_lines)
    print(f"Written {len(final_lines)} entries to {args.p2_jsonl}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Re-patch bad P2 entries using Logfire traces")
    parser.add_argument("--p2-jsonl", default="data/sft/musique/procedure2_m_patching.jsonl")
    parser.add_argument("--model", default="qwen3.5:122b")
    parser.add_argument("--provider", default="ollama")
    parser.add_argument("--mediator-model", default="gpt-oss:20b")
    parser.add_argument("--mediator-provider", default="ollama")
    parser.add_argument("--k-continuation", type=int, default=2)
    parser.add_argument("--lookback-days", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without writing any files")
    args = parser.parse_args()
    repatch(args)


if __name__ == "__main__":
    main()
