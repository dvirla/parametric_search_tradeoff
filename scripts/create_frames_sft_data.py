"""
Collect on-policy FRAMES rollouts for cue-robustness SFT.

For each audited FRAMES question we build 7 condition inputs — the 6 cue
templates plus the plain-original reference — and run K independent rollouts of
the baseline search agent (local BM25 index, free & deterministic) per
condition. EVERY rollout is saved raw (correct + wrong); correctness/closeness
curation happens downstream in curate_frames_sft_data.py so the closeness
threshold can be calibrated after the data is accumulated.

Grading is deterministic regex (heuristic_match / relaxed_match from
scripts/regrade_regex.py) — NO LLM judge, no grader quota. This matches the
offline FRAMES regrade workflow so training and eval use one identical grader.

Conditions (question field + query template):
    verbose_plain                  original_question  PLAIN     (reference + neutral anchor)
    verbose_polite                 original_question  POLITE
    terse_plain                    text (terse anchor) PLAIN
    verbose_natural                original_question  NATURAL
    verbose_elaborate              original_question  ELABORATE
    verbose_query                  original_question  QUERY
    verbose_direct                 original_question  DIRECT
    verbose_confident_parametric   original_question  CONFIDENT_PARAMETRIC

History-prefix conditions (question uses PLAIN, cue lives in a prepended conversation --
mirrors the verbose_multiturn/verbose_searchmulti wiring in run_frames_grid_experiment.sh):
    verbose_multiturn    chit_chat_multi_turn.json   (single unrelated chit-chat conversation)
    verbose_searchmulti  search_multi_turn.json      (pool of 5 mocked-search conversations)

Usage:
    uv run python scripts/create_frames_sft_data.py \
        --dataset-path data/frames_cues/neutral_audited.jsonl \
        --index-dir data/frames_index \
        --model gpt-oss:20b --provider ollama \
        --k 5 --num-workers 4 \
        --output-dir data/sft/frames --resume
"""

import argparse
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
from pydantic_ai import Tool
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from src.services.base_agent import BaseAgent
from src.services.local_index_search import LocalIndexSearchService
from src.services.common import normalize_response, history_turns_to_messages
from src.services.qa_eval import (
    PLAIN_QUERY_TEMPLATE,
    NATURAL_QUERY_TEMPLATE,
    ELABORATE_QUERY_TEMPLATE,
    POLITE_QUERY_TEMPLATE,
    DIRECT_QUERY_TEMPLATE,
    QUERY_TEMPLATE,
    CONFIDENT_PARAMETRIC_QUERY_TEMPLATE,
)
from scripts.regrade_regex import heuristic_match, relaxed_match

MAX_RETRIES = 5

# condition -> (question field on the row, query template)
CONDITIONS = {
    "verbose_plain":     ("original_question", PLAIN_QUERY_TEMPLATE),
    "verbose_polite":    ("original_question", POLITE_QUERY_TEMPLATE),
    "terse_plain":       ("text",              PLAIN_QUERY_TEMPLATE),
    "verbose_natural":   ("original_question", NATURAL_QUERY_TEMPLATE),
    "verbose_elaborate": ("original_question", ELABORATE_QUERY_TEMPLATE),
    "verbose_query":     ("original_question", QUERY_TEMPLATE),
    "verbose_direct":    ("original_question", DIRECT_QUERY_TEMPLATE),
    "verbose_confident_parametric": ("original_question", CONFIDENT_PARAMETRIC_QUERY_TEMPLATE),
}

# History-prefix conditions -> (question field, query template, history filename under --cues-dir).
# The cue lives entirely in a prepended conversation, not the question text, so both use the plain
# passthrough template -- mirrors verbose_multiturn/verbose_searchmulti in run_frames_grid_experiment.sh.
HISTORY_CONDITIONS = {
    "verbose_multiturn":   ("original_question", PLAIN_QUERY_TEMPLATE, "chit_chat_multi_turn.json"),
    "verbose_searchmulti": ("original_question", PLAIN_QUERY_TEMPLATE, "search_multi_turn.json"),
}

ALL_CONDITIONS = {**CONDITIONS, **HISTORY_CONDITIONS}


# ---------------------------------------------------------------------------
# Grading (deterministic regex, no LLM judge) — mirrors regrade_regex.grade_row
# ---------------------------------------------------------------------------

def grade_response(gold: str, response_raw: str, relaxed: bool = False) -> bool:
    """True iff the gold answer appears in the model's free-form response.

    Applies the same normalize_response pre-clean as regrade_regex.grade_row,
    then heuristic_match (strict SQuAD-style substring). With relaxed=True, also
    accepts the word-subset match.
    """
    response = normalize_response(response_raw or "")
    strict = heuristic_match(gold or "", response)
    if strict:
        return True
    return relaxed and relaxed_match(gold or "", response)


# ---------------------------------------------------------------------------
# Trace helpers
# ---------------------------------------------------------------------------

def count_search_calls(messages: list) -> int:
    """Number of search tool calls in a pydantic-ai message trace."""
    n = 0
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == "search":
                    n += 1
    return n


def messages_to_chatml(messages: list) -> list[dict]:
    """Convert a pydantic-ai ModelMessage list to ChatML dicts for SFT."""
    chatml = []
    for msg in messages:
        if isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, SystemPromptPart):
                    chatml.append({"role": "system", "content": part.content})
                elif isinstance(part, UserPromptPart):
                    content = part.content if isinstance(part.content, str) else str(part.content)
                    chatml.append({"role": "user", "content": content})
                elif isinstance(part, ToolReturnPart):
                    chatml.append({
                        "role": "tool",
                        "tool_call_id": part.tool_call_id,
                        "content": str(part.content),
                    })
        elif isinstance(msg, ModelResponse):
            thinking_parts = [p for p in msg.parts if isinstance(p, ThinkingPart)]
            text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            call_parts = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            sections = [f"<think>\n{tp.content}\n</think>" for tp in thinking_parts]
            sections += [tp.content for tp in text_parts if tp.content]
            content = "\n".join(sections) or None
            if call_parts:
                tool_calls = []
                for tcp in call_parts:
                    args = tcp.args
                    args_json = json.dumps(args) if isinstance(args, dict) else str(args)
                    tool_calls.append({
                        "id": tcp.tool_call_id,
                        "type": "function",
                        "function": {"name": tcp.tool_name, "arguments": args_json},
                    })
                chatml.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            elif content:
                chatml.append({"role": "assistant", "content": content})
    return chatml


# ---------------------------------------------------------------------------
# Rollout execution
# ---------------------------------------------------------------------------

def run_one_rollout(agent: BaseAgent, question: str, gold: str, relaxed: bool,
                     message_history: list | None = None) -> dict | None:
    """Run one agentic rollout. Returns {output, messages, is_correct, search_calls} or None."""
    for attempt in range(MAX_RETRIES):
        try:
            result = agent.agent.run_sync(question, message_history=message_history)
            output = str(result.output)
            messages = result.all_messages()
            if message_history:
                # pydantic-ai always re-emits the passed-in message_history verbatim as a literal
                # prefix of all_messages() (see commit b64da584, "Correct search_calls inflation
                # from mocked-search history injection", which fixed this post-hoc in eval
                # analysis for the grid). Fix it AT THE SOURCE here: slice off exactly the
                # injected prefix before counting search_calls, so a fake tool_call baked into
                # verbose_searchmulti's mocked history doesn't inflate the count of genuinely-new
                # search calls made this rollout -- search_calls directly drives
                # curate_frames_sft_data.py's closeness-to-plain-ref filter.
                assert messages[:len(message_history)] == message_history, (
                    "message_history is not a literal prefix of all_messages() -- pydantic-ai's "
                    "history handling must have altered it; the search_calls slice below would "
                    "silently mis-count. Investigate before trusting this run."
                )
                new_messages = messages[len(message_history):]
            else:
                new_messages = messages
            return {
                "output": output,
                "messages": messages,
                "is_correct": grade_response(gold, output, relaxed=relaxed),
                "search_calls": count_search_calls(new_messages),
            }
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Network error after {MAX_RETRIES} attempts: {e}", flush=True)
                return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {MAX_RETRIES} attempts: {e}", flush=True)
                return None


# ---------------------------------------------------------------------------
# Progress (resume at example_id::condition granularity)
# ---------------------------------------------------------------------------

def load_progress(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def save_progress(path: str, done: set[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(sorted(done), f)
    os.replace(tmp, path)


def load_history_pool(path: str) -> list[list[dict]]:
    """A history file is either ONE conversation (flat list of turns) or a POOL of alternative
    conversations (list of such lists) -- same pool-vs-single detection as
    EvaluationService.__init__ in src/services/qa_eval.py."""
    with open(path) as f:
        loaded = json.load(f)
    if not loaded:
        raise ValueError(f"Empty history file: {path}")
    return loaded if isinstance(loaded[0], list) else [loaded]


def _check_ollama() -> None:
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except (urllib.error.URLError, OSError) as e:
        sys.exit(f"ERROR: Ollama server not reachable at localhost:11434 — {e}\n"
                 "Start Ollama before running this script.")


# ---------------------------------------------------------------------------
# Work units
# ---------------------------------------------------------------------------

def process_unit(agent, row, condition, k, relaxed, history_pools):
    """Run K rollouts for one (example_id, condition). Returns list of raw records."""
    ex_id = str(row["example_id"])
    if condition in HISTORY_CONDITIONS:
        q_field, template, _ = HISTORY_CONDITIONS[condition]
        # Seeded by example_id only (not example_id+condition), same convention as
        # EvaluationService._process_single: the same question always draws the same pool entry
        # across --resume re-invocations, and all K rollouts of this unit reuse it.
        turns = random.Random(ex_id).choice(history_pools[condition])
        message_history = history_turns_to_messages(turns)
    else:
        q_field, template = CONDITIONS[condition]
        message_history = None
    base_q = row.get(q_field) or ""
    question_cued = template.format(Question=base_q)
    gold = row.get("answer", "")

    records = []
    for i in range(k):
        r = run_one_rollout(agent, question_cued, gold, relaxed, message_history=message_history)
        if r is None:
            continue
        records.append({
            "example_id": ex_id,
            "condition": condition,
            "template_field": q_field,
            "question_cued": question_cued,
            "gold": gold,
            "k": i,
            "is_correct": r["is_correct"],
            "search_calls": r["search_calls"],
            "output": r["output"],
            "messages": messages_to_chatml(r["messages"]),
        })
    return ex_id, condition, records


def setup_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-path", default="data/frames_cues/neutral_audited.jsonl",
                   help="Audited FRAMES rows (text=terse anchor, original_question=verbose, answer=gold).")
    p.add_argument("--index-dir", default="data/frames_index")
    p.add_argument("--local-backend", default="bm25", choices=["bm25", "dense"])
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--provider", default="ollama")
    p.add_argument("--k", type=int, default=5, help="Rollouts per (question, condition).")
    p.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS.keys()),
                   choices=list(ALL_CONDITIONS.keys()))
    p.add_argument("--cues-dir", default="data/frames_cues",
                   help="Dir holding history-prefix JSON files for verbose_multiturn/verbose_searchmulti "
                        "(matches CUES_DIR in run_frames_grid_experiment.sh).")
    p.add_argument("--relaxed", action="store_true",
                   help="Grade with relaxed_match (word-subset) in addition to strict substring.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="Only the first N questions (smoke test).")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--output-dir", default="data/sft/frames")
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main():
    args = setup_args()
    if args.provider == "ollama":
        _check_ollama()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = []
    with open(args.dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if args.limit:
        rows = rows[:args.limit]
    print(f"Loaded {len(rows)} questions x {len(args.conditions)} conditions x K={args.k}", flush=True)

    # Load history pools lazily -- only for history-prefix conditions actually requested, so a
    # run of just template conditions never touches these files, and a missing/malformed one
    # fails fast at startup rather than mid-run inside a worker thread.
    history_pools = {}
    for cond in args.conditions:
        if cond in HISTORY_CONDITIONS:
            _, _, fname = HISTORY_CONDITIONS[cond]
            path = os.path.join(args.cues_dir, fname)
            history_pools[cond] = load_history_pool(path)
            print(f"Loaded history pool for {cond}: {len(history_pools[cond])} conversation(s) from {path}", flush=True)

    print(f"Initializing local index ({args.local_backend}) from {args.index_dir} ...", flush=True)
    search_service = LocalIndexSearchService(args.index_dir, backend=args.local_backend)
    agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[Tool(search_service.search)],
        output_type=str,
        use_thinking=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        agent_name="frames_sft_rollout_agent",
    )

    rollouts_path = os.path.join(args.output_dir, "rollouts.jsonl")
    progress_path = os.path.join(args.output_dir, "progress.json")
    done = load_progress(progress_path) if args.resume else set()

    units = []
    for row in rows:
        ex_id = str(row["example_id"])
        for cond in args.conditions:
            key = f"{ex_id}::{cond}"
            if key not in done:
                units.append((row, cond))
    print(f"{len(units)} (question,condition) units to run "
          f"({len(done)} already done).", flush=True)

    write_lock = threading.Lock()
    out_f = open(rollouts_path, "a")
    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
            futures = [ex.submit(process_unit, agent, row, cond, args.k, args.relaxed, history_pools)
                       for (row, cond) in units]
            for fut in as_completed(futures):
                ex_id, cond, records = fut.result()
                with write_lock:
                    for rec in records:
                        out_f.write(json.dumps(rec) + "\n")
                    out_f.flush()
                    done.add(f"{ex_id}::{cond}")
                    save_progress(progress_path, done)
                    completed += 1
                n_corr = sum(r["is_correct"] for r in records)
                print(f"[{completed}/{len(units)}] ex={ex_id} {cond}: "
                      f"{len(records)} rollouts, {n_corr} correct", flush=True)
    finally:
        out_f.close()

    print(f"\nDone. Raw rollouts -> {rollouts_path}", flush=True)


if __name__ == "__main__":
    main()
