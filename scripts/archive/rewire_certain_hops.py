"""
Rewire SFT conversations: convert over-searches on certain hops into verbalized
parametric recall.

For each conversation in --input-jsonl:
  1. Attribute every search query to a hop via the attribution model
     (default gpt-oss:120b via ollama).
  2. For each query attributed to a CERTAIN hop (semantic_entropy <= 0 in
     --uncertainty-json AND num_correct >= --min-correct-runs), drop the
     tool_call and its tool result.
  3. Ask the bridge model (default nemotron-mini-style) to write a thinking
     passage that REPLACES the pre-search and post-search reasoning, verbalizes
     confidence in the hop's gold answer, and continues naturally.
  4. Merge the now-tool-call-less assistant turn into the following assistant
     turn, with the bridge passage opening the new <think> block.

The result is a hard-negative-converted-to-positive: traces where the model
"correctly skips" a search it parametrically knows the answer to, with explicit
verbalized confidence in the reasoning chain.

Match key from the SFT JSONL: the user message content equals natural_jsonl[
example_id].text. Examples that cannot be matched are passed through unchanged.

Usage:
  uv run python scripts/rewire_certain_hops.py \\
      --input-jsonl data/sft/musique_onpolicy/procedure1_onpolicy_sft.jsonl \\
      --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json \\
      --natural-jsonl data/musique_train_natural.jsonl \\
      --output-jsonl data/sft/musique_onpolicy/procedure1_onpolicy_sft_rewired.jsonl \\
      --attribution-model gpt-oss:120b \\
      --bridge-model nemotron3-nano:30b \\
      --resume
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import Literal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from pydantic import BaseModel

from src.services.base_agent import BaseAgent

load_dotenv()

MAX_RETRIES = 3


# ─── Pydantic models ──────────────────────────────────────────────────────────

class QueryAttribution(BaseModel):
    hop_index: int
    confidence: Literal["high", "medium", "low"]
    reasoning: str


# ─── Data structures ──────────────────────────────────────────────────────────

@dataclass
class ChatmlQuery:
    query: str
    msg_index: int          # index of assistant message in `messages`
    tool_call_id: str
    preceding_thinking: str


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_uncertainty_json(path: str) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {ex["example_id"]: ex for ex in data}


def load_natural_jsonl(path: str) -> dict[str, dict]:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[rec["example_id"]] = rec
    return out


def build_text_to_id(natural: dict[str, dict]) -> dict[str, str]:
    """Map natural-question text → example_id (the user message in SFT JSONL
    equals this text)."""
    out = {}
    for ex_id, rec in natural.items():
        text = (rec.get("text") or rec.get("natural_question") or "").strip()
        if text:
            out[text] = ex_id
    return out


# ─── Hop labels ───────────────────────────────────────────────────────────────

def get_certain_eligible_hops(
    sub_questions: list[dict], min_correct_runs: int
) -> dict[int, dict]:
    """Return {hop_index: sub_question_record} for hops that are:
       - certain (semantic_entropy <= 0)
       - AND actually known (num_correct >= min_correct_runs)

    Filtering on num_correct avoids rewiring hops where the model is
    consistently WRONG (entropy=0 because it consistently refuses or
    fabricates the same wrong answer) — those need a real search.
    """
    eligible = {}
    for sq in sub_questions:
        entropy = sq.get("semantic_entropy", 0.0) or 0.0
        n_correct = sq.get("num_correct", 0) or 0
        if entropy <= 0 and n_correct >= min_correct_runs:
            eligible[sq["hop_index"]] = sq
    return eligible


# ─── ChatML helpers ───────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def split_thinking(content: str) -> tuple[str, str]:
    """Return (thinking_text, content_after_first_think_block)."""
    if not content:
        return "", ""
    m = _THINK_RE.search(content)
    if not m:
        return "", content
    return m.group(1).strip(), content[m.end():].lstrip("\n")


def extract_chatml_queries(messages: list[dict]) -> list[ChatmlQuery]:
    out = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        thinking, _ = split_thinking(msg.get("content") or "")
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") or {}
            if fn.get("name") != "search":
                continue
            args_raw = fn.get("arguments") or "{}"
            try:
                parsed = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                query = (parsed or {}).get("query", "") if isinstance(parsed, dict) else ""
            except Exception:
                query = ""
            if not query:
                continue
            out.append(ChatmlQuery(
                query=query,
                msg_index=i,
                tool_call_id=tc.get("id", ""),
                preceding_thinking=thinking,
            ))
    return out


# ─── Prompts ──────────────────────────────────────────────────────────────────

ATTRIBUTION_PROMPT = """You are attributing a search query issued by an AI agent to one of the sub-questions (hops) that make up a multi-hop question.

Aggregate question: {aggregate_question}

Sub-questions (hops):
{sub_questions_text}

Agent's thinking before the search query:
{thinking}

Search query: {query}

Which sub-question (hop) is this search query primarily trying to answer?
Return the 0-based hop_index (0, 1, 2, ...) or -1 if the query is about the aggregate question or doesn't clearly target a specific sub-question.
"""


BRIDGE_PROMPT = """You are rewriting part of an AI agent's chain-of-thought to remove an unnecessary web search. The agent already knows the answer to the sub-question from its parametric knowledge — searching was wasted effort.

Multi-hop question being answered: "{aggregate_question}"

Removed search query: "{removed_query}"
This was meant to answer sub-question: "{hop_question}"
The correct answer (which the agent knows from training): "{hop_gold_answer}"

The agent's thinking JUST BEFORE the search (it currently justifies searching — REWRITE this):
---
{m_thinking}
---

The agent's thinking AFTER receiving the search results (it currently references those results — REWRITE this):
---
{n_thinking}
---

Write a SINGLE new first-person thinking passage that REPLACES both of the above. It must:
1. Preserve any context from the pre-search reasoning that is still relevant (e.g. references to prior hops).
2. Verbalize confidence — state explicitly that you already know "{hop_question}" → "{hop_gold_answer}", as recall, not retrieval.
3. Continue naturally toward whatever the post-search reasoning was doing.
4. NOT mention searching, search results, look-ups, or external sources of any kind.
5. Stay concise (2–5 sentences is usually enough).

Output ONLY the new thinking text. No <think> tags. No quotes. No preamble.
"""


# ─── LLM calls ────────────────────────────────────────────────────────────────

def attribute_query(
    qctx: ChatmlQuery,
    sub_questions: list[dict],
    aggregate_question: str,
    attribution_agent: BaseAgent,
) -> QueryAttribution:
    sub_q_text = "\n".join(
        f"  Hop {sq['hop_index']}: {sq['question']}" for sq in sub_questions
    )
    prompt = ATTRIBUTION_PROMPT.format(
        aggregate_question=aggregate_question,
        sub_questions_text=sub_q_text,
        thinking=qctx.preceding_thinking[:500],
        query=qctx.query,
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = attribution_agent.run(prompt)
            out = result.output
            if isinstance(out, QueryAttribution):
                return out
            if isinstance(out, dict):
                return QueryAttribution(**out)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Attribution failed for query {qctx.query!r}: {e}")
    return QueryAttribution(hop_index=-1, confidence="low", reasoning="attribution failed")


def generate_bridge(
    aggregate_question: str,
    removed_query: str,
    hop_question: str,
    hop_gold_answer: str,
    m_thinking: str,
    n_thinking: str,
    bridge_agent: BaseAgent,
) -> str:
    prompt = BRIDGE_PROMPT.format(
        aggregate_question=aggregate_question,
        removed_query=removed_query,
        hop_question=hop_question,
        hop_gold_answer=hop_gold_answer,
        m_thinking=m_thinking.strip() or "(no prior thinking)",
        n_thinking=n_thinking.strip() or "(no post-search thinking)",
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = bridge_agent.run(prompt)
            text = str(result.output).strip()
            text = re.sub(r"^<think>\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*</think>\s*$", "", text, flags=re.IGNORECASE)
            return text.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Bridge generation failed for hop {hop_question!r}: {e}")
    return ""


# ─── Rewiring ─────────────────────────────────────────────────────────────────

CONFIDENCE_RANK = {"high": 2, "medium": 1, "low": 0}


def rewire_conversation(
    messages: list[dict],
    sub_questions: list[dict],
    aggregate_question: str,
    attribution_agent: BaseAgent,
    bridge_agent: BaseAgent,
    min_confidence: str,
    min_correct_runs: int,
) -> tuple[list[dict], int]:
    """Returns (rewired_messages, n_removals)."""
    threshold = CONFIDENCE_RANK[min_confidence]

    queries = extract_chatml_queries(messages)
    if not queries:
        return messages, 0

    eligible = get_certain_eligible_hops(sub_questions, min_correct_runs)
    if not eligible:
        return messages, 0

    removable: list[tuple[ChatmlQuery, QueryAttribution]] = []
    for q in queries:
        attr = attribute_query(q, sub_questions, aggregate_question, attribution_agent)
        if attr.hop_index < 0:
            continue
        if attr.hop_index not in eligible:
            continue
        if CONFIDENCE_RANK.get(attr.confidence, 0) < threshold:
            continue
        removable.append((q, attr))

    if not removable:
        return messages, 0

    # Process from highest msg_index to lowest so earlier indices stay valid
    removable.sort(key=lambda x: x[0].msg_index, reverse=True)

    messages = [dict(m) for m in messages]  # shallow copy
    n_removed = 0

    for q, attr in removable:
        m_idx = q.msg_index
        if m_idx + 2 >= len(messages):
            continue  # search was last action — no next assistant to merge into
        M = messages[m_idx]
        TR = messages[m_idx + 1]
        N = messages[m_idx + 2]
        if TR.get("role") != "tool" or N.get("role") != "assistant":
            continue
        tool_calls = M.get("tool_calls") or []
        if len(tool_calls) != 1:
            # multi-tool turn — partial removal would leave dangling state; skip
            continue
        if tool_calls[0].get("id") != q.tool_call_id:
            continue

        hop = eligible[attr.hop_index]
        m_thinking, _ = split_thinking(M.get("content") or "")
        n_thinking, n_after = split_thinking(N.get("content") or "")

        bridge = generate_bridge(
            aggregate_question=aggregate_question,
            removed_query=q.query,
            hop_question=hop["question"],
            hop_gold_answer=hop.get("gold_answer", ""),
            m_thinking=m_thinking,
            n_thinking=n_thinking,
            bridge_agent=bridge_agent,
        )
        if not bridge:
            continue  # bridge failed — leave this search in place

        new_content = f"<think>\n{bridge}\n</think>"
        if n_after.strip():
            new_content += "\n" + n_after.strip()

        merged: dict = {
            "role": "assistant",
            "content": new_content,
            "tool_calls": N.get("tool_calls") or [],
        }
        if N.get("tool_call_id"):
            merged["tool_call_id"] = N["tool_call_id"]

        messages = messages[:m_idx] + [merged] + messages[m_idx + 3:]
        n_removed += 1

    return messages, n_removed


# ─── Main ─────────────────────────────────────────────────────────────────────

def setup_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Rewire SFT conversations: drop searches on certain hops "
                    "and replace with verbalized parametric recall."
    )
    p.add_argument("--input-jsonl", required=True,
                   help="ChatML JSONL produced by create_musique_sft_onpolicy.py "
                        "(or any compatible SFT data file).")
    p.add_argument("--uncertainty-json", required=True,
                   help="Output of run_musique_parametric_uncertainty.py — "
                        "supplies per-hop semantic_entropy and num_correct.")
    p.add_argument("--natural-jsonl", required=True,
                   help="Output of paraphrase_musique_natural.py — used to "
                        "recover example_id from the SFT user-message text.")
    p.add_argument("--output-jsonl", required=True,
                   help="Where to write the rewired JSONL.")
    p.add_argument("--attribution-model", default="gpt-oss:120b",
                   help="Model used to attribute search queries to hops.")
    p.add_argument("--attribution-provider", default="ollama")
    p.add_argument("--attribution-ollama-url", default=None,
                   help="Override OLLAMA_BASE_URL for the attribution model.")
    p.add_argument("--bridge-model", default="nemotron3-nano:30b",
                   help="Model used to write the bridge thinking passage.")
    p.add_argument("--bridge-provider", default="ollama")
    p.add_argument("--bridge-ollama-url", default=None,
                   help="Override OLLAMA_BASE_URL for the bridge model.")
    p.add_argument("--min-confidence", default="medium",
                   choices=["high", "medium", "low"],
                   help="Minimum attribution confidence required to rewire a "
                        "query (default: medium).")
    p.add_argument("--min-correct-runs", type=int, default=3,
                   help="Hop is eligible only if the parametric probe got at "
                        "least this many runs correct (out of 5). Default 3 "
                        "guards against entropy=0 from consistent refusal.")
    p.add_argument("--resume", action="store_true",
                   help="Append to existing --output-jsonl, skipping the first "
                        "N input lines already present in the output.")
    return p.parse_args()


def main() -> None:
    args = setup_args()

    print(f"Loading uncertainty JSON: {args.uncertainty_json}")
    uncertainty_data = load_uncertainty_json(args.uncertainty_json)
    print(f"  {len(uncertainty_data)} examples with uncertainty labels")

    print(f"Loading natural paraphrases: {args.natural_jsonl}")
    natural = load_natural_jsonl(args.natural_jsonl)
    text_to_id = build_text_to_id(natural)
    print(f"  {len(text_to_id)} natural-question texts indexed")

    print(f"Attribution agent: {args.attribution_model} via {args.attribution_provider}")
    attribution_agent = BaseAgent(
        provider_name=args.attribution_provider,
        model_name=args.attribution_model,
        output_type=QueryAttribution,
        ollama_base_url=args.attribution_ollama_url,
        use_thinking=False,
    )
    print(f"Bridge agent:      {args.bridge_model} via {args.bridge_provider}")
    bridge_agent = BaseAgent(
        provider_name=args.bridge_provider,
        model_name=args.bridge_model,
        output_type=str,
        ollama_base_url=args.bridge_ollama_url,
        use_thinking=False,
    )

    n_already_written = 0
    if args.resume and os.path.exists(args.output_jsonl):
        with open(args.output_jsonl) as f:
            for _ in f:
                n_already_written += 1
        print(f"  Resume: skipping first {n_already_written} input lines")

    out_f = open(args.output_jsonl, "a" if args.resume else "w")

    stats = {
        "total": 0, "rewired": 0, "removals": 0,
        "unmatched": 0, "no_eligible_hops": 0, "skipped_resume": 0,
    }

    try:
        with open(args.input_jsonl) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                stats["total"] += 1
                if i < n_already_written:
                    stats["skipped_resume"] += 1
                    continue

                example = json.loads(line)
                messages = example["messages"]

                user_msg = next((m for m in messages if m.get("role") == "user"), None)
                if not user_msg:
                    out_f.write(json.dumps(example) + "\n")
                    out_f.flush()
                    continue
                user_text = (user_msg.get("content") or "").strip()
                ex_id = text_to_id.get(user_text)
                if ex_id is None:
                    stats["unmatched"] += 1
                    out_f.write(json.dumps(example) + "\n")
                    out_f.flush()
                    continue

                u_record = uncertainty_data.get(ex_id)
                if u_record is None:
                    stats["unmatched"] += 1
                    out_f.write(json.dumps(example) + "\n")
                    out_f.flush()
                    continue

                sub_questions = u_record.get("sub_questions_results") or []
                aggregate_question = (
                    u_record.get("aggregate_question") or user_text
                )

                eligible = get_certain_eligible_hops(sub_questions, args.min_correct_runs)
                if not eligible:
                    stats["no_eligible_hops"] += 1
                    out_f.write(json.dumps(example) + "\n")
                    out_f.flush()
                    continue

                try:
                    new_messages, n_removed = rewire_conversation(
                        messages=messages,
                        sub_questions=sub_questions,
                        aggregate_question=aggregate_question,
                        attribution_agent=attribution_agent,
                        bridge_agent=bridge_agent,
                        min_confidence=args.min_confidence,
                        min_correct_runs=args.min_correct_runs,
                    )
                except Exception as e:
                    print(f"  [error] example {i} (id {ex_id}): {e}")
                    out_f.write(json.dumps(example) + "\n")
                    out_f.flush()
                    continue

                if n_removed > 0:
                    stats["rewired"] += 1
                    stats["removals"] += n_removed
                    example["messages"] = new_messages

                out_f.write(json.dumps(example) + "\n")
                out_f.flush()

                if stats["total"] % 25 == 0:
                    print(
                        f"  [{stats['total']}] rewired={stats['rewired']} "
                        f"removals={stats['removals']} "
                        f"unmatched={stats['unmatched']} "
                        f"no_eligible={stats['no_eligible_hops']}"
                    )
    finally:
        out_f.close()

    print("\n─── Summary ────────────────────────────────────")
    print(f"  Total input examples:    {stats['total']}")
    print(f"  Skipped (resume):        {stats['skipped_resume']}")
    print(f"  Rewired (≥1 removal):    {stats['rewired']}")
    print(f"  Total searches removed:  {stats['removals']}")
    print(f"  Pass-through (unmatched):{stats['unmatched']}")
    print(f"  Pass-through (no certain hops): {stats['no_eligible_hops']}")
    print(f"  Output:                  {args.output_jsonl}")


if __name__ == "__main__":
    main()
