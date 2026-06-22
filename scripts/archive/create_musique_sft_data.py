"""
Create MusiQue SFT dataset via K=5 rejection sampling — two procedures.

Procedure 1 — Calibration-Filtered Natural SFT:
    K=5 rollouts on the natural-paraphrased question; keep traces where the
    final answer is correct AND all uncertain hops (semantic_entropy > 0) were
    searched. No synthesis — trace is kept verbatim. Optional R3 (--r3-strict):
    also require no certain hop was over-searched.

Procedure 2 — Targeted M-Patching:
    K=5 rollouts on the natural question; for each rollout that is correct but
    has ≥1 missed uncertain hop: find the canonical query for that hop in a
    formal trace, verify the natural trace's context is consistent with the
    patch, splice the search at the correct reasoning position, run continuation
    rollouts, keep if correct and no new M-violations for that hop.

    Each (rollout, missed-hop) pair is patched independently, so multi-M rollouts
    yield multiple training examples (one per missed uncertain hop).

    Formal rollout is only run when at least one P2-eligible natural rollout
    exists (budget optimisation).

Grading for natural questions checks whether the gold answer is explicitly
stated in the response (open-ended criterion, not format-matching).

Prerequisites:
    1. run_musique_parametric_uncertainty.py --staleness_csv data/musique_train_staleness.csv
       → results/musique_parametric_train/musique_parametric_uncertainty_<model>.json
    2. paraphrase_musique_natural.py --source <above> --output data/musique_train_natural.jsonl

Usage:
    uv run python scripts/create_musique_sft_data.py \\
        --uncertainty-json results/musique_parametric_train/musique_parametric_uncertainty_<model>.json \\
        --natural-jsonl data/musique_train_natural.jsonl \\
        --model <qwen-model> \\
        --provider ollama \\
        --k 5 \\
        --output-dir data/sft/musique \\
        --resume
"""

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import httpx
from pydantic import BaseModel
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
from src.services.brave_search import BraveSearchService
from scripts.archive.run_musique_experiment import build_grader
from src.services.qa_eval import STANDARD_GRADER_TEMPLATE

MAX_RETRIES = 5


# ---------------------------------------------------------------------------
# Pydantic models (mirroring analyze_parametric_search_interplay.py)
# ---------------------------------------------------------------------------

class QueryAttribution(BaseModel):
    hop_index: int  # 0-based; -1 = aggregate/general (not clearly a specific hop)
    confidence: str  # "high" | "medium" | "low"
    reasoning: str


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class QueryWithContext:
    query: str
    preceding_thinking: str
    query_index: int
    turn_index: int
    search_result_text: str = ""
    search_result_snippets: list = field(default_factory=list)
    tool_call_id: str = ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_uncertainty_json(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def load_natural_jsonl(path: str) -> dict[str, dict]:
    """Return {example_id: record} from the natural-paraphrase JSONL."""
    result = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                result[rec["example_id"]] = rec
    return result


# ---------------------------------------------------------------------------
# Grading — open-ended natural-answer criterion
# ---------------------------------------------------------------------------

NATURAL_GRADER_TEMPLATE = """
Judge whether the following [response] correctly answers the [question].

[question]: {question}

[correct answer]: {gold_answer}

[response]: {response}

The response is written in natural prose (2-4 sentences). The correct answer is a specific factual value.

Does the response explicitly state the correct answer (or an unambiguous equivalent) somewhere in its text as the answer to the question?

correct: Answer "yes" if the gold answer is explicitly stated in the response as the answer. Answer "no" otherwise.
""".strip()


def grade_natural_response(grader, question: str, gold_answer: str, response: str) -> bool:
    prompt = STANDARD_GRADER_TEMPLATE.format(
        question=question, correct_answer=gold_answer, response=response,
    )
    prompt_messages = [grader._pack_message(content=prompt, role="user")]
    sampler_response = grader(prompt_messages)
    grading_text = str(sampler_response.response_text.output)
    match = re.search(r"correct:\s*(yes|no)", grading_text, re.IGNORECASE)
    return match.group(1).lower() == "yes" if match else False


# ---------------------------------------------------------------------------
# Query extraction from pydantic-ai messages
# ---------------------------------------------------------------------------

def extract_queries_from_messages(messages: list) -> list[QueryWithContext]:
    """Extract all search queries with their preceding thinking and search results."""
    results = []
    query_index = 0

    for turn_idx, msg in enumerate(messages):
        if not isinstance(msg, ModelResponse):
            continue

        thinking_text = "\n".join(
            p.content for p in msg.parts if isinstance(p, (ThinkingPart, TextPart))
        )

        result_map: dict[str, tuple[str, list[str]]] = {}
        if turn_idx + 1 < len(messages):
            next_msg = messages[turn_idx + 1]
            if isinstance(next_msg, ModelRequest):
                for part in next_msg.parts:
                    if isinstance(part, ToolReturnPart):
                        content = str(part.content)
                        snippets = [s.strip() for s in content.split("\n") if s.strip()]
                        result_map[part.tool_call_id] = (content, snippets)

        for part in msg.parts:
            if not (isinstance(part, ToolCallPart) and part.tool_name == "search"):
                continue
            args = part.args
            if isinstance(args, dict):
                query = args.get("query", "")
            else:
                try:
                    query = json.loads(str(args)).get("query", "")
                except Exception:
                    query = str(args)
            if not query:
                continue
            result_entry = result_map.get(part.tool_call_id, ("", []))
            results.append(QueryWithContext(
                query=query,
                preceding_thinking=thinking_text,
                query_index=query_index,
                turn_index=turn_idx,
                search_result_text=result_entry[0],
                search_result_snippets=result_entry[1],
                tool_call_id=part.tool_call_id,
            ))
            query_index += 1

    return results


# ---------------------------------------------------------------------------
# LLM-based hop attribution (same prompt as analyze_parametric_search_interplay.py)
# ---------------------------------------------------------------------------

ATTRIBUTION_PROMPT = """\
You are attributing a search query issued by an AI agent to one of the sub-questions \
(hops) that make up a multi-hop question.

Aggregate question: {aggregate_question}

Sub-questions (hops):
{sub_questions_text}

Agent's thinking before the search query:
{thinking}

Search query: {query}

Which sub-question (hop) is this search query primarily trying to answer?
Return the 0-based hop_index (0, 1, 2, ...) or -1 if the query is about the \
aggregate question or doesn't clearly target a specific sub-question.\
"""


def attribute_queries(
    query_ctxs: list[QueryWithContext],
    sub_questions: list[dict],
    aggregate_question: str,
    attribution_agent: BaseAgent,
) -> list[QueryAttribution]:
    """Attribute each search query to a hop index via LLM."""
    sub_q_text = "\n".join(
        f"  Hop {sq['hop_index']}: {sq['question']}" for sq in sub_questions
    )
    attributions = []
    for qctx in query_ctxs:
        prompt = ATTRIBUTION_PROMPT.format(
            aggregate_question=aggregate_question,
            sub_questions_text=sub_q_text,
            thinking=qctx.preceding_thinking[:500],
            query=qctx.query,
        )
        for attempt in range(MAX_RETRIES):
            try:
                result = attribution_agent.run(prompt)
                attributions.append(result.output)
                break
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"    Attribution failed: {e} — assigning hop -1")
                    attributions.append(QueryAttribution(
                        hop_index=-1, confidence="low", reasoning=f"Error: {e}"
                    ))
    return attributions


def compute_rollout_attribution(
    messages: list,
    sub_questions: list[dict],
    aggregate_question: str,
    attribution_agent: BaseAgent,
) -> tuple[list[QueryWithContext], list[QueryAttribution]]:
    """Extract queries and run LLM attribution for a rollout. Returns (qctxs, attrs)."""
    qctxs = extract_queries_from_messages(messages)
    if not qctxs:
        return [], []
    attrs = attribute_queries(qctxs, sub_questions, aggregate_question, attribution_agent)
    return qctxs, attrs


# ---------------------------------------------------------------------------
# Uncertainty labels and hop classification
# ---------------------------------------------------------------------------

def get_hop_uncertainty_labels(sub_questions: list[dict]) -> dict[int, str]:
    """Return {hop_index: 'uncertain'|'certain'} based on semantic_entropy > 0."""
    labels = {}
    for sq in sub_questions:
        hi = sq["hop_index"]
        entropy = sq.get("semantic_entropy", 0.0) or 0.0
        labels[hi] = "uncertain" if entropy > 0 else "certain"
    return labels


def classify_hops_from_attributions(
    attributions: list[QueryAttribution],
    sub_questions: list[dict],
    uncertainty_labels: dict[int, str],
) -> dict[int, str]:
    """Return {hop_index: 'E'|'CP'|'PR'|'M'} given pre-computed attributions.

    E  = uncertain + searched (correct search)
    CP = certain   + not searched (correct parametric skip)
    PR = certain   + searched (over-search)
    M  = uncertain + not searched (missed search)
    """
    covered = {attr.hop_index for attr in attributions if attr.hop_index >= 0}
    result = {}
    for sq in sub_questions:
        hi = sq["hop_index"]
        uncertain = uncertainty_labels.get(hi, "uncertain") == "uncertain"
        searched = hi in covered
        if uncertain and searched:
            result[hi] = "E"
        elif not uncertain and not searched:
            result[hi] = "CP"
        elif not uncertain and searched:
            result[hi] = "PR"
        else:
            result[hi] = "M"
    return result


def get_missed_uncertain_hops(
    attributions: list[QueryAttribution],
    sub_questions: list[dict],
    uncertainty_labels: dict[int, str],
) -> list[int]:
    """Return hop indices that are uncertain (entropy > 0) AND were not searched."""
    covered = {attr.hop_index for attr in attributions if attr.hop_index >= 0}
    return [
        sq["hop_index"] for sq in sub_questions
        if uncertainty_labels.get(sq["hop_index"], "uncertain") == "uncertain"
        and sq["hop_index"] not in covered
    ]


# ---------------------------------------------------------------------------
# Mediator LLM — bridge thinking and consistency check
# ---------------------------------------------------------------------------

ARM3_BRIDGE_PROMPT = """\
You are writing a short connecting reasoning paragraph for an AI agent solving a multi-hop question.

The agent is answering: "{natural_question}"

Conversation so far:
{prior_context}

The agent now needs to search: "{canonical_query}"

Write 2-4 sentences that:
1. Acknowledge what has been established from prior searches.
2. Naturally conclude that the agent should now search for "{canonical_query}".

Important: Reference only entities and facts already present in the conversation context above.
Do NOT introduce new entities or mention formal sub-question wording.
Return only the reasoning text. No preamble, no quotes.\
"""

CONSISTENCY_CHECK_PROMPT = """\
An AI agent has been reasoning about the following question:
"{natural_question}"

Here is a summary of the agent's conversation so far:
{prior_context}

The agent is now being asked to search for: "{canonical_query}"

Does this search query follow naturally from what the agent has established so far?
Answer "yes" if the entities and facts referenced in the search query are consistent \
with the trace (i.e., the trace already identified the correct intermediate answer \
that makes this search meaningful).
Answer "no" if the trace reached a different intermediate answer, making this search \
a non-sequitur (e.g. the trace found entity X but the query asks about entity Y).

consistent: yes or no\
"""


def _format_prior_context(messages: list) -> str:
    """Summarise the last few assistant turns and search results for the mediator."""
    items = []
    for msg in messages:
        if isinstance(msg, ModelResponse):
            for part in msg.parts:
                if isinstance(part, TextPart) and part.content.strip():
                    items.append(f"[Reasoning]: {part.content.strip()[:300]}")
                elif isinstance(part, ToolCallPart) and part.tool_name == "search":
                    args = part.args
                    q = args.get("query", "") if isinstance(args, dict) else str(args)
                    items.append(f"[Searched]: {q}")
        elif isinstance(msg, ModelRequest):
            for part in msg.parts:
                if isinstance(part, ToolReturnPart):
                    items.append(f"[Result snippet]: {str(part.content)[:250]}")
    return "\n".join(items[-12:]) if items else "(no prior context)"


def synthesize_arm3_bridge(
    natural_question: str,
    canonical_query: str,
    prior_messages: list,
    mediator_agent: BaseAgent,
) -> str:
    """Generate a connecting reasoning paragraph before the patched search call."""
    context = _format_prior_context(prior_messages)
    prompt = ARM3_BRIDGE_PROMPT.format(
        natural_question=natural_question,
        prior_context=context,
        canonical_query=canonical_query,
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = mediator_agent.run(prompt)
            return result.output.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Mediator (bridge) failed: {e} — using empty bridge")
                return ""


def check_patch_consistency(
    prior_messages: list,
    canonical_query: str,
    natural_question: str,
    mediator_agent: BaseAgent,
) -> bool:
    """Return True if the canonical_query is consistent with the natural trace's context.

    Returns False (reject the patch) when the trace established the wrong intermediate
    entity and the canonical_query would be a non-sequitur.
    """
    context = _format_prior_context(prior_messages)
    prompt = CONSISTENCY_CHECK_PROMPT.format(
        natural_question=natural_question,
        prior_context=context,
        canonical_query=canonical_query,
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = mediator_agent.run(prompt)
            text = str(result.output).lower()
            match = re.search(r"consistent:\s*(yes|no)", text, re.IGNORECASE)
            return match.group(1).lower() == "yes" if match else True  # default allow
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Consistency check failed: {e} — assuming consistent")
                return True


# ---------------------------------------------------------------------------
# ChatML conversion
# ---------------------------------------------------------------------------

def messages_to_chatml(messages: list) -> list[dict]:
    """Convert pydantic-ai ModelMessage list to ChatML dicts for SFT."""
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
# Trace patching helpers
# ---------------------------------------------------------------------------

def _last_tool_return_idx(messages: list) -> int:
    last = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) and any(
            isinstance(p, ToolReturnPart) for p in msg.parts
        ):
            last = i
    return last


def find_splice_index(
    messages: list,
    missed_hop_idx: int,
    query_ctxs: list[QueryWithContext],
    attributions: list[QueryAttribution],
) -> int:
    """Return exclusive end index for slicing messages before the patch.

    Finds the turn_index of the first search attributed to any hop with
    hop_index > missed_hop_idx (reasoning-order proxy). Slicing messages[:result]
    leaves everything up to and including the preceding tool return.
    Falls back to after the last tool return if no later-hop search exists.
    """
    attr_by_qidx = {qctx.query_index: attr for qctx, attr in zip(query_ctxs, attributions)}
    sorted_queries = sorted(query_ctxs, key=lambda q: (q.turn_index, q.query_index))

    for qctx in sorted_queries:
        attr = attr_by_qidx.get(qctx.query_index)
        if attr is not None and attr.hop_index > missed_hop_idx:
            candidate = qctx.turn_index
            # Only splice here if the prefix contains at least one assistant turn.
            # When missed_hop_idx=0 and the first action is already a later-hop search,
            # candidate==2 and the prefix is [system, user] — an unnatural starting
            # point. Fall through to the end-of-trace fallback instead, which places
            # the patch after all existing reasoning.
            if any(isinstance(m, ModelResponse) for m in messages[:candidate]):
                return candidate
            break  # prefix would be empty — use fallback below

    idx = _last_tool_return_idx(messages)
    return idx + 1 if idx >= 0 else len(messages)


def make_patch_messages(
    canonical_query: str,
    bridge_thinking: str,
    search_service: BraveSearchService,
) -> tuple[ModelResponse, ModelRequest]:
    """Build a synthetic (thinking + tool-call) + tool-return pair for the missed hop."""
    call_id = f"patch_{uuid.uuid4().hex[:8]}"
    parts = []
    if bridge_thinking:
        parts.append(ThinkingPart(content=bridge_thinking))
    parts.append(ToolCallPart(
        tool_name="search",
        args={"query": canonical_query},
        tool_call_id=call_id,
    ))
    call_msg = ModelResponse(parts=parts)

    results = search_service.search(canonical_query, max_results=5)
    results_text = "\n".join(str(r) for r in results)
    return_msg = ModelRequest(parts=[
        ToolReturnPart(tool_name="search", content=results_text, tool_call_id=call_id)
    ])
    return call_msg, return_msg


# ---------------------------------------------------------------------------
# Rollout execution
# ---------------------------------------------------------------------------

def run_one_rollout(
    agent: BaseAgent,
    question: str,
    gold_answer: str,
    grader,
    *,
    message_history: list | None = None,
) -> dict | None:
    """Run one agentic rollout. Returns {output, messages, is_correct} or None on failure."""
    for attempt in range(MAX_RETRIES):
        try:
            if message_history is not None:
                result = agent.agent.run_sync(None, message_history=message_history)
            else:
                result = agent.agent.run_sync(question)
            output = result.output
            is_correct = grade_natural_response(grader, question, gold_answer, str(output))
            return {
                "output": str(output),
                "messages": result.all_messages(),
                "is_correct": is_correct,
            }
        except (httpx.ConnectError, httpx.RemoteProtocolError, ConnectionError) as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Network error after {MAX_RETRIES} attempts: {e}")
                return None
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Failed after {MAX_RETRIES} attempts: {e}")
                return None


def run_k_rollouts(
    agent: BaseAgent, question: str, gold_answer: str, grader, k: int
) -> list[dict]:
    """Run k independent rollouts; return all results (including incorrect ones)."""
    rollouts = []
    for i in range(k):
        print(f"      rollout {i + 1}/{k} ...", end=" ", flush=True)
        r = run_one_rollout(agent, question, gold_answer, grader)
        if r is not None:
            rollouts.append(r)
            print("correct" if r["is_correct"] else "wrong")
        else:
            print("error")
    return rollouts


# ---------------------------------------------------------------------------
# Canonical query extraction from formal trace
# ---------------------------------------------------------------------------

def find_canonical_query(
    missed_hop_idx: int,
    formal_qctxs: list[QueryWithContext],
    formal_attrs: list[QueryAttribution],
) -> str | None:
    """Return the first search query in the formal trace attributed to missed_hop_idx."""
    for qctx, attr in zip(formal_qctxs, formal_attrs):
        if attr.hop_index == missed_hop_idx:
            return qctx.query
    return None


def acquire_formal_trace(
    agent: BaseAgent,
    formal_q: str,
    gold_answer: str,
    grader,
    sub_questions: list[dict],
    aggregate_question: str,
    attribution_agent: BaseAgent,
) -> tuple[list[QueryWithContext], list[QueryAttribution]] | None:
    """Run one formal rollout and return its (query_ctxs, attributions).

    Returns None if the rollout fails entirely. Correctness is not required —
    we only need the search queries for canonical query extraction.
    """
    r = run_one_rollout(agent, formal_q, gold_answer, grader)
    if r is None:
        return None
    qctxs, attrs = compute_rollout_attribution(
        r["messages"], sub_questions, aggregate_question, attribution_agent
    )
    return qctxs, attrs


# ---------------------------------------------------------------------------
# Procedure 1 — Calibration-Filtered Natural SFT
# ---------------------------------------------------------------------------

def process_procedure1(
    natural_rollouts: list[dict],
    rollout_attribution_data: list[tuple[list[QueryWithContext], list[QueryAttribution]]],
    sub_questions: list[dict],
    uncertainty_labels: dict[int, str],
    r3_strict: bool = False,
) -> list[dict]:
    """Keep correct natural rollouts where all uncertain hops were searched.

    R1: answer correct
    R2: no uncertain hop was missed (no M in hop classification)
    R3 (optional, r3_strict=True): no certain hop was over-searched (no PR)
    """
    examples = []
    for r, (_, attrs) in zip(natural_rollouts, rollout_attribution_data):
        if not r["is_correct"]:
            continue
        hop_cls = classify_hops_from_attributions(attrs, sub_questions, uncertainty_labels)
        if any(v == "M" for v in hop_cls.values()):
            continue
        if r3_strict and any(v == "PR" for v in hop_cls.values()):
            continue
        examples.append({"messages": messages_to_chatml(r["messages"])})
    return examples


# ---------------------------------------------------------------------------
# Procedure 2 — Targeted M-Patching
# ---------------------------------------------------------------------------

def process_procedure2(
    natural_rollouts: list[dict],
    rollout_attribution_data: list[tuple[list[QueryWithContext], list[QueryAttribution]]],
    sub_questions: list[dict],
    uncertainty_labels: dict[int, str],
    formal_qctxs: list[QueryWithContext],
    formal_attrs: list[QueryAttribution],
    natural_question: str,
    formal_question: str,
    gold_answer: str,
    agent: BaseAgent,
    search_service: BraveSearchService,
    mediator_agent: BaseAgent,
    attribution_agent: BaseAgent,
    grader,
    k_continuation: int,
) -> list[dict]:
    """Patch missed uncertain hops in correct natural rollouts.

    Per rollout:
      R1: answer correct
      For each missed uncertain hop:
        R3: formal trace has a canonical query for this hop
        Consistency: natural trace context is compatible with the canonical query
        Splice, patch, run k_continuation continuations
        Accept if correct + no new M-violation for this hop
    """
    examples = []
    for r_idx, (r, (qctxs, attrs)) in enumerate(zip(natural_rollouts, rollout_attribution_data)):
        if not r["is_correct"]:
            continue

        missed = get_missed_uncertain_hops(attrs, sub_questions, uncertainty_labels)
        if not missed:
            continue

        print(f"      [proc2] rollout {r_idx + 1}: {len(missed)} missed uncertain hop(s): {missed}")

        for missed_idx in missed:
            canonical_query = find_canonical_query(missed_idx, formal_qctxs, formal_attrs)
            if canonical_query is None:
                print(f"        hop {missed_idx}: no canonical query in formal trace — skip")
                continue

            splice_idx = find_splice_index(r["messages"], missed_idx, qctxs, attrs)
            prior_messages = r["messages"][:splice_idx]

            print(f"        hop {missed_idx}: canonical='{canonical_query[:60]}' — checking consistency...")
            if not check_patch_consistency(prior_messages, canonical_query, natural_question, mediator_agent):
                print(f"        hop {missed_idx}: inconsistent — skip")
                continue

            bridge = synthesize_arm3_bridge(natural_question, canonical_query, prior_messages, mediator_agent)
            call_msg, return_msg = make_patch_messages(canonical_query, bridge, search_service)
            patched_history = prior_messages + [call_msg, return_msg]

            # Index in the final ChatML list where the patch begins (for loss masking).
            patch_start_idx = len(messages_to_chatml(prior_messages))

            for ci in range(k_continuation):
                print(f"        continuation {ci + 1}/{k_continuation} ...", end=" ", flush=True)
                cont = run_one_rollout(
                    agent, natural_question, gold_answer, grader,
                    message_history=patched_history,
                )
                if cont is None:
                    print("error")
                    continue
                print("correct" if cont["is_correct"] else "wrong")
                if not cont["is_correct"]:
                    continue
                # Verify no new M-violation for the patched hop
                _, cont_attrs = compute_rollout_attribution(
                    cont["messages"], sub_questions, formal_question, attribution_agent
                )
                cont_missed = get_missed_uncertain_hops(cont_attrs, sub_questions, uncertainty_labels)
                if missed_idx in cont_missed:
                    print(f"        hop {missed_idx}: continuation still misses this hop — skip")
                    continue
                examples.append({
                    "messages": messages_to_chatml(cont["messages"]),
                    "patch_start_idx": patch_start_idx,
                })
                break

    return examples


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return set(json.load(f))


def save_progress(path: str, done_ids: set[str]) -> None:
    with open(path, "w") as f:
        json.dump(sorted(done_ids), f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create MusiQue SFT dataset via rejection sampling.")
    p.add_argument("--uncertainty-json", required=True,
                   help="Output of run_musique_parametric_uncertainty.py")
    p.add_argument("--natural-jsonl", required=True,
                   help="Output of paraphrase_musique_natural.py")
    p.add_argument("--model", required=True,
                   help="Rollout model name (e.g. qwen2.5:32b)")
    p.add_argument("--provider", default="ollama",
                   help="Rollout model provider (default: ollama)")
    p.add_argument("--k", type=int, default=5,
                   help="Rollouts per question (default: 5)")
    p.add_argument("--k-continuation", type=int, default=2,
                   help="Procedure 2: continuation attempts per patched hop (default: 2)")
    p.add_argument("--k-formal", type=int, default=1,
                   help="Procedure 2: formal rollouts for canonical query extraction (default: 1)")
    p.add_argument("--attribution-model", default="gpt-oss:20b",
                   help="Model for LLM hop attribution (default: gpt-oss:20b)")
    p.add_argument("--attribution-provider", default="ollama",
                   help="Provider for attribution model (default: ollama)")
    p.add_argument("--mediator-model", default="gpt-oss:20b",
                   help="Model for bridge thinking and consistency checks (default: gpt-oss:20b)")
    p.add_argument("--mediator-provider", default="ollama",
                   help="Provider for mediator model (default: ollama)")
    p.add_argument("--output-dir", default="data/sft/musique",
                   help="Output directory (default: data/sft/musique)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Rollout sampling temperature (default: 1.0)")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Rollout nucleus sampling top-p (default: 0.95)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Rollout top-k sampling; Ollama/vLLM only (default: 20)")
    p.add_argument("--r3-strict", action="store_true",
                   help="Procedure 1: also reject rollouts that over-searched certain hops (off by default)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-processed example_ids")
    return p.parse_args()


def _check_ollama() -> None:
    """Fail fast if the local Ollama server is not reachable."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
    except (urllib.error.URLError, OSError) as e:
        import sys
        sys.exit(f"ERROR: Ollama server not reachable at localhost:11434 — {e}\n"
                 "Start Ollama before running this script.")


def main() -> None:
    args = setup_args()
    _check_ollama()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading uncertainty JSON...")
    uncertainty_data = load_uncertainty_json(args.uncertainty_json)
    print(f"  {len(uncertainty_data)} examples")

    print("Loading natural paraphrases...")
    natural_map = load_natural_jsonl(args.natural_jsonl)
    print(f"  {len(natural_map)} natural questions")

    examples = [ex for ex in uncertainty_data if ex["example_id"] in natural_map]
    print(f"  {len(examples)} examples with both inputs\n")

    progress_path = os.path.join(args.output_dir, "progress.json")
    done_ids = load_progress(progress_path) if args.resume else set()

    proc1_path = os.path.join(args.output_dir, "procedure1_natural_sft.jsonl")
    proc2_path = os.path.join(args.output_dir, "procedure2_m_patching.jsonl")

    if not args.resume:
        for path in [proc1_path, proc2_path]:
            open(path, "w").close()

    print(f"Initializing rollout agent ({args.model} via {args.provider})...")
    search_service = BraveSearchService()
    rollout_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[Tool(search_service.search)],
        output_type=str,
        use_thinking=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        agent_name="sft_rollout_agent",
    )

    print(f"Initializing attribution agent ({args.attribution_model} via {args.attribution_provider})...")
    attribution_agent = BaseAgent(
        provider_name=args.attribution_provider,
        model_name=args.attribution_model,
        output_type=QueryAttribution,
        use_thinking=False,
        agent_name="sft_attribution_agent",
    )

    print(f"Initializing mediator agent ({args.mediator_model} via {args.mediator_provider})...")
    mediator_agent = BaseAgent(
        provider_name=args.mediator_provider,
        model_name=args.mediator_model,
        output_type=str,
        use_thinking=False,
        agent_name="sft_mediator_agent",
    )

    print("Initializing grader (gpt-oss:120b)...")
    grader = build_grader()

    print(f"\n--- SFT data generation: K={args.k}, {len(examples)} examples ---\n")

    proc1_total = proc2_total = 0

    for i, ex in enumerate(examples):
        eid = ex["example_id"]
        if eid in done_ids:
            print(f"[{i + 1}/{len(examples)}] Skipping {eid} (done)")
            continue

        natural_rec = natural_map[eid]
        natural_q = natural_rec["text"]
        formal_q = ex["aggregate_question"]
        gold_answer = ex["aggregate_answer"]
        sub_questions = ex["sub_questions_results"]
        uncertainty_labels = get_hop_uncertainty_labels(sub_questions)

        print(f"[{i + 1}/{len(examples)}] {eid}")
        print(f"  formal:  {formal_q[:72]}")
        print(f"  natural: {natural_q[:72]}")
        print(f"  uncertainty: { {sq['hop_index']: uncertainty_labels[sq['hop_index']] for sq in sub_questions} }")

        print(f"  [proc1/2] {args.k} natural rollouts...")
        natural_rollouts = run_k_rollouts(
            rollout_agent, natural_q, gold_answer, grader, args.k
        )

        print(f"  [proc1/2] attributing correct rollouts (skipping wrong)...")
        rollout_attribution_data = []
        for r_idx, r in enumerate(natural_rollouts):
            if not r["is_correct"]:
                rollout_attribution_data.append(([], []))
                continue
            print(f"      attributing rollout {r_idx + 1}/{len(natural_rollouts)}...")
            qctxs, attrs = compute_rollout_attribution(
                r["messages"], sub_questions, formal_q, attribution_agent
            )
            rollout_attribution_data.append((qctxs, attrs))

        proc1_ex = process_procedure1(
            natural_rollouts, rollout_attribution_data,
            sub_questions, uncertainty_labels, r3_strict=args.r3_strict,
        )

        # Budget-optimise: only run formal rollout if P2 candidates exist
        p2_candidates_exist = any(
            r["is_correct"] and get_missed_uncertain_hops(attrs, sub_questions, uncertainty_labels)
            for r, (_, attrs) in zip(natural_rollouts, rollout_attribution_data)
        )

        proc2_ex = []
        if p2_candidates_exist:
            print(f"  [proc2] running {args.k_formal} formal rollout(s) for canonical queries...")
            formal_result = None
            for _ in range(args.k_formal):
                formal_result = acquire_formal_trace(
                    rollout_agent, formal_q, gold_answer, grader,
                    sub_questions, formal_q, attribution_agent,
                )
                if formal_result is not None:
                    break

            if formal_result is not None:
                formal_qctxs, formal_attrs = formal_result
                proc2_ex = process_procedure2(
                    natural_rollouts, rollout_attribution_data,
                    sub_questions, uncertainty_labels,
                    formal_qctxs, formal_attrs,
                    natural_q, formal_q, gold_answer,
                    rollout_agent, search_service, mediator_agent,
                    attribution_agent, grader, args.k_continuation,
                )
            else:
                print(f"  [proc2] formal rollout failed — skipping procedure 2")
        else:
            print(f"  [proc2] no P2 candidates — skipping formal rollout")

        with open(proc1_path, "a") as f:
            for ex_out in proc1_ex:
                f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
        with open(proc2_path, "a") as f:
            for ex_out in proc2_ex:
                f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")

        proc1_total += len(proc1_ex)
        proc2_total += len(proc2_ex)

        done_ids.add(eid)
        save_progress(progress_path, done_ids)

        print(
            f"  -> +proc1={len(proc1_ex)} +proc2={len(proc2_ex)} "
            f"(totals: {proc1_total} / {proc2_total})"
        )

    print("\n--- Done ---")
    print(f"Procedure 1 (natural SFT):    {proc1_total:5d} examples  →  {proc1_path}")
    print(f"Procedure 2 (M-patching):     {proc2_total:5d} examples  →  {proc2_path}")


if __name__ == "__main__":
    main()
