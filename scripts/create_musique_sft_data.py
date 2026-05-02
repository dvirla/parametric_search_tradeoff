"""
Create MusiQue SFT dataset via K=5 rejection sampling — three arms.

Arm 1 — Naive Natural SFT:
    K=5 rollouts on the natural-paraphrased question; keep traces where the
    final answer is correct. All assistant turns go into the loss.

Arm 2 — Prompt-Relabeled Formal:
    K=5 rollouts on the formal MusiQue question; keep correct traces; swap
    the user message with the natural paraphrase; a mediator LLM rewrites
    the first assistant thinking paragraph to bridge natural→formal style.

Arm 3 — Targeted M-Patching:
    K=5 rollouts on the natural question; find rollouts with exactly 1 missed
    hop (via LLM attribution, as in analyze_parametric_search_interplay.py);
    a mediator LLM writes a connecting thinking paragraph, then the missed
    hop's canonical sub-question is spliced in as a real search call;
    the continuation is re-sampled and kept if the final answer is correct.

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
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from src.services.base_agent import BaseAgent
from src.services.brave_search import BraveSearchService
from scripts.run_musique_experiment import build_grader

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
    prompt = NATURAL_GRADER_TEMPLATE.format(
        question=question, gold_answer=gold_answer, response=response,
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
    """Extract all search queries with their preceding thinking and search results.

    Works on pydantic-ai ModelMessage objects (not the logfire JSON format used
    in analyze_parametric_search_interplay.py).
    """
    results = []
    query_index = 0

    for turn_idx, msg in enumerate(messages):
        if not isinstance(msg, ModelResponse):
            continue

        # Thinking = all TextPart content in this assistant turn
        thinking_text = "\n".join(
            p.content for p in msg.parts if isinstance(p, TextPart)
        )

        # Build tool_call_id → (full_text, snippets) from the immediately following ModelRequest
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


def detect_missed_hops(
    messages: list,
    sub_questions: list[dict],
    aggregate_question: str,
    attribution_agent: BaseAgent,
) -> list[int]:
    """Return hop indices not covered by any attributed search query.

    A hop is 'covered' when at least one query is attributed to its hop_index
    (confidence low is still accepted — the LLM made a positive attribution).
    """
    query_ctxs = extract_queries_from_messages(messages)
    if not query_ctxs:
        # No searches at all → all hops missed; caller will filter len != 1
        return [sq["hop_index"] for sq in sub_questions]

    attributions = attribute_queries(
        query_ctxs, sub_questions, aggregate_question, attribution_agent
    )
    covered = {attr.hop_index for attr in attributions if attr.hop_index >= 0}
    return [sq["hop_index"] for sq in sub_questions if sq["hop_index"] not in covered]


# ---------------------------------------------------------------------------
# Mediator LLM — thinking synthesis
# ---------------------------------------------------------------------------

ARM2_MEDIATOR_PROMPT = """\
You are rewriting the opening reasoning of an AI agent.

The agent was originally responding to this formal question:
"{formal_question}"

It is now responding to this natural-language question instead (same information need, different phrasing):
"{natural_question}"

The agent's original opening reasoning:
"{original_thinking}"

Rewrite ONLY the opening reasoning so it reads naturally as a response to the NATURAL question.
Preserve the intended reasoning structure and search strategy, but adapt the language to match the \
conversational phrasing of the natural question.
Return only the rewritten reasoning text. No preamble, no quotes.\
"""

ARM3_BRIDGE_PROMPT = """\
You are writing a short connecting reasoning paragraph for an AI agent solving a multi-hop question.

The agent is answering this question:
"{natural_question}"

Conversation so far:
{prior_context}

The agent is about to search for: "{hop_question}"

Write a brief reasoning paragraph (2-4 sentences) that:
1. Acknowledges what has been established from the prior searches.
2. Identifies the remaining gap: the answer to '{hop_question}' is still needed.
3. Concludes by stating that the agent will now search for it.

Return only the reasoning text. No preamble, no quotes.\
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
    # Keep the last 12 items to stay within context
    return "\n".join(items[-12:]) if items else "(no prior context)"


def synthesize_arm2_thinking(
    formal_question: str,
    natural_question: str,
    original_thinking: str,
    mediator_agent: BaseAgent,
) -> str:
    """Rewrite the first assistant thinking to bridge natural question → formal search style."""
    if not original_thinking.strip():
        return original_thinking
    prompt = ARM2_MEDIATOR_PROMPT.format(
        formal_question=formal_question,
        natural_question=natural_question,
        original_thinking=original_thinking,
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = mediator_agent.run(prompt)
            return result.output.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Mediator (arm2) failed: {e} — keeping original thinking")
                return original_thinking


def synthesize_arm3_bridge(
    natural_question: str,
    hop_question: str,
    prior_messages: list,
    mediator_agent: BaseAgent,
) -> str:
    """Generate a connecting reasoning paragraph before the patched search call."""
    context = _format_prior_context(prior_messages)
    prompt = ARM3_BRIDGE_PROMPT.format(
        natural_question=natural_question,
        prior_context=context,
        hop_question=hop_question,
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = mediator_agent.run(prompt)
            return result.output.strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"    Mediator (arm3) failed: {e} — using empty bridge")
                return f"I still need to look up: {hop_question}"


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
            text_parts = [p for p in msg.parts if isinstance(p, TextPart)]
            call_parts = [p for p in msg.parts if isinstance(p, ToolCallPart)]
            content = "".join(p.content for p in text_parts) or None
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


def _rewrite_first_assistant_thinking(chatml: list[dict], new_thinking: str) -> list[dict]:
    """Replace the text content of the first assistant turn (Arm 2 mediation)."""
    result = list(chatml)
    for i, msg in enumerate(result):
        if msg["role"] == "assistant":
            # Preserve tool_calls if present; only replace the text content
            updated = dict(msg, content=new_thinking)
            result[i] = updated
            break
    return result


def swap_user_message(chatml: list[dict], new_content: str) -> list[dict]:
    """Replace the first user-role message content."""
    result = list(chatml)
    for i, msg in enumerate(result):
        if msg["role"] == "user":
            result[i] = dict(msg, content=new_content)
            break
    return result


# ---------------------------------------------------------------------------
# Arm 3: trace patching
# ---------------------------------------------------------------------------

def _last_tool_return_idx(messages: list) -> int:
    last = -1
    for i, msg in enumerate(messages):
        if isinstance(msg, ModelRequest) and any(
            isinstance(p, ToolReturnPart) for p in msg.parts
        ):
            last = i
    return last


def truncate_to_splice(messages: list) -> list:
    """Return messages up to and including the last tool return."""
    idx = _last_tool_return_idx(messages)
    return list(messages[: idx + 1]) if idx >= 0 else list(messages)


def make_patch_messages(
    hop_question: str,
    bridge_thinking: str,
    search_service: BraveSearchService,
) -> tuple[ModelResponse, ModelRequest]:
    """Build a synthetic (thinking + tool-call) + tool-return pair for the missed hop."""
    call_id = f"patch_{uuid.uuid4().hex[:8]}"
    parts = []
    if bridge_thinking:
        parts.append(TextPart(content=bridge_thinking))
    parts.append(ToolCallPart(tool_name="search", args={"query": hop_question}, tool_call_id=call_id))
    call_msg = ModelResponse(parts=parts)

    results = search_service.search(hop_question, max_results=5)
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
    """Run one agentic rollout. Returns {output, messages, is_correct} or None on failure.

    When message_history is provided the agent continues from that history
    (used for Arm 3 continuations after patching). The question parameter is
    used only for grading in that case.
    """
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
# Per-arm processors
# ---------------------------------------------------------------------------

def process_arm1(natural_rollouts: list[dict]) -> list[dict]:
    """Keep correct natural rollouts and convert to ChatML."""
    return [
        {"messages": messages_to_chatml(r["messages"])}
        for r in natural_rollouts
        if r["is_correct"]
    ]


def process_arm2(
    formal_rollouts: list[dict],
    natural_question: str,
    formal_question: str,
    mediator_agent: BaseAgent,
) -> list[dict]:
    """Keep correct formal rollouts; swap user → natural; rewrite first assistant thinking."""
    examples = []
    for r in formal_rollouts:
        if not r["is_correct"]:
            continue
        chatml = messages_to_chatml(r["messages"])

        # Extract the first assistant turn's thinking text for mediation
        first_thinking = ""
        for msg in chatml:
            if msg["role"] == "assistant" and msg.get("content"):
                first_thinking = msg["content"]
                break

        print(f"      [arm2] mediating first thinking ({len(first_thinking)} chars)...")
        new_thinking = synthesize_arm2_thinking(
            formal_question, natural_question, first_thinking, mediator_agent
        )

        chatml = swap_user_message(chatml, natural_question)
        chatml = _rewrite_first_assistant_thinking(chatml, new_thinking)
        examples.append({"messages": chatml})
    return examples


def process_arm3(
    natural_rollouts: list[dict],
    sub_questions: list[dict],
    aggregate_question: str,
    agent: BaseAgent,
    search_service: BraveSearchService,
    mediator_agent: BaseAgent,
    attribution_agent: BaseAgent,
    natural_question: str,
    gold_answer: str,
    grader,
    k_continuation: int,
) -> list[dict]:
    """Find 1-missed-hop rollouts; patch with bridge thinking + search; re-sample continuation."""
    examples = []
    for r_idx, r in enumerate(natural_rollouts):
        print(f"      [arm3] attributing rollout {r_idx + 1}/{len(natural_rollouts)}...")
        missed = detect_missed_hops(
            r["messages"], sub_questions, aggregate_question, attribution_agent
        )
        if len(missed) != 1:
            print(f"        skipped: {len(missed)} missed hops")
            continue

        missed_idx = missed[0]
        hop_q = sub_questions[missed_idx]["question"]
        print(f"        missed hop {missed_idx}: {hop_q[:60]}...")

        partial = truncate_to_splice(r["messages"])

        print(f"        mediating bridge thinking...")
        bridge = synthesize_arm3_bridge(natural_question, hop_q, partial, mediator_agent)

        call_msg, return_msg = make_patch_messages(hop_q, bridge, search_service)
        patched_history = partial + [call_msg, return_msg]

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
            if cont["is_correct"]:
                # cont["messages"] already contains the full trace (patched_history + continuation)
                examples.append({"messages": messages_to_chatml(cont["messages"])})
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
                   help="Rollouts per question per arm (default: 5)")
    p.add_argument("--k-continuation", type=int, default=2,
                   help="Arm 3: continuation attempts per patched rollout (default: 2)")
    p.add_argument("--attribution-model", default="gpt-oss:20b",
                   help="Model for LLM hop attribution (default: gpt-oss:20b)")
    p.add_argument("--attribution-provider", default="ollama",
                   help="Provider for attribution model (default: ollama)")
    p.add_argument("--mediator-model", default="gpt-oss:20b",
                   help="Model for thinking synthesis mediation (default: gpt-oss:20b)")
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

    arm1_path = os.path.join(args.output_dir, "arm1_naive_natural.jsonl")
    arm2_path = os.path.join(args.output_dir, "arm2_relabeled_formal.jsonl")
    arm3_path = os.path.join(args.output_dir, "arm3_m_patching.jsonl")

    if not args.resume:
        for path in [arm1_path, arm2_path, arm3_path]:
            open(path, "w").close()

    # Build agents
    print(f"Initializing rollout agent ({args.model} via {args.provider})...")
    search_service = BraveSearchService()
    rollout_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[Tool(search_service.search)],
        output_type=str,
        use_thinking=False,
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

    arm1_total = arm2_total = arm3_total = 0

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

        print(f"[{i + 1}/{len(examples)}] {eid}")
        print(f"  formal:  {formal_q[:72]}")
        print(f"  natural: {natural_q[:72]}")

        # Arms 1 & 3 share the same K natural rollouts
        print(f"  [arm1/3] {args.k} natural rollouts...")
        natural_rollouts = run_k_rollouts(
            rollout_agent, natural_q, gold_answer, grader, args.k
        )

        arm1_ex = process_arm1(natural_rollouts)
        arm3_ex = process_arm3(
            natural_rollouts, sub_questions, formal_q,
            rollout_agent, search_service, mediator_agent, attribution_agent,
            natural_q, gold_answer, grader, args.k_continuation,
        )

        # Arm 2 needs separate formal rollouts
        print(f"  [arm2]  {args.k} formal rollouts...")
        formal_rollouts = run_k_rollouts(
            rollout_agent, formal_q, gold_answer, grader, args.k
        )
        arm2_ex = process_arm2(formal_rollouts, natural_q, formal_q, mediator_agent)

        with open(arm1_path, "a") as f:
            for ex_out in arm1_ex:
                f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
        with open(arm2_path, "a") as f:
            for ex_out in arm2_ex:
                f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")
        with open(arm3_path, "a") as f:
            for ex_out in arm3_ex:
                f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")

        arm1_total += len(arm1_ex)
        arm2_total += len(arm2_ex)
        arm3_total += len(arm3_ex)

        done_ids.add(eid)
        save_progress(progress_path, done_ids)

        print(
            f"  -> +arm1={len(arm1_ex)} +arm2={len(arm2_ex)} +arm3={len(arm3_ex)} "
            f"(totals: {arm1_total} / {arm2_total} / {arm3_total})"
        )

    print("\n--- Done ---")
    print(f"Arm 1 (naive natural):    {arm1_total:5d} examples  →  {arm1_path}")
    print(f"Arm 2 (relabeled formal): {arm2_total:5d} examples  →  {arm2_path}")
    print(f"Arm 3 (M-patching):       {arm3_total:5d} examples  →  {arm3_path}")


if __name__ == "__main__":
    main()
