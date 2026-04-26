"""
Parametric–Search Interplay Analysis

Processes matched trace ↔ eval pairs: attributes search queries to hops, computes
parametric/sequential redundancy, and exports CSV summaries for downstream analysis.
Plotting is handled by unified_interplay_analysis.py.
"""

import os
import sys
import json
import glob
import argparse
import re
import hashlib
from typing import Optional, Literal
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
from scipy import stats
from pydantic import BaseModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv

load_dotenv()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze parametric–search interplay in MuSiQue traces."
    )
    parser.add_argument(
        "--eval-dir", type=str, default="results/musique_parametric_test",
        help="Directory containing musique_parametric_uncertainty_*.json and musique_search_*_traces_*.json"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="results/musique_parametric_test/interplay_analysis",
        help="Output directory for CSV summaries and attribution checkpoints."
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use LLM for query-to-hop attribution (default: heuristic token overlap)."
    )
    parser.add_argument(
        "--attribution-model", type=str, default="gpt-4.1-mini",
        help="Model name for LLM attribution (default: gpt-4.1-mini)."
    )
    parser.add_argument(
        "--attribution-provider", type=str, default="OpenAI",
        help="Provider for LLM attribution (default: OpenAI)."
    )
    parser.add_argument(
        "--reattribute", action="store_true",
        help="Ignore attribution cache and redo all LLM calls."
    )
    parser.add_argument(
        "--check-multi-hop", action="store_true",
        help="Use LLM to check if search queries are relevant to multiple hops (requires --use-llm)."
    )
    parser.add_argument(
        "--llm-gold-check", action="store_true",
        help="Use LLM to check if gold answer is semantically present in search results "
             "(default: substring match). Requires --use-llm."
    )
    return parser.parse_args()


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────

class QueryAttribution(BaseModel):
    hop_index: int  # 0-based index into sub_questions_results; -1 = aggregate/general
    confidence: Literal["high", "medium", "low"]
    reasoning: str


class MultiHopAttribution(BaseModel):
    relevant_hops: list[int]  # 0-based hop indices this query serves; [-1] = aggregate only
    reasoning: str


class GoldFoundJudgment(BaseModel):
    found: bool    # True if the gold answer is semantically present in the search results
    reasoning: str


# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────

@dataclass
class SubQuestionResult:
    hop_index: int
    question: str
    gold_answer: str
    num_correct: int
    num_runs: int
    semantic_entropy: float
    cluster_ids: list
    run_correctness: list  # list[bool], one per run (from parametric no-search runs)


@dataclass
class QueryWithContext:
    query: str
    preceding_thinking: str
    query_index: int  # global index within trace
    turn_index: int   # assistant message index
    search_result_text: str = ""        # concatenated title+snippet from search results
    search_result_snippets: list = field(default_factory=list)  # individual "title snippet" strings
    tool_call_id: str = ""              # for matching query→result


@dataclass
class QueryRecord:
    query: str
    query_index: int
    assigned_hop: int          # -1 = aggregate/general
    hop_question: str
    gold_answer_for_hop: str
    attribution_confidence: str
    attribution_reasoning: str
    search_result_text: str
    search_result_snippets: list  # individual "title snippet" strings per search result
    gold_found_in_results: bool
    is_parametric_redundant: bool
    is_sequential_redundant: bool
    multi_hop_relevant: bool = False  # True if LLM judges query relevant to >1 hop


@dataclass
class MatchedExample:
    example_id: str
    model: str
    aggregate_question: str
    aggregate_correct: bool
    sub_questions: list          # list[SubQuestionResult]
    query_records: list          # list[QueryRecord]


# ─── SERIALIZATION ────────────────────────────────────────────────────────────

import dataclasses

def _examples_to_dict(examples: list) -> list:
    return [dataclasses.asdict(ex) for ex in examples]


def _examples_from_dict(data: list) -> list:
    result = []
    for ex in data:
        sub_questions = [SubQuestionResult(**sq) for sq in ex["sub_questions"]]
        query_records = [QueryRecord(**qr) for qr in ex["query_records"]]
        result.append(MatchedExample(
            example_id=ex["example_id"],
            model=ex["model"],
            aggregate_question=ex["aggregate_question"],
            aggregate_correct=ex["aggregate_correct"],
            sub_questions=sub_questions,
            query_records=query_records,
        ))
    return result


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def clean_problem(problem: str) -> str:
    """Remove instruction suffix from problem string."""
    separator = "\n\nYour response should be in the following format:"
    popqa_separator = "Portuguese\n\nNow answer the following:\n\nQuestion: "
    natural_separator = "\n\nPlease answer in 2-4 sentences."
    if separator in problem:
        return problem.split(separator)[0].strip()
    elif popqa_separator in problem:
        problem = problem.split(popqa_separator)[1].strip().split("\nAnswer:")[0]
    elif natural_separator in problem:
        return problem.split(natural_separator)[0].strip()
    return problem.strip()


def tokenize(text: str) -> set:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def model_name_from_eval_path(path: str) -> str:
    """Extract model name from musique_parametric_uncertainty_<model>.json"""
    basename = os.path.basename(path)
    return basename.replace("musique_parametric_uncertainty_", "").replace(".json", "")


def model_name_from_trace_path(path: str) -> str:
    """Extract model name from musique_search_<model>_traces_<ts>.json"""
    basename = os.path.basename(path)
    # Remove prefix and suffix
    name = basename.replace("musique_val_search_", "")
    name = re.sub(r"_traces_\d{8}_\d{6}\.json$", "", name)
    return name


def short_model(name: str) -> str:
    """Shorten model name for display."""
    replacements = {
        "gemini-3-pro-preview": "Gemini-3.1-Pro",
        "nemotron-3-nano_30b": "Nemotron-30B",
        "qwen3.5_122b": "Qwen3.5-122B",
    }
    return replacements.get(name, name)


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_eval_data(eval_dir: str) -> dict:
    """Returns dict[model_name -> list[entry_dict]]."""
    pattern = os.path.join(eval_dir, "musique_parametric_uncertainty_*.json")
    files = sorted(glob.glob(pattern))
    result = {}
    for path in files:
        model = model_name_from_eval_path(path)
        with open(path) as f:
            result[model] = json.load(f)
        print(f"  Loaded eval: {model} ({len(result[model])} entries) from {os.path.basename(path)}")
    return result


def load_traces(eval_dir: str) -> dict:
    """Returns dict[model_name -> list[trace_dict]]."""
    pattern = os.path.join(eval_dir, "musique_val_search_*_traces_*.json")
    files = sorted(glob.glob(pattern))
    result = {}
    for path in files:
        model = model_name_from_trace_path(path)
        with open(path) as f:
            result[model] = json.load(f)
        print(f"  Loaded traces: {model} ({len(result[model])} traces) from {os.path.basename(path)}")
    return result


def load_nosearch_aggregate(eval_dir: str) -> dict:
    """Returns dict[model_name -> dict[example_id -> entry]] from musique_aggregate_nosearch_*.json files."""
    pattern = os.path.join(eval_dir, "musique_aggregate_nosearch_*.json")
    files = sorted(glob.glob(pattern))
    result = {}
    for path in files:
        basename = os.path.basename(path)
        model = basename.replace("musique_aggregate_nosearch_", "").replace(".json", "")
        with open(path) as f:
            entries = json.load(f)
        result[model] = {e["example_id"]: e for e in entries}
        print(f"  Loaded no-search aggregate: {model} ({len(result[model])} entries) from {basename}")
    return result


# ─── QUERY EXTRACTION ─────────────────────────────────────────────────────────

def extract_queries_with_context(trace: dict) -> list:
    """
    Returns list of QueryWithContext — all search queries with preceding thinking
    and concatenated search result text.
    Multiple queries can appear in the same assistant turn.
    """
    messages = trace.get("message_trace", [])
    results = []
    query_index = 0

    for turn_idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue
        parts = msg.get("parts", [])

        # Collect thinking text before queries in this turn
        thinking_parts = [p["content"] for p in parts if p.get("type") == "thinking"]
        thinking_text = "\n".join(thinking_parts)

        # Build result_map: tool_call_id → concatenated result text from next message
        result_map: dict = {}
        next_msg = messages[turn_idx + 1] if turn_idx + 1 < len(messages) else None
        if next_msg:
            for rp in next_msg.get("parts", []):
                if rp.get("type") == "tool_call_response":
                    tcid = rp.get("tool_call_id", "")
                    result_data = rp.get("result", [])
                    snippets = [
                        f"{r.get('title', '')} {r.get('snippet', '')}".strip()
                        for r in (result_data if isinstance(result_data, list) else [])
                    ]
                    result_map[tcid] = (" ".join(snippets), snippets)

        # Collect all search tool calls in this turn
        for part in parts:
            if part.get("type") == "tool_call" and part.get("tool_name") == "search":
                args = part.get("arguments", {})
                if isinstance(args, str):
                    try:
                        import ast
                        args = ast.literal_eval(args)
                    except Exception:
                        args = {}
                query = args.get("query", "")
                tool_call_id = part.get("tool_call_id", "")
                if query:
                    result_entry = result_map.get(tool_call_id, ("", []))
                    results.append(QueryWithContext(
                        query=query,
                        preceding_thinking=thinking_text,
                        query_index=query_index,
                        turn_index=turn_idx,
                        search_result_text=result_entry[0],
                        search_result_snippets=result_entry[1],
                        tool_call_id=tool_call_id,
                    ))
                    query_index += 1

    return results


# ─── ATTRIBUTION ──────────────────────────────────────────────────────────────

def heuristic_attribute(query_ctx: QueryWithContext, sub_questions: list) -> QueryAttribution:
    """
    Attribute a query to a hop by token overlap.
    Combined text = query + preceding_thinking (first 200 chars).
    """
    combined = query_ctx.query #+ " " + query_ctx.preceding_thinking[:200]
    combined_tokens = tokenize(combined)

    best_hop = -1
    best_score = 0.0

    for sq in sub_questions:
        sq_tokens = tokenize(sq.question + " " + sq.gold_answer)
        if not sq_tokens:
            continue
        overlap = len(combined_tokens & sq_tokens) / len(sq_tokens)
        if overlap > best_score:
            best_score = overlap
            best_hop = sq.hop_index

    # Confidence based on overlap score
    if best_score >= 0.4:
        confidence = "high"
    elif best_score >= 0.15:
        confidence = "medium"
    else:
        confidence = "low"
        best_hop = -1  # Fall back to aggregate when overlap too weak

    return QueryAttribution(
        hop_index=best_hop,
        confidence=confidence,
        reasoning=f"Token overlap score: {best_score:.3f}",
    )


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


def llm_attribute(
    query_ctx: QueryWithContext,
    sub_questions: list,
    aggregate_question: str,
    attribution_agent,
    cache: dict,
    cache_key: str,
    reattribute: bool,
) -> QueryAttribution:
    """Use LLM to attribute query to hop, with caching."""
    if not reattribute and cache_key in cache:
        cached = cache[cache_key]
        return QueryAttribution(**cached)

    sub_q_text = "\n".join(
        f"  Hop {sq.hop_index}: {sq.question}" for sq in sub_questions
    )
    prompt = ATTRIBUTION_PROMPT.format(
        aggregate_question=aggregate_question,
        sub_questions_text=sub_q_text,
        thinking=query_ctx.preceding_thinking[:500],
        query=query_ctx.query,
    )

    try:
        result = attribution_agent.run(prompt)
        attribution = result.output
    except Exception as e:
        print(f"    LLM attribution error: {e} — falling back to heuristic")
        attribution = heuristic_attribute(query_ctx, sub_questions)

    cache[cache_key] = {
        "hop_index": attribution.hop_index,
        "confidence": attribution.confidence,
        "reasoning": attribution.reasoning,
    }
    return attribution


MULTI_HOP_ATTRIBUTION_PROMPT = """You are analyzing a search query issued by an AI agent solving a multi-hop question.

Aggregate question: {aggregate_question}

Sub-questions (hops):
{sub_questions_text}

Agent's thinking before the search query:
{thinking}

Search query: {query}

Which sub-questions (hops) does this search query serve or is relevant to?
A query is relevant to a hop if its results could help answer that hop's question.
Return a list of 0-based hop indices (e.g. [0], [0, 2], [-1] for aggregate only).
Be conservative: only include a hop if the query clearly targets or would help with it.
"""


def check_multi_hop_relevance(
    query_ctx: QueryWithContext,
    sub_questions: list,
    aggregate_question: str,
    multi_hop_agent,
    cache: dict,
    cache_key: str,
    reattribute: bool,
) -> MultiHopAttribution:
    """Use LLM to determine all hops a query is relevant to, with caching."""
    if not reattribute and cache_key in cache:
        cached = cache[cache_key]
        return MultiHopAttribution(**cached)

    sub_q_text = "\n".join(
        f"  Hop {sq.hop_index}: {sq.question}" for sq in sub_questions
    )
    prompt = MULTI_HOP_ATTRIBUTION_PROMPT.format(
        aggregate_question=aggregate_question,
        sub_questions_text=sub_q_text,
        thinking=query_ctx.preceding_thinking[:500],
        query=query_ctx.query,
    )

    try:
        result = multi_hop_agent.run(prompt)
        attribution = result.output
    except Exception as e:
        print(f"    LLM multi-hop attribution error: {e} — defaulting to single-hop")
        attribution = MultiHopAttribution(relevant_hops=[-1], reasoning=f"Error: {e}")

    cache[cache_key] = {
        "relevant_hops": attribution.relevant_hops,
        "reasoning": attribution.reasoning,
    }
    return attribution


GOLD_FOUND_PROMPT = """You are checking whether a specific answer is semantically present in search results.

Gold answer: {gold_answer}

Search results:
{search_text}

Is the gold answer (or an equivalent phrasing/abbreviation) present in the search results above?
Answer True if the search results contain the answer or a clear equivalent.
Answer False if the answer is absent or only tangentially mentioned without being stated as a fact.
"""


def llm_check_gold_in_results(
    gold_answer: str,
    snippets: list,
    gold_judge_agent,
    cache: dict,
    cache_key: str,
    reattribute: bool,
) -> bool:
    """
    LLM judge: is gold_answer semantically present in any of the search result snippets?
    Makes a single LLM call with all snippets presented as a numbered list.
    """
    if not reattribute and cache_key in cache:
        return cache[cache_key]["found"]

    numbered = "\n".join(
        f"{i + 1}. {s}" for i, s in enumerate(snippets) if s
    )
    prompt = GOLD_FOUND_PROMPT.format(
        gold_answer=gold_answer,
        search_text=numbered,
    )
    try:
        result = gold_judge_agent.run(prompt)
        found = result.output.found
    except Exception as e:
        print(f"    LLM gold-found error: {e} — falling back to string match")
        joined = " ".join(snippets)
        found = gold_answer.lower() in joined.lower()

    cache[cache_key] = {"found": found}
    return found


# ─── REDUNDANCY FLAGS ─────────────────────────────────────────────────────────

def compute_redundancy_flags(
    attributed_queries: list,  # list of (QueryWithContext, QueryAttribution)
    sub_questions: list,       # list[SubQuestionResult]
    gold_judge_agent=None,
    gold_judge_cache: dict = None,
    reattribute: bool = False,
) -> list:
    """
    Returns list[QueryRecord] with redundancy flags set.

    Parametric redundancy: hop has semantic_entropy == 0 AND num_correct == num_runs
    Sequential redundancy: gold answer for the hop was already found in a prior
                           query's search results for the same hop
    """
    sq_map = {sq.hop_index: sq for sq in sub_questions}
    hop_gold_found: dict = {}  # hop_index -> True once gold found in results

    records = []
    for qctx, attr in attributed_queries:
        hop = attr.hop_index

        # Resolve hop metadata
        if hop >= 0 and hop in sq_map:
            sq = sq_map[hop]
            hop_question = sq.question
            gold_answer = sq.gold_answer
            is_parametric_redundant = (
                sq.semantic_entropy <= 0.0 and sq.num_correct == sq.num_runs
            )
        else:
            hop_question = ""
            gold_answer = ""
            is_parametric_redundant = False

        # Check if gold answer appears in this query's search results
        if (gold_judge_agent is not None and gold_judge_cache is not None
                and gold_answer and qctx.search_result_snippets):
            _snippets_digest = hashlib.md5("|".join(s[:100] for s in qctx.search_result_snippets).encode()).hexdigest()[:16]
            gf_cache_key = f"gf:{hashlib.md5(gold_answer.encode()).hexdigest()[:16]}:{_snippets_digest}"
            gold_found_here = llm_check_gold_in_results(
                gold_answer, qctx.search_result_snippets,
                gold_judge_agent, gold_judge_cache, gf_cache_key, reattribute,
            )
        else:
            gold_found_here = bool(
                gold_answer and gold_answer.lower() in qctx.search_result_text.lower()
            )

        # Sequential redundancy: gold was already found by a prior query for this hop
        is_sequential_redundant = hop_gold_found.get(hop, False)

        # Update gold-found state after computing redundancy flag
        if gold_found_here:
            hop_gold_found[hop] = True

        records.append(QueryRecord(
            query=qctx.query,
            query_index=qctx.query_index,
            assigned_hop=hop,
            hop_question=hop_question,
            gold_answer_for_hop=gold_answer,
            attribution_confidence=attr.confidence,
            attribution_reasoning=attr.reasoning,
            search_result_text=qctx.search_result_text,
            search_result_snippets=qctx.search_result_snippets,
            gold_found_in_results=gold_found_here,
            is_parametric_redundant=is_parametric_redundant,
            is_sequential_redundant=is_sequential_redundant,
        ))

    return records


# ─── MATCHING ─────────────────────────────────────────────────────────────────

def match_and_build(
    eval_entries: list,
    traces: list,
    model: str,
    use_llm: bool,
    attribution_agent,
    cache: dict,
    reattribute: bool,
    multi_hop_agent=None,
    multi_hop_cache: dict = None,
    gold_judge_agent=None,
    gold_judge_cache: dict = None,
) -> list:
    """
    Match traces to eval entries, extract queries, attribute to hops, compute flags.
    Returns list[MatchedExample].
    """
    eval_map = {e["aggregate_question"]: e for e in eval_entries}
    matched_examples = []
    unmatched = 0

    for trace in traces:
        raw_problem = trace["problem"]
        problem = clean_problem(raw_problem)

        entry = eval_map.get(problem)
        if entry is None:
            unmatched += 1
            continue

        example_id = entry.get("example_id", problem[:40])

        # Build sub-questions
        sub_questions = []
        for sr in entry.get("sub_questions_results", []):
            runs = sr.get("runs", [])
            num_runs = len(runs) if runs else sr.get("num_runs") or 0
            sub_questions.append(SubQuestionResult(
                hop_index=sr["hop_index"],
                question=sr["question"],
                gold_answer=sr.get("gold_answer", ""),
                num_correct=sr.get("num_correct", 0),
                num_runs=num_runs,
                semantic_entropy=float(sr.get("semantic_entropy") or 0.0),
                cluster_ids=sr.get("cluster_ids", []),
                run_correctness=[bool(r.get("is_correct", False)) for r in runs],
            ))

        # Aggregate correctness
        agg_result = entry.get("aggregate_result", {})
        aggregate_correct = bool(agg_result.get("is_correct", False))

        # Extract queries with context
        query_ctxs = extract_queries_with_context(trace)

        # Attribute each query
        attributed = []
        for qctx in query_ctxs:
            if use_llm:
                cache_key = f"{model}:{example_id}:{qctx.query_index}"
                attr = llm_attribute(
                    qctx, sub_questions, problem,
                    attribution_agent, cache, cache_key, reattribute,
                )
            else:
                attr = heuristic_attribute(qctx, sub_questions)
            attributed.append((qctx, attr))

        # Compute redundancy
        query_records = compute_redundancy_flags(
            attributed, sub_questions,
            gold_judge_agent=gold_judge_agent,
            gold_judge_cache=gold_judge_cache,
            reattribute=reattribute,
        )

        # Multi-hop relevance check (LLM as judge)
        if multi_hop_agent is not None and multi_hop_cache is not None:
            for (qctx, _attr), qr in zip(attributed, query_records):
                mh_cache_key = f"mh:{model}:{example_id}:{qctx.query_index}"
                mh_attr = check_multi_hop_relevance(
                    qctx, sub_questions, problem,
                    multi_hop_agent, multi_hop_cache, mh_cache_key, reattribute,
                )
                qr.multi_hop_relevant = len(mh_attr.relevant_hops) > 1

        matched_examples.append(MatchedExample(
            example_id=str(example_id),
            model=model,
            aggregate_question=problem,
            aggregate_correct=aggregate_correct,
            sub_questions=sub_questions,
            query_records=query_records,
        ))

    if unmatched:
        print(f"  Warning: {unmatched} unmatched traces for model {model}")
    print(f"  Matched {len(matched_examples)}/{len(traces)} traces for {model}")
    return matched_examples


# ─── METRICS COMPUTATION ──────────────────────────────────────────────────────

def compute_hop_metrics(
    all_examples: list,
    gold_judge_agent=None,
    gold_judge_cache: dict = None,
    reattribute: bool = False,
) -> pd.DataFrame:
    """Returns DataFrame with one row per (model, example_id, hop_index)."""
    rows = []
    for ex in all_examples:
        num_hops = len(ex.sub_questions)
        for sq in ex.sub_questions:
            # Count queries assigned to this hop
            queries_for_hop = [
                qr for qr in ex.query_records if qr.assigned_hop == sq.hop_index
            ]
            parametric_accuracy = sq.num_correct / sq.num_runs if sq.num_runs > 0 else float("nan")

            # Check if a "missed search" hop's gold answer was implicitly found via another hop's queries
            is_missed_search = (
                not (sq.semantic_entropy <= 0.0 and sq.num_correct == sq.num_runs)
                and len(queries_for_hop) == 0
            )
            missed_search_cross_hop_covered = False
            if is_missed_search and sq.gold_answer:
                other_snippets = [
                    s for qr in ex.query_records
                    if qr.assigned_hop != sq.hop_index
                    for s in qr.search_result_snippets
                    if s
                ]
                if other_snippets:
                    if gold_judge_agent is not None and gold_judge_cache is not None:
                        _xhop_digest = hashlib.md5("|".join(s[:100] for s in other_snippets).encode()).hexdigest()[:16]
                        gf_cache_key = f"gf_xhop:{ex.example_id}:{sq.hop_index}:{_xhop_digest}"
                        missed_search_cross_hop_covered = llm_check_gold_in_results(
                            sq.gold_answer, other_snippets,
                            gold_judge_agent, gold_judge_cache, gf_cache_key, reattribute,
                        )
                    else:
                        other_text = " ".join(other_snippets)
                        missed_search_cross_hop_covered = sq.gold_answer.lower() in other_text.lower()

            rows.append({
                "model": ex.model,
                "example_id": ex.example_id,
                "hop_index": sq.hop_index,
                "num_hops": num_hops,
                "question": sq.question,
                "semantic_entropy": sq.semantic_entropy,
                "num_correct": sq.num_correct,
                "num_runs": sq.num_runs,
                "parametric_accuracy": parametric_accuracy,
                "parametric_certain": (sq.semantic_entropy <= 0.0 and sq.num_correct == sq.num_runs),
                "searched": len(queries_for_hop) > 0,
                "seq_redundant": any(qr.is_sequential_redundant for qr in queries_for_hop),
                "queries_assigned_count": len(queries_for_hop),
                "aggregate_correct": ex.aggregate_correct,
                "missed_search_answer_found_cross_hop": missed_search_cross_hop_covered,
            })
    return pd.DataFrame(rows)


def compute_example_metrics(all_examples: list, nosearch_by_model: Optional[dict] = None) -> pd.DataFrame:
    """Returns DataFrame with one row per (model, example_id).

    nosearch_by_model: optional dict[model_name -> dict[example_id -> entry]] from
    load_nosearch_aggregate(). When present, uses actual no-search aggregate runs
    instead of the independence-composition proxy.
    """
    rows = []
    for ex in all_examples:
        total = len(ex.query_records)
        par_red = sum(1 for qr in ex.query_records if qr.is_parametric_redundant)
        seq_red = sum(1 for qr in ex.query_records if qr.is_sequential_redundant)
        both_red = sum(1 for qr in ex.query_records
                       if qr.is_parametric_redundant and qr.is_sequential_redundant)
        # Exclusive categories (parametric takes priority to avoid double-counting)
        par_only = par_red  # includes those that are also seq-redundant
        seq_only = sum(1 for qr in ex.query_records
                       if qr.is_sequential_redundant and not qr.is_parametric_redundant)
        total_red = sum(1 for qr in ex.query_records
                        if qr.is_parametric_redundant or qr.is_sequential_redundant)
        necessary = total - total_red
        par_accs = [
            sq.num_correct / sq.num_runs for sq in ex.sub_questions if sq.num_runs > 0
        ]
        mean_par_acc = float(np.mean(par_accs)) if par_accs else float("nan")
        frac_certain = sum(
            1 for sq in ex.sub_questions
            if sq.semantic_entropy <= 0.0 and sq.num_correct == sq.num_runs
        ) / len(ex.sub_questions) if ex.sub_questions else float("nan")

        # Prefer actual no-search aggregate runs; fall back to independence-composition proxy.
        nosearch_entry = None
        if nosearch_by_model:
            model_map = nosearch_by_model.get(ex.model, {})
            nosearch_entry = model_map.get(ex.example_id)

        if nosearch_entry is not None:
            runs = nosearch_entry.get("runs", [])
            n_runs = len(runs)
            no_search_agg_runs_correct = sum(1 for r in runs if r.get("is_correct", False))
            no_search_agg_accuracy = no_search_agg_runs_correct / n_runs if n_runs > 0 else float("nan")
            nosearch_source = "actual"
        else:
            # Independence-composition proxy:
            # assume aggregate correct iff ALL hops correct in the same run.
            hop_runs = [sq.run_correctness for sq in ex.sub_questions if sq.run_correctness]
            if hop_runs:
                n_runs = len(hop_runs[0])
                agg_correct_per_run = [all(hop_runs[h][i] for h in range(len(hop_runs))) for i in range(n_runs)]
                no_search_agg_runs_correct = sum(agg_correct_per_run)
                no_search_agg_accuracy = no_search_agg_runs_correct / n_runs
            else:
                n_runs = 0
                no_search_agg_runs_correct = 0
                no_search_agg_accuracy = float("nan")
            nosearch_source = "composed"

        rows.append({
            "model": ex.model,
            "example_id": ex.example_id,
            "num_hops": len(ex.sub_questions),
            "total_search_calls": total,
            "parametric_redundant_count": par_red,
            "sequential_redundant_count": seq_red,
            "both_redundant_count": both_red,
            "parametric_only_count": par_only,
            "sequential_only_count": seq_only,
            "total_redundant_count": total_red,
            "necessary_count": necessary,
            "mean_parametric_accuracy": mean_par_acc,
            "frac_hops_certain": frac_certain,
            "aggregate_correct": ex.aggregate_correct,
            "no_search_agg_runs_correct": no_search_agg_runs_correct,
            "no_search_agg_accuracy": no_search_agg_accuracy,
            "n_runs": n_runs,
            "nosearch_source": nosearch_source,
        })
    return pd.DataFrame(rows)


# ─── WILSON CI ────────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple:
    """Wilson score confidence interval for proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ─── ATTRIBUTION EXPORT ───────────────────────────────────────────────────────

def save_attributions(all_examples: list, output_dir: str):
    """Write attributions_<model>.json per model into output_dir."""
    from collections import defaultdict
    by_model: dict = defaultdict(list)
    for ex in all_examples:
        by_model[ex.model].append(ex)

    for model, examples in by_model.items():
        records = []
        for ex in examples:
            queries = []
            for qr in ex.query_records:
                queries.append({
                    "query_index": qr.query_index,
                    "query": qr.query,
                    "assigned_hop": qr.assigned_hop,
                    "hop_question": qr.hop_question,
                    "gold_answer_for_hop": qr.gold_answer_for_hop,
                    "attribution_confidence": qr.attribution_confidence,
                    "attribution_reasoning": qr.attribution_reasoning,
                    "search_result_text": qr.search_result_text,
                    "gold_found_in_results": qr.gold_found_in_results,
                    "is_parametric_redundant": qr.is_parametric_redundant,
                    "is_sequential_redundant": qr.is_sequential_redundant,
                    "multi_hop_relevant": qr.multi_hop_relevant,
                })
            records.append({
                "example_id": ex.example_id,
                "aggregate_question": ex.aggregate_question,
                "aggregate_correct": ex.aggregate_correct,
                "queries": queries,
            })
        out_path = os.path.join(output_dir, f"attributions_{model}.json")
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        print(f"  Saved: {out_path} ({len(records)} examples)")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading eval data...")
    eval_by_model = load_eval_data(args.eval_dir)

    print("Loading traces...")
    traces_by_model = load_traces(args.eval_dir)

    print("Loading no-search aggregate data (if available)...")
    nosearch_by_model = load_nosearch_aggregate(args.eval_dir)

    # Set up LLM attribution agent if needed
    attribution_agent = None
    if args.use_llm:
        from src.services.base_agent import BaseAgent
        print(f"Initializing LLM attribution agent: {args.attribution_provider}/{args.attribution_model}")
        attribution_agent = BaseAgent(
            model_name=args.attribution_model,
            provider_name=args.attribution_provider,
            output_type=QueryAttribution,
            agent_name="QueryAttributionAgent",
            use_thinking=False,
        )

    # Load attribution cache
    cache_path = os.path.join(args.output_dir, "attribution_cache.json")
    cache: dict = {}
    if os.path.exists(cache_path) and not args.reattribute:
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Loaded {len(cache)} cached attributions from {cache_path}")

    # Set up multi-hop attribution agent if needed
    multi_hop_agent = None
    multi_hop_cache: dict = {}
    multi_hop_cache_path = os.path.join(args.output_dir, "multi_hop_cache.json")
    if args.check_multi_hop:
        if not args.use_llm:
            print("Warning: --check-multi-hop requires --use-llm. Skipping multi-hop analysis.")
        else:
            from src.services.base_agent import BaseAgent
            print(f"Initializing multi-hop attribution agent: {args.attribution_provider}/{args.attribution_model}")
            multi_hop_agent = BaseAgent(
                model_name=args.attribution_model,
                provider_name=args.attribution_provider,
                output_type=MultiHopAttribution,
                agent_name="MultiHopAttributionAgent",
                use_thinking=False,
            )
            if os.path.exists(multi_hop_cache_path) and not args.reattribute:
                with open(multi_hop_cache_path) as f:
                    multi_hop_cache = json.load(f)
                print(f"  Loaded {len(multi_hop_cache)} cached multi-hop attributions")

    # Set up gold-answer LLM judge if needed
    gold_judge_agent = None
    gold_judge_cache: dict = {}
    gold_judge_cache_path = os.path.join(args.output_dir, "gold_check_cache.json")
    if args.llm_gold_check:
        if not args.use_llm:
            print("Warning: --llm-gold-check requires --use-llm. Using string match.")
        else:
            from src.services.base_agent import BaseAgent
            print(f"Initializing gold-found judge: {args.attribution_provider}/{args.attribution_model}")
            gold_judge_agent = BaseAgent(
                model_name=args.attribution_model,
                provider_name=args.attribution_provider,
                output_type=GoldFoundJudgment,
                agent_name="GoldFoundJudgeAgent",
                use_thinking=False,
            )
            if os.path.exists(gold_judge_cache_path) and not args.reattribute:
                with open(gold_judge_cache_path) as f:
                    gold_judge_cache = json.load(f)
                print(f"  Loaded {len(gold_judge_cache)} cached gold-check results")

    # Match and process each model
    all_examples = []
    for model in sorted(eval_by_model.keys()):
        if model not in traces_by_model:
            print(f"  Warning: no traces found for model {model} — skipping")
            continue

        checkpoint_path = os.path.join(args.output_dir, f"matched_examples_{model}.json")
        if not args.reattribute and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                examples = _examples_from_dict(json.load(f))
            print(f"\nLoaded checkpoint for {model}: {len(examples)} examples from {os.path.basename(checkpoint_path)}")
        else:
            print(f"\nProcessing model: {model}")
            examples = match_and_build(
                eval_entries=eval_by_model[model],
                traces=traces_by_model[model],
                model=model,
                use_llm=args.use_llm,
                attribution_agent=attribution_agent,
                cache=cache,
                reattribute=args.reattribute,
                multi_hop_agent=multi_hop_agent,
                multi_hop_cache=multi_hop_cache,
                gold_judge_agent=gold_judge_agent,
                gold_judge_cache=gold_judge_cache,
            )
            with open(checkpoint_path, "w") as f:
                json.dump(_examples_to_dict(examples), f)
            print(f"  Saved checkpoint: {os.path.basename(checkpoint_path)}")

        all_examples.extend(examples)

    if not all_examples:
        print("No matched examples found. Exiting.")
        return

    # Save intermediate attributions
    print("\nSaving attributions...")
    save_attributions(all_examples, args.output_dir)

    # Save attribution cache
    if args.use_llm:
        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\nSaved attribution cache ({len(cache)} entries) to {cache_path}")

    # Save multi-hop cache
    if multi_hop_agent is not None:
        with open(multi_hop_cache_path, "w") as f:
            json.dump(multi_hop_cache, f, indent=2)
        print(f"Saved multi-hop cache ({len(multi_hop_cache)} entries) to {multi_hop_cache_path}")

    # Compute metrics
    print("\nComputing metrics...")
    hop_df = compute_hop_metrics(
        all_examples,
        gold_judge_agent=gold_judge_agent,
        gold_judge_cache=gold_judge_cache,
        reattribute=args.reattribute,
    )
    example_df = compute_example_metrics(all_examples, nosearch_by_model=nosearch_by_model or None)

    # Save gold-check cache (after compute_hop_metrics, which adds gf_xhop: keys)
    if gold_judge_agent is not None:
        with open(gold_judge_cache_path, "w") as f:
            json.dump(gold_judge_cache, f, indent=2)
        print(f"Saved gold-check cache ({len(gold_judge_cache)} entries) to {gold_judge_cache_path}")

    # Save hop-level summary CSV
    csv_path = os.path.join(args.output_dir, "interplay_summary.csv")
    hop_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(hop_df)} rows)")

    # Save example-level metrics CSV
    example_csv_path = os.path.join(args.output_dir, "example_metrics.csv")
    example_df.to_csv(example_csv_path, index=False)
    print(f"  Saved: {example_csv_path} ({len(example_df)} rows)")

    # Print quick summary
    print(f"\nHop-level rows: {len(hop_df)}")
    print(f"Example-level rows: {len(example_df)}")
    for model in sorted(hop_df["model"].unique()):
        sub_h = hop_df[hop_df["model"] == model]
        sub_e = example_df[example_df["model"] == model]
        total_q = sub_e["total_search_calls"].sum()
        par = sub_e["parametric_redundant_count"].sum()
        seq = sub_e["sequential_redundant_count"].sum()
        acc = sub_e["aggregate_correct"].mean()
        if len(sub_h) > 3:
            rho, pval = stats.spearmanr(sub_h["semantic_entropy"], sub_h["queries_assigned_count"])
            print(f"  {short_model(model)}: queries={total_q}, par_red={par} ({100*par/total_q:.1f}%), "
                  f"seq_red={seq} ({100*seq/total_q:.1f}%), acc={100*acc:.1f}%, ρ={rho:.3f} (p={pval:.3f})")
        else:
            print(f"  {short_model(model)}: queries={total_q}, par_red={par}, seq_red={seq}, acc={100*acc:.1f}%")

    print(f"\nDone. CSVs and checkpoints in: {args.output_dir}")
    print("Run unified_interplay_analysis.py with --datasets and --example-metrics to generate plots.")


if __name__ == "__main__":
    main()
