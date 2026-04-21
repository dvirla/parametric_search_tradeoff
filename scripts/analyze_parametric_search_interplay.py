"""
Parametric–Search Interplay Analysis

Analyzes how per-hop parametric uncertainty drives search behavior and query redundancy.
For each matched trace ↔ eval pair: attributes search queries to hops, computes
parametric/sequential redundancy, and generates correlation plots and a report.
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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from pydantic import BaseModel

matplotlib.use("Agg")

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
        help="Output directory for plots, CSV, and report."
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


# ─── PLOTTING ─────────────────────────────────────────────────────────────────

PALETTE = sns.color_palette("tab10")
MODEL_COLORS = {}


def _model_colors(models):
    global MODEL_COLORS
    for i, m in enumerate(sorted(models)):
        MODEL_COLORS[m] = PALETTE[i % len(PALETTE)]


def short_model(name: str) -> str:
    """Shorten model name for plot labels."""
    replacements = {
        "gemini-3-pro-preview": "Gemini-3.1-Pro",
        "nemotron-3-nano_30b": "Nemotron-30B",
        "qwen3.5_122b": "Qwen3.5-122B",
    }
    return replacements.get(name, name)


def plot_certain_vs_uncertain_queries(hop_df: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart: mean queries assigned for certain vs uncertain hops, per model.
    Directly answers: do models issue more queries on uncertain hops?
    """
    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = hop_df[hop_df["model"] == model].copy()
        groups = [("Certain", True), ("Uncertain", False)]
        x = np.arange(len(groups))
        means, errors, ns = [], [], []

        for _, is_certain in groups:
            grp = sub[sub["parametric_certain"] == is_certain]
            if grp.empty:
                means.append(0.0); errors.append(0.0); ns.append(0)
            else:
                vals = grp["queries_assigned_count"]
                means.append(vals.mean())
                errors.append(vals.sem() if len(vals) > 1 else 0.0)
                ns.append(len(vals))

        bars = ax.bar(x, means, yerr=errors, color=[PALETTE[2], PALETTE[3]],
                      capsize=4, alpha=0.85, width=0.5)
        for bar, m, cnt in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{m:.2f}\nn={cnt}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(["Certain", "Uncertain"], fontsize=10)
        ax.set_title(short_model(model), fontsize=11)
        ax.set_ylabel("Mean Queries Assigned ± SE", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Mean Queries per Hop: Certain vs. Uncertain", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "certain_vs_uncertain_queries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_entropy_buckets(hop_df: pd.DataFrame, output_dir: str):
    """Bar chart: mean queries per hop by entropy bucket, per model."""
    def bucket(e):
        if e <= 0.0:
            return "certain (=0)"
        elif e <= 0.5:
            return "low (0–0.5]"
        elif e <= 1.0:
            return "medium (0.5–1.0]"
        else:
            return "high (>1.0)"

    bucket_order = ["certain (=0)", "low (0–0.5]", "medium (0.5–1.0]", "high (>1.0)"]
    df = hop_df.copy()
    df["entropy_bucket"] = df["semantic_entropy"].apply(bucket)
    df["entropy_bucket"] = pd.Categorical(df["entropy_bucket"], categories=bucket_order, ordered=True)

    models = sorted(df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = df[df["model"] == model]
        agg = sub.groupby("entropy_bucket", observed=True)["queries_assigned_count"].agg(["mean", "sem"]).reindex(bucket_order)
        x = np.arange(len(bucket_order))
        bars = ax.bar(x, agg["mean"].fillna(0), yerr=agg["sem"].fillna(0),
                      color=[PALETTE[i] for i in range(len(bucket_order))],
                      capsize=4, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bucket_order, rotation=20, ha="right", fontsize=8)
        ax.set_title(short_model(model), fontsize=11)
        ax.set_xlabel("Entropy Bucket", fontsize=9)
        ax.set_ylabel("Mean Queries Assigned ± SE", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

        # Annotate counts
        counts = sub.groupby("entropy_bucket", observed=True).size().reindex(bucket_order).fillna(0)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"n={int(cnt)}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Mean Queries per Hop by Entropy Bucket", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "entropy_bucket_queries.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_redundancy_breakdown(example_df: pd.DataFrame, output_dir: str):
    """Stacked bar chart: proportion of queries by type per model."""
    models = sorted(example_df["model"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), 5))

    x = np.arange(len(models))
    width = 0.5

    model_list = sorted(example_df["model"].unique())
    totals   = example_df.groupby("model")["total_search_calls"].sum().reindex(model_list).fillna(0)
    necessary = example_df.groupby("model")["necessary_count"].sum().reindex(model_list).fillna(0)
    # Use exclusive counts so segments sum exactly to 1
    par_only = example_df.groupby("model")["parametric_only_count"].sum().reindex(model_list).fillna(0)
    seq_only = example_df.groupby("model")["sequential_only_count"].sum().reindex(model_list).fillna(0)

    denom = totals.replace(0, np.nan)
    prop_nec = (necessary / denom).fillna(0)
    prop_par = (par_only  / denom).fillna(0)
    prop_seq = (seq_only  / denom).fillna(0)

    ax.bar(x, prop_nec, width, label="Necessary", color="#2ecc71", alpha=0.85)
    ax.bar(x, prop_par, width, bottom=prop_nec, label="Parametrically Redundant", color="#e74c3c", alpha=0.85)
    ax.bar(x, prop_seq, width, bottom=prop_nec + prop_par, label="Sequentially Redundant (only)", color="#f39c12", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in model_list], rotation=15, ha="right")
    ax.set_ylabel("Proportion of Queries")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Query Redundancy Breakdown by Model")
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate raw totals
    for i, m in enumerate(model_list):
        ax.text(i, 1.02, f"n={int(totals.get(m, 0))}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "redundancy_breakdown.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_sequential_redundancy_distribution(example_df: pd.DataFrame, output_dir: str):
    """
    For examples with at least one sequentially redundant query, show the distribution
    of how many redundant queries were issued per example.
    """
    models = sorted(example_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = example_df[example_df["model"] == model]
        seq_pos = sub[sub["sequential_redundant_count"] > 0]["sequential_redundant_count"]
        if seq_pos.empty:
            ax.text(0.5, 0.5, "No sequential redundancy", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(short_model(model), fontsize=11)
            continue

        counts = seq_pos.value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=PALETTE[2], alpha=0.85, edgecolor="white")
        if len(counts) <= 10:
            ax.set_xticks(counts.index)
        else:
            from matplotlib.ticker import MaxNLocator
            ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.set_xlabel("Number of sequentially redundant queries per example", fontsize=9)
        ax.set_ylabel("Number of examples", fontsize=9)
        ax.set_title(f"{short_model(model)}\n(n={len(seq_pos)} examples with ≥1 seq. redundant query)", fontsize=10)
        ax.grid(True, axis="y", alpha=0.3)
        for xi, yi in zip(counts.index, counts.values):
            ax.text(xi, yi + 0.1, str(yi), ha="center", va="bottom", fontsize=8)

    fig.suptitle("Distribution of Sequentially Redundant Query Counts\n"
                 "(among examples with at least one sequentially redundant query)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "sequential_redundancy_distribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_redundancy_overlap(example_df: pd.DataFrame, output_dir: str):
    """
    Stacked bar showing proportion of queries in 4 exclusive categories:
    Necessary / Parametric-only / Both (par+seq) / Sequential-only.
    Reveals the overlap between the two redundancy types.
    """
    models = sorted(example_df["model"].unique())
    fig, ax = plt.subplots(figsize=(max(6, 2.5 * len(models)), 5))

    x = np.arange(len(models))
    width = 0.5

    totals    = example_df.groupby("model")["total_search_calls"].sum().reindex(models).fillna(0)
    necessary = example_df.groupby("model")["necessary_count"].sum().reindex(models).fillna(0)
    both      = example_df.groupby("model")["both_redundant_count"].sum().reindex(models).fillna(0)
    seq_only  = example_df.groupby("model")["sequential_only_count"].sum().reindex(models).fillna(0)
    # par_only here means par AND NOT seq (exclusive)
    par_only  = example_df.groupby("model")["parametric_only_count"].sum().reindex(models).fillna(0) - both

    denom = totals.replace(0, np.nan)
    prop_nec  = (necessary / denom).fillna(0)
    prop_par  = (par_only  / denom).fillna(0)
    prop_both = (both      / denom).fillna(0)
    prop_seq  = (seq_only  / denom).fillna(0)

    ax.bar(x, prop_nec,  width, label="Necessary",            color="#2ecc71", alpha=0.85)
    ax.bar(x, prop_par,  width, bottom=prop_nec,              label="Parametric-only",  color="#e74c3c", alpha=0.85)
    ax.bar(x, prop_both, width, bottom=prop_nec + prop_par,   label="Both (par + seq)", color="#9b59b6", alpha=0.85)
    ax.bar(x, prop_seq,  width, bottom=prop_nec + prop_par + prop_both,
           label="Sequential-only", color="#f39c12", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("Proportion of Queries")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Query Redundancy Overlap: Parametric vs. Sequential")
    ax.grid(True, axis="y", alpha=0.3)

    for i, m in enumerate(models):
        ax.text(i, 1.04, f"n={int(totals.get(m, 0))}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = os.path.join(output_dir, "redundancy_overlap.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_accuracy_outcomes(hop_df: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart per model: aggregate accuracy broken down by
    (parametric_certain × searched).  Shows whether searching rescues
    uncertain hops and whether it wastes effort on certain ones.
    """
    groups = {
        "Certain\n+Searched": (True, True),
        "Certain\n+Not Searched": (True, False),
        "Uncertain\n+Searched": (False, True),
        "Uncertain\n+Not Searched": (False, False),
    }
    group_colors = ["#e74c3c", "#c0392b", "#2ecc71", "#27ae60"]

    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = hop_df[hop_df["model"] == model].copy()
        x = np.arange(len(groups))
        means, errors, ns = [], [], []

        for (certain, searched) in groups.values():
            mask = (sub["parametric_certain"] == certain) & (sub["searched"] == searched)
            grp = sub[mask]
            if grp.empty:
                means.append(0.0)
                errors.append(0.0)
                ns.append(0)
            else:
                acc_vals = grp["aggregate_correct"].astype(float)
                m = acc_vals.mean()
                se = acc_vals.sem() if len(acc_vals) > 1 else 0.0
                means.append(m)
                errors.append(se)
                ns.append(len(grp))

        bars = ax.bar(x, means, yerr=errors, color=group_colors,
                      capsize=4, alpha=0.85, width=0.6)

        # Annotate n and accuracy
        for bar, m, n_cnt in zip(bars, means, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{100*m:.0f}%\nn={n_cnt}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x)
        ax.set_xticklabels(list(groups.keys()), fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Aggregate Accuracy")
        ax.set_title(short_model(model), fontsize=11)
        ax.axhline(sub["aggregate_correct"].mean(), color="gray",
                   linestyle="--", linewidth=1, alpha=0.6, label="Overall avg")
        ax.legend(fontsize=7)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Aggregate Accuracy by Parametric Certainty × Search Decision\n"
                 "(hop-level groups, outcome = aggregate question correctness)",
                 fontsize=12)
    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_by_certainty_search.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_accuracy_vs_queries_bars(hop_df: pd.DataFrame, output_dir: str):
    """
    Grouped bar chart: mean queries per hop by parametric accuracy bin, per model.
    Bins: perfect (=1.0), partial (0–1), zero (=0).
    Directly answers: do models search more when parametric accuracy was low?
    """
    def acc_bin(v):
        if pd.isna(v):
            return None
        if v >= 1.0:
            return "perfect (=1.0)"
        elif v <= 0.0:
            return "zero (=0)"
        else:
            return "partial (0–1)"

    bin_order = ["perfect (=1.0)", "partial (0–1)", "zero (=0)"]
    bin_colors = [PALETTE[2], PALETTE[1], PALETTE[3]]

    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = hop_df[hop_df["model"] == model].copy()
        sub["acc_bin"] = sub["parametric_accuracy"].apply(acc_bin)
        sub = sub.dropna(subset=["acc_bin"])
        sub["acc_bin"] = pd.Categorical(sub["acc_bin"], categories=bin_order, ordered=True)

        agg = sub.groupby("acc_bin", observed=True)["queries_assigned_count"].agg(
            ["mean", "sem", "count"]
        ).reindex(bin_order)
        x = np.arange(len(bin_order))

        bars = ax.bar(x, agg["mean"].fillna(0), yerr=agg["sem"].fillna(0),
                      color=bin_colors, capsize=4, alpha=0.85)
        for bar, (_, row) in zip(bars, agg.iterrows()):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{row['mean']:.2f}\nn={int(row['count']) if not pd.isna(row['count']) else 0}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(bin_order, rotation=10, ha="right", fontsize=9)
        ax.set_title(short_model(model), fontsize=11)
        ax.set_ylabel("Mean Queries Assigned ± SE", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Mean Queries per Hop by Parametric Accuracy Bin", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_vs_queries_bars.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_searches_per_certainty_level(hop_df: pd.DataFrame, output_dir: str):
    """
    Bar chart: mean queries_assigned_count per certainty level (num_correct/num_runs),
    with 95% CI error bars (±1.96 * SEM).  One subplot per model.
    """
    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = hop_df[hop_df["model"] == model].copy()
        sub = sub.dropna(subset=["num_correct", "num_runs"])
        sub = sub[sub["num_runs"] > 0]
        n_runs = int(sub["num_runs"].mode()[0]) if not sub.empty else 5

        # Collapse into 3 groups: low (0/n_runs), middle (1..n_runs-1), high (n_runs/n_runs)
        def _cert_group(nc, nr):
            if nc == 0:
                return f"0/{nr}"
            elif nc == nr:
                return f"{nr}/{nr}"
            else:
                return f"1–{nr-1}/{nr}\n(middle)"

        sub["cert_group"] = sub.apply(lambda r: _cert_group(int(r["num_correct"]), int(r["num_runs"])), axis=1)
        group_order = [f"0/{n_runs}", f"1–{n_runs-1}/{n_runs}\n(middle)", f"{n_runs}/{n_runs}"]

        means, errs, ns, labels = [], [], [], []
        for grp_label in group_order:
            grp = sub[sub["cert_group"] == grp_label]["queries_assigned_count"]
            if grp.empty:
                continue
            means.append(grp.mean())
            sem = grp.sem() if len(grp) > 1 else 0.0
            errs.append(1.96 * sem)
            ns.append(len(grp))
            labels.append(grp_label)

        x = np.arange(len(labels))
        palette = plt.cm.RdYlGn(np.linspace(0.15, 0.85, len(labels)))
        bars = ax.bar(x, means, yerr=errs, capsize=4, alpha=0.85, color=palette)
        for bar, m, e, cnt in zip(bars, means, errs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + e + 0.02,
                    f"{m:.2f}\nn={cnt}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
        ax.set_title(short_model(model), fontsize=11)
        ax.set_xlabel("Certainty level (correct runs / total runs)", fontsize=9)
        ax.set_ylabel("Mean Queries Assigned ± 95% CI", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Mean Searches per Hop by Certainty Level", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "searches_per_certainty_level.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_searches_per_entropy_level(hop_df: pd.DataFrame, output_dir: str):
    """
    Bar chart: mean queries_assigned_count per semantic entropy bucket
    (certain / low / medium / high), with 95% CI error bars. One subplot per model.
    """
    bucket_order = ["certain\n(=0)", "low\n(0–0.5]", "medium\n(0.5–1.0]", "high\n(>1.0)"]

    def _entropy_bucket(e: float) -> str:
        if e <= 0.0:
            return "certain\n(=0)"
        elif e <= 0.5:
            return "low\n(0–0.5]"
        elif e <= 1.0:
            return "medium\n(0.5–1.0]"
        else:
            return "high\n(>1.0)"

    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    for ax, model in zip(axes[0], models):
        sub = hop_df[hop_df["model"] == model].dropna(subset=["semantic_entropy"]).copy()
        sub["entropy_bucket"] = sub["semantic_entropy"].apply(_entropy_bucket)

        means, errs, ns, labels = [], [], [], []
        for bucket in bucket_order:
            grp = sub[sub["entropy_bucket"] == bucket]["queries_assigned_count"]
            if grp.empty:
                continue
            means.append(grp.mean())
            sem = grp.sem() if len(grp) > 1 else 0.0
            errs.append(1.96 * sem)
            ns.append(len(grp))
            labels.append(bucket)

        if not means:
            continue

        x = np.arange(len(labels))
        palette = plt.cm.RdYlGn(np.linspace(0.85, 0.15, len(labels)))  # certain=green, high=red
        bars = ax.bar(x, means, yerr=errs, capsize=4, alpha=0.85, color=palette)
        for bar, m, e, cnt in zip(bars, means, errs, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + e + 0.02,
                    f"{m:.2f}\nn={cnt}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
        ax.set_title(short_model(model), fontsize=11)
        ax.set_xlabel("Semantic entropy bucket", fontsize=9)
        ax.set_ylabel("Mean Queries Assigned ± 95% CI", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Mean Searches per Hop by Semantic Entropy Level", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "searches_per_entropy_level.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_missed_search_cross_hop_coverage(hop_df: pd.DataFrame, output_dir: str):
    """
    For hops classified as 'missed search' (uncertain + not searched),
    show what fraction had their gold answer implicitly covered by searches on other hops.
    """
    missed = hop_df[(~hop_df["parametric_certain"]) & (~hop_df["searched"])].copy()

    models = sorted(hop_df["model"].unique())
    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(models)), 5))
    x = np.arange(len(models))
    truly_missed_pcts, cross_covered_pcts, ns = [], [], []

    for model in models:
        sub = missed[missed["model"] == model]
        n_total = len(sub)
        n_covered = int(sub["missed_search_answer_found_cross_hop"].sum())
        n_truly = n_total - n_covered
        truly_missed_pcts.append(100 * n_truly / n_total if n_total else 0.0)
        cross_covered_pcts.append(100 * n_covered / n_total if n_total else 0.0)
        ns.append(n_total)

    ax.bar(x, truly_missed_pcts, 0.5, label="Truly missed", color="#e67e22", alpha=0.85)
    ax.bar(x, cross_covered_pcts, 0.5, bottom=truly_missed_pcts,
           label="Cross-hop covered", color="#9b59b6", alpha=0.85)

    for i, (n, tp, cp) in enumerate(zip(ns, truly_missed_pcts, cross_covered_pcts)):
        ax.text(i, tp + cp + 1.5, f"n={n}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("% of Missed-Search Hops")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.set_title("Missed Search: Truly Missed vs. Cross-Hop Covered\n"
                 "(cross-hop covered = gold answer found in another hop's search results)")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "missed_search_cross_hop_coverage.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_multi_hop_query_attribution(all_examples: list, output_dir: str):
    """
    Stacked bar chart per model: proportion of queries that are
    single-hop relevant vs. multi-hop relevant (per LLM judge).
    """
    from collections import defaultdict
    model_counts: dict = defaultdict(lambda: {"single": 0, "multi": 0})

    for ex in all_examples:
        for qr in ex.query_records:
            key = "multi" if qr.multi_hop_relevant else "single"
            model_counts[ex.model][key] += 1

    models = sorted(model_counts.keys())
    if not models:
        return

    fig, ax = plt.subplots(figsize=(max(5, 2.5 * len(models)), 5))
    x = np.arange(len(models))

    singles = [model_counts[m]["single"] for m in models]
    multis  = [model_counts[m]["multi"]  for m in models]
    totals  = [s + mt for s, mt in zip(singles, multis)]

    single_pcts = [100 * s / t if t else 0 for s, t in zip(singles, totals)]
    multi_pcts  = [100 * mt / t if t else 0 for mt, t in zip(multis, totals)]

    ax.bar(x, single_pcts, 0.5, label="Single-hop relevant", color="#3498db", alpha=0.85)
    ax.bar(x, multi_pcts,  0.5, bottom=single_pcts,
           label="Multi-hop relevant", color="#e67e22", alpha=0.85)

    for i, (t, sp, mp) in enumerate(zip(totals, single_pcts, multi_pcts)):
        ax.text(i, sp + mp + 1.5, f"n={t}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([short_model(m) for m in models], rotation=15, ha="right")
    ax.set_ylabel("% of Queries")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=9)
    ax.set_title("Multi-Hop Query Attribution (LLM Judge)\n"
                 "Multi-hop = query judged relevant to >1 sub-question")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "multi_hop_query_attribution.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_multi_hop_query_attribution_by_hop_type(all_examples: list, output_dir: str):
    """
    Like plot_multi_hop_query_attribution but segmented by question complexity
    (2-hop, 3-hop, 4-hop). One subplot per model, grouped bars per hop type.
    """
    from collections import defaultdict
    # model -> num_hops -> {"single": int, "multi": int}
    model_hop_counts: dict = defaultdict(lambda: defaultdict(lambda: {"single": 0, "multi": 0}))

    for ex in all_examples:
        num_hops = len(ex.sub_questions)
        for qr in ex.query_records:
            key = "multi" if qr.multi_hop_relevant else "single"
            model_hop_counts[ex.model][num_hops][key] += 1

    models = sorted(model_hop_counts.keys())
    if not models:
        return

    all_hop_types = sorted({h for m in models for h in model_hop_counts[m]})
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(max(5, 3.5 * len(all_hop_types)) * n, 5), squeeze=False)

    for col, model in enumerate(models):
        ax = axes[0][col]
        hop_data = model_hop_counts[model]
        x = np.arange(len(all_hop_types))
        single_pcts, multi_pcts, totals = [], [], []

        for h in all_hop_types:
            d = hop_data.get(h, {"single": 0, "multi": 0})
            t = d["single"] + d["multi"]
            totals.append(t)
            single_pcts.append(100 * d["single"] / t if t else 0)
            multi_pcts.append(100 * d["multi"] / t if t else 0)

        ax.bar(x, single_pcts, 0.5, label="Single-hop relevant", color="#3498db", alpha=0.85)
        ax.bar(x, multi_pcts, 0.5, bottom=single_pcts,
               label="Multi-hop relevant", color="#e67e22", alpha=0.85)

        for i, (t, sp, mp) in enumerate(zip(totals, single_pcts, multi_pcts)):
            ax.text(i, sp + mp + 1.5, f"n={t}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}-hop" for h in all_hop_types])
        ax.set_ylabel("% of Queries")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9)
        ax.set_title(short_model(model), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Multi-Hop Query Attribution by Question Type (LLM Judge)\n"
                 "Multi-hop = query judged relevant to >1 sub-question",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "multi_hop_query_attribution_by_hop_type.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_accuracy_per_hop(hop_df: pd.DataFrame, output_dir: str):
    """
    Grid of bar charts: parametric accuracy per hop index + aggregate accuracy.
    Rows = models, Columns = num_hops values present in data.
    Each cell uses only examples of that complexity (num_hops == n), so
    Hop 1 in a 2-hop question is never mixed with Hop 1 in a 3-hop question.
    """
    models = sorted(hop_df["model"].unique())
    unique_num_hops = sorted(hop_df["num_hops"].dropna().unique().astype(int))
    n_rows = len(models)
    n_cols = len(unique_num_hops)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(max(6, 4 * n_cols), 4 * n_rows),
        squeeze=False,
    )

    for row, model in enumerate(models):
        model_sub = hop_df[hop_df["model"] == model]
        for col, num_hops in enumerate(unique_num_hops):
            ax = axes[row][col]
            sub = model_sub[model_sub["num_hops"] == num_hops]

            if sub.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=10, color="gray")
                ax.set_title(f"{short_model(model)}\n{num_hops}-hop", fontsize=9)
                ax.axis("off")
                continue

            hop_indices = list(range(num_hops))
            labels, means, errors, ns = [], [], [], []

            for hop in hop_indices:
                grp = sub[sub["hop_index"] == hop]["parametric_accuracy"].dropna()
                labels.append(f"Hop {hop}")
                means.append(grp.mean() if len(grp) > 0 else 0.0)
                errors.append(grp.sem() if len(grp) > 1 else 0.0)
                ns.append(len(grp))

            # Aggregate accuracy — deduplicated per example
            agg_grp = sub.drop_duplicates("example_id")["aggregate_correct"].astype(float)
            labels.append("Aggregate")
            means.append(agg_grp.mean() if len(agg_grp) > 0 else 0.0)
            errors.append(agg_grp.sem() if len(agg_grp) > 1 else 0.0)
            ns.append(len(agg_grp))

            x = np.arange(len(labels))
            colors = [PALETTE[i % len(PALETTE)] for i in range(num_hops)] + ["#2c3e50"]
            bars = ax.bar(x, means, yerr=errors, color=colors, capsize=4, alpha=0.85, width=0.6)

            for bar, m, cnt in zip(bars, means, ns):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f"{100*m:.0f}%\nn={cnt}", ha="center", va="bottom", fontsize=8)

            # Vertical separator before Aggregate bar
            ax.axvline(num_hops - 0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim(0, 1.25)
            ax.set_ylabel("Accuracy", fontsize=9)
            ax.set_title(f"{short_model(model)}\n{num_hops}-hop", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Parametric Accuracy per Hop + Aggregate Accuracy\n(stratified by question complexity)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, "accuracy_per_hop.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_searched_vs_unsearched(hop_df: pd.DataFrame, output_dir: str):
    """
    2-row × n_models grid:
      Row 1: mean parametric_accuracy for Not Searched vs Searched hops.
      Row 2: mean semantic_entropy for Not Searched vs Searched hops.
    """
    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8), squeeze=False)

    bar_colors = [PALETTE[0], PALETTE[1]]  # Not Searched, Searched
    conditions = [("Not Searched", False), ("Searched", True)]

    for col, model in enumerate(models):
        sub = hop_df[hop_df["model"] == model].copy()

        for row, (metric, ylabel) in enumerate([
            ("parametric_accuracy", "Mean Parametric Accuracy ± SE"),
            ("semantic_entropy",    "Mean Semantic Entropy ± SE"),
        ]):
            ax = axes[row][col]
            means, errors, ns = [], [], []

            for _, is_searched in conditions:
                grp = sub[sub["searched"] == is_searched][metric].dropna()
                if grp.empty:
                    means.append(0.0); errors.append(0.0); ns.append(0)
                else:
                    means.append(grp.mean())
                    errors.append(grp.sem() if len(grp) > 1 else 0.0)
                    ns.append(len(grp))

            x = np.arange(len(conditions))
            bars = ax.bar(x, means, yerr=errors, color=bar_colors,
                          capsize=4, alpha=0.85, width=0.5)

            for bar, m, cnt in zip(bars, means, ns):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(errors) * 0.1 + 0.01,
                        f"{m:.3f}\nn={cnt}", ha="center", va="bottom", fontsize=8)

            # Overall mean dashed line
            overall = sub[metric].dropna().mean()
            if not np.isnan(overall):
                ax.axhline(overall, color="gray", linestyle="--", linewidth=1,
                           alpha=0.7, label=f"Overall: {overall:.3f}")
                ax.legend(fontsize=7)

            ax.set_xticks(x)
            ax.set_xticklabels([c[0] for c in conditions], fontsize=10)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)
            if row == 0:
                ax.set_title(short_model(model), fontsize=11)

    axes[0][0].set_ylabel("Mean Parametric Accuracy ± SE", fontsize=9)
    axes[1][0].set_ylabel("Mean Semantic Entropy ± SE", fontsize=9)

    fig.suptitle("Parametric Profile: Searched vs. Not-Searched Hops\n"
                 "(Row 1: parametric accuracy; Row 2: semantic entropy)",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = os.path.join(output_dir, "searched_vs_unsearched.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_search_calibration(hop_df: pd.DataFrame, output_dir: str):
    """
    4-bar chart per model showing % of hops in each calibration quadrant:
      Uncertain + Searched  → "Correct search"  (green)
      Uncertain + Not Searched → "Missed search"   (orange)
      Certain  + Not Searched → "Correct skip"   (blue)
      Certain  + Searched  → "Wasteful search" (red)
    """
    quadrants = [
        ("Correct\nsearch",  False, True,  "#2ecc71"),
        ("Missed\nsearch",   False, False, "#e67e22"),
        ("Correct\nskip",    True,  False, "#3498db"),
        ("Wasteful\nsearch", True,  True,  "#e74c3c"),
    ]

    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 5), squeeze=False)

    for ax, model in enumerate(models):
        ax_obj = axes[0][ax]
        sub = hop_df[hop_df["model"] == model].copy()
        total = len(sub)

        proportions, counts, labels, colors = [], [], [], []
        for label, certain_val, searched_val, color in quadrants:
            mask = (sub["parametric_certain"] == certain_val) & (sub["searched"] == searched_val)
            cnt = mask.sum()
            proportions.append(100 * cnt / total if total > 0 else 0.0)
            counts.append(int(cnt))
            labels.append(label)
            colors.append(color)

        x = np.arange(len(quadrants))
        bars = ax_obj.bar(x, proportions, color=colors, alpha=0.85, width=0.6)

        for bar, pct, cnt in zip(bars, proportions, counts):
            ax_obj.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.5,
                        f"{pct:.1f}%\nn={cnt}", ha="center", va="bottom", fontsize=8)

        ax_obj.set_xticks(x)
        ax_obj.set_xticklabels(labels, fontsize=9)
        ax_obj.set_ylim(0, max(proportions) * 1.3 + 5)
        ax_obj.set_ylabel("% of Total Hops", fontsize=9)
        ax_obj.set_title(short_model(model), fontsize=11)
        ax_obj.grid(True, axis="y", alpha=0.3)

        # Annotate total
        ax_obj.text(0.98, 0.98, f"Total hops: {total}", transform=ax_obj.transAxes,
                    ha="right", va="top", fontsize=8, color="gray")

    fig.suptitle("Search Calibration: Are Searches Triggered by Genuine Uncertainty?\n"
                 "(Proportions sum to 100% per model)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "search_calibration.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_search_calibration_extended(hop_df: pd.DataFrame, output_dir: str):
    """
    6-bar chart per model: splits "Correct search" into necessary vs. sequentially
    redundant, and splits "Missed search" into truly missed vs. cross-hop covered.

      Necessary search       — uncertain + searched + not seq-redundant            (green)
      Correct skip           — certain  + not searched                             (blue)
      Truly missed           — uncertain + not searched + NOT cross-hop covered     (orange)
      Cross-hop covered      — uncertain + not searched + cross-hop covered         (purple)
      Seq. redundant search  — uncertain + searched + seq-redundant                 (gold)
      Wasteful search        — certain  + searched (parametric redundant)           (red)
    """
    # tuple: (label, certain_val, searched_val, seq_val, cross_hop_val, color)
    # None means "no filter on that column"
    sextets = [
        ("Necessary\nsearch",  False, True,  False, None,  "#2ecc71"),  # good: uncertain+searched
        ("Correct\nskip",      True,  False, None,  None,  "#3498db"),  # good: certain+not searched
        ("Truly\nmissed",      False, False, None,  False, "#e67e22"),  # bad: uncertain+not searched+not cross-covered
        ("Cross-hop\ncovered", False, False, None,  True,  "#9b59b6"),  # info: uncertain+not searched+cross-covered
        ("Seq.\nredundant",    False, True,  True,  None,  "#f1c40f"),  # wasteful: seq redundant
        ("Wasteful\nsearch",   True,  True,  None,  None,  "#e74c3c"),  # bad: certain+searched
    ]

    models = sorted(hop_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 5), squeeze=False)

    for col, model in enumerate(models):
        ax = axes[0][col]
        sub = hop_df[hop_df["model"] == model].copy()
        total = len(sub)

        proportions, counts, labels, colors = [], [], [], []
        for label, certain_val, searched_val, seq_val, cross_hop_val, color in sextets:
            mask = (sub["parametric_certain"] == certain_val) & (sub["searched"] == searched_val)
            if seq_val is not None:
                mask &= (sub["seq_redundant"] == seq_val)
            if cross_hop_val is not None:
                mask &= (sub["missed_search_answer_found_cross_hop"] == cross_hop_val)
            cnt = int(mask.sum())
            proportions.append(100 * cnt / total if total > 0 else 0.0)
            counts.append(cnt)
            labels.append(label)
            colors.append(color)

        x = np.arange(len(sextets))
        bars = ax.bar(x, proportions, color=colors, alpha=0.85, width=0.6)

        for bar, pct, cnt in zip(bars, proportions, counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{pct:.1f}%\nn={cnt}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0, max(proportions) * 1.3 + 5)
        ax.set_ylabel("% of Total Hops", fontsize=9)
        ax.set_title(short_model(model), fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.text(0.98, 0.98, f"Total hops: {total}", transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="gray")

        # Vertical separator between search-type waste and missed/skip/parametric-waste
        ax.axvline(1.5, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    fig.suptitle("Search Calibration (Extended): Missed Search Split by Cross-Hop Coverage\n"
                 "(left of dashed line: correct decisions; right: search errors)",
                 fontsize=11, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "search_calibration_extended.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_queries_violin(hop_df: pd.DataFrame, output_dir: str):
    """Violin plot: queries_assigned_count by entropy quartile × model."""
    df = hop_df.copy()
    # Compute entropy quartiles across all models together for consistency
    labels_4 = ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
    labels_2 = ["Low\nhalf", "High\nhalf"]
    try:
        df["entropy_q"] = pd.qcut(df["semantic_entropy"], q=4, labels=labels_4, duplicates="drop")
    except ValueError:
        try:
            df["entropy_q"] = pd.qcut(df["semantic_entropy"], q=2, labels=labels_2, duplicates="drop")
        except ValueError:
            df["entropy_q"] = (df["semantic_entropy"] > 0).map({True: "Uncertain", False: "Certain"})

    fig, ax = plt.subplots(figsize=(10, 5))
    models = sorted(df["model"].unique())
    _model_colors(models)

    sns.violinplot(
        data=df,
        x="entropy_q",
        y="queries_assigned_count",
        hue="model",
        ax=ax,
        palette={m: MODEL_COLORS[m] for m in models},
        cut=0,
        inner="quartile",
        scale="width",
    )

    # Rename legend entries
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [short_model(l) for l in labels], fontsize=8, loc="upper left")

    ax.set_xlabel("Entropy Quartile")
    ax.set_ylabel("Queries Assigned to Hop")
    ax.set_title("Queries per Hop by Entropy Quartile × Model")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = os.path.join(output_dir, "queries_violin.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ─── ATTRIBUTIONS EXPORT ──────────────────────────────────────────────────────

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


# ─── REPORT ───────────────────────────────────────────────────────────────────

def generate_report(hop_df: pd.DataFrame, example_df: pd.DataFrame, output_dir: str):
    """Write interplay_report.md."""
    lines = ["# Parametric–Search Interplay Analysis\n"]
    lines.append(f"Generated from {len(hop_df)} hop observations across "
                 f"{len(hop_df['model'].unique())} models.\n")

    lines.append("\n## Aggregate Metrics per Model\n")
    lines.append("| Model | Examples | Total Queries | Necessary | Par. Redundant | Seq. Redundant | "
                 "Agg. Accuracy |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    for model in sorted(example_df["model"].unique()):
        sub = example_df[example_df["model"] == model]
        n_ex = len(sub)
        total = sub["total_search_calls"].sum()
        nec = sub["necessary_count"].sum()
        par = sub["parametric_redundant_count"].sum()
        seq = sub["sequential_redundant_count"].sum()
        acc = sub["aggregate_correct"].mean()

        par_pct = 100 * par / total if total else 0
        seq_pct = 100 * seq / total if total else 0
        nec_pct = 100 * nec / total if total else 0
        par_lo, par_hi = wilson_ci(int(par), int(total))
        seq_lo, seq_hi = wilson_ci(int(seq), int(total))

        lines.append(
            f"| {short_model(model)} | {n_ex} | {total} | "
            f"{nec_pct:.1f}% | "
            f"{par_pct:.1f}% [{100*par_lo:.1f}–{100*par_hi:.1f}%] | "
            f"{seq_pct:.1f}% [{100*seq_lo:.1f}–{100*seq_hi:.1f}%] | "
            f"{100*acc:.1f}% |\n"
        )

    lines.append("\n## Spearman Correlation: Entropy vs. Queries per Hop\n")
    lines.append("| Model | ρ | p-value | n hops |\n")
    lines.append("|---|---|---|---|\n")

    for model in sorted(hop_df["model"].unique()):
        sub = hop_df[hop_df["model"] == model]
        if len(sub) > 3:
            rho, pval = stats.spearmanr(sub["semantic_entropy"], sub["queries_assigned_count"])
            sig = " *" if pval < 0.05 else ""
            lines.append(f"| {short_model(model)} | {rho:.3f}{sig} | {pval:.4f} | {len(sub)} |\n")

    lines.append("\n## Accuracy by Parametric Certainty × Search Decision\n")
    lines.append("| Model | Condition | Hops (n) | Agg. Accuracy |\n")
    lines.append("|---|---|---|---|\n")
    conditions = [
        ("Certain + Searched", True, True),
        ("Certain + Not Searched", True, False),
        ("Uncertain + Searched", False, True),
        ("Uncertain + Not Searched", False, False),
    ]
    for model in sorted(hop_df["model"].unique()):
        sub = hop_df[hop_df["model"] == model]
        for label, certain, searched in conditions:
            mask = (sub["parametric_certain"] == certain) & (sub["searched"] == searched)
            grp = sub[mask]
            if grp.empty:
                continue
            acc = grp["aggregate_correct"].mean()
            lines.append(f"| {short_model(model)} | {label} | {len(grp)} | {100*acc:.1f}% |\n")

    lines.append("\n## Parametric Accuracy per Model\n")
    lines.append("| Model | Mean Par. Accuracy | % Hops Certain | Agg. Accuracy |\n")
    lines.append("|---|---|---|---|\n")
    for model in sorted(example_df["model"].unique()):
        sub = example_df[example_df["model"] == model]
        par_acc = sub["mean_parametric_accuracy"].mean()
        frac_cert = sub["frac_hops_certain"].mean()
        agg_acc = sub["aggregate_correct"].mean()
        lines.append(
            f"| {short_model(model)} | {100*par_acc:.1f}% | {100*frac_cert:.1f}% | {100*agg_acc:.1f}% |\n"
        )

    lines.append("\n## Searched vs. Not-Searched Hop Profile\n")
    lines.append("| Model | Condition | Hops (n) | % of Total | Mean Par. Acc | Mean Entropy |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for model in sorted(hop_df["model"].unique()):
        sub = hop_df[hop_df["model"] == model]
        total_hops = len(sub)
        for label, searched_val in [("Not Searched", False), ("Searched", True)]:
            grp = sub[sub["searched"] == searched_val]
            n_grp = len(grp)
            pct = 100 * n_grp / total_hops if total_hops > 0 else 0.0
            mean_par_acc = grp["parametric_accuracy"].mean()
            mean_entropy = grp["semantic_entropy"].mean()
            lines.append(
                f"| {short_model(model)} | {label} | {n_grp} | {pct:.0f}% | "
                f"{100*mean_par_acc:.1f}% | {mean_entropy:.3f} |\n"
            )

    lines.append("\n## Breakdown by Num Hops\n")
    lines.append("| Model | Num Hops | Examples | Mean Queries/Hop | % Uncertain | Mean Par. Acc | Agg. Acc |\n")
    lines.append("|---|---|---|---|---|---|---|\n")

    for model in sorted(hop_df["model"].unique()):
        for nh in sorted(hop_df["num_hops"].unique()):
            sub = hop_df[(hop_df["model"] == model) & (hop_df["num_hops"] == nh)]
            if sub.empty:
                continue
            n_ex = sub["example_id"].nunique()
            mean_q = sub["queries_assigned_count"].mean()
            par_rate = sub["parametric_certain"].mean()
            mean_par_acc = sub["parametric_accuracy"].mean()
            agg_acc = sub["aggregate_correct"].mean()
            lines.append(
                f"| {short_model(model)} | {nh} | {n_ex} | "
                f"{mean_q:.2f} | {100*(1-par_rate):.1f}% | "
                f"{100*mean_par_acc:.1f}% | {100*agg_acc:.1f}% |\n"
            )

    lines.append("\n## Plots\n")
    for fname, title in [
        ("accuracy_per_hop.png", "Parametric Accuracy per Hop + Aggregate"),
        ("accuracy_by_certainty_search.png", "Accuracy by Certainty × Search Decision"),
        ("searched_vs_unsearched.png", "Searched vs. Not-Searched Hop Profile"),
        ("search_calibration.png", "Search Calibration Quadrants"),
        ("search_calibration_extended.png", "Search Calibration Extended (with Sequential Redundancy)"),
        ("certain_vs_uncertain_queries.png", "Certain vs. Uncertain Queries Bar Chart"),
        ("accuracy_vs_queries_bars.png", "Parametric Accuracy vs. Queries Bar Chart"),
        ("entropy_bucket_queries.png", "Entropy Bucket Bar Chart"),
        ("redundancy_breakdown.png", "Redundancy Breakdown"),
        ("queries_violin.png", "Queries Violin Plot"),
        ("searches_per_certainty_level.png", "Mean Searches per Certainty Level"),
        ("missed_search_cross_hop_coverage.png", "Missed Search: Truly Missed vs. Cross-Hop Covered"),
    ]:
        lines.append(f"\n### {title}\n")
        lines.append(f"![{title}]({fname})\n")

    report_path = os.path.join(output_dir, "interplay_report.md")
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"  Saved: {report_path}")


# ─── SEARCH USEFULNESS BY CERTAINTY ───────────────────────────────────────────

def plot_search_usefulness_by_certainty(example_df: pd.DataFrame, output_dir: str):
    """
    For each hop-level certainty group (no hops certain / some / all), compare
    the no-search parametric accuracy against the search-run accuracy.

    Certainty = fraction of hops with zero semantic entropy (from 5-run per-hop data).
    No-search accuracy = mean of no_search_agg_accuracy (actual parametric runs when
    available, independence-composition fallback otherwise).
    Search accuracy = aggregate_correct from the single search run.
    Delta = search_acc - no_search_acc → positive means search helped.
    """
    models = sorted(example_df["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    all_rows = []

    for ax, model in zip(axes[0], models):
        sub = example_df[example_df["model"] == model].copy()
        sub = sub.dropna(subset=["no_search_agg_accuracy", "frac_hops_certain"])

        # Build 3 groups based on hop-level certainty fraction
        group_specs = [
            ("No hops\ncertain (0%)",   sub[sub["frac_hops_certain"] == 0.0]),
            ("Some hops\ncertain",       sub[(sub["frac_hops_certain"] > 0.0) & (sub["frac_hops_certain"] < 1.0)]),
            ("All hops\ncertain (100%)", sub[sub["frac_hops_certain"] == 1.0]),
        ]
        groups = []
        for label, grp in group_specs:
            if grp.empty:
                continue
            n_q = len(grp)
            no_acc = grp["no_search_agg_accuracy"].mean()
            s_k = int(grp["aggregate_correct"].sum())
            s_acc = s_k / n_q
            lo_s, hi_s = wilson_ci(s_k, n_q)
            delta = s_acc - no_acc
            groups.append({
                "label": label,
                "n_q": n_q,
                "no_acc": no_acc,
                "s_acc": s_acc,
                "s_ci_lo": s_acc - lo_s,
                "s_ci_hi": hi_s - s_acc,
                "delta": delta,
            })
            all_rows.append({
                "model": model,
                "certainty_level": label.replace("\n", " "),
                "n_questions": n_q,
                "no_search_agg_accuracy": round(no_acc, 4),
                "search_agg_accuracy": round(s_acc, 4),
                "delta": round(delta, 4),
            })

        no_search_accs = [g["no_acc"] for g in groups]
        search_accs    = [g["s_acc"] for g in groups]
        search_cis_lo  = [g["s_ci_lo"] for g in groups]
        search_cis_hi  = [g["s_ci_hi"] for g in groups]
        labels         = [g["label"] for g in groups]

        x = np.arange(len(groups))
        w = 0.35
        bars_no   = ax.bar(x - w/2, no_search_accs, w, label="No-search (parametric)",
                           color=PALETTE[0], alpha=0.85)
        bars_srch = ax.bar(x + w/2, search_accs,    w, label="With search",
                           color=PALETTE[1], alpha=0.85,
                           yerr=[search_cis_lo, search_cis_hi], capsize=4)

        for bar, g, no_acc in zip(bars_srch, groups, no_search_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"n={g['n_q']}", ha="center", va="bottom", fontsize=7)
            delta = g["delta"]
            color = "#2ca02c" if delta > 0 else "#d62728"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), no_acc) + 0.10,
                    f"{delta:+.2f}", ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Hop-level parametric certainty\n(fraction of hops with zero entropy)", fontsize=9)
        ax.set_ylabel("Aggregate accuracy ± 95% CI (search bar)", fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.set_title(short_model(model), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Search Usefulness by Hop-Level Parametric Certainty",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "search_usefulness_by_certainty.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # Save CSV
    df_out = pd.DataFrame(all_rows)
    csv_out = os.path.join(output_dir, "search_usefulness_by_certainty.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # Print summary table
    print("\nSearch Usefulness by Certainty Level:")
    print(f"  {'Model':<25} {'Cert':>6} {'N':>5} {'No-search acc':>14} {'Search acc':>11} {'Delta':>8}")
    print("  " + "-" * 75)
    for row in all_rows:
        print(f"  {row['model']:<25} {row['certainty_level']:>6} {row['n_questions']:>5}"
              f" {row['no_search_agg_accuracy']:>14.3f} {row['search_agg_accuracy']:>11.3f}"
              f" {row['delta']:>+8.3f}")


# ─── SEARCH USEFULNESS BY ENTROPY ────────────────────────────────────────────

def plot_search_usefulness_by_entropy(example_df: pd.DataFrame, hop_df: pd.DataFrame, output_dir: str):
    """
    Same layout as plot_search_usefulness_by_certainty, but the x-axis shows
    mean semantic entropy buckets (certain / low / medium / high) instead of
    correctness-based certainty levels.
    """
    # Compute mean semantic entropy per (model, example_id) from hop-level data
    mean_entropy = (
        hop_df.groupby(["model", "example_id"])["semantic_entropy"]
        .mean()
        .reset_index()
        .rename(columns={"semantic_entropy": "mean_entropy"})
    )
    merged = example_df.merge(mean_entropy, on=["model", "example_id"], how="left")

    def _entropy_bucket(e: float) -> str:
        if e <= 0.0:
            return "certain\n(=0)"
        elif e <= 0.5:
            return "low\n(0–0.5]"
        elif e <= 1.0:
            return "medium\n(0.5–1.0]"
        else:
            return "high\n(>1.0)"

    bucket_order = ["certain\n(=0)", "low\n(0–0.5]", "medium\n(0.5–1.0]", "high\n(>1.0)"]
    merged["entropy_bucket"] = merged["mean_entropy"].apply(
        lambda e: _entropy_bucket(e) if pd.notna(e) else None
    )

    models = sorted(merged["model"].unique())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), squeeze=False)

    all_rows = []

    for ax, model in zip(axes[0], models):
        sub = merged[merged["model"] == model].dropna(subset=["entropy_bucket", "no_search_agg_accuracy"])

        groups = []
        for bucket in bucket_order:
            grp = sub[sub["entropy_bucket"] == bucket]
            if grp.empty:
                continue
            n_q = len(grp)
            no_acc = grp["no_search_agg_accuracy"].mean()
            s_k = int(grp["aggregate_correct"].sum())
            s_acc = s_k / n_q
            lo_s, hi_s = wilson_ci(s_k, n_q)
            delta = s_acc - no_acc
            groups.append({
                "label": bucket,
                "n_q": n_q,
                "no_acc": no_acc,
                "s_acc": s_acc,
                "s_ci_lo": s_acc - lo_s,
                "s_ci_hi": hi_s - s_acc,
                "delta": delta,
            })
            all_rows.append({
                "model": model,
                "entropy_bucket": bucket.replace("\n", " "),
                "n_questions": n_q,
                "no_search_agg_accuracy": round(no_acc, 4),
                "search_agg_accuracy": round(s_acc, 4),
                "delta": round(delta, 4),
            })

        if not groups:
            continue

        no_search_accs = [g["no_acc"] for g in groups]
        search_accs    = [g["s_acc"] for g in groups]
        search_cis_lo  = [g["s_ci_lo"] for g in groups]
        search_cis_hi  = [g["s_ci_hi"] for g in groups]
        labels         = [g["label"] for g in groups]

        x = np.arange(len(groups))
        w = 0.35
        ax.bar(x - w/2, no_search_accs, w, label="No-search (parametric)",
               color=PALETTE[0], alpha=0.85)
        bars_srch = ax.bar(x + w/2, search_accs, w, label="With search",
                           color=PALETTE[1], alpha=0.85,
                           yerr=[search_cis_lo, search_cis_hi], capsize=4)

        for bar, g, no_acc in zip(bars_srch, groups, no_search_accs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"n={g['n_q']}", ha="center", va="bottom", fontsize=7)
            delta = g["delta"]
            color = "#2ca02c" if delta > 0 else "#d62728"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    max(bar.get_height(), no_acc) + 0.10,
                    f"{delta:+.2f}", ha="center", va="bottom", fontsize=8, color=color, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_xlabel("Mean semantic entropy bucket\n(averaged over hops per example)", fontsize=9)
        ax.set_ylabel("Aggregate accuracy ± 95% CI (search bar)", fontsize=9)
        ax.set_ylim(0, 1.25)
        ax.set_title(short_model(model), fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Search Usefulness by Semantic Entropy Level",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    out = os.path.join(output_dir, "search_usefulness_by_entropy.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    df_out = pd.DataFrame(all_rows)
    csv_out = os.path.join(output_dir, "search_usefulness_by_entropy.csv")
    df_out.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")


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

    # Save summary CSV
    csv_path = os.path.join(args.output_dir, "interplay_summary.csv")
    hop_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path} ({len(hop_df)} rows)")

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

    # Generate plots
    print("\nGenerating plots...")
    plot_accuracy_per_hop(hop_df, args.output_dir)
    plot_accuracy_outcomes(hop_df, args.output_dir)
    plot_searched_vs_unsearched(hop_df, args.output_dir)
    plot_search_calibration(hop_df, args.output_dir)
    plot_search_calibration_extended(hop_df, args.output_dir)
    plot_certain_vs_uncertain_queries(hop_df, args.output_dir)
    plot_accuracy_vs_queries_bars(hop_df, args.output_dir)
    plot_entropy_buckets(hop_df, args.output_dir)
    plot_redundancy_breakdown(example_df, args.output_dir)
    plot_sequential_redundancy_distribution(example_df, args.output_dir)
    plot_redundancy_overlap(example_df, args.output_dir)
    plot_search_usefulness_by_certainty(example_df, args.output_dir)
    plot_search_usefulness_by_entropy(example_df, hop_df, args.output_dir)
    plot_searches_per_certainty_level(hop_df, args.output_dir)
    plot_searches_per_entropy_level(hop_df, args.output_dir)
    plot_missed_search_cross_hop_coverage(hop_df, args.output_dir)
    if multi_hop_agent is not None:
        plot_multi_hop_query_attribution(all_examples, args.output_dir)
        plot_multi_hop_query_attribution_by_hop_type(all_examples, args.output_dir)
    try:
        plot_queries_violin(hop_df, args.output_dir)
    except Exception as e:
        print(f"  Violin plot skipped: {e}")

    # Save gold-check cache
    if gold_judge_agent is not None:
        with open(gold_judge_cache_path, "w") as f:
            json.dump(gold_judge_cache, f, indent=2)
        print(f"Saved gold-check cache ({len(gold_judge_cache)} entries) to {gold_judge_cache_path}")

    # Generate report
    print("\nGenerating report...")
    generate_report(hop_df, example_df, args.output_dir)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
