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
    return parser.parse_args()


# ─── PYDANTIC MODELS ──────────────────────────────────────────────────────────

class QueryAttribution(BaseModel):
    hop_index: int  # 0-based index into sub_questions_results; -1 = aggregate/general
    confidence: Literal["high", "medium", "low"]
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


@dataclass
class QueryWithContext:
    query: str
    preceding_thinking: str
    query_index: int  # global index within trace
    turn_index: int   # assistant message index
    search_result_text: str = ""  # concatenated title+snippet from search results
    tool_call_id: str = ""        # for matching query→result


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
    gold_found_in_results: bool
    is_parametric_redundant: bool
    is_sequential_redundant: bool


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
    if separator in problem:
        return problem.split(separator)[0].strip()
    elif popqa_separator in problem:
        problem = problem.split(popqa_separator)[1].strip().split("\nAnswer:")[0]
    return problem.strip()


def tokenize(text: str) -> set:
    """Lowercase word tokens, stripping punctuation."""
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def model_name_from_eval_path(path: str) -> str:
    """Extract model name from musique_parametric_uncertainty_<model>.json"""
    basename = os.path.basename(path)
    return basename.replace("musique_parametric_uncertainty_", "").replace(".json", "").replace("_v2", "")


def model_name_from_trace_path(path: str) -> str:
    """Extract model name from musique_search_<model>_traces_<ts>.json"""
    basename = os.path.basename(path)
    # Remove prefix and suffix
    name = basename.replace("musique_search_", "")
    name = re.sub(r"_traces_\d{8}_\d{6}\.json$", "", name)
    return name


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_eval_data(eval_dir: str) -> dict:
    """Returns dict[model_name -> list[entry_dict]]."""
    pattern = os.path.join(eval_dir, "musique_parametric_uncertainty_*v2.json")
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
    pattern = os.path.join(eval_dir, "musique_search_*_traces_*.json")
    files = sorted(glob.glob(pattern))
    result = {}
    for path in files:
        model = model_name_from_trace_path(path)
        with open(path) as f:
            result[model] = json.load(f)
        print(f"  Loaded traces: {model} ({len(result[model])} traces) from {os.path.basename(path)}")
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
                    text = " ".join(
                        f"{r.get('title', '')} {r.get('snippet', '')}"
                        for r in (result_data if isinstance(result_data, list) else [])
                    )
                    result_map[tcid] = text

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
                    results.append(QueryWithContext(
                        query=query,
                        preceding_thinking=thinking_text,
                        query_index=query_index,
                        turn_index=turn_idx,
                        search_result_text=result_map.get(tool_call_id, ""),
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


# ─── REDUNDANCY FLAGS ─────────────────────────────────────────────────────────

def compute_redundancy_flags(
    attributed_queries: list,  # list of (QueryWithContext, QueryAttribution)
    sub_questions: list,       # list[SubQuestionResult]
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
                    attribution_agent, cache, cache_key, reattribute
                )
            else:
                attr = heuristic_attribute(qctx, sub_questions)
            attributed.append((qctx, attr))

        # Compute redundancy
        query_records = compute_redundancy_flags(attributed, sub_questions)

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

def compute_hop_metrics(all_examples: list) -> pd.DataFrame:
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
                "queries_assigned_count": len(queries_for_hop),
                "aggregate_correct": ex.aggregate_correct,
            })
    return pd.DataFrame(rows)


def compute_example_metrics(all_examples: list) -> pd.DataFrame:
    """Returns DataFrame with one row per (model, example_id)."""
    rows = []
    for ex in all_examples:
        total = len(ex.query_records)
        par_red = sum(1 for qr in ex.query_records if qr.is_parametric_redundant)
        seq_red = sum(1 for qr in ex.query_records if qr.is_sequential_redundant)
        # Exclusive categories (parametric takes priority to avoid double-counting)
        par_only = par_red
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
        rows.append({
            "model": ex.model,
            "example_id": ex.example_id,
            "num_hops": len(ex.sub_questions),
            "total_search_calls": total,
            "parametric_redundant_count": par_red,
            "sequential_redundant_count": seq_red,
            "parametric_only_count": par_only,
            "sequential_only_count": seq_only,
            "total_redundant_count": total_red,
            "necessary_count": necessary,
            "mean_parametric_accuracy": mean_par_acc,
            "frac_hops_certain": frac_certain,
            "aggregate_correct": ex.aggregate_correct,
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
        "gemini-3-pro-preview": "Gemini-3-Pro",
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
        ("certain_vs_uncertain_queries.png", "Certain vs. Uncertain Queries Bar Chart"),
        ("accuracy_vs_queries_bars.png", "Parametric Accuracy vs. Queries Bar Chart"),
        ("entropy_bucket_queries.png", "Entropy Bucket Bar Chart"),
        ("redundancy_breakdown.png", "Redundancy Breakdown"),
        ("queries_violin.png", "Queries Violin Plot"),
    ]:
        lines.append(f"\n### {title}\n")
        lines.append(f"![{title}]({fname})\n")

    report_path = os.path.join(output_dir, "interplay_report.md")
    with open(report_path, "w") as f:
        f.writelines(lines)
    print(f"  Saved: {report_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = setup_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading eval data...")
    eval_by_model = load_eval_data(args.eval_dir)

    print("Loading traces...")
    traces_by_model = load_traces(args.eval_dir)

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

    # Compute metrics
    print("\nComputing metrics...")
    hop_df = compute_hop_metrics(all_examples)
    example_df = compute_example_metrics(all_examples)

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
    plot_certain_vs_uncertain_queries(hop_df, args.output_dir)
    plot_accuracy_vs_queries_bars(hop_df, args.output_dir)
    plot_entropy_buckets(hop_df, args.output_dir)
    plot_redundancy_breakdown(example_df, args.output_dir)
    try:
        plot_queries_violin(hop_df, args.output_dir)
    except Exception as e:
        print(f"  Violin plot skipped: {e}")

    # Generate report
    print("\nGenerating report...")
    generate_report(hop_df, example_df, args.output_dir)

    print(f"\nDone. All outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
