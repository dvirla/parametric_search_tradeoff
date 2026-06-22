"""
Commitment-locus probe: pure analysis on existing traces (no rollouts).

For every (example, hop) that appears in BOTH the formal-benchmark trace and the natural-
paraphrase trace of the same model, an LLM judge identifies the first turn index at which
the agent commits to a specific answer for that hop's sub-question. Combined with the
first turn at which the agent issued a search query attributed to that hop, this yields
"committed_before_search" per (example, hop, variant).

If the model's commitment systematically precedes (or replaces) search on the natural
variant relative to the benchmark variant — for the same underlying hop — the failure
mode is a framing-induced confidence prior, not in-trace overconfidence.

Sharded for distributed running:
  --num-shards N --shard-index I  partitions by hash(example_id).
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from typing import Optional

from pydantic import BaseModel, Field

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()


class CommitmentJudgment(BaseModel):
    first_committed_turn_index: int = Field(
        description="0-based turn index of the FIRST turn at which the agent commits to a "
                    "specific answer for the target sub-question, treating it as established "
                    "(e.g. 'X is Y', 'we found that...', or using the answer in subsequent "
                    "reasoning). Return -1 if the agent never commits within the transcript."
    )
    committed_answer: str = Field(
        description="The specific answer the agent committed to (verbatim or short paraphrase). "
                    "Empty string if first_committed_turn_index = -1."
    )
    matches_gold: bool = Field(
        description="True iff the committed_answer is semantically equivalent to the gold "
                    "answer (e.g. 'USA' ~ 'United States')."
    )
    justification: str = Field(
        description="One-sentence justification citing the relevant turn content."
    )


JUDGE_PROMPT = """\
You are analyzing a transcript of an AI agent solving a multi-hop question by interleaving \
reasoning, search-tool calls, and search results. Each turn in the transcript is prefixed \
with [TURN i: ROLE ...].

Aggregate question: {aggregate_question}

Target sub-question (one hop of the chain): {hop_question}
Gold answer for this sub-question: {gold_answer}

Your task: identify the FIRST turn at which the agent commits to a specific answer for \
the TARGET SUB-QUESTION above (not the aggregate question and not any other hop).

"Commits" means:
  - Asserts a definite answer ("X is Y", "the answer is Z"), OR
  - Treats an answer as established and reuses it in subsequent reasoning (e.g. forms the \
next hop's search query as if the prior answer is settled).

If a turn merely lists candidate answers, expresses doubt, or fetches information without \
concluding, it is NOT a commitment. The very first turn that satisfies the above is the \
commitment turn.

Notes:
  - Turn indices are 0-based and match the [TURN i: ...] prefix.
  - Return -1 for first_committed_turn_index if no commitment occurs anywhere in the transcript.
  - matches_gold compares the COMMITTED answer to the GOLD answer; commitment exists even \
when wrong.

==== TRANSCRIPT ====
{transcript}
==== END TRANSCRIPT ====
"""


def setup_args():
    parser = argparse.ArgumentParser(description="Commitment-locus probe over existing traces.")
    parser.add_argument("--benchmark-matched", required=True,
                        help="results/musique_parametric/interplay_analysis/matched_examples_<slug>.json")
    parser.add_argument("--natural-matched", required=True,
                        help="results/musique-natural/interplay_analysis/matched_examples_<slug>.json")
    parser.add_argument("--benchmark-traces", required=True,
                        help="results/musique_parametric/musique_val_search_<slug>_traces_*.json")
    parser.add_argument("--natural-traces", required=True,
                        help="results/musique-natural/musique_val_search_<slug>_baseline_agent_run_1_traces_*.json")
    parser.add_argument("--judge_model", default="gpt-oss:120b",
                        help="LLM judge model (default: gpt-oss:120b via ollama).")
    parser.add_argument("--judge_provider", default="ollama")
    parser.add_argument("--model-slug", required=True,
                        help="Slug for output filename (e.g. qwen3.5_122b).")
    parser.add_argument("--output-dir", default="results/commitment_locus")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of (example, hop, variant) probes (smoke testing).")
    parser.add_argument("--max-search-result-chars", type=int, default=1200,
                        help="Truncate each search result block in the transcript (default: 1200).")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def strip_answer_format_suffix(problem: str) -> str:
    for sep in ("\n\nPlease answer in", "\n\nYour response should be in"):
        if sep in problem:
            return problem.split(sep)[0].strip()
    return problem.strip()


def render_trace_with_turn_indices(message_trace: list[dict], max_search_chars: int) -> tuple[str, list[Optional[str]]]:
    """Render each message as a numbered turn. Returns (text, search_queries_per_turn) where
    search_queries_per_turn[i] is the search query issued at turn i (or None)."""
    lines: list[str] = []
    queries_by_turn: list[Optional[str]] = []
    for i, msg in enumerate(message_trace):
        role = msg.get("role", "")
        parts = msg.get("parts", []) or []
        turn_search_queries: list[str] = []
        for p in parts:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text" and role == "system":
                lines.append(f"[TURN {i}: SYSTEM] {p.get('content', '')}")
            elif ptype == "text" and role == "user":
                lines.append(f"[TURN {i}: USER] {p.get('content', '')}")
            elif ptype == "thinking":
                lines.append(f"[TURN {i}: ASSISTANT THINKING] {p.get('content', '')}")
            elif ptype == "text" and role == "assistant":
                lines.append(f"[TURN {i}: ASSISTANT] {p.get('content', '')}")
            elif ptype == "tool_call":
                args = p.get("arguments", "")
                # Try to extract just the query string for the search-query map
                try:
                    parsed = json.loads(args) if isinstance(args, str) else args
                    if isinstance(parsed, dict) and p.get("tool_name") == "search":
                        q = parsed.get("query", "")
                        if q:
                            turn_search_queries.append(q)
                except Exception:
                    pass
                lines.append(f"[TURN {i}: ASSISTANT TOOL_CALL] {p.get('tool_name', '')}({args})")
            elif ptype == "tool_call_response":
                result = p.get("result", "")
                if not isinstance(result, str):
                    result = json.dumps(result, default=str)
                if len(result) > max_search_chars:
                    result = result[:max_search_chars] + f" ... [truncated, total {len(result)} chars]"
                lines.append(f"[TURN {i}: TOOL RESULT] {result}")
        # Pack all queries on this turn into the map; if multiple, store first (we look them up per query string)
        queries_by_turn.append(turn_search_queries[0] if turn_search_queries else None)
    return "\n\n".join(lines), queries_by_turn


def first_search_turn_for_hop(query_records: list[dict], queries_by_turn: list[Optional[str]],
                              hop_index: int) -> Optional[int]:
    """Find the smallest turn index whose search query is attributed to hop_index."""
    hop_queries = [qr["query"] for qr in query_records if qr.get("assigned_hop") == hop_index]
    if not hop_queries:
        return None
    hop_query_set = set(hop_queries)
    for i, q in enumerate(queries_by_turn):
        if q is not None and q in hop_query_set:
            return i
    return None


def shard_for(example_id: str, num_shards: int) -> int:
    return int(hashlib.md5(example_id.encode()).hexdigest(), 16) % num_shards


def main():
    args = setup_args()

    print(f"Loading benchmark matched_examples: {args.benchmark_matched}")
    with open(args.benchmark_matched) as f:
        bench_matched = json.load(f)
    bench_by_eid = {r["example_id"]: r for r in bench_matched}

    print(f"Loading natural matched_examples: {args.natural_matched}")
    with open(args.natural_matched) as f:
        nat_matched = json.load(f)
    nat_by_eid = {r["example_id"]: r for r in nat_matched}

    print(f"Loading benchmark traces: {args.benchmark_traces}")
    with open(args.benchmark_traces) as f:
        bench_traces = json.load(f)
    bench_traces_by_q = {strip_answer_format_suffix(t["problem"]): t for t in bench_traces}

    print(f"Loading natural traces: {args.natural_traces}")
    with open(args.natural_traces) as f:
        nat_traces = json.load(f)
    nat_traces_by_q = {strip_answer_format_suffix(t["problem"]): t for t in nat_traces}

    common_eids = sorted(set(bench_by_eid) & set(nat_by_eid))
    print(f"  {len(common_eids)} example_ids present in both matched_examples files.")
    common_eids = [eid for eid in common_eids if shard_for(eid, args.num_shards) == args.shard_index]
    print(f"  {len(common_eids)} after shard {args.shard_index}/{args.num_shards}.")

    # Build the list of (eid, hop, variant) probe rows
    targets: list[dict] = []
    for eid in common_eids:
        bm = bench_by_eid[eid]
        nm = nat_by_eid[eid]
        b_trace = bench_traces_by_q.get(bm["aggregate_question"])
        n_trace = nat_traces_by_q.get(nm["aggregate_question"])
        if b_trace is None or n_trace is None:
            continue
        # Hops should align by hop_index; iterate over hops present in both
        bm_hops_by_idx = {sq["hop_index"]: sq for sq in bm["sub_questions"]}
        nm_hops_by_idx = {sq["hop_index"]: sq for sq in nm["sub_questions"]}
        for hop_idx in sorted(set(bm_hops_by_idx) & set(nm_hops_by_idx)):
            for variant, matched, trace in (("benchmark", bm, b_trace), ("natural", nm, n_trace)):
                sq = matched["sub_questions"][hop_idx] if hop_idx < len(matched["sub_questions"]) else None
                if sq is None or sq.get("hop_index") != hop_idx:
                    # fallback to dict lookup
                    sq = bm_hops_by_idx[hop_idx] if variant == "benchmark" else nm_hops_by_idx[hop_idx]
                targets.append({
                    "example_id": eid,
                    "variant": variant,
                    "hop_index": hop_idx,
                    "hop_question": sq["question"],
                    "gold_answer": sq["gold_answer"],
                    "semantic_entropy": sq.get("semantic_entropy"),
                    "aggregate_question": matched["aggregate_question"],
                    "query_records": matched["query_records"],
                    "message_trace": trace["message_trace"],
                })

    print(f"  {len(targets)} (example, hop, variant) targets.")
    if args.limit:
        targets = targets[: args.limit]
        print(f"  {len(targets)} after --limit.")

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(
        args.output_dir,
        f"commitment_locus_{args.model_slug}_shard_{args.shard_index}_of_{args.num_shards}.csv",
    )
    out_jsonl = out_csv.replace(".csv", ".jsonl")

    done_keys: set[tuple[str, int, str]] = set()
    if args.resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done_keys.add((row["example_id"], int(row["hop_index"]), row["variant"]))
        print(f"  Resuming: {len(done_keys)} probes already in {out_csv}.")

    targets = [t for t in targets
               if (t["example_id"], t["hop_index"], t["variant"]) not in done_keys]
    print(f"  {len(targets)} probes remaining.\n")

    if not targets:
        print("Nothing to do.")
        return

    print(f"Initializing judge {args.judge_provider}/{args.judge_model}...")
    judge = BaseAgent(
        provider_name=args.judge_provider,
        model_name=args.judge_model,
        output_type=CommitmentJudgment,
        agent_name=f"commitment_locus_judge_{args.model_slug}",
    )

    fieldnames = [
        "example_id", "variant", "hop_index", "hop_question", "gold_answer",
        "semantic_entropy", "num_turns",
        "first_committed_turn_index", "committed_answer", "matches_gold",
        "first_search_turn_for_hop", "committed_before_search",
        "justification",
    ]
    write_header = not os.path.exists(out_csv)
    csv_f = open(out_csv, "a", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        csv_w.writeheader()
    jsonl_f = open(out_jsonl, "a", encoding="utf-8")

    try:
        for i, tgt in enumerate(targets):
            transcript, queries_by_turn = render_trace_with_turn_indices(
                tgt["message_trace"], args.max_search_result_chars
            )
            prompt = JUDGE_PROMPT.format(
                aggregate_question=tgt["aggregate_question"],
                hop_question=tgt["hop_question"],
                gold_answer=tgt["gold_answer"],
                transcript=transcript,
            )

            try:
                resp = judge.run(prompt)
                judgment: CommitmentJudgment = resp.output
                committed_idx = judgment.first_committed_turn_index
                committed_answer = judgment.committed_answer
                matches_gold = judgment.matches_gold
                justification = judgment.justification
            except Exception as e:
                print(f"  [{i+1}/{len(targets)}] judge error on {tgt['example_id']} hop={tgt['hop_index']} "
                      f"variant={tgt['variant']}: {e}")
                committed_idx = -1
                committed_answer = ""
                matches_gold = False
                justification = f"JUDGE_ERROR: {e}"

            first_search_turn = first_search_turn_for_hop(
                tgt["query_records"], queries_by_turn, tgt["hop_index"]
            )
            if committed_idx < 0:
                committed_before_search = None  # never committed
            elif first_search_turn is None:
                committed_before_search = True  # committed without ever searching for this hop
            else:
                committed_before_search = committed_idx < first_search_turn

            row = {
                "example_id": tgt["example_id"],
                "variant": tgt["variant"],
                "hop_index": tgt["hop_index"],
                "hop_question": tgt["hop_question"],
                "gold_answer": tgt["gold_answer"],
                "semantic_entropy": tgt["semantic_entropy"],
                "num_turns": len(tgt["message_trace"]),
                "first_committed_turn_index": committed_idx,
                "committed_answer": committed_answer,
                "matches_gold": matches_gold,
                "first_search_turn_for_hop": first_search_turn if first_search_turn is not None else -1,
                "committed_before_search": committed_before_search,
                "justification": justification,
            }
            csv_w.writerow(row)
            csv_f.flush()
            jsonl_f.write(json.dumps(row, default=str) + "\n")
            jsonl_f.flush()
            print(f"  [{i+1}/{len(targets)}] {tgt['example_id']} hop={tgt['hop_index']} "
                  f"variant={tgt['variant']}: committed@{committed_idx} "
                  f"search@{first_search_turn} before={committed_before_search} match={matches_gold}")
    finally:
        csv_f.close()
        jsonl_f.close()

    print(f"\n--- Done. Wrote {out_csv} and {out_jsonl} ---")


if __name__ == "__main__":
    main()
