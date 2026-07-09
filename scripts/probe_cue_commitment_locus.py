"""Phase 1 of docs/PLAN_cue_mechanism_analysis.md: commitment-locus probe for the PLAIN vs
ELABORATE cue experiment.

For each trajectory (one trace = one model answering one question under one cue), an LLM judge
reads the full transcript (thinking, search tool-calls, search results, final answer) and
identifies the FIRST turn at which the agent commits to a specific final answer. We then map that
turn index onto the trajectory's ordered search-call sequence to get `commitment_step` (how many
searches preceded commitment; 0 = committed before any search) and `post_commitment_searches`
(searches issued at/after commitment — verification/elaboration ritual rather than
evidence-gathering).

Adapted from scripts/archive/probe_commitment_locus.py: same judge-prompt structure, per-turn
transcript replay, --max-search-result-chars truncation, and shard/resume machinery, but for
single-shot QA trajectories (no MusiQue hop decomposition) and a single overall commitment per
trajectory rather than one per hop.

Cells (see docs/PLAN_cue_mechanism_analysis.md Phase 1): facts_open x {gemma4:31b, qwen3.5:122b},
medqa x {qwen3.5:122b, gemini-3.1-pro-preview}. Both cues each -> 8 trace sets. Traces come from
Phase 0 (scripts/download_cue_traces.py -> results/cue_traces/*_traces.json), already normalized
so `trace["problem"]` == the eval JSON's base `problem` string (no cue suffix) for both cues.

Judge: gemini-3-flash-preview via BaseAgent's "Google" provider (note: BaseAgent checks the exact
string "Google", not "google"). Cheap (~600 calls total for the full run).

Usage:
    # spot-check 5 trajectories per cell (run this FIRST, inspect the JSONL, then go full)
    uv run python scripts/probe_cue_commitment_locus.py --limit 5

    # full run, all 4 cells x 2 cues
    uv run python scripts/probe_cue_commitment_locus.py

    # one cell only
    uv run python scripts/probe_cue_commitment_locus.py --cells facts_open_gemma

    # paired summary once CSVs exist (no judge calls)
    uv run python scripts/probe_cue_commitment_locus.py --summarize-only
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import sys
import time
from typing import Any, Optional

import numpy as np
from pydantic import BaseModel, Field
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.services.base_agent import BaseAgent  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

TRACES_DIR = "results/cue_traces"
OUTPUT_DIR = "results/cue_commitment_locus"

# cell -> (dataset, model_name, model_slug, plain_eval_json, elaborate_eval_json)
CELL_CONFIGS: dict[str, tuple[str, str, str, str, str]] = {
    "facts_open_gemma": (
        "facts-open", "gemma4:31b", "gemma4_31b",
        "results/facts_gemma100/facts-open_baseline_gemma4:31b_plain.json",
        "results/facts_gemma100/facts-open_baseline_gemma4:31b_elaborate.json",
    ),
    "facts_open_qwen": (
        "facts-open", "qwen3.5:122b", "qwen3.5_122b",
        "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_plain.json",
        "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_elaborate.json",
    ),
    "medqa_qwen": (
        "medqa", "qwen3.5:122b", "qwen3.5_122b",
        "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_plain.json",
        "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_elaborate.json",
    ),
    "medqa_gemini": (
        "medqa", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview",
        "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_plain.json",
        "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_elaborate.json",
    ),
}


class CommitmentJudgment(BaseModel):
    first_committed_turn_index: int = Field(
        description="0-based turn index of the FIRST turn at which the agent commits to a "
                    "specific final answer to the question, treating it as established (e.g. "
                    "'X is Y', 'so the answer is Z', or using the answer as settled fact in "
                    "subsequent reasoning). The transcript always ends with a final answer, so "
                    "if no earlier turn qualifies, use the final answering turn. Only return -1 "
                    "if no answer, tentative or definite, appears anywhere."
    )
    committed_answer: str = Field(
        description="The specific answer the agent committed to at that first commitment turn "
                    "(verbatim or short paraphrase). Empty string if first_committed_turn_index = -1."
    )
    matches_gold: bool = Field(
        description="True iff the committed_answer is semantically equivalent to the gold "
                    "answer (e.g. 'USA' ~ 'United States')."
    )
    justification: str = Field(
        description="One-sentence justification citing the relevant turn content."
    )


JUDGE_PROMPT = """\
You are analyzing a transcript of an AI agent answering a question. The agent interleaves \
reasoning ("thinking"), search-tool calls, and search results before producing a final answer. \
Each turn in the transcript is prefixed with [TURN i: ROLE ...].

Question: {question}
Gold answer: {gold_answer}

Your task: identify the FIRST turn at which the agent commits to a specific answer to the \
question above.

"Commits" means:
  - Asserts a definite answer ("X is Y", "the answer is Z", "so it must be..."), OR
  - Treats a specific answer as established and reuses it as settled fact in subsequent \
reasoning or search queries (not merely floating it as one candidate among several).

A turn that only lists candidate answers, expresses uncertainty/hedges without settling, or \
issues a search query without yet concluding is NOT a commitment. The agent may keep searching \
AFTER committing (e.g. to verify, elaborate, or double-check) — that does not undo the \
commitment; find the FIRST turn satisfying the definition above, even if later turns add \
caveats, more detail, or (rarely) revise the answer.

The transcript always ends with a final answer. If no earlier turn qualifies as a commitment, \
use the final answering turn as the commitment turn — do not return -1 in that case. Only \
return -1 if you truly cannot find any answer, tentative or definite, anywhere in the transcript.

Notes:
  - Turn indices are 0-based and match the [TURN i: ...] prefix.
  - matches_gold compares the COMMITTED answer (at the FIRST commitment turn) to the gold \
answer; commitment exists even when the committed answer turns out to be wrong.

==== TRANSCRIPT ====
{transcript}
==== END TRANSCRIPT ====
"""


def safe_parse(val: Any) -> Any:
    """Normalize a trace field that should be a dict/list. Handles the (empirically confirmed
    on this data) common case of an already-native dict/list, with JSON- and Python-repr-string
    fallbacks for robustness (per docs/PLAN_cue_mechanism_analysis.md's note to verify this)."""
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def render_trace_with_turn_indices(message_trace: list[dict], max_search_chars: int
                                    ) -> tuple[str, list[int], list[str]]:
    """Render each message as a numbered turn. Returns (transcript_text, search_turn_indices,
    search_queries) where search_turn_indices[k] is the turn index of the k-th search tool_call
    (chronological), and search_queries[k] is its query string."""
    lines: list[str] = []
    search_turn_indices: list[int] = []
    search_queries: list[str] = []
    for i, msg in enumerate(message_trace):
        role = msg.get("role", "")
        parts = msg.get("parts", []) or []
        for p in parts:
            if not isinstance(p, dict):
                continue
            ptype = p.get("type")
            if ptype == "text" and role == "system":
                content = p.get("content", "")
                if content:
                    lines.append(f"[TURN {i}: SYSTEM] {content}")
            elif ptype == "text" and role == "user":
                lines.append(f"[TURN {i}: USER] {p.get('content', '')}")
            elif ptype == "thinking":
                lines.append(f"[TURN {i}: ASSISTANT THINKING] {p.get('content', '')}")
            elif ptype == "text" and role == "assistant":
                lines.append(f"[TURN {i}: ASSISTANT] {p.get('content', '')}")
            elif ptype == "tool_call":
                args = safe_parse(p.get("arguments", ""))
                tool_name = p.get("tool_name", "")
                lines.append(f"[TURN {i}: ASSISTANT TOOL_CALL] {tool_name}({args})")
                if tool_name == "search" and isinstance(args, dict) and args.get("query"):
                    search_turn_indices.append(i)
                    search_queries.append(str(args["query"]))
            elif ptype == "tool_call_response":
                result = safe_parse(p.get("result", ""))
                result_str = result if isinstance(result, str) else json.dumps(result, default=str)
                if len(result_str) > max_search_chars:
                    result_str = result_str[:max_search_chars] + \
                        f" ... [truncated, total {len(result_str)} chars]"
                lines.append(f"[TURN {i}: TOOL RESULT] {result_str}")
    return "\n\n".join(lines), search_turn_indices, search_queries


def shard_for(example_id: str, num_shards: int) -> int:
    return int(hashlib.md5(str(example_id).encode()).hexdigest(), 16) % num_shards


def load_eval_rows(path: str) -> dict[str, dict]:
    with open(path) as f:
        rows = json.load(f)
    return {r["problem"].strip(): r for r in rows if "problem" in r}


def load_traces(path: str) -> dict[str, dict]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        traces = json.load(f)
    return {t["problem"].strip(): t for t in traces if "problem" in t}


def run_probe(cell: str, cue: str, judge: BaseAgent, args) -> str:
    dataset, model_name, model_slug, plain_json, elab_json = CELL_CONFIGS[cell]
    eval_json = plain_json if cue == "plain" else elab_json
    eval_by_problem = load_eval_rows(eval_json)

    trace_path = os.path.join(args.traces_dir, f"{dataset}_{model_slug}_{cue}_traces.json")
    traces_by_problem = load_traces(trace_path)

    os.makedirs(args.output_dir, exist_ok=True)
    out_csv = os.path.join(args.output_dir, f"commitment_{cell}_{cue}.csv")
    out_jsonl = out_csv.replace(".csv", ".jsonl")

    # Build targets: eval rows with stop_reason == None (drop runaway UsageLimitExceeded rows per
    # the pairing convention), joined to their trace by exact `problem` string.
    targets = []
    n_no_trace = 0
    for problem, row in eval_by_problem.items():
        if row.get("stop_reason") is not None:
            continue
        trace = traces_by_problem.get(problem)
        if trace is None:
            n_no_trace += 1
            continue
        targets.append((problem, row, trace))
    if n_no_trace:
        print(f"  [{cell}/{cue}] WARNING: {n_no_trace} eval rows had no matching trace "
              f"(of {len(eval_by_problem)} total, after stop_reason filter).")

    targets.sort(key=lambda t: t[1]["example_id"])
    targets = [t for t in targets if shard_for(t[1]["example_id"], args.num_shards) == args.shard_index]

    fieldnames = [
        "example_id", "cell", "cue", "dataset", "model_slug", "problem_excerpt", "gold_answer",
        "num_turns", "n_searches", "eval_search_calls", "commitment_step",
        "post_commitment_searches", "committed_turn_index", "committed_answer", "matches_gold",
        "final_correct", "stop_reason", "justification",
    ]

    done_keys: set[str] = set()
    if args.resume and os.path.exists(out_csv):
        with open(out_csv) as f:
            for row in csv.DictReader(f):
                done_keys.add(row["example_id"])
        print(f"  [{cell}/{cue}] Resuming: {len(done_keys)} already done.")

    targets = [t for t in targets if str(t[1]["example_id"]) not in done_keys]
    if args.limit:
        targets = targets[: args.limit]
    print(f"  [{cell}/{cue}] {len(targets)} trajectories to probe.")

    if not targets:
        return out_csv

    write_header = not os.path.exists(out_csv)
    csv_f = open(out_csv, "a", newline="", encoding="utf-8")
    csv_w = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        csv_w.writeheader()
    jsonl_f = open(out_jsonl, "a", encoding="utf-8")

    try:
        for i, (problem, row, trace) in enumerate(targets):
            message_trace = trace.get("message_trace", [])
            transcript, search_turn_indices, search_queries = render_trace_with_turn_indices(
                message_trace, args.max_search_result_chars
            )
            gold_answer = row.get("correct_answer", "")
            prompt = JUDGE_PROMPT.format(question=problem, gold_answer=gold_answer,
                                         transcript=transcript)
            judgment = None
            last_err = None
            for attempt in range(args.judge_max_retries):
                try:
                    resp = judge.run(prompt)
                    judgment = resp.output
                    break
                except Exception as e:
                    last_err = e
                    if attempt < args.judge_max_retries - 1:
                        wait = min(2 ** attempt * 2, 30)
                        print(f"  [{i+1}/{len(targets)}] judge error on {row['example_id']} "
                              f"(attempt {attempt+1}/{args.judge_max_retries}): {e} "
                              f"-- retrying in {wait}s")
                        time.sleep(wait)
            if judgment is not None:
                committed_idx = judgment.first_committed_turn_index
                committed_answer = judgment.committed_answer
                matches_gold = judgment.matches_gold
                justification = judgment.justification
            else:
                print(f"  [{i+1}/{len(targets)}] judge FAILED on {row['example_id']} after "
                      f"{args.judge_max_retries} attempts: {last_err}")
                committed_idx = -1
                committed_answer = ""
                matches_gold = False
                justification = f"JUDGE_ERROR: {last_err}"

            n_searches = len(search_turn_indices)
            if committed_idx < 0:
                # Never committed per the judge (shouldn't happen per prompt instructions, but
                # guard defensively): treat as committed only after all observed searches.
                commitment_step = n_searches
            else:
                commitment_step = sum(1 for t in search_turn_indices if t < committed_idx)
            post_commitment_searches = n_searches - commitment_step

            out_row = {
                "example_id": row["example_id"],
                "cell": cell,
                "cue": cue,
                "dataset": dataset,
                "model_slug": model_slug,
                "problem_excerpt": problem[:120],
                "gold_answer": gold_answer,
                "num_turns": len(message_trace),
                "n_searches": n_searches,
                "eval_search_calls": row.get("sampler_search_calls"),
                "commitment_step": commitment_step,
                "post_commitment_searches": post_commitment_searches,
                "committed_turn_index": committed_idx,
                "committed_answer": committed_answer,
                "matches_gold": matches_gold,
                "final_correct": row.get("sampler_correct"),
                "stop_reason": row.get("stop_reason"),
                "justification": justification,
            }
            csv_w.writerow(out_row)
            csv_f.flush()
            jsonl_f.write(json.dumps({**out_row, "search_queries": search_queries}, default=str) + "\n")
            jsonl_f.flush()
            print(f"  [{i+1}/{len(targets)}] {row['example_id']}: n_searches={n_searches} "
                  f"commit_step={commitment_step} post_commit={post_commitment_searches} "
                  f"match={matches_gold} correct={row.get('sampler_correct')}")
    finally:
        csv_f.close()
        jsonl_f.close()

    print(f"  [{cell}/{cue}] Wrote {out_csv}")
    return out_csv


# ── summary analysis ─────────────────────────────────────────────────────────
def load_csv(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def summarize(cells: list[str], output_dir: str):
    import pandas as pd
    pd.set_option("display.width", 160)

    for cell in cells:
        plain_path = os.path.join(output_dir, f"commitment_{cell}_plain.csv")
        elab_path = os.path.join(output_dir, f"commitment_{cell}_elaborate.csv")
        dp, de = load_csv(plain_path), load_csv(elab_path)
        print(f"\n{'='*90}\nCELL: {cell}\n{'='*90}")
        if dp is None or de is None:
            print(f"  Missing CSV(s): plain={dp is not None} elaborate={de is not None} — skipping.")
            continue

        # 1. PLAIN post-commitment search share
        dp_searched = dp[dp["n_searches"] > 0]
        if len(dp_searched):
            share = dp_searched["post_commitment_searches"] / dp_searched["n_searches"]
            print(f"  PLAIN post-commitment search share (of {len(dp_searched)} rows with "
                  f">=1 search): mean={share.mean():.2f} median={share.median():.2f}")
        else:
            print("  PLAIN: no rows with >=1 search.")

        # paired join on example_id
        merged = dp.merge(de, on="example_id", suffixes=("_plain", "_elab"))
        n = len(merged)
        print(f"  Paired examples (plain ∩ elaborate, both stop_reason=None): {n}")
        if n == 0:
            continue

        # 2. Δ commitment_step, Δ post_commitment_searches (elaborate - plain)
        d_commit = merged["commitment_step_elab"] - merged["commitment_step_plain"]
        d_post = merged["post_commitment_searches_elab"] - merged["post_commitment_searches_plain"]

        def _wilcoxon_report(delta, label):
            nz = delta[delta != 0]
            if len(nz) < 1:
                print(f"  Δ{label} (ELAB-PLAIN): all zero, n={len(delta)}")
                return
            stat, p = wilcoxon(delta)
            print(f"  Δ{label} (ELAB-PLAIN): median={delta.median():.2f} mean={delta.mean():.2f} "
                  f"Wilcoxon p={p:.4g} (n={len(delta)}, n_nonzero={len(nz)})")

        _wilcoxon_report(d_commit, "commitment_step")
        _wilcoxon_report(d_post, "post_commitment_searches")

        # 3. flip x commitment-shift cross-tab
        def _flip(row):
            p_c, e_c = bool(row["final_correct_plain"]), bool(row["final_correct_elab"])
            if p_c and not e_c:
                return "R->W"
            if not p_c and e_c:
                return "W->R"
            if p_c and e_c:
                return "stable_correct"
            return "stable_wrong"

        merged["flip"] = merged.apply(_flip, axis=1)

        def _shift(row):
            d = row["commitment_step_elab"] - row["commitment_step_plain"]
            if d < 0:
                return "earlier"
            if d > 0:
                return "later"
            return "same"

        merged["commit_shift"] = merged.apply(_shift, axis=1)
        ct = pd.crosstab(merged["flip"], merged["commit_shift"])
        print("\n  Flip x commitment-shift cross-tab (ELAB relative to PLAIN):")
        print(ct.to_string().replace("\n", "\n  "))

        # bonus: were pruned searches (plain_n_searches > elab_n_searches) mostly
        # pre- or post-commitment (using PLAIN's own commitment_step as the boundary)?
        merged["pruned"] = merged["n_searches_plain"] - merged["n_searches_elab"]
        pruned_rows = merged[merged["pruned"] > 0]
        if len(pruned_rows):
            # upper bound on how many of the pruned searches could have been pre-commitment,
            # assuming truncation removes searches from the tail first.
            pre = np.minimum(pruned_rows["pruned"], pruned_rows["post_commitment_searches_plain"])
            print(f"\n  Rows with pruned searches (PLAIN>ELAB n_searches): {len(pruned_rows)}/{n}; "
                  f"of the pruned searches, at most (tail-first assumption) "
                  f"{pre.sum()}/{pruned_rows['pruned'].sum()} were PLAIN post-commitment "
                  f"(i.e. verification-only pruning, harmless-by-construction upper bound).")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", default=",".join(CELL_CONFIGS.keys()),
                    help="Comma-separated cell names (default: all 4).")
    ap.add_argument("--cues", default="plain,elaborate")
    ap.add_argument("--traces-dir", default=TRACES_DIR)
    ap.add_argument("--output-dir", default=OUTPUT_DIR)
    ap.add_argument("--judge-model", default="gemini-3-flash-preview")
    ap.add_argument("--judge-provider", default="Google",
                    help="BaseAgent provider string — must be exactly 'Google' (case-sensitive).")
    ap.add_argument("--max-search-result-chars", type=int, default=1200)
    ap.add_argument("--judge-max-retries", type=int, default=4,
                    help="Retries with exponential backoff on transient judge errors (429/5xx).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Cap trajectories probed per cell/cue (spot-checking).")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    ap.add_argument("--summarize-only", action="store_true",
                    help="Skip judge calls; just print the paired summary from existing CSVs.")
    args = ap.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        if c not in CELL_CONFIGS:
            raise SystemExit(f"Unknown cell '{c}'. Valid: {list(CELL_CONFIGS)}")
    cues = [c.strip() for c in args.cues.split(",") if c.strip()]

    if not args.summarize_only:
        print(f"Initializing judge {args.judge_provider}/{args.judge_model}...")
        judge = BaseAgent(
            provider_name=args.judge_provider,
            model_name=args.judge_model,
            output_type=CommitmentJudgment,
            agent_name="cue_commitment_locus_judge",
        )
        for cell in cells:
            for cue in cues:
                run_probe(cell, cue, judge, args)

    summarize(cells, args.output_dir)


if __name__ == "__main__":
    main()
