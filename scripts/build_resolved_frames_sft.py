"""
Build the "resolved" FRAMES cue-robustness SFT set: cue-invariant search usage WITHOUT
losing the ability to answer parametrically when no tool is offered.

Fixes three defects of curate_frames_sft_data.py's output (see
docs/frames_cue_robustness_sft.md and the gemma-SFT channel-leak investigation):

1. **No tool schema in the prompt.** Collection registered a real Tool(search.search), but
   training rendered `apply_chat_template(...)` with no `tools=`, so every training prompt
   was tool-free while 83% of decision turns emitted a search call. Tool availability was
   invisible to the model, and its prior on any tool-free prompt became "search" -- which is
   exactly "cannot answer without a tool". Each record here carries an explicit
   `tools_available` flag; train_sft.py renders the real search schema when it is true.

2. **No tool-ABSENT examples.** The zero-search rollouts in the old set came from runs where
   the tool WAS offered and the model declined it (type B below) -- a different condition from
   "no tool exists". Genuine tool-absent trajectories come from the no_search parametric
   probe's own Logfire traces (type C), which carry `thinking` and the answer separately.

3. **The plain anchor was unfiltered while every cue was clipped** to |search-plain_ref|<=1,
   so plain kept a fatter search tail (21.3% of its examples at >=6 calls vs 5.6-16.2% for the
   cues). That trains a residual "cue -> search less" effect, the very thing the SFT removes.
   `--clip-plain` applies the same closeness filter to the anchor; `--cap-per-cue` bounds each
   (question, cue) so the cues' own accuracy differences do not skew the mix.

Three example types are emitted:
    A  tools_available=true,  trajectory searches          <- cue-invariant search usage
    B  tools_available=true,  answers directly             <- "tool there, didn't need it"
    C  tools_available=false, answers directly             <- parametric answer, no tool
A and B come from --rollouts; C from --traces. B and C have identical `messages` and differ
only in the flag, which is why the flag cannot be derived from the trajectory.

Output is in `<think>`-in-content form, identical to curate_frames_sft_data.py, so the existing
gemmify_sft_chatml.py pass converts it unchanged.

Usage:
    uv run python scripts/build_resolved_frames_sft.py \
        --rollouts data/sft/frames_gemma4/rollouts.jsonl \
        --traces results/parametric_traces_raw/gemma4_31b_no_search_frames_alruns/*.json \
        --eval-json "results/frames_parametric/gemma4_31b/frames-cues_no_search_gemma4:31b_run_1.json" \
        --output-dir data/sft/frames_gemma4_resolved
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.curate_frames_sft_data import (
    PLAIN_CONDITION, SFT_FILENAME, group_by_example, is_test, load_rollouts, plain_reference,
)
from scripts.regrade_regex import heuristic_match

# The 7 conditions the resolved arm trains on. multiturn / searchmulti /
# confident_parametric are deliberately held out so the FRAMES test-set eval over all 10
# conditions measures generalisation to unseen cues.
SEVEN_CONDITIONS = ["verbose_plain", "verbose_polite", "terse_plain", "verbose_natural",
                    "verbose_elaborate", "verbose_query", "verbose_direct"]

# Cue suffix -> the training condition the no_search trace belongs to. Only cues inside the
# trained set may contribute tool-absent examples, or the held-out conditions leak.
TRACE_CUE_SUFFIX = {
    "verbose_elaborate": "Please answer with a detailed explanation",
    "verbose_direct": "Just answer the question directly",
    "verbose_polite": "If it isn't too much trouble",
}


def search_calls(messages) -> int:
    """Search calls the model itself made after the last user message.

    Counted past the last user message so the mocked history that the multiturn /
    searchmulti conditions prepend cannot inflate the count (the HISTORY_SEARCH_OFFSET bug).
    """
    last_user = max(i for i, m in enumerate(messages) if m.get("role") == "user")
    return sum(len(m.get("tool_calls") or []) for m in messages[last_user + 1:]
               if m.get("role") == "assistant")


def load_traces(patterns) -> list[dict]:
    rows = []
    for pat in patterns:
        for path in sorted(glob.glob(pat)):
            with open(path) as f:
                rows.extend(json.load(f))
    return rows


def trace_condition(trace) -> str | None:
    """Recover the cue condition from the RAW first user message.

    The trace's own `problem` field is already cleaned -- download_traces.clean_problem()
    strips the elaborate/polite/direct suffixes -- so the condition is NOT recoverable from
    it. Traces with >1 user message are the history-prefix conditions (multiturn /
    searchmulti), which are held out here.
    """
    users = [m for m in trace["message_trace"] if m.get("role") == "user"]
    if not users:
        return None
    if len(users) > 1:
        return None
    raw = "".join(p.get("content", "") for p in users[0].get("parts", [])
                  if p.get("type") == "text")
    for cond, suffix in TRACE_CUE_SUFFIX.items():
        if suffix in raw:
            return cond
    return PLAIN_CONDITION


def trace_user_message(trace) -> str:
    """The user message EXACTLY as the probe sent it, cue suffix included.

    Never reconstruct this from a suffix table: the trace's `problem` field is already
    cleaned, and a rebuilt string that differs by even one character would train a prompt
    the eval never sends.
    """
    for m in trace["message_trace"]:
        if m.get("role") == "user":
            return "".join(p.get("content", "") for p in m.get("parts", [])
                           if p.get("type") == "text")
    return ""


def trace_answer(trace) -> tuple[str, str]:
    """(thinking, answer_text) of the assistant turn."""
    for m in trace["message_trace"]:
        if m.get("role") == "assistant":
            th = "".join(p.get("content", "") for p in m.get("parts", []) if p.get("type") == "thinking")
            tx = "".join(p.get("content", "") for p in m.get("parts", []) if p.get("type") == "text")
            return th.strip(), tx.strip()
    return "", ""


def build_search_side(rollouts, conditions, threshold, test_pct, cap_per_cue, clip_plain):
    """Types A and B, from the on-policy search rollouts."""
    wanted = set(conditions) | {PLAIN_CONDITION}
    rs = [r for r in rollouts if r["condition"] in wanted]
    grouped = group_by_example(rs)
    train_ids = [e for e in grouped if not is_test(e, test_pct)]
    out = []
    for e in train_ids:
        ref = plain_reference(grouped[e], require_correct=True)
        if ref is None:
            continue
        for cond, rws in grouped[e].items():
            cand = [r for r in rws if r["is_correct"]]
            # The anchor is clipped too when --clip-plain, so plain's marginal search
            # distribution matches the cues' instead of keeping a fatter tail.
            if cond != PLAIN_CONDITION or clip_plain:
                cand = [r for r in cand if abs(search_calls(r["messages"]) - ref) <= threshold]
            if cap_per_cue:
                cand = cand[:cap_per_cue]
            for r in cand:
                out.append({"messages": r["messages"], "tools_available": True,
                            "condition": cond, "example_id": str(r["example_id"]),
                            "kind": "A_search" if search_calls(r["messages"]) else "B_declined"})
    return out, set(train_ids)


def build_tool_absent_side(traces, eval_json, train_ids, conditions, solve_rate_min,
                           cap_per_cue):
    """Type C, from the no_search probe's own traces.

    Label and trajectory come from the SAME events: a question counts as answerable without
    search when its solve rate over these traces clears --solve-rate-min, and the kept
    trajectories are the correct ones among them. Only correct trajectories are emitted, so
    the model is never taught to state a wrong parametric answer confidently.
    """
    ref = {}
    for r in json.load(open(eval_json)):
        ref[r["problem"].strip()] = (str(r["example_id"]), r["correct_answer"])

    allowed = set(conditions)
    per = defaultdict(list)          # (eid, cond) -> [(thinking, answer, correct)]
    for t in traces:
        cond = trace_condition(t)
        if cond is None or cond not in allowed:
            continue
        hit = ref.get(t["problem"].strip())
        if not hit:
            continue
        eid, gold = hit
        if eid not in train_ids:
            continue
        th, tx = trace_answer(t)
        if not tx:
            continue
        per[(eid, cond)].append((th, tx, heuristic_match(gold, tx), trace_user_message(t)))

    # Solve rate is pooled over every trace of that question, across the cues present.
    by_q = defaultdict(list)
    for (eid, _cond), rows in per.items():
        by_q[eid].extend(rows)
    solve_rate = {e: sum(1 for r in v if r[2]) / len(v) for e, v in by_q.items()}

    out = []
    for (eid, cond), rows in sorted(per.items()):
        if solve_rate.get(eid, 0.0) < solve_rate_min:
            continue
        good = [(th, tx, user) for th, tx, ok, user in rows if ok and th and user]
        if cap_per_cue:
            good = good[:cap_per_cue]
        for th, tx, user in good:
            # Emit in <think>-in-content form so gemmify_sft_chatml.py converts it unchanged.
            content = f"<think>\n{th}\n</think>\n{tx}"
            out.append({"messages": [
                {"role": "system", "content": ""},
                {"role": "user", "content": user},
                {"role": "assistant", "content": content, "tool_calls": []},
            ], "tools_available": False, "condition": cond, "example_id": eid,
                "kind": "C_no_tool"})
    return out, solve_rate


def write_tool_schema(output_dir: str) -> dict:
    """Emit the search tool schema EXACTLY as pydantic-ai sends it at inference.

    Generated from the same code path the eval uses -- Tool(LocalIndexSearchService.search) --
    rather than hand-written, because the whole point of the fix is that the training prompt
    and the inference prompt render byte-identically. It is written next to the dataset so
    train_sft.py reads the schema that this dataset was built against and cannot drift.

    Note the schema carries a pydantic-ai docstring-parser wart: the Returns section splits on
    the colon inside `{"title": str}`, giving `<type>A list of {"title"</type>`. That is what
    inference sends, so training must reproduce it verbatim.
    """
    from pydantic_ai.tools import Tool
    from src.services.local_index_search import LocalIndexSearchService

    class _Stub:
        search = LocalIndexSearchService.search

    td = Tool(_Stub().search).tool_def
    schema = {"type": "function", "function": {
        "name": td.name, "description": td.description,
        "parameters": td.parameters_json_schema}}
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "search_tool_schema.json")
    with open(path, "w") as f:
        json.dump([schema], f, indent=2)
    print(f"  wrote tool schema -> {path}")
    return schema


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollouts", default="data/sft/frames_gemma4/rollouts.jsonl")
    p.add_argument("--traces", nargs="+", required=True,
                   help="no_search trace JSON(s) from download_traces.py --no-dedup.")
    p.add_argument("--eval-json", required=True,
                   help="A parametric result JSON, used only for example_id + gold + question text.")
    p.add_argument("--output-dir", default="data/sft/frames_gemma4_resolved")
    p.add_argument("--conditions", nargs="+", default=SEVEN_CONDITIONS)
    p.add_argument("--threshold", type=int, default=1)
    p.add_argument("--test-pct", type=int, default=20)
    p.add_argument("--cap-per-cue", type=int, default=None,
                   help="Max kept examples per (question, condition) on the search side.")
    p.add_argument("--clip-plain", action="store_true",
                   help="Apply the closeness filter to the plain anchor too (recommended).")
    p.add_argument("--solve-rate-min", type=float, default=0.6,
                   help="Min parametric solve rate for a question to contribute tool-absent examples.")
    p.add_argument("--absent-cap-per-cue", type=int, default=2,
                   help="Max tool-absent examples per (question, condition).")
    p.add_argument("--dry-run", action="store_true", help="Report only, write nothing.")
    args = p.parse_args()

    rollouts = load_rollouts(args.rollouts)
    search_side, train_ids = build_search_side(
        rollouts, args.conditions, args.threshold, args.test_pct,
        args.cap_per_cue, args.clip_plain)

    traces = load_traces(args.traces)
    absent_side, solve_rate = build_tool_absent_side(
        traces, args.eval_json, train_ids, args.conditions,
        args.solve_rate_min, args.absent_cap_per_cue)

    records = search_side + absent_side
    kinds = Counter(r["kind"] for r in records)
    print(f"\n{'='*88}\nRESOLVED FRAMES SFT -- {len(records)} examples\n{'='*88}")
    print(f"  A  tools + searches      : {kinds['A_search']:>5}")
    print(f"  B  tools + answered      : {kinds['B_declined']:>5}")
    print(f"  C  NO tools + answered   : {kinds['C_no_tool']:>5}"
          f"   ({100.0*kinds['C_no_tool']/max(1,len(records)):.1f}% of the set)")
    print(f"  trained conditions       : {', '.join(args.conditions)}")
    print(f"  clip_plain={args.clip_plain} cap_per_cue={args.cap_per_cue} "
          f"solve_rate_min={args.solve_rate_min} absent_cap={args.absent_cap_per_cue}")

    print(f"\n  {'condition':<26} {'A':>6} {'B':>6} {'C':>6} {'sc_med':>7} {'sc_p90':>7}")
    per = defaultdict(Counter); sc = defaultdict(list)
    for r in records:
        per[r["condition"]][r["kind"]] += 1
        if r["tools_available"]:
            sc[r["condition"]].append(search_calls(r["messages"]))
    for c in args.conditions:
        v = sc.get(c) or [0]
        p90 = statistics.quantiles(v, n=10)[-1] if len(v) > 10 else max(v)
        print(f"  {c:<26} {per[c]['A_search']:>6} {per[c]['B_declined']:>6} "
              f"{per[c]['C_no_tool']:>6} {statistics.median(v):>7.1f} {p90:>7.1f}")

    qs = {r["example_id"] for r in records}
    qc = {r["example_id"] for r in records if r["kind"] == "C_no_tool"}
    print(f"\n  questions: {len(qs)} total | {len(qc)} contribute tool-absent examples")
    leaked = qs & {e for e in {str(x['example_id']) for x in rollouts} if is_test(e, args.test_pct)}
    print(f"  TEST-SPLIT LEAKAGE CHECK: {len(leaked)} test questions present "
          f"({'OK' if not leaked else 'FAIL'})")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return
    os.makedirs(args.output_dir, exist_ok=True)
    write_tool_schema(args.output_dir)
    # NOT SFT_FILENAME: this is the <think>-in-content form. gemmify_sft_chatml.py converts
    # it into SFT_FILENAME (the name train_sft.py's _ARM_FILES looks for), exactly as the
    # earlier gemma arms did. Training directly off this file would learn literal <think> tags.
    path = os.path.join(args.output_dir, "procedure1_resolved_raw.jsonl")
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps({"messages": r["messages"],
                                "tools_available": r["tools_available"]}) + "\n")
    test_ids = sorted(e for e in {str(x["example_id"]) for x in rollouts}
                      if is_test(e, args.test_pct))
    with open(os.path.join(args.output_dir, "test_ids.json"), "w") as f:
        json.dump(test_ids, f, indent=2)
    print(f"\n  wrote {len(records)} examples -> {path}")
    print(f"  NEXT: uv run python scripts/gemmify_sft_chatml.py --in {path} \\\n"
          f"          --out {os.path.join(args.output_dir, SFT_FILENAME)}")
    print(f"  held out {len(test_ids)} test questions -> {args.output_dir}/test_ids.json")


if __name__ == "__main__":
    main()
