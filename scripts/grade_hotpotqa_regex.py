r"""
Deterministic regex (SQuAD-style) grading for the HotpotQA cue grid + parametric probe.

All HotpotQA runs were produced with `--no_grader`, so `sampler_correct` is None everywhere.
This script decides correctness OFFLINE from the saved `sampler_response`, reusing the EXACT
match functions the project already trusts (`scripts/regrade_regex.py`: SQuAD normalization with
the Unicode-typography fold, strict normalized-substring, one-directional word-subset relaxed
match) so HotpotQA grades are semantically comparable to the FRAMES/MedQA regex regrades.

Unlike regrade_regex.py this does NOT compare against an LLM judge (there is none here) — it just
produces verdicts and accuracy tables.

THREE GRADERS, reported separately:
  * strict   — normalized gold appears as a substring of the normalized response.
  * relaxed  — every normalized gold word appears somewhere in the response (superset of strict).
  * boolean  — for the ~4.7% of hotpotqa-300 whose gold is literally "yes"/"no" (14 of 300),
               substring matching is meaningless: "no" occurs inside ordinary prose constantly,
               so `strict` scores them almost arbitrarily. These rows instead match on the FIRST
               standalone yes/no token in the response, and are reported as their own segment.
               Headline accuracy is quoted over NON-boolean rows; the boolean segment is shown
               beside it rather than folded in.

KNOWN BIAS, state it when reporting: substring/word-subset grading rewards VERBOSITY — a longer
answer has more chances to contain the gold string, and can also "match" while explicitly
rejecting the answer. The cue conditions deliberately manipulate response length (median words:
direct ~2-3, plain ~31-97, elaborate ~215-275), so cross-condition accuracy differences from this
grader are CONFOUNDED with length. Per-condition median response length is emitted alongside
accuracy so the confound stays visible. An LLM-judge pass is the fix; this is the cheap first look.

Outputs (under --output-dir):
    per_row.csv       one row per (model, run_name, example_id) with all verdicts
    by_condition.csv  accuracy per (model, run_name), split boolean/non-boolean, + median words
    summary.md        readable table

Usage:
    uv run python scripts/grade_hotpotqa_regex.py --results-root results/hotpotqa_cue_grid
    uv run python scripts/grade_hotpotqa_regex.py --results-root results/hotpotqa_parametric \
        --agent-type no_search --output-dir results/hotpotqa_parametric_regex
"""

from __future__ import annotations

import os
import re
import csv
import sys
import glob
import json
import argparse
import statistics
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from scripts.regrade_regex import normalize, heuristic_match, relaxed_match  # noqa: E402
from src.services.common import normalize_response  # noqa: E402

# Longest-first so "confident_parametric" is not shadowed by a shorter name.
KNOWN_CUES = ["confident_parametric", "searchmulti", "multiturn", "elaborate",
              "natural", "polite", "direct", "query", "plain"]

_YESNO_RE = re.compile(r"\b(yes|no)\b")

# Search calls contributed by a condition's injected conversation HISTORY, which older runs
# wrongly counted as searches the model chose to make. `AgentAsSampler.acall` passes the history
# to pydantic-ai as `message_history`, and `response.all_messages()` returns the full conversation
# INCLUDING it -- so every mocked `search` tool call in the prefix inflated `sampler_search_calls`
# by a constant. Fixed at the source (agent_sampler.py now subtracts it and records
# `history_search_calls`), but rows collected BEFORE that fix need correcting here.
# Verified empirically on hotpotqa-300: data/frames_cues/search_multi_turn.json has exactly 1
# search tool_call in each of its 5 conversations, and every model's searchmulti minimum is >=1
# with 0.0% zero-search rows, while plain has 0.3-17% zero-search rows.
# The chit-chat history (multiturn) has 0 tool calls and needs no correction.
HISTORY_SEARCH_OFFSET = {"searchmulti": 1}


def history_offset_for(run_name: str) -> int:
    """Constant to subtract from a legacy row's search_calls for this condition."""
    base = re.sub(r"(_rep\d+|_run_\d+)$", "", run_name)
    return HISTORY_SEARCH_OFFSET.get(base, 0)


def setup_args():
    p = argparse.ArgumentParser(description="Regex/EM grading for HotpotQA runs.")
    p.add_argument("--results-root", default="results/hotpotqa_cue_grid")
    p.add_argument("--dataset", default="hotpotqa-300")
    p.add_argument("--agent-type", default="baseline", choices=["baseline", "no_search"])
    p.add_argument("--subset-file", default=None,
                   help="Tier JSONL for type/boolean metadata (default derived from --dataset).")
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def parse_run_name(stem: str, dataset: str, agent: str) -> str | None:
    """`<dataset>_<agent>_<model>_<run_name>` -> run_name. Model may contain '_', so anchor on
    the known cue vocabulary and allow the `_rep2` / `_run_<n>` suffixes the drivers append."""
    prefix = f"{dataset}_{agent}_"
    if not stem.startswith(prefix):
        return None
    rest = stem[len(prefix):]
    for cue in KNOWN_CUES:
        m = re.search(r"_(" + re.escape(cue) + r")(_rep\d+|_run_\d+)?$", rest)
        if m:
            return m.group(1) + (m.group(2) or "")
    return None


def boolean_match(gold: str, response: str) -> bool:
    """First standalone yes/no token in the response must equal the gold."""
    m = _YESNO_RE.search(normalize(response))
    return bool(m) and m.group(1) == normalize(gold)


def main():
    args = setup_args()
    subset = args.subset_file or f"data/{args.dataset.replace('-', '_')}.jsonl"
    meta = {}
    if os.path.exists(subset):
        for line in open(subset):
            r = json.loads(line)
            meta[r["example_id"]] = r
    else:
        print(f"[warn] {subset} missing -- no type/boolean segmentation")

    out_dir = args.output_dir or f"{args.results_root.rstrip('/')}_regex"
    os.makedirs(out_dir, exist_ok=True)

    pattern = os.path.join(args.results_root, "*", f"{args.dataset}_{args.agent_type}_*.json")
    paths = sorted(glob.glob(pattern))
    # Skip directories parked aside by hand (e.g. "<model>.athena-superseded-<date>", "*.bak"):
    # they hold real result files but are NOT a model and must not appear as a row.
    skipped = [p for p in paths if re.search(r"superseded|\.bak", os.path.dirname(p))]
    paths = [p for p in paths if p not in skipped]
    if skipped:
        dirs = sorted({os.path.basename(os.path.dirname(p)) for p in skipped})
        print(f"  [skip] set-aside dirs not graded: {', '.join(dirs)}")
    if not paths:
        raise SystemExit(f"no result files matched {pattern}")

    per_row = []
    for path in paths:
        model = os.path.basename(os.path.dirname(path))
        run_name = parse_run_name(os.path.basename(path)[:-len(".json")], args.dataset, args.agent_type)
        if run_name is None:
            print(f"  [skip] unparseable: {os.path.basename(path)}")
            continue
        try:
            rows = json.load(open(path))
        except Exception as e:
            print(f"  [skip] unreadable {path}: {e}")
            continue
        for row in rows:
            gold = row.get("correct_answer") or ""
            raw = row.get("sampler_response") or ""
            resp = normalize_response(raw)
            ex_id = row.get("example_id")
            info = meta.get(ex_id, {})
            is_bool = bool(info.get("answer_is_boolean", normalize(gold) in ("yes", "no")))
            strict = heuristic_match(gold, resp)
            raw_sc = row.get("sampler_search_calls")
            # Prefer the value the runner itself recorded once the source fix landed; fall back
            # to the known per-condition constant for rows collected before it.
            offset = row.get("history_search_calls")
            offset = history_offset_for(run_name) if offset is None else 0
            corrected_search_calls = (max(0, raw_sc - offset)
                                      if isinstance(raw_sc, (int, float)) else raw_sc)
            per_row.append({
                "model": model, "run_name": run_name, "example_id": ex_id,
                "type": info.get("type", ""), "answer_is_boolean": is_bool,
                "gold": gold, "n_words": len(str(raw).split()),
                "search_calls": corrected_search_calls,
                "search_calls_raw": raw_sc,
                "history_search_calls": offset,
                "strict": int(strict),
                "relaxed": int(strict or relaxed_match(gold, resp)),
                "boolean": int(boolean_match(gold, resp)) if is_bool else "",
            })

    rows_csv = os.path.join(out_dir, "per_row.csv")
    with open(rows_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_row[0].keys()))
        w.writeheader(); w.writerows(per_row)

    groups = defaultdict(list)
    for r in per_row:
        groups[(r["model"], r["run_name"])].append(r)

    table = []
    for (model, run_name), rs in sorted(groups.items()):
        nb = [r for r in rs if not r["answer_is_boolean"]]
        bo = [r for r in rs if r["answer_is_boolean"]]
        table.append({
            "model": model, "run_name": run_name, "n": len(rs),
            "n_nonbool": len(nb), "n_bool": len(bo),
            "strict_nonbool": round(sum(r["strict"] for r in nb) / len(nb), 4) if nb else "",
            "relaxed_nonbool": round(sum(r["relaxed"] for r in nb) / len(nb), 4) if nb else "",
            "boolean_acc": round(sum(r["boolean"] for r in bo) / len(bo), 4) if bo else "",
            "median_words": int(statistics.median([r["n_words"] for r in rs])) if rs else 0,
            "mean_search_calls": round(
                statistics.mean([r["search_calls"] for r in rs if isinstance(r["search_calls"], (int, float))]), 3)
            if any(isinstance(r["search_calls"], (int, float)) for r in rs) else "",
        })

    cond_csv = os.path.join(out_dir, "by_condition.csv")
    with open(cond_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        w.writeheader(); w.writerows(table)

    md = [f"# HotpotQA regex grading — {args.results_root} ({args.agent_type})", "",
          f"{len(per_row)} rows over {len(groups)} (model, condition) cells.", "",
          "Accuracy is over NON-boolean rows; the yes/no segment is separate because substring",
          "matching is meaningless there. **Substring grading rewards verbosity**, and the cues",
          "manipulate length — read accuracy next to `median_words`, not alone.", "",
          "| model | condition | n | strict | relaxed | bool acc | med words | mean search |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        md.append(f"| {r['model']} | {r['run_name']} | {r['n']} | {r['strict_nonbool']} | "
                  f"{r['relaxed_nonbool']} | {r['boolean_acc']} | {r['median_words']} | "
                  f"{r['mean_search_calls']} |")
    open(os.path.join(out_dir, "summary.md"), "w").write("\n".join(md) + "\n")

    print(f"graded {len(per_row)} rows over {len(groups)} cells")
    print(f"  {rows_csv}\n  {cond_csv}\n  {os.path.join(out_dir, 'summary.md')}")


if __name__ == "__main__":
    main()
