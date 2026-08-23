"""
Phase 2 of the modal-answer redirection check (Phase 1:
analyze_modal_answer_shift.py, which found the eligible pairs and produced
results/modal_answer_shift/eligible_modal_answer_pairs.jsonl). The sibling
session has since judged all 20 available cells directly, at
results/{frames,medqa}_parametric/<model>/<prefix>_<tag>_<cue>_vs_plain_modal_change.json
(schema: example_id, correct_answer, plain_modal_run, cue_modal_run,
plain_modal_response, cue_modal_response, changed [bool]).

This script aggregates those verdicts and answers three questions:
  1. What fraction of examples redirect to a different canonical answer under
     the cue, per (dataset, model, cue)?
  2. Does redirection happen even in cells where aggregate entropy is FLAT
     (analyze_entropy_under_cue.py's sign test found no net shift)? Entropy is
     symmetric to WHICH cluster is the majority, not what it contains, so a
     cell can look perfectly calibrated on entropy alone while still silently
     reassigning which answer is canonical for a meaningful fraction of
     examples -- this is the one thing the entropy-only test structurally
     cannot see.
  3. Among redirected examples, does the cue move the canonical answer TOWARD
     or AWAY FROM correctness (regex-graded via heuristic_match/relaxed_match,
     reused from regrade_regex.py -- same convention as
     analyze_volume_accuracy_decoupling.py: appropriate here because this is
     short-gold-vs-response matching, not response-vs-response semantic
     comparison, which is what the judge itself was for)?

Usage:
    uv run python scripts/analyze_modal_answer_shift_judged.py
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "scripts"))
OUT_DIR = os.path.join(REPO, "results", "modal_answer_shift")

from regrade_regex import heuristic_match, relaxed_match  # noqa: E402

ENTROPY_CSV = os.path.join(REPO, "results", "entropy_under_cue", "entropy_under_cue.csv")
TRANSITIONS_CSV = os.path.join(OUT_DIR, "cluster_count_transitions.csv")


def rc(gold, resp):
    return heuristic_match(gold, resp) or relaxed_match(gold, resp)


def parse_fname(path):
    """Recover (dataset, model, cue) from a *_vs_plain_modal_change.json path."""
    parts = path.split(os.sep)
    model = parts[-2]
    fname = parts[-1]
    ds = "frames" if fname.startswith("frames-cues") else "medqa"
    prefix = "frames-cues_no_search_" if ds == "frames" else "medqa-500_no_search_"
    rest = fname[len(prefix):-len("_vs_plain_modal_change.json")]
    # rest = "<tag>_<cue>", tag itself may contain ':' but not '_', cue is
    # whatever remains after the first '_' following the tag -- since tag has
    # no underscores in this project's naming, split on first '_'.
    tag, cue = rest.split("_", 1)
    return ds, model, cue


def main():
    entropy_lookup = {}
    if os.path.exists(ENTROPY_CSV):
        for r in csv.DictReader(open(ENTROPY_CSV)):
            entropy_lookup[(r["dataset"], r["model"], r["cue"])] = r

    transitions_lookup = {}
    if os.path.exists(TRANSITIONS_CSV):
        for r in csv.DictReader(open(TRANSITIONS_CSV)):
            transitions_lookup[(r["dataset"], r["model"], r["cue"])] = r

    rows = []
    for path in sorted(glob.glob(os.path.join(REPO, "results", "*_parametric", "*", "*_vs_plain_modal_change.json"))):
        ds, model, cue = parse_fname(path)
        data = json.load(open(path))
        n = len(data)
        n_changed = sum(1 for r in data if r["changed"])

        # Direction of redirection, among changed==True rows only.
        n_toward_correct = n_away_from_correct = n_correct_to_correct = n_wrong_to_wrong = 0
        for r in data:
            if not r["changed"]:
                continue
            gold = r["correct_answer"] or ""
            plain_ok = rc(gold, r["plain_modal_response"] or "")
            cue_ok = rc(gold, r["cue_modal_response"] or "")
            if plain_ok and not cue_ok:
                n_away_from_correct += 1
            elif not plain_ok and cue_ok:
                n_toward_correct += 1
            elif plain_ok and cue_ok:
                n_correct_to_correct += 1
            else:
                n_wrong_to_wrong += 1

        key = (ds, model, cue)
        ent = entropy_lookup.get(key, {})
        trans = transitions_lookup.get(key, {})

        rows.append(dict(
            dataset=ds, model=model, cue=cue, n=n,
            n_changed=n_changed, pct_changed=round(100 * n_changed / n, 1),
            n_toward_correct=n_toward_correct, n_away_from_correct=n_away_from_correct,
            n_correct_to_correct=n_correct_to_correct, n_wrong_to_wrong=n_wrong_to_wrong,
            entropy_sign_test_p=ent.get("sign_test_p", ""),
            entropy_mean_delta=ent.get("mean_delta", ""),
            entropy_flat=(ent.get("sign_test_p", "") != "" and float(ent.get("sign_test_p", 1)) >= 0.05),
            mechanism=ent.get("mechanism", ""),
            pct_consensus_breakdown=trans.get("pct_consensus_breakdown", ""),
        ))

    out_path = os.path.join(OUT_DIR, "modal_answer_shift_judged.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}  ({len(rows)} rows)\n")

    print("=== modal-answer redirection rate, cross-referenced against the entropy-flat verdict ===")
    for r in sorted(rows, key=lambda r: -r["pct_changed"]):
        flat_flag = "FLAT entropy" if r["entropy_flat"] else "entropy SHIFTED" if r["entropy_flat"] is False else "entropy: n/a"
        net_direction = r["n_toward_correct"] - r["n_away_from_correct"]
        dir_str = f"net {net_direction:+d} toward correct" if r["n_changed"] > 0 else "n/a"
        print(f"  {r['dataset']:6s} {r['model']:20s} {r['cue']:22s} n={r['n']:4d}  "
              f"changed={r['n_changed']:3d} ({r['pct_changed']:5.1f}%)  [{flat_flag}]  "
              f"toward={r['n_toward_correct']} away={r['n_away_from_correct']} ({dir_str})  "
              f"mech={r['mechanism'] or '?'}")

    flat_rows = [r for r in rows if r["entropy_flat"] is True]
    shifted_rows = [r for r in rows if r["entropy_flat"] is False]
    if flat_rows:
        avg_changed_flat = sum(r["pct_changed"] for r in flat_rows) / len(flat_rows)
        print(f"\nAmong {len(flat_rows)} cells with FLAT aggregate entropy (no net shift by the sign test), "
              f"mean modal-answer redirection rate = {avg_changed_flat:.1f}%.")
        print("This is the key number: entropy-flat does NOT mean answer-stable -- entropy is symmetric to")
        print("WHICH cluster is the majority, not what it contains, so this redirection is invisible to the")
        print("aggregate entropy test by construction, not because it's absent.")
    if shifted_rows:
        avg_changed_shifted = sum(r["pct_changed"] for r in shifted_rows) / len(shifted_rows)
        print(f"\nAmong {len(shifted_rows)} cells with a significant entropy SHIFT, "
              f"mean modal-answer redirection rate = {avg_changed_shifted:.1f}%.")

    total_toward = sum(r["n_toward_correct"] for r in rows)
    total_away = sum(r["n_away_from_correct"] for r in rows)
    total_changed = sum(r["n_changed"] for r in rows)
    overall_p = stats.binomtest(total_toward, total_toward + total_away, 0.5).pvalue if (total_toward + total_away) else float("nan")
    print(f"\nAcross all {len(rows)} cells: {total_changed} redirected examples total -- "
          f"{total_toward} moved toward the correct answer, {total_away} moved away "
          f"(net {total_toward - total_away:+d}, binomial p={overall_p:.3g}). [regex-graded correctness -- "
          f"directional signal only, known to undercount MedQA absolute accuracy, see accuracy_revision.md caveat 8]")

    print("\n=== how much the modal answer changed, per cue, pooled across models/datasets ===")
    print("(this is the trustworthy number -- N changed / N eligible, no correctness grading involved)")
    by_cue_change = defaultdict(lambda: {"n": 0, "changed": 0, "n_cells": 0})
    for r in rows:
        by_cue_change[r["cue"]]["n"] += r["n"]
        by_cue_change[r["cue"]]["changed"] += r["n_changed"]
        by_cue_change[r["cue"]]["n_cells"] += 1
    total_n = sum(d["n"] for d in by_cue_change.values())
    total_changed_pooled = sum(d["changed"] for d in by_cue_change.values())
    for cue, d in sorted(by_cue_change.items(), key=lambda kv: -kv[1]["changed"] / kv[1]["n"]):
        pct = 100 * d["changed"] / d["n"]
        print(f"  {cue:22s} ({d['n_cells']} cells, n={d['n']:5d})  changed={d['changed']:4d}  ({pct:5.1f}%)")
    print(f"  {'ALL CUES POOLED':22s} ({len(rows)} cells, n={total_n:5d})  changed={total_changed_pooled:4d}  "
          f"({100 * total_changed_pooled / total_n:5.1f}%)")

    print("\n=== direction of redirection (toward/away from correct), per cue -- REGEX-GRADED, CAVEAT BELOW ===")
    print("CAVEAT: 'direct' cuts response length by 400-1000+ chars in every cell (see entropy_under_cue.csv's")
    print("len_delta column) -- the same mechanism already documented to make regex substring/word-subset")
    print("matching undercount short, reworded-but-correct responses (see accuracy_revision.md caveat 8, the")
    print("MedQA regex-vs-LLM-judge gap). 'direct's skew below is likely that grading artifact resurfacing, NOT")
    print("a real accuracy effect -- treat it as unreliable, not as evidence 'direct' pushes toward wrong answers.")
    by_cue = defaultdict(lambda: {"toward": 0, "away": 0, "n_cells": 0})
    for r in rows:
        by_cue[r["cue"]]["toward"] += r["n_toward_correct"]
        by_cue[r["cue"]]["away"] += r["n_away_from_correct"]
        by_cue[r["cue"]]["n_cells"] += 1
    for cue, d in sorted(by_cue.items(), key=lambda kv: -abs(kv[1]["toward"] - kv[1]["away"])):
        t, a = d["toward"], d["away"]
        if t + a == 0:
            continue
        p = stats.binomtest(t, t + a, 0.5).pvalue
        flag = " *** SIGNIFICANT DIRECTIONAL SKEW" if p < 0.05 else ""
        suspect = "  [SUSPECT: large response-length shift, see caveat]" if cue == "direct" else ""
        direction = "TOWARD WRONG" if t < a else "toward correct"
        print(f"  {cue:22s} ({d['n_cells']} cells)  toward={t:3d} away={a:3d} net={t-a:+4d}  "
              f"p={p:.3g}  [{direction}]{flag}{suspect}")


if __name__ == "__main__":
    main()
