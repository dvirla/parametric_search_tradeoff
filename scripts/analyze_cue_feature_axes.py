"""
Given that neither entropy (analyze_entropy_under_cue.py) nor the model's own
canonical answer (analyze_modal_answer_shift_judged.py) changes under a cue in
any way that explains the huge search-volume shifts already documented
(cue_suppression_mechanism.csv's level_shift), the search-triggering policy
must be responding to something about the cue ITSELF, not the model's
epistemic state on the question. This script tests the most obvious
candidate: does suppression magnitude track simple, hand-coded TEXTUAL
features of the cue -- explicit reference to search/confidence vs. purely
stylistic instructions -- plus a dedicated "conversation state" axis for
`multiturn`/`searchmulti{,2,3}`, which are structurally different from every
other cue (they inject prior CONVERSATION TURNS via --history_path, not a
question-text wrapper -- see src/services/qa_eval.py L196 and
scripts/build_multiround_search_history.py).

Cue feature coding (hand-coded from the actual template/history-file source,
not guessed -- see SOURCE comments below each entry):
  - mentions_search: does the cue's text explicitly name the search tool?
  - epistemic_direction / epistemic_strength: does the cue directly assert
    something about the model's own knowledge state, and how forcefully?
    none < mild (baked into the question's own phrasing) < strong (an
    explicit meta-instruction about the model's competence).
  - output_length: none / short ("2-4 sentences") / long ("8-10 sentences").
  - answer_commit: a maximal "final answer only" output directive (`direct`).
  - politeness / structured_format: the two remaining QUERY_TEMPLATE axes.
  - conversation_state: none / chitchat (unrelated prior turns, NO search
    content at all) / mocked_search (prior turns show the assistant already
    used the search tool and got results). THE NEW AXIS.
  - conversation_rounds: 0 for non-conversational cues; 1/2/3 for
    searchmulti/searchmulti2/searchmulti3 (round count is literally the
    manipulated variable there -- see build_multiround_search_history.py).

CAVEAT stated up front, not buried: there are only 12 distinct cues in this
data. Cue-level features cannot support a trustworthy multi-covariate
regression at that n (cluster-robust inference needs far more independent
clusters than that) -- this script reports descriptive contrasts and simple
rank correlations at the cue level, explicitly, rather than dressing up an
underpowered regression as confirmatory. Treat every number here as
suggestive, not as a hypothesis-tested claim.

Usage:
    uv run python scripts/analyze_cue_feature_axes.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "results", "cue_feature_axes")
os.makedirs(OUT_DIR, exist_ok=True)

MECHANISM_CSV = os.path.join(REPO, "results", "cue_suppression_mechanism", "cue_suppression_mechanism.csv")
DELTA_CSV = os.path.join(REPO, "results", "cue_suppression_mechanism", "volume_vs_accuracy_delta.csv")

# base cue name (phrasing prefix verbose_/orig_/terse_ stripped) -> features.
# EPISTEMIC_STRENGTH: 0=none, 1=mild (baked into question phrasing), 2=strong
# (explicit meta-instruction about the model's own competence/task process).
CUE_FEATURES = {
    "natural": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                     output_length="short", answer_commit=0, politeness=0, structured_format=0,
                     conversation_state="none", conversation_rounds=0,
                     source="NATURAL_QUERY_TEMPLATE, qa_eval.py L32-34: '...Please answer in 2-4 sentences.'"),
    "elaborate": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                       output_length="long", answer_commit=0, politeness=0, structured_format=0,
                       conversation_state="none", conversation_rounds=0,
                       source="ELABORATE_QUERY_TEMPLATE, qa_eval.py L37-39: '...at least 8-10 sentences'"),
    "polite": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                    output_length="none", answer_commit=0, politeness=1, structured_format=0,
                    conversation_state="none", conversation_rounds=0,
                    source="POLITE_QUERY_TEMPLATE, qa_eval.py L45-47: extreme deferential wrapper, no length/format directive"),
    "direct": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                    output_length="none", answer_commit=1, politeness=0, structured_format=0,
                    conversation_state="none", conversation_rounds=0,
                    source="DIRECT_QUERY_TEMPLATE, qa_eval.py L54-56: 'Just answer...directly — final answer only.'"),
    "query": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                   output_length="none", answer_commit=0, politeness=0, structured_format=1,
                   conversation_state="none", conversation_rounds=0,
                   source="QUERY_TEMPLATE, qa_eval.py L22-29: Explanation/Exact Answer/Confidence structured format"),
    "confident_parametric": dict(mentions_search=1, epistemic_direction="boost", epistemic_strength=2,
                                   output_length="none", answer_commit=0, politeness=0, structured_format=0,
                                   conversation_state="none", conversation_rounds=0,
                                   source="CONFIDENT_PARAMETRIC_QUERY_TEMPLATE, qa_eval.py L62-64: 'You already have the "
                                          "relevant knowledge...You do not need to use the search tool'"),
    "epi_strong_boost": dict(mentions_search=0, epistemic_direction="boost", epistemic_strength=1,
                               output_length="none", answer_commit=0, politeness=0, structured_format=0,
                               conversation_state="none", conversation_rounds=0,
                               source="data/frames_cues/epi_strong_boost.jsonl: question rewritten with 'You must know "
                                      "this one: ...' prefix baked into the question text itself"),
    "epi_strong_hedge": dict(mentions_search=0, epistemic_direction="hedge", epistemic_strength=1,
                               output_length="none", answer_commit=0, politeness=0, structured_format=0,
                               conversation_state="none", conversation_rounds=0,
                               source="data/frames_cues/epi_strong_hedge.jsonl: 'I'm really not sure, but...' prefix "
                                      "baked into the question text itself"),
    "multiturn": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                        output_length="none", answer_commit=0, politeness=0, structured_format=0,
                        conversation_state="chitchat", conversation_rounds=0,
                        source="data/frames_cues/chit_chat_multi_turn.json via --history_path: unrelated prior turn "
                               "(e.g. drafting a business email), NO search/tool content at all"),
    "searchmulti": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                          output_length="none", answer_commit=0, politeness=0, structured_format=0,
                          conversation_state="mocked_search", conversation_rounds=1,
                          source="data/frames_cues/search_multi_turn.json via --history_path: 1 prior round showing "
                                 "the assistant already called the search tool and got a result"),
    "searchmulti2": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                           output_length="none", answer_commit=0, politeness=0, structured_format=0,
                           conversation_state="mocked_search", conversation_rounds=2,
                           source="data/frames_cues/search_multi_turn_2round.json: 2 concatenated mocked-search rounds"),
    "searchmulti3": dict(mentions_search=0, epistemic_direction="none", epistemic_strength=0,
                           output_length="none", answer_commit=0, politeness=0, structured_format=0,
                           conversation_state="mocked_search", conversation_rounds=3,
                           source="data/frames_cues/search_multi_turn_3round.json: 3 concatenated mocked-search rounds"),
}

PHRASING_PREFIXES = ("verbose_", "orig_", "terse_")


def strip_phrasing(cue):
    for p in PHRASING_PREFIXES:
        if cue.startswith(p):
            return cue[len(p):]
    return cue


def main():
    delta_lookup = {}
    for r in csv.DictReader(open(DELTA_CSV)):
        delta_lookup[(r["dataset"], r["model"], r["cue"])] = r

    cell_rows = []
    for r in csv.DictReader(open(MECHANISM_CSV)):
        base_cue = strip_phrasing(r["cue"])
        if base_cue not in CUE_FEATURES:
            print(f"  ! no feature coding for cue '{base_cue}' (from '{r['cue']}') -- skipping")
            continue
        feat = CUE_FEATURES[base_cue]
        delta = delta_lookup.get((r["dataset"], r["model"], r["cue"]))
        base_calls = float(delta["base_calls"]) if delta else None
        d_calls = float(delta["d_calls"]) if delta else None
        rel_shift = (d_calls / base_calls) if (delta and base_calls and base_calls > 1e-6) else None

        cell_rows.append(dict(
            dataset=r["dataset"], model=r["model"], cue=r["cue"], base_cue=base_cue,
            level_shift=float(r["level_shift"]), p_level_shift=float(r["p_level_shift"]),
            base_calls=base_calls, d_calls=d_calls, rel_shift=rel_shift,
            mechanism=r["mechanism"], **feat,
        ))

    cell_path = os.path.join(OUT_DIR, "cue_feature_axes_cells.csv")
    with open(cell_path, "w", newline="") as f:
        fieldnames = [k for k in cell_rows[0].keys() if k != "source"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in cell_rows:
            w.writerow({k: v for k, v in row.items() if k != "source"})
    print(f"wrote {cell_path}  ({len(cell_rows)} rows)\n")

    # Aggregate to one row per cue (mean across (dataset, model) cells) -- the
    # actual unit of independent variation for cue-level TEXTUAL features.
    by_cue = defaultdict(list)
    for r in cell_rows:
        by_cue[r["base_cue"]].append(r)

    cue_summary = []
    for cue, rows in sorted(by_cue.items()):
        level_shifts = [r["level_shift"] for r in rows]
        rel_shifts = [r["rel_shift"] for r in rows if r["rel_shift"] is not None]
        frames_ls = [r["level_shift"] for r in rows if r["dataset"] == "frames"]
        medqa_ls = [r["level_shift"] for r in rows if r["dataset"] == "medqa"]
        feat = CUE_FEATURES[cue]
        cue_summary.append(dict(
            cue=cue, n_cells=len(rows),
            mean_level_shift=round(float(np.mean(level_shifts)), 3),
            mean_abs_level_shift=round(float(np.mean(np.abs(level_shifts))), 3),
            mean_level_shift_frames=round(float(np.mean(frames_ls)), 3) if frames_ls else "",
            mean_level_shift_medqa=round(float(np.mean(medqa_ls)), 3) if medqa_ls else "",
            mean_rel_shift=round(float(np.mean(rel_shifts)), 3) if rel_shifts else "",
            **feat,
        ))

    summary_path = os.path.join(OUT_DIR, "cue_feature_axes_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = [k for k in cue_summary[0].keys() if k != "source"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in cue_summary:
            w.writerow({k: v for k, v in row.items() if k != "source"})
    print(f"wrote {summary_path}  ({len(cue_summary)} cues)\n")

    print("=== per-cue mean |level shift| (search calls), by feature coding ===")
    print(f"  n = {len(cue_summary)} distinct cues -- descriptive only, not enough units for confirmatory inference\n")
    for r in sorted(cue_summary, key=lambda r: -r["mean_abs_level_shift"]):
        print(f"  {r['cue']:22s} n_cells={r['n_cells']:2d}  mean|shift|={r['mean_abs_level_shift']:.3f} "
              f"(signed={r['mean_level_shift']:+.3f})  "
              f"(FRAMES={r['mean_level_shift_frames']}, MedQA={r['mean_level_shift_medqa']})  "
              f"mentions_search={r['mentions_search']} epistemic={r['epistemic_direction']}/{r['epistemic_strength']} "
              f"len={r['output_length']} commit={r['answer_commit']} polite={r['politeness']} "
              f"fmt={r['structured_format']} conv={r['conversation_state']}/{r['conversation_rounds']}")

    print("\n=== epistemic strength axis (ordinal 0/1/2) vs. mean |level shift| ===")
    strength_vals = [r["epistemic_strength"] for r in cue_summary]
    shift_vals = [r["mean_abs_level_shift"] for r in cue_summary]
    rho, p = stats.spearmanr(strength_vals, shift_vals)
    print(f"  Spearman rho={rho:+.3f} (p={p:.3g}), n={len(cue_summary)} cues")
    for lvl, label in [(0, "none"), (1, "mild (baked into question text)"), (2, "strong (explicit meta-instruction)")]:
        vals = [r["mean_abs_level_shift"] for r in cue_summary if r["epistemic_strength"] == lvl]
        if vals:
            print(f"    strength={lvl} ({label}): n={len(vals)}, mean|shift|={np.mean(vals):.3f} -- cues: "
                  f"{[r['cue'] for r in cue_summary if r['epistemic_strength'] == lvl]}")

    print("\n=== explicit search-mention (confident_parametric) vs. every other cue ===")
    mentions = [r["mean_abs_level_shift"] for r in cue_summary if r["mentions_search"] == 1]
    no_mentions = [r["mean_abs_level_shift"] for r in cue_summary if r["mentions_search"] == 0]
    print(f"  mentions_search=1: n={len(mentions)}, mean|shift|={np.mean(mentions):.3f}")
    print(f"  mentions_search=0: n={len(no_mentions)}, mean|shift|={np.mean(no_mentions):.3f}  "
          f"(n=1 vs n={len(no_mentions)} -- not a real hypothesis test, just the raw contrast)")

    print("\n=== THE NEW AXIS: conversation state (none / chitchat / mocked_search) -- SIGNED, direction matters ===")
    for state in ("none", "chitchat", "mocked_search"):
        vals = [r["mean_level_shift"] for r in cue_summary if r["conversation_state"] == state]
        if vals:
            cues = [r["cue"] for r in cue_summary if r["conversation_state"] == state]
            print(f"  {state:14s} n={len(vals):2d}  mean SIGNED shift={np.mean(vals):+.3f}  "
                  f"mean|shift|={np.mean(np.abs(vals)):.3f}  cues={cues}")

    print("\n  Chit-chat (unrelated prior turn, NO search content) vs. mocked-search (prior turn already shows")
    print("  the assistant using the tool) isolates whether ANY conversational context suppresses search, or")
    print("  specifically the IMPLICATION 'you already searched recently' -- and the SIGN is the finding here,")
    print("  not just the magnitude:")
    chitchat_signed = [r["mean_level_shift"] for r in cue_summary if r["conversation_state"] == "chitchat"]
    mocked_signed = [r["mean_level_shift"] for r in cue_summary if r["conversation_state"] == "mocked_search"]
    if chitchat_signed and mocked_signed:
        chit_dir = 'SUPPRESSES' if chitchat_signed[0] < 0 else 'INCREASES'
        mock_dir = 'SUPPRESSES' if np.mean(mocked_signed) < 0 else 'INCREASES'
        print(f"    chitchat (multiturn): {chitchat_signed[0]:+.3f}  -- {chit_dir} search")
        print(f"    mocked_search (avg of searchmulti/2/3): {np.mean(mocked_signed):+.3f}  -- {mock_dir} search")
        if chit_dir == mock_dir:
            print(f"    SAME direction, different magnitude: both {chit_dir.lower()} search. Any conversational")
            print("    context (not specifically 'you already searched') appears to suppress search somewhat --")
            print("    chit-chat suppresses more than mocked-search does. NOTE: an earlier, buggy version of this")
            print("    analysis (raw, uncorrected sampler_search_calls) found mocked-search INCREASING search")
            print("    with a clean round dose-response -- that was almost entirely the mock-history-call")
            print("    counting bug (see accuracy_revision.md Sec 1.0's correction note), not a real effect.")
        else:
            print(f"    OPPOSITE directions: chitchat {chit_dir.lower()}s, mocked-search {mock_dir.lower()}s search.")

    print("\n  Dose-response within mocked-search rounds (searchmulti=1 round -> searchmulti3=3 rounds), SIGNED:")
    dose_rows = [(r["conversation_rounds"], r["mean_level_shift"], r["cue"]) for r in cue_summary
                 if r["conversation_state"] == "mocked_search"]
    for rounds, shift, cue in sorted(dose_rows):
        print(f"    {rounds} round(s) ({cue}): mean signed shift={shift:+.3f}")
    if len(dose_rows) >= 3:
        rounds_arr = [d[0] for d in dose_rows]
        shift_arr = [d[1] for d in dose_rows]
        rho2, p2 = stats.spearmanr(rounds_arr, shift_arr)
        print(f"    Spearman rho(rounds, signed shift)={rho2:+.3f} (p={p2:.3g}, n={len(dose_rows)} -- "
              f"3 points, not enough to claim a dose-response either way)")

    print("\n=== output-length / stylistic axis (natural=short, elaborate=long) vs. epistemic/conversation cues ===")
    stylistic = [r["mean_abs_level_shift"] for r in cue_summary
                 if r["output_length"] != "none" or r["politeness"] == 1 or r["structured_format"] == 1 or r["answer_commit"] == 1]
    epistemic_or_conv = [r["mean_abs_level_shift"] for r in cue_summary
                          if r["epistemic_strength"] > 0 or r["conversation_state"] != "none"]
    print(f"  purely stylistic (natural/elaborate/polite/direct/query): n={len(stylistic)}, mean|shift|={np.mean(stylistic):.3f}")
    print(f"  epistemic or conversation-state cues: n={len(epistemic_or_conv)}, mean|shift|={np.mean(epistemic_or_conv):.3f}")


if __name__ == "__main__":
    main()
