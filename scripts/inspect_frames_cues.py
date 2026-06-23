"""Streamlit inspector — compare FRAMES phrasings side by side.

For each non-stale FRAMES question it shows, stacked top to bottom:
  1. ORIGINAL FRAMES question (+ gold answer)
  2. NEUTRAL anchor (the formal benchmark rewrite — strict if available, else the original
     frames_benchmark.jsonl) with its audit badges (equivalence / leak / neutral-register),
     cross-referenced against results/frames_benchmark_audit.csv (the standalone auditor verdict)
  3. CUE VARIATIONS grouped by dimension (epistemic, framing, mitigation, structural, output).
     Conditions that haven't been generated yet show a placeholder, so the panel fills in as
     scripts/paraphrase_frames_cues.py produces them.

This is the manual QA surface for the prompt-cue experiment: confirm the neutral anchor preserves
the information need with no leak, and eyeball each cue variant for drift / leak / cue isolation.

Run:
    uv run streamlit run scripts/inspect_frames_cues.py

Optional flags (after `--`):
    uv run streamlit run scripts/inspect_frames_cues.py -- \
        --cues-dir data/frames_cues --anchor data/frames_benchmark_strict.jsonl \
        --audit-csv results/frames_benchmark_audit.csv
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import streamlit as st

DEFAULT_CUES_DIR = "data/frames_cues"
DEFAULT_STALENESS = "results/frames_staleness_flash.csv"
DEFAULT_AUDIT = "results/frames_benchmark_audit.csv"
# Prefer the strict anchor if it exists; fall back to the original benchmark rewrite.
ANCHOR_CANDIDATES = ["data/frames_benchmark_strict.jsonl", "data/frames_benchmark.jsonl"]

# Cue grouping mirrors CUE_SPECS in scripts/paraphrase_frames_cues.py. (label, [slugs]).
CUE_GROUPS = [
    ("Epistemic stance",
     ["epi_strong_boost", "epi_mild_boost", "epi_mild_hedge", "epi_strong_hedge"]),
    ("Framing / personalization", ["framing_first_person"]),
    ("Pragmatic mitigation", ["mitigation_polite"]),
    ("Structural explicitness", ["structural_implicit"]),
    ("Output constraints", ["output_final_only", "output_detailed"]),
]
KNOWN_CUE_SLUGS = [s for _, slugs in CUE_GROUPS for s in slugs]


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cues-dir", default=DEFAULT_CUES_DIR)
    p.add_argument("--anchor", default=None, help="Neutral anchor JSONL (default: strict if present)")
    p.add_argument("--staleness-csv", default=DEFAULT_STALENESS)
    p.add_argument("--audit-csv", default=DEFAULT_AUDIT,
                   help="Standalone auditor verdicts on the formal rewrite (keyed by example_id)")
    # streamlit passes its own args; ignore unknowns
    args, _ = p.parse_known_args()
    return args


@st.cache_data(show_spinner="Loading FRAMES from HuggingFace…")
def load_originals() -> dict[str, dict]:
    from datasets import load_dataset
    df = load_dataset("google/frames-benchmark", split="test").to_pandas()
    return {str(i): {"question": r["Prompt"], "answer": str(r["Answer"]),
                     "reasoning_types": r.get("reasoning_types")}
            for i, r in df.iterrows()}


@st.cache_data
def load_nonstale_questions(staleness_csv: str) -> set[str]:
    df = pd.read_csv(staleness_csv)
    return set(df.loc[df["is_stale"] == False, "aggregate_question"].str.strip())  # noqa: E712


@st.cache_data
def load_jsonl_by_id(path: str) -> dict[str, dict]:
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                out[str(r["example_id"])] = r
            except (json.JSONDecodeError, KeyError):
                pass
    return out


@st.cache_data
def load_audit(path: str) -> dict[str, dict]:
    """example_id -> auditor verdict row from results/frames_benchmark_audit.csv."""
    if not path or not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {str(r["example_id"]): r.to_dict() for _, r in df.iterrows()}


def pick_anchor_path(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    for c in ANCHOR_CANDIDATES:
        if os.path.exists(c):
            return c
    return None


def discover_cue_files(cues_dir: str) -> dict[str, str]:
    """slug -> path for every <slug>.jsonl in the cues dir (excluding neutral.jsonl)."""
    found = {}
    for path in sorted(glob.glob(os.path.join(cues_dir, "*.jsonl"))):
        slug = os.path.splitext(os.path.basename(path))[0]
        if slug == "neutral":
            continue
        found[slug] = path
    return found


# ── badge rendering ─────────────────────────────────────────────────────────────
def _flag(ok: bool | None, label_ok: str, label_bad: str) -> str:
    if ok is None:
        return f"⚪ {label_ok}?"
    return (f"🟢 {label_ok}" if ok else f"🔴 {label_bad}")


def anchor_badges(rec: dict) -> str:
    bits = []
    status = rec.get("validation_status")
    if status:
        bits.append(f"**status:** `{status}`")
    if "equivalent" in rec:
        bits.append(_flag(rec.get("equivalent"), "equivalent", "DRIFT"))
    if "leaks_intermediate" in rec:
        leaks = rec.get("leaks_intermediate")
        bits.append(_flag(None if leaks is None else (not leaks), "no-leak", "LEAK"))
    if "is_neutral_register" in rec:
        bits.append(_flag(rec.get("is_neutral_register"), "neutral-reg", "NON-NEUTRAL"))
    elif "is_benchmark_style" in rec:  # old benchmark file
        bits.append(_flag(rec.get("is_benchmark_style"), "benchmark-style", "conversational"))
    drift = rec.get("drift")
    if drift and drift != "none":
        bits.append(f"⚠️ drift=`{drift}`")
    return "  ·  ".join(bits)


def cue_badges(rec: dict) -> str:
    bits = []
    status = rec.get("validation_status")
    if status:
        bits.append(f"**status:** `{status}`")
    if "equivalent" in rec:
        bits.append(_flag(rec.get("equivalent"), "equivalent", "DRIFT"))
    if "leaks_intermediate" in rec:
        leaks = rec.get("leaks_intermediate")
        bits.append(_flag(None if leaks is None else (not leaks), "no-leak", "LEAK"))
    if "cue_applied" in rec:
        bits.append(_flag(rec.get("cue_applied"), "cue-applied", "cue-absent"))
    if "only_target_cue" in rec:
        bits.append(_flag(rec.get("only_target_cue"), "isolated", "OTHER-CUE"))
    return "  ·  ".join(bits)


def _drift_set(rec: dict) -> bool:
    return str(rec.get("drift")) not in ("none", "nan", "None", "")


def audit_csv_badges(rec: dict) -> str:
    """Badges for a row of results/frames_benchmark_audit.csv."""
    bits = []
    status = rec.get("validation_status")
    if status and str(status) != "nan":
        bits.append(f"**audit:** `{status}`")
    bits.append(_flag(bool(rec.get("equivalent")), "equivalent", "DRIFT"))
    bits.append(_flag(not bool(rec.get("leaks_intermediate")), "no-leak", "LEAK"))
    bits.append(_flag(bool(rec.get("is_neutral_register")), "neutral-reg", "NON-NEUTRAL"))
    if _drift_set(rec):
        bits.append(f"⚠️ drift=`{rec.get('drift')}`")
    return "  ·  ".join(bits)


def _audit_leaks(rec: dict | None) -> bool:
    return bool(rec) and bool(rec.get("leaks_intermediate"))


def _audit_flagged(rec: dict | None) -> bool:
    if not rec:
        return False
    return (bool(rec.get("leaks_intermediate")) or not bool(rec.get("equivalent"))
            or not bool(rec.get("is_neutral_register")) or _drift_set(rec))


def _has_failed_cue(ex_id: str, cues: dict[str, dict]) -> bool:
    for slug, by_id in cues.items():
        rec = by_id.get(ex_id)
        if rec and rec.get("validation_status") == "fail":
            return True
    return False


def main():
    args = _parse_args()
    st.set_page_config(page_title="FRAMES cue inspector", layout="wide")

    originals = load_originals()
    nonstale_q = load_nonstale_questions(args.staleness_csv)
    anchor_path = pick_anchor_path(args.anchor)
    anchor = load_jsonl_by_id(anchor_path)
    audit = load_audit(args.audit_csv)

    cue_files = discover_cue_files(args.cues_dir)
    cues = {slug: load_jsonl_by_id(path) for slug, path in cue_files.items()}

    # Non-stale example ids, ordered by HF index. Restrict to ids present in originals.
    nonstale_ids = [eid for eid, o in originals.items() if o["question"].strip() in nonstale_q]
    nonstale_ids.sort(key=lambda e: int(e) if e.isdigit() else e)

    # ── sidebar ──────────────────────────────────────────────────────────────
    st.sidebar.title("FRAMES cue inspector")
    st.sidebar.caption(f"Anchor: `{anchor_path or '— none found —'}`")
    st.sidebar.caption(f"Cue dir: `{args.cues_dir}` — {len(cue_files)} file(s)")
    n_leaks = sum(1 for a in audit.values() if _audit_leaks(a))
    st.sidebar.caption(f"Audit: `{args.audit_csv}` — {len(audit)} row(s), {n_leaks} leak(s)")
    generated = [s for s in KNOWN_CUE_SLUGS if s in cues] + [s for s in cues if s not in KNOWN_CUE_SLUGS]
    st.sidebar.caption("Generated conditions: " + (", ".join(generated) if generated else "none yet"))

    only_leaks = st.sidebar.checkbox("Only audit-flagged LEAKS", value=False)
    only_flagged = st.sidebar.checkbox(
        "Only audit-flagged (leak/drift/non-equiv/non-neutral)", value=False)
    only_failed_cue = st.sidebar.checkbox("Only examples with a failed cue", value=False)

    ids = nonstale_ids
    if only_leaks:
        ids = [e for e in ids if _audit_leaks(audit.get(e))]
    if only_flagged:
        ids = [e for e in ids if _audit_flagged(audit.get(e))]
    if only_failed_cue:
        ids = [e for e in ids if _has_failed_cue(e, cues)]

    st.sidebar.markdown(f"**{len(ids)}** / {len(nonstale_ids)} non-stale examples match filters")
    if not ids:
        st.warning("No examples match the current filters.")
        return

    # navigation
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    st.session_state.idx = max(0, min(st.session_state.idx, len(ids) - 1))

    c1, c2, c3 = st.sidebar.columns(3)
    if c1.button("◀ Prev"):
        st.session_state.idx = (st.session_state.idx - 1) % len(ids)
    if c3.button("Next ▶"):
        st.session_state.idx = (st.session_state.idx + 1) % len(ids)
    jump = st.sidebar.selectbox("Jump to example_id", ids,
                                index=st.session_state.idx,
                                format_func=lambda e: f"{e}")
    if jump != ids[st.session_state.idx]:
        st.session_state.idx = ids.index(jump)

    ex_id = ids[st.session_state.idx]
    st.sidebar.markdown(f"Position **{st.session_state.idx + 1} / {len(ids)}**")

    # ── main panel ───────────────────────────────────────────────────────────
    orig = originals[ex_id]
    st.subheader(f"Example `{ex_id}`")
    rt = orig.get("reasoning_types")
    st.caption(f"gold answer: **{orig['answer']}**" + (f"  ·  reasoning_types: {rt}" if rt is not None else ""))

    # 1. original
    st.markdown("### 1 · Original FRAMES question")
    st.info(orig["question"])

    # 2. neutral anchor
    st.markdown("### 2 · Neutral anchor")
    arec = anchor.get(ex_id)
    if arec:
        st.success(arec.get("text", "—"))
        badges = anchor_badges(arec)
        if badges:
            st.markdown(badges)
        reason = arec.get("audit_reasoning")
        ri = arec.get("register_issues")
        if reason:
            with st.expander("auditor reasoning"):
                st.write(reason)
                if ri:
                    st.write("register issues:", ri)
    else:
        st.warning("— neutral anchor not found for this example —")

    # 2b. standalone audit.csv verdict (cross-reference)
    arec_audit = audit.get(ex_id)
    if arec_audit:
        st.markdown("**audit.csv:** " + audit_csv_badges(arec_audit))
        ar = arec_audit.get("audit_reasoning")
        ft = arec_audit.get("formal_text")
        if (ar and str(ar) != "nan") or (ft and str(ft) != "nan"):
            with st.expander("audit.csv reasoning / formal_text"):
                if ft and str(ft) != "nan":
                    st.caption("formal_text:")
                    st.write(ft)
                if ar and str(ar) != "nan":
                    st.write(ar)

    # 3. cue variations
    st.markdown("### 3 · Cue variations")
    for group_label, slugs in CUE_GROUPS:
        st.markdown(f"#### {group_label}")
        for slug in slugs:
            rec = cues.get(slug, {}).get(ex_id)
            col_l, col_r = st.columns([1, 4])
            col_l.markdown(f"`{slug}`")
            if rec is None:
                if slug in cues:
                    col_r.caption("— not present for this example —")
                else:
                    col_r.caption("⏳ placeholder — condition not generated yet")
            else:
                status = rec.get("validation_status")
                text = rec.get("text", "—")
                if status == "fail":
                    col_r.error(text)
                else:
                    col_r.write(text)
                b = cue_badges(rec)
                if b:
                    col_r.markdown(b)
                issues = rec.get("issues")
                if issues:
                    col_r.caption("issues: " + "; ".join(issues))

    # any discovered slugs outside the known groups
    extra = [s for s in cues if s not in KNOWN_CUE_SLUGS]
    if extra:
        st.markdown("#### Other / unrecognized conditions")
        for slug in extra:
            rec = cues[slug].get(ex_id)
            col_l, col_r = st.columns([1, 4])
            col_l.markdown(f"`{slug}`")
            col_r.write(rec.get("text", "—") if rec else "— not present —")


if __name__ == "__main__":
    main()
