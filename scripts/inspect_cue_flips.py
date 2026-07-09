"""Phase 2 of docs/PLAN_cue_mechanism_analysis.md: Streamlit flip-forensics inspector for the
PLAIN vs ELABORATE cue experiment (facts_open + MedQA x {gemini-3.1-pro-preview, qwen3.5:122b,
gemma4:31b} — 6 cells).

Manual forensics for:
  (a) MedQA W->R flips — was PLAIN's wrong answer *induced by retrieved content*?
  (b) facts_open R->W flips (Gemma/Qwen) — was the fact ELABORATE got wrong present in PLAIN's
      pruned-query results?

Follows the pattern of scripts/inspect_frames_cues.py (argparse after `--`) and
scripts/archive/inspect_missed_leakage.py (headless --export mode, annotation upsert).

Run:
    uv run streamlit run scripts/inspect_cue_flips.py -- \\
        --traces-dir results/cue_traces --commitment-dir results/cue_commitment_locus \\
        --annotations-csv results/cue_flip_annotations.csv

    # headless export of a flipped-question digest per cell (no UI):
    uv run python scripts/inspect_cue_flips.py --export results/cue_flip_digest
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd

# ── cell config (6 cells: facts_open + medqa x {gemini, qwen, gemma}) ──────────
# cell -> (dataset, model_name, model_slug, plain_eval_json, elaborate_eval_json)
CELLS: dict[str, tuple[str, str, str, str, str]] = {
    "facts_open_gemini": (
        "facts-open", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview",
        "results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_plain.json",
        "results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_elaborate.json"),
    "facts_open_qwen": (
        "facts-open", "qwen3.5:122b", "qwen3.5_122b",
        "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_plain.json",
        "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_elaborate.json"),
    "facts_open_gemma": (
        "facts-open", "gemma4:31b", "gemma4_31b",
        "results/facts_gemma100/facts-open_baseline_gemma4:31b_plain.json",
        "results/facts_gemma100/facts-open_baseline_gemma4:31b_elaborate.json"),
    "medqa_gemini": (
        "medqa", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview",
        "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_plain.json",
        "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_elaborate.json"),
    "medqa_qwen": (
        "medqa", "qwen3.5:122b", "qwen3.5_122b",
        "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_plain.json",
        "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_elaborate.json"),
    "medqa_gemma": (
        "medqa", "gemma4:31b", "gemma4_31b",
        "results/cue_smoke_medqa/medqa_baseline_gemma4:31b_plain.json",
        "results/cue_smoke_medqa/medqa_baseline_gemma4:31b_elaborate.json"),
}

DEFAULT_TRACES_DIR = "results/cue_traces"
DEFAULT_COMMITMENT_DIR = "results/cue_commitment_locus"
DEFAULT_ANNOTATIONS_CSV = "results/cue_flip_annotations.csv"

ELABORATE_SUFFIX = "\n\nPlease answer with a detailed explanation - at least 8-10 sentences"

ANNOTATION_CATEGORIES = [
    "pruned evidence was load-bearing",
    "retrieval-induced error",
    "parametric recall compensated",
    "grader issue",
    "unclear",
]


# ── data loading ─────────────────────────────────────────────────────────────
def load_eval_rows(path: str) -> dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {r["example_id"]: r for r in rows if "example_id" in r}


def load_traces(path: str) -> dict[str, dict]:
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        traces = json.load(f)
    return {t["problem"].strip(): t for t in traces if "problem" in t}


def load_commitment_csv(cell: str, cue: str, commitment_dir: str) -> dict[str, dict]:
    path = os.path.join(commitment_dir, f"commitment_{cell}_{cue}.csv")
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {str(r["example_id"]): r.to_dict() for _, r in df.iterrows()}


def usable(row: dict) -> bool:
    sr = row.get("stop_reason")
    return sr is None or (isinstance(sr, float) and pd.isna(sr)) or str(sr) in ("None", "nan", "")


def flip_type(plain_correct, elab_correct) -> str:
    p, e = bool(plain_correct), bool(elab_correct)
    if p and not e:
        return "R->W"
    if not p and e:
        return "W->R"
    return "stable_correct" if p else "stable_wrong"


def build_cell_data(cell: str, traces_dir: str, commitment_dir: str) -> list[dict]:
    """Return one dict per paired (usable both sides) example_id for this cell."""
    dataset, model_name, model_slug, plain_json, elab_json = CELLS[cell]
    plain_eval = load_eval_rows(plain_json)
    elab_eval = load_eval_rows(elab_json)
    plain_traces = load_traces(os.path.join(traces_dir, f"{dataset}_{model_slug}_plain_traces.json"))
    elab_traces = load_traces(os.path.join(traces_dir, f"{dataset}_{model_slug}_elaborate_traces.json"))
    plain_commit = load_commitment_csv(cell, "plain", commitment_dir)
    elab_commit = load_commitment_csv(cell, "elaborate", commitment_dir)

    rows = []
    for eid, prow in plain_eval.items():
        erow = elab_eval.get(eid)
        if erow is None or not usable(prow) or not usable(erow):
            continue
        problem = prow["problem"].strip()
        rows.append({
            "example_id": eid,
            "cell": cell,
            "dataset": dataset,
            "model_name": model_name,
            "problem": problem,
            "gold_answer": prow.get("correct_answer"),
            "plain": {
                "response": prow.get("sampler_response", ""),
                "correct": prow.get("sampler_correct"),
                "search_calls": prow.get("sampler_search_calls"),
                "trace": plain_traces.get(problem),
                "commit": plain_commit.get(str(eid)),
            },
            "elaborate": {
                "response": erow.get("sampler_response", ""),
                "correct": erow.get("sampler_correct"),
                "search_calls": erow.get("sampler_search_calls"),
                "trace": elab_traces.get(problem),
                "commit": elab_commit.get(str(eid)),
            },
            "flip": flip_type(prow.get("sampler_correct"), erow.get("sampler_correct")),
        })
    return rows


def cells_with_traces(traces_dir: str) -> list[str]:
    """Cells for which at least the plain traces file exists on disk."""
    out = []
    for cell, (dataset, _, model_slug, _, _) in CELLS.items():
        p = os.path.join(traces_dir, f"{dataset}_{model_slug}_plain_traces.json")
        if os.path.exists(p):
            out.append(cell)
    return out


# ── short-answer extraction ─────────────────────────────────────────────────
_ANSWER_AFTER_RE = re.compile(
    r"(?:answer is|answer:|exact answer:)\s*\**\s*(.+?)(?:\n|\.|$)", re.I)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def extract_short_answer(text: str) -> str:
    if not text:
        return ""
    m = _ANSWER_AFTER_RE.search(text)
    if m:
        return m.group(1).strip(" *").strip()
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text.strip()) if s.strip()]
    if sents:
        return sents[-1][:200]
    return text[:200]


# ── query normalization / fuzzy alignment ───────────────────────────────────
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(q: str) -> set[str]:
    return set(_TOKEN_RE.findall((q or "").lower()))


def token_overlap(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def align_queries(plain_qs: list[str], elab_qs: list[str], threshold: float = 0.6
                   ) -> tuple[list[dict], list[str]]:
    """Greedy best-match alignment. Returns (plain_alignment, new_elab_queries) where
    plain_alignment is a list of {"query": str, "status": "kept"|"pruned", "match": str|None}."""
    remaining_elab = list(elab_qs)
    plain_alignment = []
    for pq in plain_qs:
        best_score, best_q = 0.0, None
        for eq in remaining_elab:
            score = token_overlap(pq, eq)
            if score > best_score:
                best_score, best_q = score, eq
        if best_score >= threshold:
            plain_alignment.append({"query": pq, "status": "kept", "match": best_q})
            remaining_elab.remove(best_q)
        else:
            plain_alignment.append({"query": pq, "status": "pruned", "match": None})
    return plain_alignment, remaining_elab  # remaining_elab = "new" queries


# ── trajectory step extraction (for rendering) ──────────────────────────────
def safe_parse(val):
    if not isinstance(val, str):
        return val
    try:
        return json.loads(val)
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        import ast
        return ast.literal_eval(val)
    except Exception:
        return val


def extract_steps(message_trace: list[dict] | None) -> list[dict]:
    """Ordered list of {type: 'search'|'final', thinking, query, results, text}."""
    if not message_trace:
        return []
    steps = []
    n = len(message_trace)
    for i, msg in enumerate(message_trace):
        if msg.get("role") != "assistant":
            continue
        parts = msg.get("parts", []) or []
        thinking = " ".join(p.get("content", "") for p in parts if p.get("type") == "thinking")
        texts = [p.get("content", "") for p in parts if p.get("type") == "text" and p.get("content")]
        tool_calls = [p for p in parts if p.get("type") == "tool_call"]
        if tool_calls:
            for tc in tool_calls:
                args = safe_parse(tc.get("arguments", {}))
                query = args.get("query", "") if isinstance(args, dict) else str(args)
                results = _find_tool_response(message_trace, i + 1, tc.get("tool_call_id"))
                steps.append({"type": "search", "thinking": thinking, "query": query,
                             "tool_name": tc.get("tool_name", ""), "results": results,
                             "turn_index": i})
        if texts:
            steps.append({"type": "final", "thinking": thinking, "text": " ".join(texts),
                         "turn_index": i})
    return steps


def _find_tool_response(message_trace: list[dict], start: int, tool_call_id) -> list:
    for j in range(start, min(start + 3, len(message_trace))):
        for p in message_trace[j].get("parts", []) or []:
            if p.get("type") == "tool_call_response" and p.get("tool_call_id") == tool_call_id:
                result = safe_parse(p.get("result", []))
                return result if isinstance(result, list) else ([result] if result else [])
    return []


def search_queries_of(trace: dict | None) -> list[str]:
    return [s["query"] for s in extract_steps((trace or {}).get("message_trace")) if s["type"] == "search" and s.get("query")]


# ── annotations: load / upsert (kept UI-free so it's directly testable) ────
ANNOTATION_COLUMNS = ["cell", "example_id", "flip_type", "category", "note", "updated_at"]


def load_annotations(path: str) -> pd.DataFrame:
    if os.path.exists(path):
        df = pd.read_csv(path, dtype={"example_id": str})
        for c in ANNOTATION_COLUMNS:
            if c not in df.columns:
                df[c] = None
        return df[ANNOTATION_COLUMNS]
    return pd.DataFrame(columns=ANNOTATION_COLUMNS)


def upsert_annotation(path: str, cell: str, example_id: str, flip_type_: str,
                       category: str, note: str) -> pd.DataFrame:
    """Upsert one (cell, example_id) annotation row and write the CSV immediately. Returns the
    resulting full DataFrame. UI-free so it can be exercised directly in tests."""
    df = load_annotations(path)
    example_id = str(example_id)
    mask = (df["cell"] == cell) & (df["example_id"] == example_id)
    new_row = {
        "cell": cell, "example_id": example_id, "flip_type": flip_type_,
        "category": category, "note": note,
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if mask.any():
        df.loc[mask, list(new_row.keys())] = list(new_row.values())
    else:
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# ── highlighting ─────────────────────────────────────────────────────────────
_HIGHLIGHT_STYLES = {
    "gold": "background-color:#1e5c1e;color:#eaffea;",
    "plain": "background-color:#1a3d6b;color:#e8f0ff;",
    "elaborate": "background-color:#7a4a10;color:#fff4e5;",
}


def highlight_html(text: str, terms: dict[str, str]) -> str:
    """terms: label -> term string (label in {'gold','plain','elaborate'}). Case-insensitive
    substring highlight; escapes HTML first. Longer terms are highlighted first to avoid partial
    overlaps clobbering a longer match."""
    import html as _html
    if not text:
        return ""
    escaped = _html.escape(text)
    ordered = sorted(((k, v) for k, v in terms.items() if v and len(v.strip()) >= 2),
                     key=lambda kv: -len(kv[1]))
    for label, term in ordered:
        term_esc = re.escape(_html.escape(term.strip()))
        style = _HIGHLIGHT_STYLES.get(label, "background-color:#555;")
        escaped = re.sub(
            f"({term_esc})",
            f'<mark style="{style}" title="{label}">\\1</mark>',
            escaped, flags=re.I,
        )
    return escaped


# ── headless export ─────────────────────────────────────────────────────────
def export(outdir: str, traces_dir: str, commitment_dir: str):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    cells = cells_with_traces(traces_dir)
    print(f"Cells with traces: {cells}")
    for cell in cells:
        rows = build_cell_data(cell, traces_dir, commitment_dir)
        df = pd.DataFrame([{
            "example_id": r["example_id"], "flip": r["flip"], "gold_answer": r["gold_answer"],
            "plain_correct": r["plain"]["correct"], "elaborate_correct": r["elaborate"]["correct"],
            "plain_search_calls": r["plain"]["search_calls"],
            "elaborate_search_calls": r["elaborate"]["search_calls"],
            "problem": r["problem"][:200],
        } for r in rows])
        flipped = df[df["flip"].isin(["R->W", "W->R"])].sort_values("flip")
        out_csv = os.path.join(outdir, f"{cell}_flips.csv")
        flipped.to_csv(out_csv, index=False)
        out_md = os.path.join(outdir, f"{cell}_flips.md")
        with open(out_md, "w") as f:
            f.write(f"# {cell} — flipped questions (PLAIN -> ELABORATE)\n\n")
            f.write(f"Total paired usable examples: {len(df)}. "
                   f"R->W: {(df.flip=='R->W').sum()}  W->R: {(df.flip=='W->R').sum()}\n\n")
            for _, r in flipped.iterrows():
                f.write(f"## `{r.example_id}` — {r.flip}\n\n")
                f.write(f"- gold: {r.gold_answer}\n")
                f.write(f"- search calls: PLAIN {r.plain_search_calls} -> "
                       f"ELABORATE {r.elaborate_search_calls}\n")
                f.write(f"- question: {r.problem}\n\n")
        print(f"  {cell}: {len(df)} paired, {len(flipped)} flipped -> {out_csv}, {out_md}")


# ── streamlit UI ─────────────────────────────────────────────────────────────
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--traces-dir", default=DEFAULT_TRACES_DIR)
    p.add_argument("--commitment-dir", default=DEFAULT_COMMITMENT_DIR)
    p.add_argument("--annotations-csv", default=DEFAULT_ANNOTATIONS_CSV)
    args, _ = p.parse_known_args()
    return args


def run_streamlit():
    import streamlit as st

    args = _parse_args()
    st.set_page_config(page_title="Cue flip-forensics inspector", layout="wide")
    st.title("Cue flip-forensics inspector — PLAIN vs ELABORATE")

    @st.cache_data(show_spinner="Loading cells...")
    def _cached_cells(traces_dir, commitment_dir):
        cells = cells_with_traces(traces_dir)
        return {c: build_cell_data(c, traces_dir, commitment_dir) for c in cells}

    data_by_cell = _cached_cells(args.traces_dir, args.commitment_dir)
    if not data_by_cell:
        st.error(f"No cells with traces found under `{args.traces_dir}`. "
                 f"Run scripts/download_cue_traces.py first.")
        return

    st.sidebar.header("Filters")
    cell = st.sidebar.selectbox("Cell (dataset x model)", list(data_by_cell.keys()))
    rows = data_by_cell[cell]
    st.sidebar.caption(f"{len(rows)} paired examples (usable both sides)")

    flip_filter = st.sidebar.selectbox(
        "Flip type", ["all", "R->W", "W->R", "stable_correct", "stable_wrong"])
    shown = rows if flip_filter == "all" else [r for r in rows if r["flip"] == flip_filter]
    st.sidebar.caption(f"{len(shown)} match filter")

    if not shown:
        st.warning("No examples match the current filters.")
        return

    badge = {"R->W": "🔴", "W->R": "🟢", "stable_correct": "⚪", "stable_wrong": "⚫"}
    labels = [f"{badge.get(r['flip'],'')} {r['example_id']} — {r['problem'][:80]}" for r in shown]
    idx = st.sidebar.selectbox("Question", range(len(shown)), format_func=lambda i: labels[i])
    row = shown[idx]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Highlight terms (editable)")
    default_gold = str(row["gold_answer"] or "")
    default_plain = extract_short_answer(row["plain"]["response"])
    default_elab = extract_short_answer(row["elaborate"]["response"])
    term_gold = st.sidebar.text_input("Gold answer (green)", value=default_gold, key=f"g_{cell}_{row['example_id']}")
    term_plain = st.sidebar.text_input("PLAIN short answer (blue)", value=default_plain, key=f"p_{cell}_{row['example_id']}")
    term_elab = st.sidebar.text_input("ELABORATE short answer (orange)", value=default_elab, key=f"e_{cell}_{row['example_id']}")
    terms = {"gold": term_gold, "plain": term_plain, "elaborate": term_elab}

    # ── header ───────────────────────────────────────────────────────────
    st.subheader(f"`{row['example_id']}` — {row['flip']}")
    st.markdown(f"**Question (base):** {row['problem']}")
    st.caption("ELABORATE sends this question PLUS the suffix: "
              f"*\"{ELABORATE_SUFFIX.strip()}\"*")
    st.markdown(f"**Gold answer:** {row['gold_answer']}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("PLAIN correct", str(row["plain"]["correct"]))
    c2.metric("ELABORATE correct", str(row["elaborate"]["correct"]))
    c3.metric("PLAIN searches", row["plain"]["search_calls"])
    c4.metric("ELABORATE searches", row["elaborate"]["search_calls"])

    pc, ec = row["plain"]["commit"], row["elaborate"]["commit"]
    if pc or ec:
        st.markdown("**Phase 1 commitment-locus:**")
        cc1, cc2 = st.columns(2)
        for col, rec, name in ((cc1, pc, "PLAIN"), (cc2, ec, "ELABORATE")):
            if rec:
                col.caption(
                    f"{name}: commit_step={rec.get('commitment_step')} "
                    f"post_commit={rec.get('post_commitment_searches')} "
                    f"matches_gold={rec.get('matches_gold')} "
                    f"committed_answer=\"{str(rec.get('committed_answer'))[:80]}\"")
            else:
                col.caption(f"{name}: — not probed (cell not in Phase 1 set) —")

    # ── query-alignment panel ───────────────────────────────────────────
    plain_qs = search_queries_of(row["plain"]["trace"])
    elab_qs = search_queries_of(row["elaborate"]["trace"])
    st.markdown("### Query alignment (PLAIN -> ELABORATE, token-overlap >= 0.6)")
    if not plain_qs and not elab_qs:
        st.caption("No searches in either condition.")
    else:
        alignment, new_qs = align_queries(plain_qs, elab_qs)
        qa1, qa2 = st.columns(2)
        with qa1:
            st.markdown("**PLAIN queries**")
            if not alignment:
                st.caption("— none —")
            for a in alignment:
                if a["status"] == "kept":
                    st.markdown(f"🟢 kept — `{a['query']}`")
                else:
                    st.markdown(f":red[🔴 pruned — `{a['query']}`]")
        with qa2:
            st.markdown("**ELABORATE queries**")
            if not elab_qs:
                st.caption("— none —")
            kept_matches = {a["match"] for a in alignment if a["status"] == "kept"}
            for q in elab_qs:
                tag = "🟢 kept" if q in kept_matches else "🆕 new"
                st.markdown(f"{tag} — `{q}`")

    # ── two-column trajectory view ──────────────────────────────────────
    st.markdown("### Trajectories")
    col_p, col_e = st.columns(2)
    for col, side, label in ((col_p, "plain", "PLAIN"), (col_e, "elaborate", "ELABORATE")):
        with col:
            st.markdown(f"#### {label}")
            resp = row[side]["response"] or ""
            with st.expander("Final answer", expanded=len(resp) < 600):
                st.markdown(highlight_html(resp, terms), unsafe_allow_html=True)

            trace = row[side]["trace"]
            if trace is None:
                st.info("No trace found for this condition (Logfire download miss).")
                continue
            steps = extract_steps(trace.get("message_trace"))
            search_steps = [s for s in steps if s["type"] == "search"]
            if not search_steps:
                st.caption("0 searches in this condition.")
            for k, s in enumerate(search_steps):
                with st.expander(f"Step {k+1}: search — `{s['query']}`"):
                    if s["thinking"]:
                        st.markdown("**Thinking (excerpt):**")
                        st.markdown(highlight_html(s["thinking"][:1500], terms),
                                   unsafe_allow_html=True)
                    st.markdown("**Query:**")
                    st.code(s["query"], language=None)
                    st.markdown("**Results:**")
                    if not s["results"]:
                        st.caption("— no results —")
                    for r in s["results"][:8]:
                        if isinstance(r, dict):
                            title = r.get("title", "")
                            content = r.get("snippet") or r.get("content") or r.get("url") or ""
                        else:
                            title, content = "", str(r)
                        content = content[:800]
                        st.markdown(f"**{highlight_html(title, terms)}**", unsafe_allow_html=True)
                        st.markdown(highlight_html(content, terms), unsafe_allow_html=True)
                        st.markdown("---")

    # ── annotation widget ────────────────────────────────────────────────
    st.markdown("### Annotation")
    existing = load_annotations(args.annotations_csv)
    prior = existing[(existing.cell == cell) & (existing.example_id == str(row["example_id"]))]
    prior_category = prior["category"].iloc[0] if len(prior) else ANNOTATION_CATEGORIES[-1]
    prior_note = prior["note"].iloc[0] if len(prior) else ""
    prior_note = "" if pd.isna(prior_note) else prior_note

    cat_idx = ANNOTATION_CATEGORIES.index(prior_category) if prior_category in ANNOTATION_CATEGORIES else len(ANNOTATION_CATEGORIES) - 1
    category = st.radio("Category", ANNOTATION_CATEGORIES, index=cat_idx,
                        key=f"cat_{cell}_{row['example_id']}")
    note = st.text_area("Note", value=prior_note, key=f"note_{cell}_{row['example_id']}")
    st.caption(f"Flip type (auto): {row['flip']}")
    if st.button("Save annotation", key=f"save_{cell}_{row['example_id']}"):
        upsert_annotation(args.annotations_csv, cell, row["example_id"], row["flip"], category, note)
        st.success(f"Saved to {args.annotations_csv}")


if __name__ == "__main__":
    in_streamlit = False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        in_streamlit = get_script_run_ctx() is not None
    except Exception:
        pass

    if in_streamlit:
        run_streamlit()
    else:
        ap = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
        ap.add_argument("--export", metavar="OUTDIR",
                        help="Write a static per-cell flip digest (CSV + Markdown), no UI.")
        ap.add_argument("--traces-dir", default=DEFAULT_TRACES_DIR)
        ap.add_argument("--commitment-dir", default=DEFAULT_COMMITMENT_DIR)
        args = ap.parse_args()
        if args.export:
            export(args.export, args.traces_dir, args.commitment_dir)
        else:
            print("Run the UI with:")
            print("  uv run streamlit run scripts/inspect_cue_flips.py -- "
                  "--traces-dir results/cue_traces --commitment-dir results/cue_commitment_locus")
            print("Or export a digest:")
            print("  uv run python scripts/inspect_cue_flips.py --export results/cue_flip_digest")
