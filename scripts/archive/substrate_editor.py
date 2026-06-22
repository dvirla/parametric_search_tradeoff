"""
Streamlit app for reviewing and editing ShareChat substrate questions.

Usage:
  uv run streamlit run scripts/substrate_editor.py
  uv run streamlit run scripts/substrate_editor.py -- --file data/sharechat_substrates.jsonl
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))

from scripts.archive.label_sharechat_substrates import (
    extract_placeholder_refs,
    validate_substrates,
)

st.set_page_config(
    page_title="Substrate Editor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_PATH = "data/sharechat_substrates_gemini-3.1-pro-preview_with_depends.jsonl"


# ── I/O ──────────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


# ── DataFrame helpers ────────────────────────────────────────────────────────

def _format_deps(deps: list[int]) -> str:
    return ", ".join(str(d) for d in sorted(set(deps)))


def _parse_deps(raw: str) -> list[int]:
    """Parse 'Depends on' cell — comma/space separated ints; empty → []."""
    if raw is None:
        return []
    tokens = [t.strip() for t in str(raw).replace(",", " ").split() if t.strip()]
    out: list[int] = []
    for t in tokens:
        try:
            out.append(int(t))
        except ValueError:
            # Skip non-int tokens; validation will flag the mismatch via placeholders.
            continue
    return sorted(set(out))


def substrates_to_df(substrates: list[dict]) -> pd.DataFrame:
    return pd.DataFrame({
        "Probe question": [sq["question"] for sq in substrates],
        "Depends on": [_format_deps(sq.get("depends_on", [])) for sq in substrates],
    })


def df_to_substrates(df: pd.DataFrame) -> list[dict]:
    """Convert data_editor DataFrame back to substrate list, dropping blank rows."""
    result = []
    for i, (_, row) in enumerate(df.iterrows()):
        text = str(row.get("Probe question", "")).strip()
        if text:
            result.append({
                "hop_index": len(result),
                "question": text,
                "depends_on": _parse_deps(row.get("Depends on", "")),
            })
    return result


# ── Session state init ───────────────────────────────────────────────────────

def _init_state(path: str) -> None:
    records = load_jsonl(path)
    st.session_state.records = records
    st.session_state.loaded_path = path
    st.session_state.dirty = set()   # set of indices with unsaved edits
    st.session_state.idx = 0


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🔬 Substrate Editor")

    # Resolve default path relative to repo root (script is in scripts/)
    repo_root = Path(__file__).parent.parent
    default_abs = str(repo_root / DEFAULT_PATH)

    file_path = st.text_input("JSONL file", value=default_abs)

    # Auto-load on first run or path change
    if (
        "records" not in st.session_state
        or st.session_state.get("loaded_path") != file_path
    ):
        if os.path.exists(file_path):
            _init_state(file_path)
        else:
            st.error(f"File not found:\n`{file_path}`")
            st.stop()

    if st.button("↺ Reload from disk", use_container_width=True):
        _init_state(file_path)
        st.rerun()

    records: list[dict] = st.session_state.records
    dirty: set = st.session_state.dirty

    # ── Stats ─────────────────────────────────────────────────────────────
    st.divider()
    n_subs = [len(r["substrate_questions"]) for r in records]
    total_subs = sum(n_subs)
    dependent_subs = sum(
        1 for r in records for sq in r["substrate_questions"]
        if sq.get("depends_on") or extract_placeholder_refs(sq.get("question", ""))
    )
    col_a, col_b = st.columns(2)
    col_a.metric("Questions", len(records))
    col_b.metric("Unsaved", len(dirty), delta_color="inverse")
    col_c, col_d = st.columns(2)
    col_c.metric("Avg probes", f"{sum(n_subs)/len(n_subs):.1f}")
    col_d.metric("Range", f"{min(n_subs)}–{max(n_subs)}")
    col_e, col_f = st.columns(2)
    col_e.metric("Probes total", total_subs)
    col_f.metric("Dependent probes", dependent_subs)

    # ── Navigation ────────────────────────────────────────────────────────
    st.divider()

    search = st.text_input("🔍 Filter questions", placeholder="keyword…")

    def _label(i: int) -> str:
        r = records[i]
        marker = "✏️ " if i in dirty else f"{i+1}. "
        q = r["question"]
        return f"{marker}{q[:58]}{'…' if len(q) > 58 else ''}"

    if search:
        indices = [
            i for i, r in enumerate(records)
            if search.lower() in r["question"].lower()
        ]
        if not indices:
            st.caption("No matches.")
            indices = list(range(len(records)))
    else:
        indices = list(range(len(records)))

    nav_idx = st.selectbox(
        "Select question",
        options=indices,
        index=0 if st.session_state.idx not in indices else indices.index(st.session_state.idx),
        format_func=_label,
    )
    st.session_state.idx = nav_idx

    # ── Save ──────────────────────────────────────────────────────────────
    st.divider()
    n_dirty = len(dirty)
    if st.button(
        f"💾  Save {n_dirty} change{'s' if n_dirty != 1 else ''} to disk",
        disabled=(n_dirty == 0),
        use_container_width=True,
        type="primary",
    ):
        save_jsonl(file_path, records)
        st.session_state.dirty = set()
        st.toast(f"Saved {len(records)} records to {Path(file_path).name}", icon="✅")
        st.rerun()


# ── Main ─────────────────────────────────────────────────────────────────────

idx: int = st.session_state.idx
record: dict = records[idx]

# ── Header with prev / next ──────────────────────────────────────────────────
col_prev, col_hdr, col_next = st.columns([1, 10, 1])
with col_prev:
    st.write("")  # vertical spacer
    if st.button("◀", disabled=(idx == 0), use_container_width=True, help="Previous question"):
        st.session_state.idx = idx - 1
        st.rerun()
with col_hdr:
    status = " ✏️" if idx in dirty else ""
    st.subheader(f"Question {idx + 1} / {len(records)}{status}")
    st.caption(f"ID: `{record['example_id']}`")
with col_next:
    st.write("")
    if st.button("▶", disabled=(idx == len(records) - 1), use_container_width=True, help="Next question"):
        st.session_state.idx = idx + 1
        st.rerun()

# ── Question text (read-only) ────────────────────────────────────────────────
st.markdown(
    f"""<div style="
        background:#f0f4ff;
        border-left:4px solid #4a6fa5;
        padding:10px 16px;
        border-radius:4px;
        font-size:1.05rem;
        margin-bottom:8px;
    ">{record['question']}</div>""",
    unsafe_allow_html=True,
)

# ── Substrate editor ─────────────────────────────────────────────────────────
st.markdown(
    f"**Substrate questions** &nbsp;—&nbsp; "
    f"<span style='color:grey'>{len(record['substrate_questions'])} probes</span>",
    unsafe_allow_html=True,
)
st.caption(
    "Edit cells inline · add rows with ＋ at the bottom · "
    "select a row and press **Delete** to remove it · changes auto-save when you click away · "
    "use `{hop_K}` (0-indexed) in a probe to reference a previous probe's answer and list `K` "
    "in **Depends on** (comma-separated). Adding/deleting rows renumbers hop indices but does "
    "**not** rewrite placeholders in other rows — fix them manually."
)

edited_df = st.data_editor(
    substrates_to_df(record["substrate_questions"]),
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "Probe question": st.column_config.TextColumn(
            "Probe question",
            width="large",
        ),
        "Depends on": st.column_config.TextColumn(
            "Depends on",
            help="Comma-separated hop indices this probe depends on (e.g. '0, 2'). Must match {hop_K} placeholders in the question.",
            width="small",
        ),
    },
    key=f"editor_{idx}",
    hide_index=True,
)

# Auto-commit: update session_state whenever the editor content differs from stored.
# Runs on every interaction; no explicit Apply button needed.
new_substrates = df_to_substrates(edited_df)
if new_substrates != record["substrate_questions"]:
    st.session_state.records[idx]["substrate_questions"] = new_substrates
    st.session_state.dirty.add(idx)
    # Force sidebar metric refresh
    st.rerun()

# ── Validation banner ────────────────────────────────────────────────────────
_validation_errors = validate_substrates(record["substrate_questions"])
if _validation_errors:
    st.error("⚠️ Validation issues — fix before relying on this record:\n\n"
             + "\n".join(f"- {e}" for e in _validation_errors))

# ── Footer: jump bar ─────────────────────────────────────────────────────────
st.divider()
jump_col, _, info_col = st.columns([2, 4, 4])
with jump_col:
    jump_to = st.number_input(
        "Jump to #", min_value=1, max_value=len(records), value=idx + 1, step=1
    )
    if jump_to - 1 != idx:
        st.session_state.idx = jump_to - 1
        st.rerun()

with info_col:
    if dirty:
        edited_qs = sorted(dirty)
        preview = ", ".join(str(i + 1) for i in edited_qs[:6])
        if len(edited_qs) > 6:
            preview += f" … (+{len(edited_qs)-6} more)"
        st.info(f"Unsaved edits on questions: {preview}")
