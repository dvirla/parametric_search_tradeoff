"""
Streamlit app for manually curating the ShareChat dataset.

Loads tucnguyen/ShareChat (claude, gemini, perplexity subsets), deduplicates
by text (first occurrence wins), and lets you include or skip each entry.
Selected entries are saved to data/sharechat_curated.jsonl.

Run:
    uv run streamlit run scripts/sharechat_curator.py
"""

import json
import os
import sys

import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

OUTPUT_FILE = "data/sharechat_curated.jsonl"
HF_DATASET = "tucnguyen/ShareChat"
SUBSETS = ("claude", "gemini", "perplexity")

st.set_page_config(
    page_title="ShareChat Curator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Light readable card styles ────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .question-card {
        background: #ffffff;
        border: 1px solid #d0d7de;
        border-left: 5px solid #7c6af7;
        border-radius: 8px;
        padding: 1.4em 1.8em;
        font-size: 1.08em;
        line-height: 1.8;
        color: #1a1a2e;
        white-space: pre-wrap;
        word-break: break-word;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading ShareChat from HuggingFace…")
def load_data() -> list[dict]:
    from datasets import load_dataset

    all_messages = []
    for subset in SUBSETS:
        ds = load_dataset(HF_DATASET, subset, split="train")
        for row in ds:
            if (
                row.get("detected_language_final") == "English"
                and row.get("role") == "user"
                and row.get("message_index") == 1
            ):
                all_messages.append({"text": row["plain_text"], "subset": subset})

    # Deduplicate — keep first occurrence across subsets
    seen: set[str] = set()
    deduped: list[dict] = []
    for msg in all_messages:
        if msg["text"] not in seen:
            seen.add(msg["text"])
            deduped.append(msg)

    # Remove messages with placeholder tags
    clean = [
        m
        for m in deduped
        if "<REDACTED>" not in m["text"]
        and "<URL>" not in m["text"]
        and "<DATE_TIME>" not in m["text"]
    ]
    return clean


def load_decisions(output_path: str) -> dict[str, dict]:
    """Return {text: {included, reasoning_hops, explanation}} for already-decided entries."""
    decisions: dict[str, dict] = {}
    if not os.path.exists(output_path):
        return decisions
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj["text"]
                # Support both old bool format and new dict format
                if isinstance(obj.get("included"), bool):
                    decisions[text] = {
                        "included": obj["included"],
                        "reasoning_hops": obj.get("reasoning_hops"),
                        "explanation": obj.get("explanation", ""),
                    }
            except (json.JSONDecodeError, KeyError):
                pass
    return decisions


def save_all(data: list[dict], decisions: dict[str, dict], output_path: str) -> None:
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            if item["text"] in decisions:
                record = {**item, **decisions[item["text"]]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Session state init ────────────────────────────────────────────────────────

if "data" not in st.session_state:
    st.session_state.data = load_data()

if "decisions" not in st.session_state:
    st.session_state.decisions = load_decisions(OUTPUT_FILE)

if "idx" not in st.session_state:
    data = st.session_state.data
    decisions = st.session_state.decisions
    first_undecided = next(
        (i for i, item in enumerate(data) if item["text"] not in decisions), 0
    )
    st.session_state.idx = first_undecided

data: list[dict] = st.session_state.data
decisions: dict[str, dict] = st.session_state.decisions
total = len(data)
idx: int = st.session_state.idx


# ── Helpers ───────────────────────────────────────────────────────────────────

def decide(include: bool, reasoning_hops: int | None, explanation: str) -> None:
    text = data[st.session_state.idx]["text"]
    st.session_state.decisions[text] = {
        "included": include,
        "reasoning_hops": reasoning_hops if reasoning_hops and reasoning_hops > 0 else None,
        "explanation": explanation.strip() if explanation.strip() else None,
    }
    save_all(data, st.session_state.decisions, OUTPUT_FILE)
    next_undecided = next(
        (
            i
            for i in range(st.session_state.idx + 1, total)
            if data[i]["text"] not in st.session_state.decisions
        ),
        None,
    )
    if next_undecided is not None:
        st.session_state.idx = next_undecided


def go_to(new_idx: int) -> None:
    st.session_state.idx = max(0, min(total - 1, new_idx))


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("ShareChat Curator")

    decided = len(decisions)
    included = sum(1 for v in decisions.values() if v.get("included"))
    skipped = decided - included
    remaining = total - decided

    st.metric("Total", total)
    col1, col2 = st.columns(2)
    col1.metric("Included", included)
    col2.metric("Skipped", skipped)
    st.metric("Remaining", remaining)
    st.progress(decided / total if total else 0, text=f"{decided}/{total} reviewed")

    st.divider()
    st.subheader("Jump to entry")
    jump = st.number_input(
        "Entry number (1-based)", min_value=1, max_value=total, value=idx + 1, step=1
    )
    if st.button("Go"):
        go_to(int(jump) - 1)
        st.rerun()

    if st.button("Jump to first undecided"):
        first = next(
            (i for i, item in enumerate(data) if item["text"] not in decisions), 0
        )
        go_to(first)
        st.rerun()

    st.divider()
    st.subheader("Output file")
    st.code(OUTPUT_FILE, language=None)

# ── Main area ─────────────────────────────────────────────────────────────────

if total == 0:
    st.error("No data loaded. Check HuggingFace connectivity.")
    st.stop()

item = data[idx]
text = item["text"]
subset = item["subset"]
prior = decisions.get(text, {})
current_decision = prior.get("included")

# Header
hcol1, hcol2 = st.columns([3, 1])
hcol1.subheader(f"Entry {idx + 1} of {total}")
decision_label = {True: "✅ Included", False: "⛔ Skipped", None: "⬜ Undecided"}[current_decision]
hcol2.markdown(f"**Status:** {decision_label}")
st.caption(f"Subset: `{subset}`")

st.divider()

# Question text
st.markdown(
    f'<div class="question-card">{text}</div>',
    unsafe_allow_html=True,
)

st.divider()

# ── Optional tags ─────────────────────────────────────────────────────────────
tag_col1, tag_col2 = st.columns([1, 3])

with tag_col1:
    hops_options = [0, 1, 2, 3, 4, 5]
    hops_default = prior.get("reasoning_hops") or 0
    hops_index = hops_options.index(hops_default) if hops_default in hops_options else 0
    reasoning_hops = st.selectbox(
        "Reasoning hops (optional)",
        options=hops_options,
        index=hops_index,
        format_func=lambda x: "— not set —" if x == 0 else str(x),
        key=f"hops_{idx}",
    )

with tag_col2:
    explanation = st.text_area(
        "Explanation / notes (optional)",
        value=prior.get("explanation") or "",
        height=90,
        placeholder="Why include or skip? Any notes about this question…",
        key=f"explanation_{idx}",
    )

st.divider()

# ── Action buttons ────────────────────────────────────────────────────────────
bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([2, 2, 1, 1, 2])

with bcol1:
    if st.button("✅  Include", use_container_width=True, type="primary"):
        decide(True, reasoning_hops, explanation)
        st.rerun()

with bcol2:
    if st.button("⛔  Skip", use_container_width=True):
        decide(False, reasoning_hops, explanation)
        st.rerun()

with bcol3:
    if st.button("◀  Prev", use_container_width=True):
        go_to(idx - 1)
        st.rerun()

with bcol4:
    if st.button("Next  ▶", use_container_width=True):
        go_to(idx + 1)
        st.rerun()

with bcol5:
    if st.button("⏩  Next undecided", use_container_width=True):
        nxt = next(
            (i for i in range(idx + 1, total) if data[i]["text"] not in decisions),
            None,
        )
        if nxt is not None:
            go_to(nxt)
            st.rerun()

st.caption("Include / Skip saves immediately and jumps to the next undecided entry.")
