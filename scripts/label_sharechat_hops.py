"""
Streamlit app for annotating reasoning hops in the curated ShareChat dataset.

Produces data/curated_sharechat_hops.json compatible with
run_sharechat_parametric_uncertainty.py and analyze_parametric_search_interplay.py.

Usage:
    uv run streamlit run scripts/label_sharechat_hops.py -- \
        --suggest-model gpt-oss:20b --suggest-provider ollama
"""

import argparse
import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel, Field

load_dotenv()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suggest-model", default="gpt-oss:20b")
    parser.add_argument("--suggest-provider", default="ollama")
    parser.add_argument("--ollama-url", default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--results-dir", default="results/curated_sharechat")
    parser.add_argument("--output-file", default="data/curated_sharechat_hops.json")
    args, _ = parser.parse_known_args()
    return args


ARGS = parse_args()


# ─── PYDANTIC MODELS FOR AUTO-SUGGEST ─────────────────────────────────────────

class HopSubQuestion(PydanticBaseModel):
    question: str = Field(
        description=(
            "A general, abstract sub-question describing the TYPE of fact this hop "
            "retrieves. Do not bake a specific candidate answer into the question."
        )
    )
    gold_answer: str = Field(
        description=(
            "Expected concise answer (entity, fact, date, or value) if it is known "
            "with confidence. Use an empty string when the question is open-ended, "
            "has multiple valid answers, or the answer is not known. Do not fabricate."
        )
    )


class HopDecomposition(PydanticBaseModel):
    sub_questions: list[HopSubQuestion] = Field(
        description="Sub-questions in order, one per reasoning hop"
    )
    aggregate_answer: str = Field(
        description=(
            "Expected answer to the benchmark (aggregate) question if a single "
            "canonical answer exists. Use an empty string for open-ended questions."
        )
    )


_SUGGEST_PROMPT = """\
Decompose the following question into atomic reasoning hops.

Natural question: {natural}
Benchmark version: {benchmark}
Reasoning explanation: {reasoning}
Expected number of hops: {n_hops}

GUIDELINES
- Sub-questions must be GENERAL and ABSTRACT. They should describe the *category*
  of fact each hop retrieves, NOT a specific historical instance, person, place,
  or event that you happen to think satisfies the question.
- For open-ended questions ("name any time when X", "give an example of Y",
  "have there been cases where Z"), the first hop should describe the abstract
  entity class. Do not anchor on one candidate.
- Never copy named entities from a specific candidate answer into the
  sub-question text. The sub-question must be answerable on its own without
  having already chosen the candidate.
- If you do not know the gold answer with confidence, leave gold_answer as "".
  Empty answers are fine and preferred over fabrication.
- Similarly leave aggregate_answer as "" for open-ended questions that have
  many valid answers.

EXAMPLES

Closed multi-hop (gold answers known):
  Natural: "What is the capital of the country that won the 2022 World Cup?"
  Hop 1: Q="Which country won the 2022 FIFA World Cup?"   A="Argentina"
  Hop 2: Q="What is the capital of Argentina?"            A="Buenos Aires"
  Aggregate: "Buenos Aires"

Open-ended (good — abstract hops, no fabricated answers):
  Natural: "Can you name any times in history where a military had the chance
            to concede peace, but chose war to reclaim lost territory and ended
            up losing the entire country?"
  Hop 1: Q="Which sovereign entity was offered peace but chose war in order to
            reclaim previously lost territory?"
        A=""
  Hop 2: Q="Did that entity ultimately lose the war and forfeit its entire
            country?"
        A=""
  Aggregate: ""

Open-ended (BAD — anchors on a specific candidate, do NOT do this):
  Hop 1: Q="Which historical military uprising in 1794 attempted to reclaim
            territories lost in the first two partitions of Poland?"
"""


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

@st.cache_data
def load_questions() -> list[dict]:
    benchmark_path = os.path.join(ARGS.data_dir, "curated_sharechat_wildchat_benchmark.csv")
    df = pd.read_csv(benchmark_path)

    reasoning_map: dict[str, dict] = {}
    jsonl_path = os.path.join(ARGS.data_dir, "sharechat_benchmark.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                d = json.loads(line)
                reasoning_map[d["text"]] = {
                    "reasoning": d.get("reasoning", ""),
                    "reasoning_hops": max(1, int(d.get("reasoning_hops", 2) or 2)),
                }

    questions = []
    for _, row in df.iterrows():
        text = str(row["text"])
        meta = reasoning_map.get(text, {"reasoning": "", "reasoning_hops": 2})
        questions.append({
            "text": text,
            "benchmark_question": str(row.get("benchmark_question", "")),
            "reasoning": meta["reasoning"],
            "reasoning_hops": meta["reasoning_hops"],
        })
    return questions


@st.cache_data
def load_model_responses() -> dict:
    """Returns dict[problem_text -> dict[model_label -> {response, search_calls}]]."""
    responses: dict[str, dict] = {}
    pattern = os.path.join(ARGS.results_dir, "curated-sharechat_baseline_*.json")
    for path in sorted(glob.glob(pattern)):
        basename = os.path.basename(path)
        model_label = (
            basename.replace("curated-sharechat_baseline_", "").replace("_run_1.json", "")
        )
        with open(path) as f:
            data = json.load(f)
        for entry in data:
            problem = entry.get("problem", "")
            if problem not in responses:
                responses[problem] = {}
            responses[problem][model_label] = {
                "response": entry.get("sampler_response", ""),
                "search_calls": entry.get("sampler_search_calls", 0),
            }
    return responses


def load_annotations() -> dict:
    path = ARGS.output_file
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return data.get("annotations", {})
    return {}


def save_annotations(annotations: dict) -> None:
    data = {"version": "1.0", "annotations": annotations}
    os.makedirs(os.path.dirname(os.path.abspath(ARGS.output_file)), exist_ok=True)
    with open(ARGS.output_file, "w") as f:
        json.dump(data, f, indent=2)


# ─── SESSION STATE HELPERS ────────────────────────────────────────────────────

def _init_session_state() -> None:
    if "current_idx" not in st.session_state:
        st.session_state.current_idx = 0
    if "annotations" not in st.session_state:
        st.session_state.annotations = load_annotations()
    if "last_loaded_idx" not in st.session_state:
        st.session_state.last_loaded_idx = -1
    if "suggest_status" not in st.session_state:
        st.session_state.suggest_status = ""


def _load_question_into_form(idx: int, questions: list[dict]) -> None:
    """Pre-populate form session state from saved annotation for question idx."""
    ann = st.session_state.annotations.get(str(idx), {})
    default_hops = questions[idx]["reasoning_hops"]
    num_hops = ann.get("num_hops", default_hops)
    st.session_state["form_num_hops"] = num_hops

    sub_qs = ann.get("sub_questions", [])
    for i in range(num_hops):
        sq = sub_qs[i] if i < len(sub_qs) else {}
        st.session_state[f"form_hop_{i}_q"] = sq.get("question", "")
        st.session_state[f"form_hop_{i}_a"] = sq.get("gold_answer", "")
    # Clear hop fields beyond current num_hops
    for i in range(num_hops, 6):
        st.session_state[f"form_hop_{i}_q"] = ""
        st.session_state[f"form_hop_{i}_a"] = ""

    st.session_state["form_aggregate_answer"] = ann.get("aggregate_answer", "")
    st.session_state["form_notes"] = ann.get("notes", "")
    st.session_state.last_loaded_idx = idx


def _collect_form_data(idx: int, questions: list[dict]) -> dict:
    """Read current form values into an annotation dict."""
    num_hops = st.session_state.get("form_num_hops", questions[idx]["reasoning_hops"])
    sub_questions = []
    for i in range(num_hops):
        sub_questions.append({
            "hop_index": i,
            "question": st.session_state.get(f"form_hop_{i}_q", "").strip(),
            "gold_answer": st.session_state.get(f"form_hop_{i}_a", "").strip(),
        })
    return {
        "example_id": f"curated_sharechat_{idx}",
        "problem": questions[idx]["text"],
        "benchmark_question": questions[idx]["benchmark_question"],
        "num_hops": num_hops,
        "sub_questions": sub_questions,
        "aggregate_answer": st.session_state.get("form_aggregate_answer", "").strip(),
        "notes": st.session_state.get("form_notes", "").strip(),
        "labeled": True,
        "timestamp": datetime.utcnow().isoformat(),
    }


def _is_labeled(idx: int) -> bool:
    return st.session_state.annotations.get(str(idx), {}).get("labeled", False)


def _labeled_count() -> int:
    return sum(1 for ann in st.session_state.annotations.values() if ann.get("labeled", False))


# ─── AUTO-SUGGEST ─────────────────────────────────────────────────────────────

def run_auto_suggest(idx: int, questions: list[dict]) -> None:
    from src.services.base_agent import BaseAgent

    q = questions[idx]
    num_hops = st.session_state.get("form_num_hops", q["reasoning_hops"])
    prompt = _SUGGEST_PROMPT.format(
        natural=q["text"],
        benchmark=q["benchmark_question"],
        reasoning=q["reasoning"],
        n_hops=num_hops,
    )

    try:
        agent = BaseAgent(
            provider_name=ARGS.suggest_provider,
            model_name=ARGS.suggest_model,
            output_type=HopDecomposition,
            agent_name="sharechat_hop_suggester",
            use_thinking=False,
            ollama_base_url=ARGS.ollama_url,
        )
        response = agent.run(prompt)
        decomp: HopDecomposition = response.output

        suggested_hops = min(len(decomp.sub_questions), 6)
        st.session_state["_pending_suggestion"] = {
            "num_hops": suggested_hops,
            "sub_questions": [
                {"q": sq.question, "a": sq.gold_answer}
                for sq in decomp.sub_questions[:6]
            ],
            "aggregate_answer": decomp.aggregate_answer,
        }
        st.session_state.suggest_status = "ok"
    except Exception as e:
        st.session_state.suggest_status = f"error: {e}"


# ─── MAIN APP ─────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="ShareChat Hop Labeler", layout="wide")

    _init_session_state()
    questions = load_questions()
    model_responses = load_model_responses()
    total = len(questions)

    # ── Sidebar ───────────────────────────────────────────────────────────────

    with st.sidebar:
        st.title("ShareChat Hop Labeler")
        labeled = _labeled_count()
        st.progress(labeled / total, text=f"{labeled} / {total} labeled")
        st.divider()

        filter_opt = st.radio("Show", ["All", "Unlabeled", "Labeled"], horizontal=True)
        if filter_opt == "Labeled":
            visible_indices = [i for i in range(total) if _is_labeled(i)]
        elif filter_opt == "Unlabeled":
            visible_indices = [i for i in range(total) if not _is_labeled(i)]
        else:
            visible_indices = list(range(total))

        st.caption(f"{len(visible_indices)} questions in view")

        for vis_i in visible_indices:
            q = questions[vis_i]
            label_icon = "✅" if _is_labeled(vis_i) else "○"
            short = q["benchmark_question"][:55] + "…" if len(q["benchmark_question"]) > 55 else q["benchmark_question"]
            if st.button(f"{label_icon} [{vis_i}] {short}", key=f"nav_{vis_i}"):
                st.session_state.current_idx = vis_i
                st.rerun()

    # ── Ensure current_idx is valid ───────────────────────────────────────────

    idx = st.session_state.current_idx
    if idx >= total:
        idx = 0
        st.session_state.current_idx = 0

    # Load form state when navigating to a new question
    if st.session_state.last_loaded_idx != idx:
        _load_question_into_form(idx, questions)

    # Apply a pending auto-suggest result before any widgets are instantiated,
    # so we can legally write to widget-backed keys like form_num_hops.
    if "_pending_suggestion" in st.session_state:
        pending = st.session_state.pop("_pending_suggestion")
        st.session_state["form_num_hops"] = pending["num_hops"]
        for i, sq in enumerate(pending["sub_questions"]):
            st.session_state[f"form_hop_{i}_q"] = sq["q"]
            st.session_state[f"form_hop_{i}_a"] = sq["a"]
        for i in range(len(pending["sub_questions"]), 6):
            st.session_state[f"form_hop_{i}_q"] = ""
            st.session_state[f"form_hop_{i}_a"] = ""
        st.session_state["form_aggregate_answer"] = pending["aggregate_answer"]

    q = questions[idx]

    # ── Header ────────────────────────────────────────────────────────────────

    col_title, col_nav = st.columns([4, 1])
    with col_title:
        status = "✅ Labeled" if _is_labeled(idx) else "○ Unlabeled"
        st.subheader(f"Question {idx + 1} / {total}  —  {status}")
    with col_nav:
        nav_c1, nav_c2 = st.columns(2)
        with nav_c1:
            if st.button("← Prev", use_container_width=True) and idx > 0:
                st.session_state.current_idx = idx - 1
                st.rerun()
        with nav_c2:
            if st.button("Next →", use_container_width=True) and idx < total - 1:
                st.session_state.current_idx = idx + 1
                st.rerun()

    # ── Question display ──────────────────────────────────────────────────────

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown("**Natural question**")
        st.info(q["text"])

        st.markdown("**Benchmark question**")
        st.success(q["benchmark_question"])

        with st.expander("Reasoning hint (click to expand)", expanded=False):
            st.markdown(q["reasoning"])
            st.caption(f"Suggested hops: {q['reasoning_hops']}")

        # Model responses
        st.markdown("**Model responses**")
        problem_key = q["text"]
        model_data = model_responses.get(problem_key, {})
        if model_data:
            model_tabs = st.tabs(list(model_data.keys()))
            for tab, (model_label, info) in zip(model_tabs, model_data.items()):
                with tab:
                    st.caption(f"Search calls: {info['search_calls']}")
                    st.markdown(info["response"][:3000])
        else:
            st.caption("No model responses found for this question.")

    # ── Hop annotation form ───────────────────────────────────────────────────

    with right_col:
        st.markdown("**Hop Annotation**")

        num_hops = st.selectbox(
            "Number of reasoning hops",
            options=list(range(1, 7)),
            index=min(st.session_state.get("form_num_hops", q["reasoning_hops"]), 6) - 1,
            key="form_num_hops",
        )

        # Auto-suggest button
        suggest_col, status_col = st.columns([1, 2])
        with suggest_col:
            if st.button("🤖 Auto-suggest", use_container_width=True):
                with st.spinner(f"Calling {ARGS.suggest_model}…"):
                    run_auto_suggest(idx, questions)
                st.rerun()
        with status_col:
            if st.session_state.suggest_status:
                if st.session_state.suggest_status == "ok":
                    st.success("Suggestions loaded — review and edit below")
                else:
                    st.error(st.session_state.suggest_status)
                st.session_state.suggest_status = ""

        st.divider()

        for i in range(num_hops):
            st.markdown(f"**Hop {i + 1}**")
            h_col1, h_col2 = st.columns([3, 2])
            with h_col1:
                st.text_area(
                    f"Sub-question",
                    key=f"form_hop_{i}_q",
                    height=70,
                    placeholder="What specific fact does this hop retrieve?",
                    label_visibility="collapsed",
                )
            with h_col2:
                st.text_input(
                    f"Gold answer",
                    key=f"form_hop_{i}_a",
                    placeholder="Expected answer",
                    label_visibility="collapsed",
                )

        st.divider()
        st.text_input(
            "Aggregate answer (final answer to benchmark question)",
            key="form_aggregate_answer",
            placeholder="Final answer",
        )
        st.text_area(
            "Notes (optional)",
            key="form_notes",
            height=60,
            placeholder="Any caveats or observations",
        )

        # Save buttons
        save_col, skip_col = st.columns([2, 1])
        with save_col:
            if st.button("💾 Save & Next", type="primary", use_container_width=True):
                ann = _collect_form_data(idx, questions)
                st.session_state.annotations[str(idx)] = ann
                save_annotations(st.session_state.annotations)
                if idx < total - 1:
                    st.session_state.current_idx = idx + 1
                st.rerun()
        with skip_col:
            if st.button("Skip →", use_container_width=True):
                if idx < total - 1:
                    st.session_state.current_idx = idx + 1
                    st.rerun()

        # Existing annotation preview
        if _is_labeled(idx):
            existing = st.session_state.annotations.get(str(idx), {})
            ts = existing.get("timestamp", "")[:19].replace("T", " ")
            st.caption(f"Last saved: {ts}")


if __name__ == "__main__":
    main()
