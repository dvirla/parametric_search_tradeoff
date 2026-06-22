"""Streamlit investigation tool — paraphrase leakage on "missed-but-correct" hops.

Surfaces the suspicious set behind Test 1: examples that are answered CORRECTLY
under natural2 phrasing yet have MORE truly-missed (uncertain & unsearched) hops
than under formal phrasing. If the natural2 paraphrase accidentally leaked an
intermediate hop's gold answer into the question text, the model can skip that
hop (a "miss") and still answer correctly — a confounder that would inflate the
Missed cell without hurting accuracy.

For each example it shows the formal vs natural2 question side by side, the
per-hop gold decomposition, which hops were missed under each phrasing, and an
automatic LEAKAGE flag = an intermediate hop's gold answer appears verbatim in
the natural2 question text.

Run:
    uv run streamlit run scripts/inspect_missed_leakage.py
    # headless CSV export (no UI):
    uv run python scripts/inspect_missed_leakage.py --export results/natural2_paper_figures

Final-answer correctness is sourced from the reevaluated baseline JSONs
(paired_eval_files), NOT the buggy interplay aggregate_correct column.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import src.viz as viz
import src.results_files as rf

VARIANT_TO_PHRASING = {"benchmark": "formal", "natural": "natural2"}
GRADING = "reevaluated"
CERTAIN_RULE = "entropy"
MUS_SUMMARY = "results/musique_parametric/interplay_analysis/interplay_summary.csv"
NAT2_SUMMARY = "results/musique-natural2/interplay_analysis/interplay_summary.csv"
NAT2_COMMIT_DIR = "results/musique-natural2/commitment_locus"


# ── text normalisation / leakage detection ─────────────────────────────────────
def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _leak_hit(question: str, answer: str) -> bool:
    """True if the (normalised) gold answer appears as a token-substring in the
    question. Skips very short / numeric-only answers that match by chance."""
    q, a = _norm(question), _norm(answer)
    if len(a) < 4 or a in {"yes", "no", "true", "false", "none"}:
        return False
    # whole-phrase match, padded so we match token boundaries
    return f" {a} " in f" {q} "


def _coerce_correct(v):
    if v is True or v == 1 or str(v) == "True":
        return 1
    if v is False or v == 0 or str(v) == "False":
        return 0
    return np.nan


# ── data loading ────────────────────────────────────────────────────────────--
def _interplay_frame() -> pd.DataFrame:
    mus = viz.load_interplay_summary(MUS_SUMMARY, "musique", certain_rule=CERTAIN_RULE)
    nat2 = viz.load_interplay_summary(NAT2_SUMMARY, "musique-natural2", certain_rule=CERTAIN_RULE)
    nat2["model"] = (nat2["model"].astype(str)
                     .str.replace(r"^musique-natural2_", "", regex=True)
                     .str.replace(r"_baseline_agent_run_\d+$", "", regex=True))
    frames = []
    for variant, d in [("benchmark", mus), ("natural", nat2)]:
        d = viz.add_cost_cells(d)
        d = d[d["model"].isin(viz.BASE_MODEL_SLUGS)].copy()
        d["variant"] = variant
        frames.append(d[["model", "variant", "example_id", "hop_index", "num_hops",
                         "uncertain", "truly_missed", "search_attributed", "entropy"]])
    return pd.concat(frames, ignore_index=True)


def _commitment_frame() -> pd.DataFrame:
    rows = []
    for slug in viz.BASE_MODEL_SLUGS:
        for f in glob.glob(os.path.join(NAT2_COMMIT_DIR, f"commitment_locus_{slug}_shard_*.csv")):
            c = pd.read_csv(f)
            c["model"] = slug
            rows.append(c[["model", "variant", "example_id", "hop_index", "hop_question",
                           "gold_answer", "committed_answer", "matches_gold"]])
    cl = pd.concat(rows, ignore_index=True)
    cl["intrace_correct"] = cl["matches_gold"].apply(_coerce_correct)
    return cl


def _question_text() -> pd.DataFrame:
    """example_id -> formal/natural2 full question text + final gold answer."""
    out = []
    for slug in viz.BASE_MODEL_SLUGS:
        canon = viz.canonical_model(slug)
        rec = {"model": slug}
        per_ph = {}
        for ph in ("formal", "natural2"):
            dirpath = rf.PHRASING_DIRS.get(ph)
            path = rf._find_baseline_file(dirpath, canon, GRADING) if dirpath else None
            if path is None:
                per_ph[ph] = None
                continue
            be = rf.load_baseline_eval(path).dropna(subset=["example_id"])
            per_ph[ph] = be[["example_id", "problem", "correct_answer", "sampler_response",
                             "sampler_correct"]].rename(columns={
                "problem": f"{ph}_q", "correct_answer": f"{ph}_gold",
                "sampler_response": f"{ph}_resp", "sampler_correct": f"{ph}_correct"})
        if per_ph["formal"] is None or per_ph["natural2"] is None:
            continue
        merged = per_ph["formal"].merge(per_ph["natural2"], on="example_id", how="inner")
        merged["model"] = slug
        out.append(merged)
    return pd.concat(out, ignore_index=True)


def build():
    ip = _interplay_frame()
    cl = _commitment_frame()
    qt = _question_text()

    # per-hop detail (one row per model/example/hop, both variants folded wide)
    keys = ["model", "variant", "example_id", "hop_index"]
    hops = ip.merge(cl[keys + ["hop_question", "gold_answer", "committed_answer",
                               "intrace_correct"]], on=keys, how="left")

    # example-level missed counts per variant
    summ = (hops.assign(tm=hops["truly_missed"].fillna(False))
                .groupby(["model", "variant", "example_id"])
                .agg(n_missed=("tm", "sum"), num_hops=("num_hops", "max"))
                .reset_index()
                .pivot_table(index=["model", "example_id"], columns="variant",
                             values=["n_missed", "num_hops"], aggfunc="first"))
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ = summ.reset_index()

    df = summ.merge(qt[["model", "example_id", "formal_q", "natural2_q", "natural2_gold",
                        "formal_correct", "natural2_correct", "formal_resp", "natural2_resp"]],
                    on=["model", "example_id"], how="left")
    df["natural2_correct"] = df["natural2_correct"].apply(_coerce_correct)
    df["formal_correct"] = df["formal_correct"].apply(_coerce_correct)
    df["delta_missed"] = df["n_missed_natural"].fillna(0) - df["n_missed_benchmark"].fillna(0)

    # suspicious = correct under natural2 AND more missed hops than under formal
    df["suspicious"] = (df["natural2_correct"] == 1) & (df["delta_missed"] > 0)

    # ── leakage flag: does the natural2 question contain an intermediate gold? ───
    nat2_hops = hops[hops.variant == "natural"]
    leak = []
    for (model, eid), grp in nat2_hops.groupby(["model", "example_id"]):
        nq = df[(df.model == model) & (df.example_id == eid)]["natural2_q"]
        if nq.empty or pd.isna(nq.iloc[0]):
            continue
        nq = nq.iloc[0]
        maxhop = grp["hop_index"].max()
        leaked = []
        for _, h in grp.iterrows():
            # only intermediate (non-final) hops count as leakage
            if h["hop_index"] == maxhop:
                continue
            if pd.notna(h["gold_answer"]) and _leak_hit(nq, h["gold_answer"]):
                leaked.append((int(h["hop_index"]), str(h["gold_answer"])))
        leak.append({"model": model, "example_id": eid,
                     "leak_any": bool(leaked),
                     "leaked_hops": "; ".join(f"hop{i}:{a}" for i, a in leaked)})
    leakdf = pd.DataFrame(leak)
    df = df.merge(leakdf, on=["model", "example_id"], how="left")
    df["leak_any"] = df["leak_any"].fillna(False)
    return df, hops


# ── headless export ─────────────────────────────────────────────────────────--
def export(outdir: str):
    df, _ = build()
    sus = df[df.suspicious].copy()
    Path(outdir).mkdir(parents=True, exist_ok=True)
    cols = ["model", "example_id", "delta_missed", "n_missed_benchmark", "n_missed_natural",
            "num_hops_natural", "natural2_correct", "leak_any", "leaked_hops",
            "formal_q", "natural2_q", "natural2_gold"]
    sus = sus.sort_values(["leak_any", "delta_missed"], ascending=[False, False])
    sus[cols].to_csv(Path(outdir) / "missed_leakage_suspects.csv", index=False)
    print(f"Suspicious examples (correct under natural2, more misses than formal): {len(sus)}")
    for m, g in sus.groupby("model"):
        print(f"  {viz.display_name(m):<22} suspects={len(g):<4} leakage-flagged={int(g.leak_any.sum())}")
    print(f"Wrote {Path(outdir) / 'missed_leakage_suspects.csv'}")


# ── streamlit UI ────────────────────────────────────────────────────────────--
def run_streamlit():
    import streamlit as st
    st.set_page_config(page_title="Missed-hop leakage inspector", layout="wide")
    st.title("🔍 Paraphrase-leakage inspector — missed-but-correct hops")

    @st.cache_data(show_spinner="Loading frames...")
    def _cached():
        return build()

    df, hops = _cached()

    with st.sidebar:
        model = st.selectbox("Model", viz.BASE_MODEL_SLUGS,
                             format_func=viz.display_name, index=0)
        only_sus = st.checkbox("Only suspicious (correct & extra misses)", value=True)
        only_leak = st.checkbox("Only leakage-flagged", value=False)
        st.caption("Suspicious = correct under natural2 AND more truly-missed hops "
                   "than under formal. Leakage = an intermediate hop's gold answer "
                   "appears verbatim in the natural2 question.")

    sub = df[df.model == model].copy()
    if only_sus:
        sub = sub[sub.suspicious]
    if only_leak:
        sub = sub[sub.leak_any]
    sub = sub.sort_values(["leak_any", "delta_missed"], ascending=[False, False])

    c1, c2, c3 = st.columns(3)
    c1.metric("Examples shown", len(sub))
    c2.metric("Suspicious (this model)", int(df[(df.model == model)].suspicious.sum()))
    c3.metric("Leakage-flagged among suspicious",
              int(df[(df.model == model) & df.suspicious].leak_any.sum()))

    if sub.empty:
        st.info("No examples match the current filters.")
        return

    labels = [f"{'🚩 ' if r.leak_any else ''}{r.example_id}  (Δmiss={int(r.delta_missed)})"
              for _, r in sub.iterrows()]
    pick = st.selectbox("Example", range(len(sub)), format_func=lambda i: labels[i])
    row = sub.iloc[pick]

    if row.leak_any:
        st.error(f"🚩 LEAKAGE: intermediate gold answer appears in the natural2 "
                 f"question → {row.leaked_hops}")
    else:
        st.success("No verbatim intermediate-gold leakage detected (inspect manually below).")

    qc1, qc2 = st.columns(2)
    with qc1:
        st.subheader("Formal question")
        st.write(row.formal_q)
        st.caption(f"correct={row.formal_correct} · missed hops={int(row.get('n_missed_benchmark') or 0)}")
        st.markdown("**Response (formal):**")
        st.write(row.formal_resp)
    with qc2:
        st.subheader("Natural2 question (paraphrase)")
        # highlight any leaked gold spans
        nq = row.natural2_q or ""
        if row.leak_any and isinstance(row.leaked_hops, str):
            for chunk in row.leaked_hops.split(";"):
                ans = chunk.split(":", 1)[-1].strip()
                if ans:
                    nq = re.sub(f"({re.escape(ans)})", r"**:red[\1]**", nq, flags=re.I)
        st.markdown(nq)
        st.caption(f"correct={row.natural2_correct} · missed hops={int(row.get('n_missed_natural') or 0)}")
        st.markdown("**Response (natural2):**")
        st.write(row.natural2_resp)

    st.subheader("Per-hop decomposition")
    hd = hops[(hops.model == model) & (hops.example_id == row.example_id)].copy()
    # fold benchmark/natural side by side per hop
    piv = hd.pivot_table(index=["hop_index", "hop_question", "gold_answer"],
                         columns="variant",
                         values=["truly_missed", "search_attributed", "intrace_correct",
                                 "committed_answer", "entropy"],
                         aggfunc="first").reset_index()
    piv.columns = [f"{a}_{b}" if b else a for a, b in piv.columns]
    st.dataframe(piv, use_container_width=True)
    st.caption("Final gold answer: " + str(row.natural2_gold))


if __name__ == "__main__":
    # streamlit runs the module top-to-bottom; detect it via the runtime.
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
                        help="Write suspicious-set CSV and print a summary (no UI).")
        args = ap.parse_args()
        if args.export:
            export(args.export)
        else:
            print("Run the UI with:  uv run streamlit run scripts/inspect_missed_leakage.py")
            print("Or export a CSV:  uv run python scripts/inspect_missed_leakage.py "
                  "--export results/natural2_paper_figures")
