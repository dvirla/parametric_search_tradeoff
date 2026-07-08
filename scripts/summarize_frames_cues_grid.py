"""
Summarize one model's FRAMES prompt-cue GRID experiment into a Markdown report.

Point it at a results folder produced by scripts/run_frames_grid_experiment.sh
(results/frames_cues/<model_slug>/, holding one frames-cues_baseline_<model>_<condition>.json
per condition) and it regenerates the SUMMARY_*.md layout used for Gemini:

  Key insights — auto-generated TL;DR (significant search movers, accuracy costs
                 classified by search direction, cue text glosses)
  Table 1 — Cell descriptives (2x4 phrasing x output-template grid + epistemic cues)
  Table 2 — Paired contrasts (Wilcoxon signed-rank, paired on example_id)
  Table 3 — Interaction (difference-in-differences: phrasing x template-vs-PLAIN)
  Conclusions — auto-derived from the significance pattern.

Significant rows in Tables 2/3 are bolded and marked (🔻/🔺 search down/up, ⬇️/⬆️ accuracy
down/up) so they pop out; the Tradeoff column distinguishes an accuracy drop paid for a
search *reduction* (🟠) from an accuracy drop despite a search *increase* (🔴).

Each condition JSON now carries `example_id` per row (no cues-dir mapping needed); all
contrasts are paired on the example_ids common to the two conditions involved.

Usage:
  uv run python scripts/summarize_frames_cues_grid.py \
      --results-dir results/frames_cues/nemotron-3-nano_30b

  # multiple models at once (one report each):
  uv run python scripts/summarize_frames_cues_grid.py \
      --results-dir results/frames_cues/nemotron-3-nano_30b \
                    results/frames_cues/gemma4_31b \
                    results/frames_cues/qwen-3.5_122b
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy.stats import binomtest, wilcoxon
try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    multipletests = None

# Grid conditions: <phrasing>_<template> plus the two epistemic cues (terse anchor, PLAIN template).
PHRASINGS = ["verbose", "terse"]
TEMPLATES = ["plain", "natural", "query", "elaborate", "polite", "direct"]
EPI_CONDS = ["epi_strong_boost", "epi_strong_hedge"]

TEMPLATE_LABEL = {
    "natural": 'NATURAL ("2-4 sentences")',
    "elaborate": 'ELABORATE ("8-10 sent. detailed")',
    "plain": "PLAIN (bare question)",
    "query": "QUERY (structured Exact-Answer)",
    "polite": "POLITE (extreme politeness, no length cue)",
    "direct": "DIRECT (max answer-directive, no length/politeness)",
}
TEMPLATE_LABEL_TERSE = {**TEMPLATE_LABEL, "plain": "PLAIN (= neutral baseline)"}


def _find_result_json(results_dir: str, condition: str) -> str | None:
    hits = sorted(glob.glob(os.path.join(results_dir, f"frames-cues_baseline_*_{condition}.json")))
    return hits[0] if hits else None


# Source JSONL (problem text -> example_id) per condition, for older result files that predate
# example_id persistence. verbose/terse phrasings share one anchor file each.
def _source_jsonl(condition: str) -> str:
    if condition.startswith("verbose_"):
        return "orig_phrasing50.jsonl"
    if condition.startswith("terse_"):
        return "neutral_matched50.jsonl"
    return f"{condition}.jsonl"  # epi_strong_boost / epi_strong_hedge


def _problem_to_id(cues_dir: str, condition: str) -> dict[str, int]:
    path = os.path.join(cues_dir, _source_jsonl(condition))
    mapping = {}
    if not os.path.exists(path):
        return mapping
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                mapping[str(r["text"]).strip()] = r["example_id"]
    return mapping


def load_condition(results_dir: str, condition: str, cues_dir: str) -> dict[int, dict] | None:
    """Return {example_id: {search_calls, searched, correct}} for one condition, or None.

    example_id is read from the result row when present; otherwise recovered by mapping the
    row's `problem` text through the condition's source JSONL in cues_dir.
    """
    path = _find_result_json(results_dir, condition)
    if not path:
        return None
    with open(path) as f:
        results = json.load(f)
    prob2id = None
    out = {}
    for r in results:
        ex = r.get("example_id")
        if ex is None:
            if prob2id is None:
                prob2id = _problem_to_id(cues_dir, condition)
            ex = prob2id.get(str(r.get("problem", "")).strip())
            if ex is None:
                continue
        sc = int(r.get("sampler_search_calls", 0) or 0)
        # A row is "stopped" when the agent hit a hard limit (e.g. UsageLimitExceeded):
        # in this dataset every stopped row is uniformly (search_calls=100, correct=False),
        # i.e. it burned the full search budget without delivering a correct answer.
        stopped = bool(r.get("stop_reason"))
        # Normalize id type: some files store example_id as int, the JSONLs as str.
        out[str(ex)] = {"search_calls": sc, "searched": sc > 0,
                        "correct": bool(r.get("sampler_correct")), "stopped": stopped}
    return out or None


def _completed(cond: dict[str, dict]) -> dict[str, dict]:
    """Subset of a condition's rows that ran to completion (not budget-exhausted)."""
    return {i: v for i, v in cond.items() if not v["stopped"]}


def _paired(a: dict[int, dict], b: dict[int, dict], field: str):
    """Aligned numpy arrays over the example_ids common to both conditions."""
    ids = sorted(set(a) & set(b))
    av = np.array([a[i][field] for i in ids], dtype=float)
    bv = np.array([b[i][field] for i in ids], dtype=float)
    return av, bv, len(ids)


def _paired_view(a: dict[str, dict], b: dict[str, dict], field: str, completed: bool):
    """Aligned arrays for `field`, paired on common ids.

    completed=True restricts to ids that ran to completion in BOTH conditions (so n
    varies per contrast); completed=False uses every common id, with stopped rows
    entering at their stored values (search_calls=100, correct=False).
    """
    if completed:
        a, b = _completed(a), _completed(b)
    ids = sorted(set(a) & set(b))
    av = np.array([a[i][field] for i in ids], dtype=float)
    bv = np.array([b[i][field] for i in ids], dtype=float)
    return av, bv, ids


def _stop_contrast(a: dict[str, dict], b: dict[str, dict]):
    """Paired McNemar on the binary `stopped` outcome over common ids (Δ = a − b)."""
    ids = sorted(set(a) & set(b))
    sa = np.array([a[i]["stopped"] for i in ids], dtype=int)
    sb = np.array([b[i]["stopped"] for i in ids], dtype=int)
    return _mcnemar(sa, sb)


def _wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Wilcoxon on x - y; 1.0 when all differences are zero."""
    if len(x) == 0 or not np.any(x - y != 0):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def _apply_multitest_correction(pvalues: list[float], method: str = "fdr_bh") -> dict[int, float]:
    """Apply multiple testing correction to a list of p-values.

    Returns a dict mapping original index to corrected p-value.
    method: "bonferroni", "holm", "fdr_bh" (FDR), or None for no correction.
    """
    if method is None or not pvalues:
        return {i: p for i, p in enumerate(pvalues)}
    if multipletests is None:
        raise ImportError("statsmodels required for p-value correction. Install with: uv pip install statsmodels")
    reject, corrected, *_ = multipletests(pvalues, method=method)
    return {i: float(p) for i, p in enumerate(corrected)}


def _mcnemar(a_correct: np.ndarray, b_correct: np.ndarray) -> tuple[float, float, int, int]:
    """Paired binary-accuracy contrast a − b.

    Returns (mean_delta_pp, p, n10, n01) where the delta is in percentage points,
    n10 = (a right, b wrong), n01 = (a wrong, b right), and p is McNemar's exact
    (binomial) test on the discordant pairs. p = 1.0 when there are no discordances.
    """
    a = a_correct.astype(int)
    b = b_correct.astype(int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    mean_delta_pp = float((a - b).mean()) * 100.0
    n = n10 + n01
    p = 1.0 if n == 0 else float(binomtest(min(n10, n01), n, 0.5).pvalue)
    return mean_delta_pp, p, n10, n01


def _verdict(mean_delta: float, p: float) -> str:
    if p < 0.05:
        return "↓ sig" if mean_delta < 0 else "↑ sig"
    if p < 0.10:
        return "null (trend)"
    return "null"


def _tradeoff_flag(d_search: float, p_search: float, d_acc: float, p_acc: float) -> str:
    """Classify the search↔accuracy relationship for one contrast.

    Cross-classifies significant search movement against significant accuracy movement:
    a search *reduction* with a significant accuracy drop is a real tradeoff (🟠); a
    search *increase* with a significant accuracy drop means extra retrieval is hurting,
    not helping (🔴) — a qualitatively different failure. A significant reduction with a
    non-sig dip is a watch-item; with flat/up accuracy it's a free efficiency win. An
    accuracy drop with no significant search movement is flagged too. Everything else
    is left blank.
    """
    sig_s, sig_a_drop = p_search < 0.05, (p_acc < 0.05 and d_acc < 0)
    if sig_s and d_search < 0:
        if sig_a_drop:
            return "🟠 TRADEOFF (less search → acc drop)"
        if p_acc < 0.05 and d_acc > 0:
            return "✨ less search, acc UP"
        if d_acc < -1.0:
            return "acc dip (ns)"
        return "free ✅"
    if sig_s and d_search > 0:
        if sig_a_drop:
            return "🔴 acc drop despite MORE search"
        return ""
    if sig_a_drop:
        return "⚠️ acc drop (search flat)"
    return ""


# Human-readable gloss for each cue/template name appearing in contrast labels, quoting the
# actual prompt text (from src/services/qa_eval.py *_QUERY_TEMPLATE) where it exists.
CUE_GLOSS = {
    "PLAIN": "PLAIN = bare question, no added instruction",
    "NATURAL": 'NATURAL = adds "Please answer in 2-4 sentences."',
    "ELABORATE": 'ELABORATE = adds "Please answer with a detailed explanation - at least 8-10 sentences"',
    "POLITE": "POLITE = extreme-politeness wrapper, no length/format cue",
    "DIRECT": 'DIRECT = adds "Just answer the question directly — final answer only."',
    "QUERY": "QUERY = structured Explanation / Exact Answer / Confidence format",
    "boost": "boost = strong epistemic-confidence prefix",
    "hedge": "hedge = strong epistemic-hedge prefix",
    "neutral": "neutral = terse PLAIN baseline",
    "terse": "terse = terse rewrite of the question",
    "verbose": "verbose = original verbose FRAMES phrasing",
}


def _cue_gloss(label: str) -> str:
    """'(PLAIN = bare question…; ELABORATE = adds "…")' for a contrast label like
    'ELABORATE − PLAIN @verbose'."""
    head = label.split("@")[0]
    glosses = [CUE_GLOSS[p.strip()] for p in head.split("−") if p.strip() in CUE_GLOSS]
    return f"({'; '.join(glosses)})" if glosses else ""


def _b(s: str, sig: bool) -> str:
    """Bold a table cell when its contrast is significant."""
    return f"**{s}**" if sig else s


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def build_report(results_dir: str, model_label: str, grader: str, cues_dir: str, correction_method: str | None = "fdr_bh") -> str:
    cond = {}
    for ph in PHRASINGS:
        for tm in TEMPLATES:
            cond[f"{ph}_{tm}"] = load_condition(results_dir, f"{ph}_{tm}", cues_dir)
    for c in EPI_CONDS:
        cond[c] = load_condition(results_dir, c, cues_dir)

    missing = [k for k, v in cond.items() if v is None]
    available = {k: v for k, v in cond.items() if v is not None}
    if "terse_plain" not in available:
        raise SystemExit(f"terse_plain (neutral baseline) missing in {results_dir}; cannot pair.")

    common = set.intersection(*(set(v) for v in available.values()))
    n_common = len(common)
    n_by_cond = {k: len(v) for k, v in available.items()}
    n_max = max(n_by_cond.values())
    incomplete = {k: n for k, n in n_by_cond.items() if n < n_max}
    neutral = available["terse_plain"]

    L = []
    L.append(f"# FRAMES prompt-cue sensitivity — {model_label} results summary")
    L.append("")
    L.append(f"**Model:** {model_label} · **Agent:** baseline search · **Index:** local BM25 (`data/frames_index`)")
    L.append(f"**Grader:** {grader} · **n = {n_max}** grid questions ({n_common} common to all "
             "conditions) · **DV:** `search_calls` (per-question retrieval count)")
    if incomplete:
        inc = ", ".join(f"{k} ({n}/{n_max})" for k, n in sorted(incomplete.items()))
        L.append(f"⚠️ **Uneven coverage** — incomplete conditions: {inc}. Every contrast is paired "
                 "on the example_ids common to its two conditions (`nC` in Table 2), so partial "
                 "conditions only shrink their own contrasts; but Table 1 cell descriptives for "
                 "them cover a different question subset — don't compare those cells' raw means "
                 "against full-coverage cells. Tests: Wilcoxon signed-rank (paired per contrast).")
    else:
        L.append(f"Same {n_max} non-stale FRAMES questions throughout. "
                 "Tests: Wilcoxon signed-rank (paired).")
    correction_label = ""
    if correction_method:
        method_names = {"bonferroni": "Bonferroni", "holm": "Holm", "fdr_bh": "FDR (Benjamini-Hochberg)"}
        correction_label = f" · **Multiple-test correction: {method_names.get(correction_method, correction_method)}**"
    L.append(f"Statistical tests: Wilcoxon signed-rank (paired){correction_label}.")
    L.append("")
    L.append("> **Stopped questions:** rows that hit a hard limit (`stop_reason`, e.g. `UsageLimitExceeded`) "
             "are uniformly `search_calls=100` and incorrect — the agent burned its full search budget "
             "without answering. They are reported as a first-class **stop-rate** DV, and every "
             "search/accuracy contrast is shown in two views: **completed-only** (genuine retrieval "
             "policy, paired on ids completed in both conditions) and **all** (stopped rows folded in "
             "at 100/incorrect, so the censoring/budget-exhaustion effect is visible).")
    if missing:
        L.append("")
        L.append(f"> ⚠️ Missing conditions (skipped): {', '.join(sorted(missing))}")
    L.append("")
    # Key insights are inserted here once all contrasts are computed (see end of function).
    insight_at = len(L)

    # ---- Table 1: cell descriptives -------------------------------------------------
    L.append("## Table 1 — Cell descriptives (2×4 template grid + epistemic cues)")
    L.append("")
    L.append("`compl/all` = completed-only value / value with stopped rows folded in (100, incorrect). "
             "Median search is completed-only.")
    L.append("")
    L.append("| Phrasing | Template / cue | Stopped | Mean search (compl/all) | Median | Accuracy (compl/all) |")
    L.append("|---|---|---|---|---|---|")

    def cell_row(phrasing_label, label, data):
        comp = _completed(data)
        n, ns = len(data), len(data) - len(_completed(data))
        sc_c = np.array([d["search_calls"] for d in comp.values()], dtype=float)
        sc_a = np.array([d["search_calls"] for d in data.values()], dtype=float)
        acc_c = np.array([d["correct"] for d in comp.values()], dtype=float)
        acc_a = np.array([d["correct"] for d in data.values()], dtype=float)
        med = int(np.median(sc_c)) if len(sc_c) else 0
        stop_pct = (100.0 * ns / n) if n else 0.0
        mean_c = sc_c.mean() if len(sc_c) else float("nan")
        acc_c_m = acc_c.mean() if len(acc_c) else float("nan")
        L.append(f"| {phrasing_label} | {label} | {ns}/{n} ({stop_pct:.0f}%) | "
                 f"{_fmt(mean_c)} / {_fmt(sc_a.mean())} | {med} | {_fmt(acc_c_m)} / {_fmt(acc_a.mean())} |")

    order = [("verbose", t) for t in ["natural", "elaborate", "polite", "direct", "plain", "query"]] + \
            [("terse", t) for t in ["natural", "elaborate", "polite", "direct", "plain", "query"]]
    for ph, tm in order:
        key = f"{ph}_{tm}"
        if key in available:
            lbl = (TEMPLATE_LABEL_TERSE if ph == "terse" else TEMPLATE_LABEL)[tm]
            cell_row(ph, lbl, available[key])
    if "epi_strong_boost" in available:
        cell_row("terse", "+ epistemic boost", available["epi_strong_boost"])
    if "epi_strong_hedge" in available:
        cell_row("terse", "+ epistemic hedge", available["epi_strong_hedge"])
    L.append("")

    # ---- Table 2: paired contrasts --------------------------------------------------
    # (label, condition_a, condition_b, isolates)  -> Δ = a − b
    contrasts = [
        ("boost − neutral", "epi_strong_boost", "terse_plain", "epistemic booster"),
        ("hedge − neutral", "epi_strong_hedge", "terse_plain", "epistemic hedge"),
        ("terse − verbose @PLAIN", "terse_plain", "verbose_plain", "phrasing"),
        ("terse − verbose @NATURAL", "terse_natural", "verbose_natural", "phrasing"),
        ("terse − verbose @ELABORATE", "terse_elaborate", "verbose_elaborate", "phrasing"),
        ("terse − verbose @QUERY", "terse_query", "verbose_query", "phrasing"),
        ("QUERY − PLAIN @verbose", "verbose_query", "verbose_plain", "structured template"),
        ("QUERY − PLAIN @terse", "terse_query", "terse_plain", "structured template"),
        ("NATURAL − PLAIN @verbose", "verbose_natural", "verbose_plain", "short directive"),
        ("NATURAL − PLAIN @terse", "terse_natural", "terse_plain", "short directive"),
        ("ELABORATE − PLAIN @verbose", "verbose_elaborate", "verbose_plain", "long directive"),
        ("ELABORATE − PLAIN @terse", "terse_elaborate", "terse_plain", "long directive"),
        ("ELABORATE − NATURAL @verbose", "verbose_elaborate", "verbose_natural", "output length"),
        ("ELABORATE − NATURAL @terse", "terse_elaborate", "terse_natural", "output length"),
        ("POLITE − PLAIN @verbose", "verbose_polite", "verbose_plain", "politeness"),
        ("POLITE − PLAIN @terse", "terse_polite", "terse_plain", "politeness"),
        ("POLITE − NATURAL @verbose", "verbose_polite", "verbose_natural", "politeness vs length-directive"),
        ("POLITE − NATURAL @terse", "terse_polite", "terse_natural", "politeness vs length-directive"),
        ("DIRECT − PLAIN @verbose", "verbose_direct", "verbose_plain", "pure answer-directive"),
        ("DIRECT − PLAIN @terse", "terse_direct", "terse_plain", "pure answer-directive"),
        ("DIRECT − NATURAL @verbose", "verbose_direct", "verbose_natural", "directive minus length+please"),
        ("DIRECT − NATURAL @terse", "terse_direct", "terse_natural", "directive minus length+please"),
        ("DIRECT − POLITE @verbose", "verbose_direct", "verbose_polite", "directive vs politeness"),
        ("DIRECT − POLITE @terse", "terse_direct", "terse_polite", "directive vs politeness"),
    ]
    active = [(label, ca, cb, iso) for label, ca, cb, iso in contrasts
              if ca in available and cb in available]

    # ---- Table 2: search contrasts, completed-only vs all ---------------------------
    L.append("## Table 2 — Search contrasts (completed-only vs all)")
    L.append("")
    L.append("Δ = a − b on per-question `search_calls` (Wilcoxon signed-rank). "
             "**compl** pairs only ids that completed in both conditions (`nC`); **all** folds "
             "stopped rows in at 100. Rows with a significant completed-view search shift are "
             "**bolded** and marked 🔻 (search down) / 🔺 (search up). `Tradeoff` (computed on the "
             "completed view) cross-classifies against accuracy: 🟠 = accuracy paid for a search "
             "*reduction*, 🔴 = accuracy drops despite a search *increase*.")
    L.append("")
    L.append("| # | Comparison | Isolates | nC | Δsearch (compl) | p | Δsearch (all) | p | Tradeoff |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows2 = {}          # completed-only search contrasts -> (mean, median, p)  [drives conclusions]
    rows2_all = {}      # all-rows search contrasts
    rows2_acc = {}      # completed-only accuracy contrasts -> (Δpp, p)         [drives conclusions]
    rows2_acc_all = {}  # all-rows accuracy contrasts
    rows_stop = {}      # stop-rate contrasts -> (Δpp, p)
    for i, (label, ca, cb, isolates) in enumerate(active, 1):
        avc, bvc, idsc = _paired_view(available[ca], available[cb], "search_calls", completed=True)
        meandc, meddc, pc = float((avc - bvc).mean()), float(np.median(avc - bvc)), _wilcoxon_p(avc, bvc)
        ava, bva, _ = _paired_view(available[ca], available[cb], "search_calls", completed=False)
        meanda, pa = float((ava - bva).mean()), _wilcoxon_p(ava, bva)
        rows2[label] = (meandc, meddc, pc)
        rows2_all[label] = (meanda, float(np.median(ava - bva)), pa)
        # accuracy + stop contrasts (computed here, rendered in Table 3)
        acc_c, bcc, _ = _paired_view(available[ca], available[cb], "correct", completed=True)
        rows2_acc[label] = _mcnemar(acc_c, bcc)[:2]
        acc_a, bca, _ = _paired_view(available[ca], available[cb], "correct", completed=False)
        rows2_acc_all[label] = _mcnemar(acc_a, bca)[:2]
        rows_stop[label] = _stop_contrast(available[ca], available[cb])[:2]
        flag = _tradeoff_flag(meandc, pc, rows2_acc[label][0], rows2_acc[label][1])
        sig_c, sig_a = pc < 0.05, pa < 0.05
        mark = ("🔻 " if meandc < 0 else "🔺 ") if sig_c else ""
        L.append(f"| {i} | {mark}{_b(label, sig_c)} | {isolates} | {len(idsc)} | "
                 f"{_b(f'{meandc:+.2f}', sig_c)} | {_b(f'{pc:.3g}', sig_c)} | "
                 f"{_b(f'{meanda:+.2f}', sig_a)} | {_b(f'{pa:.3g}', sig_a)} | {flag} |")
    L.append("")

    # ---- Apply multiple-testing correction (if requested) ----------------------------
    if correction_method:
        all_pvals = (
            [rows2[l][2] for l in rows2] +
            [rows2_all[l][2] for l in rows2_all] +
            [rows2_acc[l][1] for l in rows2_acc] +
            [rows2_acc_all[l][1] for l in rows2_acc_all] +
            [rows_stop[l][1] for l in rows_stop]
        )
        corrected_dict = _apply_multitest_correction(all_pvals, method=correction_method)
        idx = 0
        for l in rows2:
            rows2[l] = (rows2[l][0], rows2[l][1], corrected_dict[idx])
            idx += 1
        for l in rows2_all:
            rows2_all[l] = (rows2_all[l][0], rows2_all[l][1], corrected_dict[idx])
            idx += 1
        for l in rows2_acc:
            rows2_acc[l] = (rows2_acc[l][0], corrected_dict[idx])
            idx += 1
        for l in rows2_acc_all:
            rows2_acc_all[l] = (rows2_acc_all[l][0], corrected_dict[idx])
            idx += 1
        for l in rows_stop:
            rows_stop[l] = (rows_stop[l][0], corrected_dict[idx])
            idx += 1

    # ---- Table 3: accuracy & stop-rate contrasts ------------------------------------
    L.append("## Table 3 — Accuracy & stop-rate contrasts")
    L.append("")
    L.append("Δ = a − b in percentage points (McNemar exact). `Acc` is shown completed-only vs all; "
             "`stop-rate` is the paired difference in the share of questions that hit the budget cap "
             "(positive ⇒ condition *a* exhausts search more often). Rows with a significant "
             "completed-view accuracy shift are **bolded** and marked ⬇️/⬆️; `Tradeoff` repeats the "
             "search-context classification from Table 2 (🟠 acc drop under *less* search, "
             "🔴 acc drop under *more* search, ⚠️ acc drop with search flat).")
    L.append("")
    L.append("| # | Comparison | ΔAcc pp (compl) | p | ΔAcc pp (all) | p | Δstop-rate pp | p | Tradeoff |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for i, (label, ca, cb, isolates) in enumerate(active, 1):
        dac, pac = rows2_acc[label]
        daa, paa = rows2_acc_all[label]
        ds, ps = rows_stop[label]
        flag = _tradeoff_flag(rows2[label][0], rows2[label][2], dac, pac)
        sig_c, sig_a, sig_s = pac < 0.05, paa < 0.05, ps < 0.05
        mark = ("⬇️ " if dac < 0 else "⬆️ ") if sig_c else ""
        L.append(f"| {i} | {mark}{_b(label, sig_c)} | {_b(f'{dac:+.2f}', sig_c)} | "
                 f"{_b(f'{pac:.3g}', sig_c)} | {_b(f'{daa:+.2f}', sig_a)} | {_b(f'{paa:.3g}', sig_a)} | "
                 f"{_b(f'{ds:+.2f}', sig_s)} | {_b(f'{ps:.3g}', sig_s)} | {flag} |")
    L.append("")

    # ---- Table 4: difference-in-differences (phrasing x template-vs-PLAIN) ----------
    L.append("## Table 4 — Interaction (difference-in-differences: phrasing × template-vs-PLAIN)")
    L.append("")
    L.append("Computed on **completed-only** rows (the 100-call spikes from stopped questions would "
             "otherwise dominate a difference-in-differences on raw search_calls).")
    L.append("")
    L.append("| Template effect | mean DiD | med | p | Reading |")
    L.append("|---|---|---|---|---|")
    did_rows = {}
    for tm in ["natural", "elaborate", "polite", "query"]:
        keys = [f"terse_{tm}", "terse_plain", f"verbose_{tm}", "verbose_plain"]
        if any(k not in available for k in keys):
            continue
        comp = {k: _completed(available[k]) for k in keys}
        ids = sorted(set.intersection(*(set(comp[k]) for k in keys)))
        tt = np.array([comp[f"terse_{tm}"][i]["search_calls"] for i in ids], float)
        tp = np.array([comp["terse_plain"][i]["search_calls"] for i in ids], float)
        vt = np.array([comp[f"verbose_{tm}"][i]["search_calls"] for i in ids], float)
        vp = np.array([comp["verbose_plain"][i]["search_calls"] for i in ids], float)
        # DiD = verbose template-effect − terse template-effect (sign convention matches the
        # original hand-written gemini SUMMARY: negative ⇒ template fires more strongly under verbose).
        did = (vt - vp) - (tt - tp)
        meand, medd = float(did.mean()), float(np.median(did))
        p = _wilcoxon_p((vt - vp), (tt - tp))
        did_rows[tm] = (meand, medd, p)
        if p < 0.05:
            reading = "phrasing-dependent (interaction)"
        else:
            reading = "no interaction"
        L.append(f"| {tm.upper()} | {meand:+.2f} | {medd:+.1f} | {p:.3g} | {reading} |")
    L.append("")

    # ---- Conclusions (auto-derived) -------------------------------------------------
    L.append("## Conclusions (auto-derived)")
    L.append("")
    sig = [(lbl, *vals) for lbl, vals in rows2.items() if vals[2] < 0.05]

    # 1. epistemic cues
    epi_bits = []
    for lbl in ("boost − neutral", "hedge − neutral"):
        if lbl in rows2:
            epi_bits.append(f"{lbl.split(' ')[0]} p={rows2[lbl][2]:.2g}")
    if epi_bits:
        epi_sig = any(rows2[l][2] < 0.05 for l in ("boost − neutral", "hedge − neutral") if l in rows2)
        verdict = "an effect — investigate" if epi_sig else "clean null"
        L.append(f"1. **Epistemic cues (boost/hedge): {verdict}** ({', '.join(epi_bits)}).")

    # 2. phrasing
    ph_labels = [l for l in rows2 if "terse − verbose" in l]
    ph_sig = [l for l in ph_labels if rows2[l][2] < 0.05]
    if ph_labels:
        msg = "null under every template" if not ph_sig else f"significant under {', '.join(ph_sig)}"
        L.append(f"2. **Phrasing (verbose vs terse): {msg}.**")

    # 3. directive (natural/elaborate vs plain) effects
    dir_labels = [l for l in rows2 if ("NATURAL − PLAIN" in l or "ELABORATE − PLAIN" in l)]
    dir_sig = [l for l in dir_labels if rows2[l][2] < 0.05]
    if dir_sig:
        signs = {("↓" if rows2[l][0] < 0 else "↑") for l in dir_sig}
        direction = "reduce" if signs == {"↓"} else ("increase" if signs == {"↑"} else "shift")
        L.append(f"3. **Directive templates (NATURAL/ELABORATE vs PLAIN) {direction} search** "
                 f"in: {', '.join(dir_sig)}.")
    elif dir_labels:
        L.append("3. **Directive templates (NATURAL/ELABORATE vs PLAIN): no significant shift.**")

    # 3b. politeness hypothesis: does extreme politeness (no length cue) reduce search vs PLAIN?
    pol_labels = [l for l in rows2 if "POLITE − PLAIN" in l]
    if pol_labels:
        pol_sig = [l for l in pol_labels if rows2[l][2] < 0.05]
        if not pol_sig:
            stat = "**no effect** — politeness alone does NOT move search (hypothesis not supported)"
        else:
            signs = {("↓" if rows2[l][0] < 0 else "↑") for l in pol_sig}
            if signs == {"↓"}:
                stat = "**reduces search** — supports 'politeness drives less search'"
            elif signs == {"↑"}:
                stat = "**increases search** — opposite of the hypothesis"
            else:
                stat = "**shifts search inconsistently** across phrasing"
        bits = ", ".join(f"{l.split('@')[1]} Δ={rows2[l][0]:+.2f} p={rows2[l][2]:.2g}" for l in pol_labels)
        L.append(f"3b. **Politeness (POLITE vs PLAIN, no length cue): {stat}** ({bits}).")

        # Dissociation test: NATURAL carries BOTH a conversational/polite framing AND an output-length
        # directive. If NATURAL reduces search but pure-politeness POLITE does not, the reduction is
        # attributable to the output directive, not politeness.
        nat_labels = [l for l in rows2 if "NATURAL − PLAIN" in l]
        nat_down = [l for l in nat_labels if rows2[l][2] < 0.05 and rows2[l][0] < 0]
        if nat_down and not pol_sig:
            L.append("    **Dissociation:** NATURAL cuts search vs PLAIN "
                     f"({', '.join(l.split('@')[1] for l in nat_down)}) while POLITE does not — so the "
                     "NATURAL/ELABORATE reduction comes from the **output-length/answer directive, "
                     "not politeness**.")

        # POLITE vs the mild NATURAL directive (trend-aware: flag consistent same-direction trends).
        pn = [l for l in rows2 if "POLITE − NATURAL" in l]
        pn_hit = [l for l in pn if rows2[l][2] < 0.10]  # sig or trend
        if pn:
            if not pn_hit:
                L.append("    POLITE ≈ NATURAL: no POLITE−NATURAL difference (extreme politeness "
                         "behaves like the mild directive).")
            elif all(rows2[l][0] > 0 for l in pn_hit) and len(pn_hit) == len(pn):
                L.append("    POLITE searches **more** than NATURAL (consistent trend "
                         f"{', '.join(f'{l.split(chr(64))[1]} p={rows2[l][2]:.2g}' for l in pn)}) — "
                         "extreme politeness does NOT reproduce the directive's search reduction.")
            else:
                L.append(f"    POLITE−NATURAL mixed: {', '.join(f'{l.split(chr(64))[1]} Δ={rows2[l][0]:+.2f} p={rows2[l][2]:.2g}' for l in pn)}.")

    # 4. accuracy (completed-only primary, all-rows in parentheses)
    accs = {k: np.mean([d["correct"] for d in _completed(v).values()]) for k, v in available.items()}
    accs_all = {k: np.mean([d["correct"] for d in v.values()]) for k, v in available.items()}
    L.append(f"4. **Accuracy (completed) range {min(accs.values()):.2f}–{max(accs.values()):.2f}** "
             f"(lowest: {min(accs, key=accs.get)}; highest: {max(accs, key=accs.get)}); "
             f"all-rows range {min(accs_all.values()):.2f}–{max(accs_all.values()):.2f} once "
             "budget-exhausted questions are folded in.")

    # 4b. stop rate: budget-exhaustion as a behavioral DV
    stops = {k: np.mean([d["stopped"] for d in v.values()]) for k, v in available.items()}
    stop_sig = [l for l in rows_stop if rows_stop[l][1] < 0.05]
    lo_k, hi_k = min(stops, key=stops.get), max(stops, key=stops.get)
    bit = (f"; significant stop-rate contrasts: " +
           ", ".join(f"{l} (Δ={rows_stop[l][0]:+.1f}pp, p={rows_stop[l][1]:.2g})" for l in stop_sig)
           if stop_sig else "; no contrast moves the stop rate significantly")
    L.append(f"4b. **Stop rate (budget exhaustion) ranges {stops[lo_k]:.0%}–{stops[hi_k]:.0%}** "
             f"(lowest: {lo_k}; highest: {hi_k}){bit}. Every stopped question is 100 search "
             "calls / incorrect, so this is the rate at which a cue pushes the agent into runaway search.")

    # 5. search↔accuracy tradeoff: does any significant search reduction cost accuracy?
    reducers = [l for l in rows2 if rows2[l][2] < 0.05 and rows2[l][0] < 0]
    if reducers:
        costly = [l for l in reducers if rows2_acc[l][0] < 0 and rows2_acc[l][1] < 0.05]
        free = [l for l in reducers if l not in costly]
        if not costly:
            L.append(f"5. **No search↔accuracy tradeoff:** all {len(reducers)} significant "
                     "search-reducing cues hold accuracy flat (none with a significant McNemar "
                     f"drop). Free efficiency wins: {', '.join(free)}.")
        else:
            L.append(f"5. **Search↔accuracy tradeoff in {len(costly)}/{len(reducers)} "
                     "search-reducing cues** — significant accuracy cost in: " +
                     ", ".join(f"{l} (ΔAcc={rows2_acc[l][0]:+.2f}pp, p={rows2_acc[l][1]:.2g})"
                               for l in costly) +
                     (f". The rest reduce search for free: {', '.join(free)}." if free else "."))
    increasers = [l for l in rows2 if rows2[l][2] < 0.05 and rows2[l][0] > 0]
    worse_more = [l for l in increasers if rows2_acc[l][0] < 0 and rows2_acc[l][1] < 0.05]
    if worse_more:
        L.append("5b. 🔴 **More search AND lower accuracy** in: " +
                 ", ".join(f"{l} (Δsearch={rows2[l][0]:+.2f}, ΔAcc={rows2_acc[l][0]:+.2f}pp, "
                           f"p={rows2_acc[l][1]:.2g})" for l in worse_more) +
                 " — the extra retrieval is hurting, not helping (distinct from a "
                 "reduction-driven tradeoff).")
    L.append("")
    if sig:
        L.append("**Significant contrasts (p<0.05):** " +
                 ", ".join(f"{l} (Δ={v[0]:+.2f}, p={v[2]:.2g})" for l, v, *_ in
                           [(l, rows2[l]) for l in rows2 if rows2[l][2] < 0.05]) + ".")
    else:
        L.append("**No contrast reached p<0.05** — flat retrieval policy across all cues.")
    L.append("")
    L.append("**Caveats:** single model; high right-tail variance in `search_calls` (medians often "
             "identical across cells); ~14 contrasts, so weight effects that *replicate* across "
             "phrasing/length over isolated hits.")
    L.append("")
    L.append("## Provenance")
    L.append("")
    L.append(f"Result files under `{results_dir}/`. Analysis: id-join on `example_id` "
             "(read from each result JSON, or recovered via the condition's source JSONL "
             "`text`→`example_id` for older files predating example_id persistence; ids normalized "
             "to str), paired per-contrast on the common example_ids. "
             "Generated by `scripts/summarize_frames_cues_grid.py`.")
    L.append("")

    # ---- Key insights (inserted at the top, after the header) -----------------------
    # Uses the completed-view contrasts (rows2 / rows2_acc), same basis as the conclusions.
    ins = ["## Key insights", ""]
    n_contrasts = len(active)
    sig_search = [l for l in rows2 if rows2[l][2] < 0.05]
    down = [l for l in sig_search if rows2[l][0] < 0]
    up = [l for l in sig_search if rows2[l][0] > 0]
    if sig_search:
        big = max(sig_search, key=lambda l: abs(rows2[l][0]))
        ins.append(f"- **{len(sig_search)}/{n_contrasts} contrasts significantly moved search usage** "
                   f"({len(down)} decreased it, {len(up)} increased it). Largest shift: {big} "
                   f"(Δ={rows2[big][0]:+.2f} calls/question, p={rows2[big][2]:.2g}).")
    else:
        ins.append(f"- **No cue moved search significantly** in any of the {n_contrasts} paired "
                   "contrasts — this model's retrieval policy is insensitive to the cue grid.")
    sig_acc = [l for l in rows2 if rows2_acc[l][1] < 0.05]
    drop_less = [l for l in sig_acc if rows2_acc[l][0] < 0 and l in down]
    drop_more = [l for l in sig_acc if rows2_acc[l][0] < 0 and l in up]
    drop_flat = [l for l in sig_acc if rows2_acc[l][0] < 0 and l not in down and l not in up]
    gains = [l for l in sig_acc if rows2_acc[l][0] > 0]
    for l in drop_less:
        ins.append(f"- 🟠 **Real tradeoff — {l}:** accuracy drops {rows2_acc[l][0]:+.2f}pp "
                   f"(p={rows2_acc[l][1]:.2g}) under a significant search *reduction* "
                   f"(Δ={rows2[l][0]:+.2f} calls/question). Cutting search here is not free. "
                   f"{_cue_gloss(l)}")
    for l in drop_more:
        ins.append(f"- 🔴 **Accuracy drops despite MORE search — {l}:** {rows2_acc[l][0]:+.2f}pp "
                   f"(p={rows2_acc[l][1]:.2g}) while search *increases* significantly "
                   f"(Δ={rows2[l][0]:+.2f} calls/question) — the extra retrieval hurts rather "
                   f"than helps. {_cue_gloss(l)}")
    for l in drop_flat:
        ins.append(f"- ⚠️ **Accuracy drop with search flat — {l}:** {rows2_acc[l][0]:+.2f}pp "
                   f"(p={rows2_acc[l][1]:.2g}) with no significant search change. {_cue_gloss(l)}")
    if gains:
        ins.append("- ⬆️ Significant accuracy gains: " +
                   ", ".join(f"{l} ({rows2_acc[l][0]:+.2f}pp, p={rows2_acc[l][1]:.2g})"
                             for l in gains) + ".")
    if not sig_acc:
        ins.append("- **No contrast changed accuracy significantly** — every search shift above "
                   "is free in accuracy terms.")
    free = [l for l in down if l not in drop_less]
    if free and drop_less:
        ins.append(f"- ✅ The other {len(free)} search-reducing cue(s) are **free** in terms of "
                   f"accuracy: {', '.join(free)}.")
    elif free:
        ins.append(f"- ✅ All {len(free)} search-reducing cue(s) hold accuracy flat — "
                   "free efficiency wins.")
    stop_sig = [l for l in rows_stop if rows_stop[l][1] < 0.05]
    if stop_sig:
        ins.append("- ⏱️ Budget exhaustion (stop rate) moves significantly in: " +
                   ", ".join(f"{l} (Δ={rows_stop[l][0]:+.1f}pp, p={rows_stop[l][1]:.2g})"
                             for l in stop_sig) + ".")
    ins.append("")
    L[insight_at:insight_at] = ins
    return "\n".join(L)


def slugify(label: str) -> str:
    return label.replace(":", "_").replace("/", "_").replace(".", "_").replace("-", "_")


def main():
    p = argparse.ArgumentParser(description="Summarize a FRAMES cue-grid results folder into Markdown")
    p.add_argument("--results-dir", nargs="+", required=True,
                   help="One or more results/frames_cues/<model_slug>/ folders")
    p.add_argument("--model-label", default=None,
                   help="Display label (default: inferred from folder name). Ignored with >1 dir.")
    p.add_argument("--grader", default="gemini-3-flash-preview", help="Grader model (for the header)")
    p.add_argument("--cues-dir", default="data/frames_cues",
                   help="Condition JSONL dir, used to recover example_id for older result files")
    p.add_argument("--output", default=None,
                   help="Output .md path. Default: <parent>/SUMMARY_<slug>.md next to the folder.")
    p.add_argument("--correction", default="fdr_bh",
                   choices=["none", "bonferroni", "holm", "fdr_bh"],
                   help="Multiple-testing p-value correction method (default: fdr_bh / False Discovery Rate)")
    args = p.parse_args()

    for rdir in args.results_dir:
        rdir = rdir.rstrip("/")
        folder = os.path.basename(rdir)
        label = args.model_label if (args.model_label and len(args.results_dir) == 1) else folder
        correction = None if args.correction == "none" else args.correction
        report = build_report(rdir, label, args.grader, args.cues_dir, correction_method=correction)
        if args.output and len(args.results_dir) == 1:
            out_path = args.output
        else:
            out_path = os.path.join(os.path.dirname(rdir) or ".", f"SUMMARY_{slugify(folder)}.md")
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
