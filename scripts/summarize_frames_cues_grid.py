"""
Summarize one model's FRAMES prompt-cue GRID experiment into a Markdown report.

Point it at a results folder produced by scripts/run_frames_grid_experiment.sh
(results/frames_cues/<model_slug>/, holding one frames-cues_baseline_<model>_<condition>.json
per condition) and it regenerates the SUMMARY_*.md layout used for Gemini:

  Table 1 — Cell descriptives (2x4 phrasing x output-template grid + epistemic cues)
  Table 2 — Paired contrasts (Wilcoxon signed-rank, paired on example_id)
  Table 3 — Interaction (difference-in-differences: phrasing x template-vs-PLAIN)
  Conclusions — auto-derived from the significance pattern.

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
from scipy.stats import wilcoxon

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
        # Normalize id type: some files store example_id as int, the JSONLs as str.
        out[str(ex)] = {"search_calls": sc, "searched": sc > 0, "correct": bool(r.get("sampler_correct"))}
    return out or None


def _paired(a: dict[int, dict], b: dict[int, dict], field: str):
    """Aligned numpy arrays over the example_ids common to both conditions."""
    ids = sorted(set(a) & set(b))
    av = np.array([a[i][field] for i in ids], dtype=float)
    bv = np.array([b[i][field] for i in ids], dtype=float)
    return av, bv, len(ids)


def _wilcoxon_p(x: np.ndarray, y: np.ndarray) -> float:
    """Paired Wilcoxon on x - y; 1.0 when all differences are zero."""
    if len(x) == 0 or not np.any(x - y != 0):
        return 1.0
    return float(wilcoxon(x, y, zero_method="wilcox").pvalue)


def _verdict(mean_delta: float, p: float) -> str:
    if p < 0.05:
        return "↓ sig" if mean_delta < 0 else "↑ sig"
    if p < 0.10:
        return "null (trend)"
    return "null"


def _fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:.{nd}f}"


def build_report(results_dir: str, model_label: str, grader: str, cues_dir: str) -> str:
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
    neutral = available["terse_plain"]

    L = []
    L.append(f"# FRAMES prompt-cue sensitivity — {model_label} results summary")
    L.append("")
    L.append(f"**Model:** {model_label} · **Agent:** baseline search · **Index:** local BM25 (`data/frames_index`)")
    L.append(f"**Grader:** {grader} · **n = {n_common}** paired on `example_id` · "
             "**DV:** `search_calls` (per-question retrieval count)")
    L.append(f"Same {n_common} non-stale FRAMES questions throughout. Tests: Wilcoxon signed-rank (paired).")
    if missing:
        L.append("")
        L.append(f"> ⚠️ Missing conditions (skipped): {', '.join(sorted(missing))}")
    L.append("")

    # ---- Table 1: cell descriptives -------------------------------------------------
    L.append("## Table 1 — Cell descriptives (2×4 template grid + epistemic cues)")
    L.append("")
    L.append("| Phrasing | Template / cue | Mean search | Median | Accuracy |")
    L.append("|---|---|---|---|---|")

    def cell_row(phrasing_label, label, data):
        sc = np.array([d["search_calls"] for d in data.values()], dtype=float)
        acc = np.array([d["correct"] for d in data.values()], dtype=float)
        med = int(np.median(sc)) if len(sc) else 0
        L.append(f"| {phrasing_label} | {label} | {_fmt(sc.mean())} | {med} | {_fmt(acc.mean())} |")

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
    L.append(f"## Table 2 — Paired contrasts (n={n_common})")
    L.append("")
    L.append("| # | Comparison | Isolates | meanΔ | medΔ | p | Verdict |")
    L.append("|---|---|---|---|---|---|---|")
    rows2 = {}
    i = 0
    for label, ca, cb, isolates in contrasts:
        if ca not in available or cb not in available:
            continue
        i += 1
        av, bv, _ = _paired(available[ca], available[cb], "search_calls")
        diff = av - bv
        meand, medd = float(diff.mean()), float(np.median(diff))
        p = _wilcoxon_p(av, bv)
        rows2[label] = (meand, medd, p)
        sign_med = f"{medd:+.1f}".rstrip("0").rstrip(".") if medd else "0"
        L.append(f"| {i} | {label} | {isolates} | {meand:+.2f} | {sign_med} | "
                 f"{p:.3g} | {_verdict(meand, p)} |")
    L.append("")

    # ---- Table 3: difference-in-differences (phrasing x template-vs-PLAIN) ----------
    L.append("## Table 3 — Interaction (difference-in-differences: phrasing × template-vs-PLAIN)")
    L.append("")
    L.append("| Template effect | mean DiD | med | p | Reading |")
    L.append("|---|---|---|---|---|")
    did_rows = {}
    for tm in ["natural", "elaborate", "polite", "query"]:
        keys = [f"terse_{tm}", "terse_plain", f"verbose_{tm}", "verbose_plain"]
        if any(k not in available for k in keys):
            continue
        ids = sorted(set.intersection(*(set(available[k]) for k in keys)))
        tt = np.array([available[f"terse_{tm}"][i]["search_calls"] for i in ids], float)
        tp = np.array([available["terse_plain"][i]["search_calls"] for i in ids], float)
        vt = np.array([available[f"verbose_{tm}"][i]["search_calls"] for i in ids], float)
        vp = np.array([available["verbose_plain"][i]["search_calls"] for i in ids], float)
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

    # 4. accuracy
    accs = {k: np.mean([d["correct"] for d in v.values()]) for k, v in available.items()}
    L.append(f"4. **Accuracy range {min(accs.values()):.2f}–{max(accs.values()):.2f}** "
             f"(lowest: {min(accs, key=accs.get)}; highest: {max(accs, key=accs.get)}).")
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
    args = p.parse_args()

    for rdir in args.results_dir:
        rdir = rdir.rstrip("/")
        folder = os.path.basename(rdir)
        label = args.model_label if (args.model_label and len(args.results_dir) == 1) else folder
        report = build_report(rdir, label, args.grader, args.cues_dir)
        if args.output and len(args.results_dir) == 1:
            out_path = args.output
        else:
            out_path = os.path.join(os.path.dirname(rdir) or ".", f"SUMMARY_{slugify(folder)}.md")
        with open(out_path, "w") as f:
            f.write(report)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
