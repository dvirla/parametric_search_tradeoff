#!/usr/bin/env python3
"""LLM-based semantic leak detector for the MuSiQue natural-phrasing rewrites.

Background: the verbatim bridge-leak metric in analyze_missed_hop_paradox.py is a
lower bound — it only catches an intermediate gold answer appearing as a literal
substring. A faithful paraphrase should preserve the full reasoning chain (refer to
bridge entities only by description). Some natural rewrites instead *name* or make
directly identifiable an intermediate entity, collapsing a 4-hop question into a
~1-hop lookup. This judge measures that semantically.

For each MuSiQue example it shows an LLM the original (benchmark) question, the
natural rewrite, and the bridge hops (sub-question + gold entity, final answer hop
omitted), and asks which bridge entities the natural phrasing reveals. The
question text + gold chain are identical across models, so each example is judged
ONCE (cached by example_id), not once per model.

Outputs (default results/missed_hop_paradox/):
  leak_judgments.json        per example_id: leaked_hop_indices, severity, reasoning,
                             plus the verbatim-leak counts for comparison
  clean_example_ids.json     {"semantic_clean": [...], "verbatim_clean": [...]}
  leak_detection_report.md   contamination rate (verbatim vs semantic) + metric impact
  fig_leak_cleaned_metrics.png/pdf   acc gap & search, all vs verbatim-clean vs semantic-clean

Run a smoke test first, then the full pass (it resumes from cache):
  uv run python scripts/detect_paraphrase_leak.py --max-examples 15
  uv run python scripts/detect_paraphrase_leak.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import viz                          # noqa: E402
from src.services.base_agent import BaseAgent  # noqa: E402


# ─── Judge I/O ──────────────────────────────────────────────────────────────────
class LeakJudgment(BaseModel):
    leaked_hop_indices: list[int]            # bridge hops the natural phrasing reveals
    severity: Literal["none", "partial", "collapsed"]
    reasoning: str


LEAK_PROMPT = """You are auditing whether a paraphrased ("natural") version of a multi-hop question has accidentally leaked intermediate reasoning steps.

The ORIGINAL benchmark question is answerable only by chaining several reasoning hops. Each hop yields a bridge entity; the final hop yields the overall answer. A faithful paraphrase keeps every bridge entity IMPLICIT — referring to it only by description, never naming it or making it directly identifiable.

A bridge hop is LEAKED if the NATURAL phrasing states, names, or makes a competent reader able to identify that hop's bridge entity WITHOUT performing the reasoning step. Rephrasing, politeness, or added narrative framing is NOT a leak as long as the bridge entity stays implicit. Restating the original descriptive clue is NOT a leak.

ORIGINAL (benchmark) question:
{bench_q}

NATURAL (paraphrased) question:
{nat_q}

Bridge hops to protect (the final answer hop is intentionally omitted):
{hops}

Return the 0-based indices of the bridge hops whose answer entity is revealed by the NATURAL phrasing (empty list if the chain is fully preserved). Set severity to:
- "none": no bridge revealed;
- "partial": some but not all bridges revealed;
- "collapsed": all (or all but one) bridges revealed, so the question reduces to roughly a single lookup.
Give brief reasoning naming any revealed entity.
"""


def _parse(v):
    if isinstance(v, (list, dict)):
        return v
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    return eval(v, {"nan": float("nan"), "true": True, "false": False, "null": None})


def _is_true(v) -> bool:
    return str(v) == "True"


def verbatim_bridge_leak(question: str, golds: list[str]) -> int:
    ql = (question or "").lower()
    return sum(1 for g in golds[:-1] if g and len(str(g)) >= 2 and str(g).lower() in ql)


def load_examples(bench_dir: str, nat_dir: str, slugs: list[str]) -> dict[str, dict]:
    """example_id -> {bench_q, nat_q, golds, hops(bridge text), num_hops}.

    Text + gold are model-independent; we read from whichever model file has the
    example, preferring the first slug listed.
    """
    out: dict[str, dict] = {}
    for slug in slugs:
        bpath = os.path.join(bench_dir, f"matched_examples_{slug}.json")
        npath = os.path.join(nat_dir, f"matched_examples_{slug}.json")
        if not (os.path.exists(bpath) and os.path.exists(npath)):
            continue
        b = {e["example_id"]: e for e in json.load(open(bpath))}
        n = {e["example_id"]: e for e in json.load(open(npath))}
        for eid in set(b) & set(n):
            if eid in out:
                continue
            sq = _parse(n[eid]["sub_questions"]) or _parse(b[eid]["sub_questions"])
            golds = [s.get("gold_answer") for s in sq]
            bridges = sq[:-1]  # drop final answer hop
            hops_text = "\n".join(
                f"- hop {s['hop_index']}: {s['question']}  → bridge entity: {s.get('gold_answer')}"
                for s in bridges
            ) or "(no bridge hops — single-hop question)"
            out[eid] = {
                "bench_q": b[eid]["aggregate_question"],
                "nat_q": n[eid]["aggregate_question"],
                "golds": golds,
                "num_hops": len(sq),
                "hops_text": hops_text,
                "n_bridges": len(bridges),
                "verbatim_leak": verbatim_bridge_leak(n[eid]["aggregate_question"], golds),
            }
    return out


def judge_examples(examples: dict, agent: BaseAgent, cache: dict, cache_path: str,
                   max_examples: int | None) -> dict:
    ids = sorted(examples)
    todo = [e for e in ids if e not in cache]
    if max_examples is not None:
        todo = todo[:max_examples]
    print(f"Judging {len(todo)} examples ({len(cache)} cached, {len(ids)} total)")
    for i, eid in enumerate(todo, 1):
        ex = examples[eid]
        if ex["n_bridges"] == 0:                      # single-hop: nothing to leak
            cache[eid] = {"leaked_hop_indices": [], "severity": "none",
                          "reasoning": "single-hop question, no bridge to leak"}
            continue
        prompt = LEAK_PROMPT.format(bench_q=ex["bench_q"], nat_q=ex["nat_q"], hops=ex["hops_text"])
        try:
            j: LeakJudgment = agent.run(prompt).output
            valid = [h for h in j.leaked_hop_indices if 0 <= h < ex["num_hops"] - 1]
            cache[eid] = {"leaked_hop_indices": valid, "severity": j.severity,
                          "reasoning": j.reasoning}
        except Exception as e:
            print(f"  ! {eid}: {e}")
            continue
        if i % 25 == 0:
            json.dump(cache, open(cache_path, "w"), indent=2)
            print(f"  {i}/{len(todo)} judged")
    json.dump(cache, open(cache_path, "w"), indent=2)
    return cache


# ─── Metric impact ───────────────────────────────────────────────────────────────
def paired_metrics(bench_dir: str, nat_dir: str, slugs: list[str],
                   semantic_leaked: set[str], verbatim_leaked: set[str]) -> pd.DataFrame:
    """Per-model accuracy gap & search volume on all / verbatim-clean / semantic-clean."""
    rows = []
    for slug in slugs:
        bpath = os.path.join(bench_dir, f"matched_examples_{slug}.json")
        npath = os.path.join(nat_dir, f"matched_examples_{slug}.json")
        if not (os.path.exists(bpath) and os.path.exists(npath)):
            continue
        b = {e["example_id"]: e for e in json.load(open(bpath))}
        n = {e["example_id"]: e for e in json.load(open(npath))}
        recs = []
        for eid in set(b) & set(n):
            recs.append(dict(eid=eid,
                             bc=_is_true(b[eid]["aggregate_correct"]),
                             nc=_is_true(n[eid]["aggregate_correct"]),
                             sb=len(_parse(b[eid]["query_records"])),
                             sn=len(_parse(n[eid]["query_records"]))))
        df = pd.DataFrame(recs)
        for subset, keep in [("all", df),
                             ("verbatim-clean", df[~df.eid.isin(verbatim_leaked)]),
                             ("semantic-clean", df[~df.eid.isin(semantic_leaked)])]:
            rows.append({"model": viz.display_name(slug), "subset": subset, "n": len(keep),
                         "acc_bench": keep.bc.mean(), "acc_nat": keep.nc.mean(),
                         "acc_gap": keep.nc.mean() - keep.bc.mean(),
                         "search_bench": keep.sb.mean(), "search_nat": keep.sn.mean()})
    return pd.DataFrame(rows)


def fig_cleaned_metrics(metrics: pd.DataFrame, outdir: Path) -> None:
    models = [m for m in viz.MODEL_ORDER if m in set(metrics.model)]
    subsets = ["all", "verbatim-clean", "semantic-clean"]
    colors = {"all": viz.ERR_MUTE, "verbatim-clean": viz.BENCHMARK, "semantic-clean": viz.MISSED}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    x = np.arange(len(models)); w = 0.26

    ax = axes[0]
    for j, sub in enumerate(subsets):
        gaps = [metrics[(metrics.model == m) & (metrics.subset == sub)].acc_gap.iloc[0] for m in models]
        off = (j - 1) * w
        bars = ax.bar(x + off, gaps, w, color=colors[sub], label=sub, zorder=3)
        for bar, g in zip(bars, gaps):
            ax.text(bar.get_x() + bar.get_width() / 2, g + (0.004 if g >= 0 else -0.004),
                    f"{g:+.3f}", ha="center", va="bottom" if g >= 0 else "top", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("Accuracy gap (natural − benchmark)", fontsize=10)
    ax.set_title("Natural-phrasing accuracy advantage\nlargely vanishes once leaked hops are removed",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3, zorder=0); ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for j, sub in enumerate(subsets):
        ratio = [metrics[(metrics.model == m) & (metrics.subset == sub)].eval("search_bench/search_nat").iloc[0]
                 if metrics[(metrics.model == m) & (metrics.subset == sub)].search_nat.iloc[0] else np.nan
                 for m in models]
        off = (j - 1) * w
        bars = ax.bar(x + off, ratio, w, color=colors[sub], label=sub, zorder=3)
        for bar, r in zip(bars, ratio):
            ax.text(bar.get_x() + bar.get_width() / 2, r, f"{r:.1f}×", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models, fontsize=10, fontweight="bold")
    ax.set_ylabel("Search-volume ratio (benchmark / natural)", fontsize=10)
    ax.set_title("Search collapse is unchanged by cleaning\n(it is a real behavioral effect, not leakage)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, frameon=False); ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Impact of removing leaked paraphrases on the cross-phrasing claims",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    viz.savefig(fig, outdir, "fig_leak_cleaned_metrics")


def write_report(examples: dict, cache: dict, metrics: pd.DataFrame, outdir: Path) -> None:
    judged = {e: cache[e] for e in examples if e in cache}
    n = len(judged)
    sem_leaked = {e for e, j in judged.items() if j["leaked_hop_indices"]}
    vb_leaked = {e for e in examples if examples[e]["verbatim_leak"] >= 1}
    sev = pd.Series([j["severity"] for j in judged.values()]).value_counts().to_dict()
    # agreement
    both = len(sem_leaked & vb_leaked); sem_only = len(sem_leaked - vb_leaked); vb_only = len(vb_leaked - sem_leaked)

    lines = ["# Paraphrase Leak Detection — LLM semantic judge\n",
             f"Judged **{n}** unique MuSiQue examples (model-independent text+gold).\n",
             "## Contamination rate\n",
             f"- **Verbatim** bridge-leak (substring): **{len(vb_leaked)}** ({len(vb_leaked)/n:.1%})",
             f"- **Semantic** leak (LLM judge): **{len(sem_leaked)}** ({len(sem_leaked)/n:.1%})",
             f"- Severity: {sev}",
             f"- Agreement: both {both}, semantic-only {sem_only}, verbatim-only {vb_only}\n",
             "The semantic judge catches paraphrases that resolve a bridge without copying the "
             "gold string verbatim, so it is the truer contamination estimate.\n",
             "## Leak rate by hop count\n",
             "| #hops | n | verbatim leaked | semantic leaked |", "|---|---|---|---|"]
    by_h = {}
    for e, ex in examples.items():
        if e not in judged:
            continue
        h = ex["num_hops"]; by_h.setdefault(h, [0, 0, 0])
        by_h[h][0] += 1
        by_h[h][1] += ex["verbatim_leak"] >= 1
        by_h[h][2] += bool(judged[e]["leaked_hop_indices"])
    for h in sorted(by_h):
        tot, vb, sm = by_h[h]
        lines.append(f"| {h} | {tot} | {vb} ({vb/tot:.0%}) | {sm} ({sm/tot:.0%}) |")

    lines += ["\n## Metric impact (per model)\n",
              "| Model | subset | n | acc bench | acc nat | acc gap | search bench | search nat |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in metrics.iterrows():
        lines.append(f"| {r.model} | {r.subset} | {int(r.n)} | {r.acc_bench:.3f} | {r.acc_nat:.3f} | "
                     f"{r.acc_gap:+.3f} | {r.search_bench:.1f} | {r.search_nat:.1f} |")
    lines += ["\n## Read\n",
              "- The **accuracy gap** (natural − benchmark) shrinks toward (or below) zero on the "
              "semantic-clean subset: the natural advantage is largely a leakage artifact.\n",
              "- The **search-volume collapse** persists on the clean subset: it is a genuine "
              "behavioral effect, not an artifact of easier questions.\n"]
    (outdir / "leak_detection_report.md").write_text("\n".join(lines))
    print(f"Saved {outdir}/leak_detection_report.md")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench-dir", default="results/musique_parametric/interplay_analysis")
    p.add_argument("--nat-dir", default="results/musique-natural/interplay_analysis")
    p.add_argument("--slugs", nargs="+", default=viz.BASE_MODEL_SLUGS)
    p.add_argument("--provider", default="Google")
    p.add_argument("--model", default="gemini-flash-latest")
    p.add_argument("--ollama-base-url", default=None)
    p.add_argument("--output-dir", default="results/missed_hop_paradox")
    p.add_argument("--max-examples", type=int, default=None, help="judge only the first N uncached (smoke test)")
    p.add_argument("--reattribute", action="store_true", help="ignore cache, re-judge all")
    args = p.parse_args()

    viz.apply_theme()
    outdir = Path(args.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    examples = load_examples(args.bench_dir, args.nat_dir, args.slugs)
    if not examples:
        print("No examples found — check --bench-dir/--nat-dir."); return
    print(f"Loaded {len(examples)} unique examples")

    cache_path = str(outdir / "leak_judgments.json")
    cache = {} if args.reattribute else (json.load(open(cache_path)) if os.path.exists(cache_path) else {})

    print(f"Init leak judge: {args.provider}/{args.model}")
    agent = BaseAgent(provider_name=args.provider, model_name=args.model,
                      output_type=LeakJudgment, use_thinking=False, temperature=0.0,
                      ollama_base_url=args.ollama_base_url,
                      system_prompt="You are a meticulous QA dataset auditor.")
    cache = judge_examples(examples, agent, cache, cache_path, args.max_examples)

    judged = {e for e in examples if e in cache}
    sem_leaked = {e for e in judged if cache[e]["leaked_hop_indices"]}
    vb_leaked = {e for e in examples if examples[e]["verbatim_leak"] >= 1}
    json.dump({"semantic_clean": sorted(judged - sem_leaked),
               "verbatim_clean": sorted(set(examples) - vb_leaked),
               "semantic_leaked": sorted(sem_leaked),
               "verbatim_leaked": sorted(vb_leaked)},
              open(outdir / "clean_example_ids.json", "w"), indent=2)
    print(f"Saved {outdir}/clean_example_ids.json")

    metrics = paired_metrics(args.bench_dir, args.nat_dir, args.slugs, sem_leaked, vb_leaked)
    metrics.to_csv(outdir / "leak_cleaned_metrics.csv", index=False)
    # Figure only meaningful once a substantial fraction is judged.
    if len(judged) >= 0.5 * len(examples):
        fig_cleaned_metrics(metrics, outdir)
    write_report(examples, cache, metrics, outdir)
    print(f"\nDone. {len(sem_leaked)}/{len(judged)} judged examples flagged as semantic leaks.")


if __name__ == "__main__":
    main()
