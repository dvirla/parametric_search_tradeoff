---
name: project_missed_hop_paradox
description: "Why natural phrasing has higher accuracy with less search — resolves the \"cost of missed hops\" worry"
metadata: 
  node_type: memory
  type: project
  originSessionId: 6f701ead-9cc4-4d49-816c-4afb3a0e88ee
---

The "cost of a missed hop" (make_paper_figures.py Rung 1/2) looked undermined because under natural/user-like phrasing models search LESS yet score HIGHER than benchmark phrasing. Drill-down (`scripts/analyze_missed_hop_paradox.py` → `results/missed_hop_paradox/`) pairs every MuSiQue example across both phrasings and resolves it — neither explanation says missed hops are free:

1. **Benchmark flails, doesn't abstain.** In the paradox cell (natural ✓ / benchmark ✗) the benchmark run issues MORE searches yet still fails: Gemini 20.7 vs 4.6, Nemotron 5.5 vs 3.1, Qwen 5.0 vs 1.3. Some Gemini benchmark runs loop to 50–70 searches. Extra search here is unproductive looping, not healthy search.
2. **Natural rewrites pre-resolve hops.** They embed/paraphrase intermediate bridge gold answers into the prompt, collapsing a 4-hop chain into a ~1-hop lookup (e.g. benchmark "region immediately north of where Israel is located…" → natural "the Kingdom of Saudi Arabia; when was it established"). Verbatim bridge-leak +0.35–0.42 in paradox cell vs near-0 in mirror cell; semantic collapse is larger still.

**Caveat:** `truly_missed` is computed from an isolated parametric probe that never sees the leaked prompt, so it OVERcounts cost when the rewrite already gave away the bridge.

**How to apply:** Rung-1/2 cost claims should be made within matched difficulty (control for leaked/collapsed hops); attribute the cross-phrasing accuracy gap to prompt leakage + benchmark looping, NOT to a benefit of searching less. Relates to [[project_no_local_ollama]] (analysis runs locally on precomputed artifacts, no LLM calls).

**Outcome-breakdown view (added 2026-06-07):** `scripts/plot_missed_hop_outcome_breakdown.py` → `results/natural2_paper_figures/missed_hop_outcome_{counts,rate,share}.png` + `_breakdown.csv` splits every truly-missed hop by the final-answer outcome of its example, across Benchmark/Nat1/Nat2. **Outcome correctness MUST come from the phrasing-specific eval run via the results DB (`rdb.load_eval`, `--grading` default `reevaluated`), NOT from the interplay summary's `aggregate_correct`** — see [[project_natural2_interplay_aggregate_bug]]. Resolves "more misses, same accuracy": the *extra* misses in natural phrasing land disproportionately inside still-correct answers. Under reevaluated grading, share of truly-missed hops sitting in a correct answer rises Benchmark→Nat1 (Gemini 48%→52%, Nemotron 14%→31%, Qwen 7%→37%); Nat2 intermediate (50/25/21%). Accuracy is flat across phrasings under reevaluated grading (the natural>formal gap was the artifact, see [[project_grading_artifact_accuracy_gap]]), yet misses still rise sharply in natural — so the extra misses are largely harmless (bridge already leaked/cross-hop covered), which is why accuracy holds.
