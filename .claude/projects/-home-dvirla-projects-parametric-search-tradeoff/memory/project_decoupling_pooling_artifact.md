---
name: project-decoupling-pooling-artifact
description: The paper's search-vs-accuracy "decoupling" is a FRAMES+MedQA pooling artifact; revision plan in docs/PLAN_results_revision.md
metadata:
  type: project
---

Reviewed 2026-09-10. Two "no relationship" claims in `main.tex` are averages of significant
**opposite-signed** per-dataset effects, because FRAMES cells sit at 4.2-6.4 baseline search calls
while MedQA cells sit at 0.09-0.25:

- §`sec:decoupling` ρ=+0.168 (p=.113, n=90) = FRAMES **+0.356 (p=.011)** pooled with MedQA −0.310 (p=.051).
  Signed ρ(Δcalls, Δacc): FRAMES +0.723, HotpotQA +0.659, MedQA −0.008.
- `app:dual_metric` Pearson 0.13/0.16 = FRAMES **+0.354** pooled with MedQA **−0.453** (LLM judge).
  Even pooled, the *Spearman* is significant (+0.187 / +0.412) — the appendix only reports Pearson.

So volume and accuracy ARE coupled wherever a non-degenerate search policy exists; MedQA's
decoupling is a zero-search-floor artifact. This is the same pooling failure already fixed once for
entropy-vs-search (see [[project-cross-dataset-uncertainty]]). The **example-level** decoupling
claim does survive — HotpotQA's raised +0.06..+0.17 band is explained by its own RERUN noise floor
of +0.08, and no cue clears that floor after correction.

**Why:** the pooled nulls read as "brittleness is free", which undercuts the paper's own thesis;
the corrected reading (suppression costs real accuracy where retrieval is load-bearing) is stronger.

**How to apply:** full handoff plan with every number, provenance and edit location is
`docs/PLAN_results_revision.md`. Regenerated per-dataset FDR outputs (the other commissioned change)
are in `results/cue_briefing_per_dataset/`. Paper is at
`/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex` (rev 15495c1).
