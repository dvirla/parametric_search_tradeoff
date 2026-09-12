---
name: project-hotpotqa-mechanism-run
description: HotpotQA mechanism classification was run 2026-09-11 (50/54 level-shift-only); paper's sec:mechanism numbers are stale
metadata:
  type: project
---

The necessity/mechanism regression needs only the **cue-free** entropy probe (a pre-treatment
covariate) plus per-condition search calls — **not** entropy-under-cue. That misreading made this
look blocked for weeks. HotpotQA had every input; it was a ~1-hour code change.

Results: level-shift-only (calibration intact) = **58/70 FRAMES, 67/73 MedQA, 50/54 HotpotQA**.
Five of six HotpotQA models are 100% level-shift; all 4 breaking cells are `nemotron-3-nano:30b`
and are labelled "inverted" only because its plain slope is already −0.128 — a near-zero-denominator
sign artifact, the cue actually moves the slope *up*.

**The paper's "50 of 62 on FRAMES, all 60 on MedQA" is stale** — the mechanism CSV lagged its own
upstream interaction CSV. MedQA is no longer "all": 6 eroded cells, all `nemotron-cascade-2`.

**Two FDR-family traps fixed, same species as [[project-decoupling-pooling-artifact]]:**
`analyze_necessity_vs_template_search_5run.py` corrected BH globally, so adding a dataset shifted
all 143 existing q-values; it and the roster are now per-dataset (FRAMES/MedQA byte-identical,
0 classification changes).

**How to apply:** full detail in `docs/PLAN_results_revision.md` §6.1; §3.2 there now shows the
3-dataset coupling table (HotpotQA +0.521 is the *strongest*, pooled-all-3 +0.275 p=.0006).
