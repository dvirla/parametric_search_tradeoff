---
name: project_staleness_hops_not_gold
description: The `hops` column in frames_staleness_flash.csv is a stale-classification byproduct, NOT gold hop decomposition
metadata:
  type: feedback
---

The `hops` field in `results/frames_staleness_flash.csv` (per-step decomposition with
step/answer_type/is_time_sensitive/reason) was produced as a byproduct of the staleness
classifier (gemini-3-flash). It is NOT ground truth and carries no gold per-hop answers.

**Why:** FRAMES ships no gold hop decomposition. Using these proxy hops to ground a
leakage/equivalence judgement (e.g. validating the neutral anchor in the cue experiment) would
mislead — they're not authoritative about what the reasoning chain actually is.

**How to apply:** ground leak/equivalence audits ONLY on the original FRAMES prompt + gold final
answer, judged holistically (see `NeutralAnchorAudit` in paraphrase_validation.py and
scripts/validate_frames_benchmark.py). Use the staleness CSV ONLY for the is_stale filter
(non-stale 541). Related: [[project_frames_cue_equivalence_invariant]].
