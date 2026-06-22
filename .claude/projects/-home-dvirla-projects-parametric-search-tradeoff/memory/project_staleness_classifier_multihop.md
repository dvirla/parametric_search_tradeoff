---
name: project_staleness_classifier_multihop
description: "classify_musique_staleness.py generalized for MusiQue+FRAMES, gemini-3-flash structured per-hop, \"any theoretical change\" calibration"
metadata: 
  node_type: memory
  type: project
  originSessionId: b25ebb35-897b-4c6c-b874-4d4b7385d1d4
---

`scripts/classify_musique_staleness.py` was generalized (2026-06-20) beyond MusiQue:
- Reads BOTH the aggregate schema (`aggregate_question`) and the eval-result schema
  (`problem`/`correct_answer`/optional `example_id`). FRAMES files have no example_id
  → stable id derived from md5(question); hop_count blank. MusiQue hop_count parsed
  from the `Nhop` example_id prefix.
- Judge is now `--provider`/`--model` (default Google / `gemini-3-flash-preview`,
  thinking ON, temp 1) with **structured per-hop output** (`StalenessClassification`
  → list of `HopAssessment`). The prompt forces explicit decomposition of every hop,
  including hops hidden in relative clauses, and emits a `hops` JSON audit column.

**Calibration decision (deliberate):** "any theoretical change" — a hop is STALE if its
answer could change *even in principle*, including administrative designations that an
authority can reassign (county seat, capital, borders, official name). So e.g. "capital
of the county where Fort Deposit is located" → STALE. The user explicitly chose this over
a "realistic recent change" rule. **Why:** keeps a broad net for the parametric-vs-search
staleness filter; the prior gpt-oss prompt was being inconsistent on multi-hop, not too
loose. **How to apply:** don't "fix" county-seat/border hops being marked STALE — that's intended.

The old `results/musique_staleness.csv` + `data/musique_train_staleness.csv` were produced
by the PRIOR prompt (gpt-oss:20b, no decomposition); re-runs go to new output paths.
Related: [[project_grading_artifact_accuracy_gap]], [[project_dataset_registration]].
