---
name: project_natural2_paraphrase_leakage
description: natural2 paraphrases sometimes leak resolved intermediate hop entities (chain collapse), inflating the Missed cell without hurting accuracy
metadata:
  type: project
---

The natural2 MuSiQue paraphraser sometimes **resolves an intermediate hop and
names the bridge entity directly** in the question, collapsing a multi-hop chain
to fewer hops. E.g. a 4-hop question became "When was the **Kingdom of Saudi
Arabia** established?" (hops 0–2 pre-resolved); "the league the **New York
Yankees** belong to" (leaked the most-titled team); "Martin of **Aragon's**
death" (leaked the region).

**Why it matters:** this is a confounder behind the Test-1 disconnect — Missed (M)
cell rises under natural2 yet aggregate accuracy is flat. When a hop's answer is
leaked into the prompt, the model legitimately skips/"misses" it (no search) and
still answers correctly. So part of the natural2 Missed increase is a
data-quality artifact, not genuine behavioral change.

**Quantified verdict (2026-06-21):** `scripts/judge_paraphrase_leakage.py`
(gemini-3-flash, per-hop "does the paraphrase still REQUIRE this hop?") + the
entropy fix [[project_natural2_entropy_override_bug]] together show the
missed-hop increase is overwhelmingly artifact. ΔM (raw → entropy-fixed →
leakage-corrected): Gemini +0.056→+0.016→**+0.007**; Nemotron +0.053→+0.031→**−0.007**;
Qwen +0.068→+0.043→**+0.025**. Gemini/Nemotron real signal ≈ 0; only Qwen keeps a
small genuine residual (+0.025). This resolves the "M up, accuracy flat" disconnect:
the extra misses were certain hops mislabeled uncertain, or hops the paraphrase had
already pre-resolved (chain collapse) — neither can cost accuracy.

**Filtering in figures (2026-06-21):** `make_paper_figures.py --mode natural2
--drop-leaky-paraphrases` drops leaky examples from every natural2 figure (taxonomy,
cell-shift, calibration, redundancy, commitment, phrasing bars). Leakage is a property
of the paraphrase (shared across models), NOT the model — so `load_leaky_paraphrases`
returns example_ids and drops the UNION across models uniformly from BOTH phrasings
(judge is occasionally inconsistent on identical paraphrases; union is conservative).
101 distinct leaky example_ids → 498 clean examples/model (from 599). Post-filter ΔM
(quadrant-M metric): Gemini 0.021→0.010, Nemotron 0.059→0.038, Qwen 0.065→0.041.

**How to apply:** before citing the Missed-cell increase as behavioral, remove
artifact misses. Two detectors: `inspect_missed_leakage.py` (streamlit, verbatim
substring — LOWER BOUND, ~4–7/model); `judge_paraphrase_leakage.py` (LLM judge,
catches semantic/elided leaks like "Martin of Aragon", SEALs named directly,
"after British India was partitioned"). Judgments cached at
results/natural2_paper_figures/paraphrase_leakage_judgments.jsonl. Final-answer
correctness must come from paired_eval_files, not interplay aggregate_correct.
Related: [[project_missed_hop_paradox]], [[project_grading_artifact_accuracy_gap]].
