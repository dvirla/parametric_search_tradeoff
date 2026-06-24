---
name: project_frames_cue_sensitivity_null
description: FRAMES prompt-cue experiment on gemini-3.1-pro — epistemic/phrasing/template cues don't move search; apparent effects were a query-template confound.
metadata:
  type: project
---

Controlled FRAMES prompt-cue sensitivity result for **gemini-3.1-pro-preview**, baseline search agent, local BM25 index (`data/frames_index`), n≈48 paired ids, dependent variable = `sampler_search_calls`. Done 2026-06-24.

**Headline: no single linguistic/template manipulation significantly changes retrieval.** Accuracy ~85–90% throughout; search medians 4–5 in every cell; SDs 8–14 (search count is dominated by question *difficulty*, maxes ~40–52). Paired Wilcoxon, all isolated factors null:
- Epistemic cue (strong boost / strong hedge vs neutral, all PLAIN): p = 0.38 / 0.41. **The epistemic stance cue is a clean null.**
- Phrasing verbose vs terse (both PLAIN): p = 0.96.
- QUERY_TEMPLATE vs PLAIN (template): p = 0.78–0.96.
- NATURAL "Please answer in 2-4 sentences" vs others: trends *down* ~−1.1 (p = 0.03–0.14) — the only directional signal, and it *reduces* search.

**The query-template confound (the real lesson).** The three "neutral-ish" FRAMES runs each use a DIFFERENT template (routing in `qa_eval.py`): `frames` → NATURAL ("...2-4 sentences"); `frames-benchmark` → default structured QUERY_TEMPLATE (Explanation/Exact-Answer/Confidence); `frames-cues` → PLAIN (passthrough, nothing appended — deliberately, so a cue isn't contaminated by an output-format cue). So `results/frames_baseline_*` (verbose+NATURAL) is NOT a clean neutral for the cue experiment. Early comparisons that crossed templates produced spurious "significant" effects.

**Compositional endpoint effect.** verbose+NATURAL (mean 6.88) vs terse+QUERY (mean 9.65) IS significant: Δ +2.77, p = 0.0099 (reproduced fresh; was p=0.0016 with the Jun-18 run). But this changes BOTH phrasing and template at once; each step alone is non-significant (drop-NATURAL +1.2, add-QUERY −0.06, terse-under-QUERY +1.6). Two small same-direction nudges stack at the extremes → significant endpoint gap with borrowed power. No single knob is responsible.

**Why:** confirms epistemic stance is `need_irrelevant` (doesn't shift retrieval); shows phrasing holds for *search behavior* too once template is fixed (project premise was about accuracy). The earlier apparent phrasing effect was entirely the template/backend packaging.

**How to apply:** (1) For any cross-condition FRAMES comparison, HOLD THE TEMPLATE FIXED — compare within PLAIN (run things as `frames-cues` conditions). To put verbose original phrasing on PLAIN, write it as a `frames_cues/<slug>.jsonl` with `text`=original_question. To get a specific template on arbitrary text: `frames-benchmark` dataset → QUERY_TEMPLATE; `frames-cues` → PLAIN; `frames` → NATURAL. (2) Don't compare new local-backend cue runs against old `frames_baseline`/`frames-benchmark` runs without matching template. (3) n≈48 is underpowered for the small per-step effects; medians + paired Wilcoxon, not means (means are variance-inflated). (4) `run_qa_eval_experiment.py` now has `--grader_model`/`--grader_provider` (default ollama/gpt-oss:20b) so the LLM grader can run remote (e.g. Google/gemini-3-flash-preview) when no local ollama. Runner passthrough: `GRADER_PROVIDER`/`GRADER_MODEL` env in `scripts/run_frames_cue_experiment.sh`.

Related: [[project_frames_cue_equivalence_invariant]], [[project_search_backends]], [[project_dataset_registration]], [[project_no_local_ollama]], [[project_gemini3_flash_judge_config]].
