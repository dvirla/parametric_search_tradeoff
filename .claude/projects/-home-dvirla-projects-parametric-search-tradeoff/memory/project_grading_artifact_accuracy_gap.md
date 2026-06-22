---
name: project_grading_artifact_accuracy_gap
description: Natural>formal MuSiQue accuracy advantage was a one-sided grading artifact; vanishes when both sides graded by gpt-oss:120b.
metadata: 
  node_type: memory
  type: project
  originSessionId: 6f701ead-9cc4-4d49-816c-4afb3a0e88ee
---

The apparent "natural phrasing is more accurate than formal" result on MuSiQue (600 val) was largely a **grading artifact**, not a phrasing effect. Discovered 2026-06-03.

**Cause:** the natural side had been re-graded with gpt-oss:120b (containment template), but the formal side still carried its original (gpt-oss:20b) grading. Both sides give free-form *prose* answers, so the weak grader under-counted both — but only natural got the fix. Re-grading the formal side with the *same* 120b containment grader raised formal accuracy sharply (gemini +22.7pts 0.435→0.662; qwen +10.7 0.376→0.483; nemotron +4.3).

**Corrected, apples-to-apples gaps (natural − formal, both graded by gpt-oss:120b):**
- gemini-3-pro: **−0.028** (ns, McNemar p=0.10)
- qwen3.5_122b: **−0.008** (ns, p=0.76)
- nemotron-3-nano_30b: **+0.070** (significant, p=0.0009) — only survivor
Deterministic normalized string-containment (gold-in-response) on both sides agrees directionally: −0.045 / −0.055 / +0.042.

**Why this matters / how to apply:**
- Drop any paper claim that natural phrasing is *more accurate*. For 2 of 3 models the gap is statistically zero.
- The core claim is *behavioral* (search-rate divergence) and is UNAFFECTED — in fact stronger: natural reaches equal accuracy with far less search (gemini 4.9 vs 15.2 searches/ex; qwen 1.3 vs 5.6), so it can't be dismissed as "natural was an easier task."
- Always grade both phrasings with the SAME judge. Use `--template natural` (containment) in re_evaluate_logs.py for prose answers; the standard extract-final-answer template under-counts prose.

**Leak-excluded (drop the 66 natural rewrites that leak a bridge entity verbatim, n=534), both graded 120b:**
- gemini: **−0.064** (***, favors FORMAL) — removing leak flips it significantly to formal
- qwen: **−0.047** (*, favors formal)
- nemotron: +0.047 (*) — natural advantage shrinks from full-set +0.070
So once graded fairly AND leak-removed, natural is NOT more accurate (worse for 2 of 3). The full-set natural edge lived in the leaked examples.

**Search behaviour is robust to all of this** (paired Wilcoxon, all p<1e-10, both full and leak-excluded): natural uses far fewer searches — gemini ~13→5, qwen ~5.6→1.4, nemotron ~5.7→3.3. Leak removal barely moves it. THIS is the paper's robust claim: same information need, equal-or-worse accuracy, 2–4× less search → behavioral non-generalization, not an easier task and not a leak artifact.

**Repro tooling:**
- `scripts/analyze_phrasing_corrected.py --output-dir results/phrasing_effect_corrected` — regenerates accuracy + search figures (full & leak-excluded) from the 120b-regraded logs. Self-contained; recomputes the 66 leak ids inline. Uses `src/viz.py` palette (BENCHMARK blue / NATURAL red).
- `scripts/extract_formal_aggregate.py` dumps `aggregate_result` from `results/musique_parametric/musique_parametric_uncertainty_<slug>.json` into the natural flat schema at `results/musique-formal/`, so the formal side flows through the same grader command.
- `scripts/re_evaluate_logs.py 'results/musique-formal/...run_1.json' --grader_model gpt-oss:120b` (default template now `natural`/containment; full bidirectional re-grade).
- Pairing natural↔formal is by `example_id` via `data/musique_val_natural.jsonl` (natural logs lack example_id; key on `text`→problem). Gold answers are 100% identical across the two phrasings.

Relates to [[project_missed_hop_paradox]] — the natural-wins-with-less-search paradox was partly this artifact; the leak story still explains the residual.
