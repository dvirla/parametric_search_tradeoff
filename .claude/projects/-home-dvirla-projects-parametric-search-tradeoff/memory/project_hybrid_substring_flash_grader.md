---
name: project_hybrid_substring_flash_grader
description: string-match + Gemini-3-Flash hybrid grader ≈ full gpt-oss judge; confirms no formal→natural2 accuracy drop
metadata: 
  node_type: memory
  type: project
  originSessionId: 183bfad5-04fe-4575-b607-542cd50e1d16
---

Two-tier grader (substring hit → correct; else Gemini-3-Flash equivalence judge) on musique-formal + musique-natural2, scored 2026-06-19. Script: `scripts/eval_substring_heuristic.py --semantic`; verdict cache `results/semantic_heuristic_cache.json`.

**Hybrid accuracy (formal / natural2):** gemini 0.628/0.618, nemotron 0.362/0.407, qwen 0.465/0.484. Lands within 1–3 pp of the gpt-oss reeval judge in every cell (substring alone was 12–18 pp too low due to low recall on verbose/aliased answers). Hybrid↔reeval-judge agreement 0.92–0.95.

**Key finding:** the formal→natural2 accuracy "drop" seen under pure string matching is a grading artifact — under semantic grading natural2 is indistinguishable from (nemotron/qwen slightly above) formal. Reinforces [[project_grading_artifact_accuracy_gap]].

Implementation notes: Gemini judge must run on a single shared asyncio event loop (`_LOOP`) — thread pools or per-file `asyncio.run` hit "bound to a different event loop" from pydantic-ai's async client. Config per [[project_gemini3_flash_judge_config]] (thinking ON, temp 1).
