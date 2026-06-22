---
name: project_gemini3_flash_judge_config
description: Gemini 3 Flash judge/grader runs must use thinking ON + temperature 1
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 183bfad5-04fe-4575-b607-542cd50e1d16
---

When using Gemini 3 Flash (`gemini-3-flash-preview`, provider `Google`) as an LLM judge/grader, run it with `use_thinking=True` and `temperature=1`.

**Why:** User guidance for this model — thinking improves equivalence judgments and temp 1 is the intended operating point. These match `BaseAgent`'s current defaults but should be set explicitly so judge behavior doesn't drift if defaults change.

**How to apply:** Pass `use_thinking=True, temperature=1` to `BaseAgent(provider_name="Google", model_name="gemini-3-flash-preview", ...)`. Already pinned in `scripts/eval_substring_heuristic.py` (the two-tier substring→Gemini semantic-equivalence grader). Related: [[project_grading_artifact_accuracy_gap]].
