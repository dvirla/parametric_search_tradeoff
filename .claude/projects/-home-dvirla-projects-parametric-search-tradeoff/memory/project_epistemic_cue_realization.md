---
name: project_epistemic_cue_realization
description: How to make epistemic-cue paraphrases (paraphrase_frames_cues.py) actually inject speaker-stance certainty/doubt, not hollow adverbs.
metadata:
  type: project
---

Lessons from fixing the epistemic booster/hedge rewrites in `scripts/paraphrase_frames_cues.py` (CUE_SPECS `epi_*` + the `epistemic` DIMENSION_DESCRIPTIONS).

**The cue must be the SPEAKER's stance, not a propositional adverb.** "What is *clearly* the full name...?" / "What conflict *possibly* took place...?" attribute (un)certainty to the asked entity — hollow, and for hedges they alter the factual claim/constraint. Realize it as the asker's own framing: boosters "Surely you know — <q>", "You must know this one:", "I'm certain there's a clear answer here — <q>"; hedges "I might be wrong, but <q>", "I could be misremembering, but <q>". Explicitly forbid in the instruction: bare adverbs inside the interrogative, detached "—this is a well-documented fact?" meta-tags, synonym stacking ("surely without a doubt well established"), and bare imperatives ("Tell me exactly" = request-directness, not epistemic stance).

**"Vary across questions" in the prompt is a no-op.** Each rewrite is an independent LLM call with an identical prompt, so the model emits its single most-likely realization every time → all 50 identical. User preference: get diversity from clearer instructions, NOT code-side rotation. The working approach: give a wide marker palette and instruct the model to pick the ONE that fits THIS sentence most naturally — diversity then emerges from the variety of the 50 source sentences, not cross-call coordination.

**Why:** without speaker-stance framing the cue is either absent (adverb on entity) or confounds the constraint (hedge adverbs loosen numbers/dates), breaking the single-cue OFAT design ([[project_frames_cue_equivalence_invariant]]).

**How to apply:** the cue-compliance auditor (`src/services/paraphrase_validation.py`, judge runs thinking ON) reads the dimension description for CHECK 3, so encoding "bare imperative/adverb does not count" there makes it reject those. The downstream behavioral result: even well-formed epistemic cues don't move gemini-3.1-pro search ([[project_frames_cue_sensitivity_null]]).
