---
name: project_frames_cue_equivalence_invariant
description: All FRAMES prompt-cue paraphrase variants must preserve information need + no leak, including the need-relevant cues
metadata:
  type: feedback
---

In the FRAMES prompt-cue sensitivity experiment, EVERY cue variant (epistemic, framing,
mitigation, AND the "need-relevant" ones — structural explicitness, output constraints) must
preserve the exact information need, resolve to the same gold answer, and never leak/short-circuit
an intermediate reasoning step.

**Why:** the experiment isolates how phrasing alone shifts the agent's retrieval policy. If
`structural_implicit` reveals an intermediate entity, or `output_*` alters question content, the
variant becomes genuinely easier and confounds the search-behavior signal — we'd no longer be
measuring phrasing.

**How to apply:** the cue-compliance audit (`CueComplianceAudit` in
[[project_dataset_registration]]'s paraphrase_validation.py) enforces `equivalent` and
`not leaks_intermediate` for ALL conditions, not just the surface-cue ones. Don't assume
need-relevant cues get a pass on equivalence/leak. structural_implicit changes how the chain is
surfaced, not what must be computed.
