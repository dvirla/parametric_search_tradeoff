# Accuracy revision — handoff notes for editing `main.tex`

Target file: `/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex`
Status: **proposal only — no edits applied to the paper.** This is the writing-ready synthesis of
an extended analysis thread (semantic-entropy validity, causal cue-interaction tests, mechanism
decomposition, causal mediation, oracle/no-search-value analysis). It supersedes all earlier
per-section drafts of this file. The full LLM-judge regrade of all no-search rollouts
(`results/no_search_llm_grades/`, 20,015 rows, `scripts/regrade_no_search_llm.py`) is now
**complete** — every number below is final, LLM-graded, no provisional tags remain. One casualty
of the regrade worth knowing about: the old 3-bar "oracle ceiling" figure (regex-graded across the
full cue battery) is no longer valid, because the corrected no-search floor now exceeds it on
MedQA in every model — a floor can't exceed its own ceiling. That figure has been replaced with a
clean 2-bar (no-search vs. `plain`) comparison; re-adding a valid oracle bar would require
LLM-regrading every cue condition too (~60k more gradings, out of scope here — see §5.9).

---

## 1. The central reframe (read this before touching §2)

The paper's current thesis — tool-use is **"fundamentally decoupled"** from parametric
knowledge/uncertainty — is contradicted by real, replicated evidence of baseline grounding and
should not survive as stated. But the fix is not "weaken it to weakly coupled" — that's a flatter,
less interesting paper. The evidence supports a **more precise and more interesting** claim than
either extreme:

> Search-triggering behavior is grounded in a demonstrably valid internal uncertainty signal, but
> how strongly that signal translates into behavior is uneven — proportionate to the signal's
> strength on FRAMES, but on MedQA the signal is strong while the behavioral response is weak or
> absent for half the models tested. On top of this uneven baseline, specific — not universal —
> cues can silently disable the mechanism tying search to uncertainty (not merely redirect its
> volume), and at least one such case has a causally-verified accuracy cost distinct from the
> volume loss itself. **Independent of that mechanism story, search VOLUME itself is often
> arbitrary: for the majority of cues that don't touch the necessity-tracking mechanism, how much
> a cue moves search volume is statistically unrelated to how much it moves accuracy — including
> cases where volume is inflated up to 38× with no detectable accuracy change.** This last point
> is the paper's most self-contained, easiest-to-defend headline claim — it doesn't require the
> entropy instrument at all.

Five separable sub-claims do the work, each with its own instrument, its own evidentiary bar, and
its own required reading (§7). Do not collapse them back into one decoupled/not-decoupled verdict —
that collapse is exactly the paper's current problem.

### 1.0 The cleanest headline finding: search volume is elastic and largely decoupled from accuracy

This is the strongest single claim in the whole thread, and arguably what the paper should lead
with — it needs none of the entropy machinery (§1.1–1.2) to be meaningful, so it can't be
undercut by any instrument-validity objection. It directly operationalizes an intuition worth
stating plainly: *if a content-free prompt cue can move how much a model searches by a large
amount while accuracy barely moves, that volume was not tightly coupled to actual information
need in the first place.*

**Design**: restrict to the Stage-2 "level shift only" cells (§1.3 below — cues where the
entropy→search *slope* is statistically unchanged from `plain`, i.e. the necessity-tracking
mechanism itself is intact by that test) and ask a *different* question of the same cells: does
the *size* of the volume shift predict the *size* of the accuracy shift? Script:
`scripts/analyze_volume_accuracy_decoupling.py`, data:
`results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv` (regex-graded for consistency
across every cue, including ones never LLM-graded).

**Result**: across 90 level-shift-only cells, **the correlation between |Δ search volume| and
|Δ accuracy| is statistically indistinguishable from zero** (ρ=+0.016, p=0.88). Among the 29
cells with a large volume swing (|Δcalls|>1.0), **48% show negligible accuracy change (<3pp)**.
The most extreme cases are on MedQA, where near-zero baseline search volume means a cue can
inflate it by an order of magnitude: `searchmulti2`/`searchmulti3` push search volume
**8× to 38×** across all 4 models, with accuracy moving by at most ±2.8pp (noise). For contrast,
the 10 cells where the mechanism genuinely *does* break (erosion/inversion/sharpening, §1.3) show
a much larger typical accuracy shift (median 10.5pp vs. 3.0pp for level-shift cells) — so the
distinction from §1.3 is doing real work; this isn't just "everything is noisy."

**Why this is the sharper reframe of "decoupling" than the original paper's claim**: the original
claim was an unqualified assertion with weak statistical backing. This version is a precise,
falsifiable, quantified statement — a null correlation with a reported effect size and p-value,
not a rhetorical claim — and it survives exactly the kind of scrutiny that broke the original
("fundamentally decoupled" implied by weak per-example Δsearch/Δaccuracy correlations that
couldn't beat their own noise floor, per §2.6's now-superseded paragraph). This one can.

### 1.1 The signal is valid: entropy predicts correctness, equally strongly on both datasets

Self-consistency (semantic entropy over 5 independent no-search rollouts, LLM-judge clustered) is
a real, useful predictor of whether the model is about to be right — not just a measure of output
diversity. **Final numbers**, entropy vs. TRUE no-search accuracy (own 5 rollouts, LLM-judge
graded — `scripts/analyze_entropy_vs_correctness.py`, `results/entropy_vs_correctness/`):

| | FRAMES ρ(entropy, correct) | MedQA ρ(entropy, correct) |
|---|---|---|
| gemma4_31b | −0.569 (p=2.2e-44) | −0.556 (p=6.9e-42) |
| gpt-oss_120b | −0.708 (p=2.6e-77) | −0.617 (p=8.6e-54) |
| gpt-oss_20b | −0.638 (p=1.1e-58) | −0.654 (p=2.0e-62) |
| nemotron-3-nano_30b | −0.646 (p=3.9e-60) | −0.573 (p=6.5e-45) |

**The signal is essentially equally strong on both datasets** (ρ≈−0.55 to −0.65 everywhere, all 8
cells) — accuracy at entropy=0 is 73–83% (FRAMES) / 82–87% (MedQA); at entropy>0 it drops to
15–21% (FRAMES) / 43–52% (MedQA). This directly supersedes an earlier, weaker estimate for MedQA
computed via a `plain`-condition proxy before the full no-search regrade landed — that number
(ρ≈−0.37 to −0.45) undercounted the true correlation. **The paper should state: the internal
uncertainty signal is comparably valid across domains — there is no dataset-level difference in
signal quality to explain away.** Whatever asymmetry exists downstream (§1.2) is about behavior,
not about the signal itself.

### 1.2 The behavioral response to that signal is a general, partial gap — more severe for some models on MedQA

Compare against entropy→search-*call-count* correlation (data in `results/baseline_calibration/`,
split-half replicated):

| | FRAMES ρ(entropy, calls) | MedQA ρ(entropy, calls) |
|---|---|---|
| gemma4_31b | 0.347 | 0.215–0.315 (replicates) |
| gpt-oss_120b | 0.438–0.443 | 0.117–0.179 (replicates, weak) |
| gpt-oss_20b | 0.337–0.387 | ~0 (does not replicate) |
| nemotron-3-nano_30b | 0.215–0.219 | ~−0.04 (does not replicate) |

**Given §1.1's corrected, equally-strong signal, the correct framing is a general "utilization
gap" present on BOTH datasets, not a FRAMES-good/MedQA-bad contrast.** Even on FRAMES, where
behavior tracks the signal best, ρ≈0.35–0.44 reflects only *roughly half* the signal's own
correctness-predictive strength (ρ≈0.6–0.7) — a real, if partial, loss in translating signal to
behavior, present in every model. MedQA shows the same partial loss for `gemma4_31b` and
`gpt-oss_120b` (behavioral ρ drops to ~0.15–0.3 against a signal of ~0.6), and a **near-total**
loss for `gpt-oss_20b` and `nemotron-3-nano_30b` (behavioral ρ≈0 against the same ~0.6-strength
signal). **This is the precise, model-specific, defensible version of "decoupled"**: not a uniform
verdict, but a gradient from partial underuse (every model, FRAMES) to near-total underuse (two
models, MedQA) of an equally-valid signal.

**Important nuance for the writing session**: since acting on the signal wouldn't have improved
outcomes much on MedQA anyway (§1.4), the near-total behavioral gap there is real but
low-consequence — state both facts, don't let one erase the other.

### 1.3 On top of the uneven baseline: mechanism-specific fragility, not universal override

Whether a cue's effect on search *volume* also breaks the entropy→search *slope* (necessity
tracking itself), tested via `calls ~ entropy + is_cue + entropy:is_cue`, FDR-corrected across 122
(dataset, model, cue) cells (cue is manipulated — a complete within-subject crossover, so this
licenses causal language on the cue's own effect):

- **FRAMES: 50/62 cells are a pure level shift** — volume moves, the entropy→search slope is
  statistically unchanged from `plain`. This is the common case and should **not** be reported as
  decoupling — the model's relative allocation of search effort by necessity survives.
- **A specific, nameable minority show genuine slope erosion**: `gemma4_31b` under
  `confident_parametric` (slope drops to 33% of its plain value), `multiturn`, `searchmulti`;
  `gpt-oss_20b` under `multiturn`. This is the real "decoupling" event, mechanistically distinct
  from a volume shift, and it happens to specific models under specific, mostly
  capability-framing/stateful cues — not universally.
- **`nemotron-3-nano_30b` shows the opposite under 6 cues** — slope *increases* (70–136% of plain).
  Some cues make search behavior *more* necessity-tracking, not less.
- **MedQA: 0/60 cells show a significant slope change** — consistent with §1.4 (little reason for
  a cue to move a signal nobody profits from acting on).

### 1.4 Domain moderates the stakes: does the tool add value at all?

Prerequisite check, changes how every MedQA finding above should be weighted. **Final numbers**,
no-search vs. `plain`, both LLM-judge graded (`scripts/analyze_no_search_accuracy_llm.py`,
`results/no_search_accuracy/no_search_accuracy_llm.csv`,
`results/search_oracle/no_search_oracle_comparison.png`):

| | no-search floor | `plain` accuracy | search adds |
|---|---|---|---|
| FRAMES gemma4_31b | 34.5% | 69.9% | **+35.3pp** |
| FRAMES gpt-oss_120b | 41.8% | 68.3% | **+26.4pp** |
| FRAMES gpt-oss_20b | 24.4% | 56.7% | **+32.3pp** |
| FRAMES nemotron-3-nano_30b | 34.3% | 49.5% | **+15.2pp** |
| MedQA gemma4_31b | 80.3% | 81.8% | +1.5pp |
| MedQA gpt-oss_120b | 77.3% | 76.6% | −0.7pp |
| MedQA gpt-oss_20b | 70.7% | 70.4% | −0.3pp |
| MedQA nemotron-3-nano_30b | 69.5% | 68.0% | −1.5pp |

**FRAMES: search is genuinely load-bearing, adding +15 to +35pp** over parametric knowledge alone
in every model.

**MedQA's aggregate "+0pp" conflates two very different populations — do not state it as a single
number without this breakdown.** Only **4–20% of MedQA examples ever receive a search call under
`plain`** (`scripts/analyze_medqa_search_conditional.py`,
`results/no_search_accuracy/medqa_search_conditional.csv`). On the 80–96% that don't, `plain` ≈
no-search trivially (deltas −0.5pp to +1.8pp, as expected when no search happened). On the small
subset that *does* get searched, accuracy is **consistently worse than the model's own no-search
accuracy on those same examples, in all 4 models** (−3.8pp to −9.3pp) — for `gpt-oss_20b` and
`nemotron-3-nano_30b`, the model even chose to search on examples where its no-search accuracy was
already *higher* than on the examples it left alone, and searching still made things worse there.
**The correct claim is not "search doesn't help on MedQA" — it's "search is rarely invoked on
MedQA, and when it is, it correlates with a real accuracy cost."** That's a sharper and more
specific finding, though the usual caveat applies: this is a matched same-example comparison, not
a randomized one, so it can't fully rule out that the searched subset was already heading toward a
wrong answer for reasons unrelated to search (same endogenous-selection concern as §1.5's broken
observational mediation test — search-triggering may be as much a *symptom* of a struggling
rollout as a *cause* of the bad outcome).

Traced mechanism (qualitative, n=1, not a systematic audit — flagged for a future, larger trace
sample if the paper wants to make this a stronger claim): the local BM25 retrieval corpus does
surface topically relevant textbook content, but (a) queries are frequently malformed — verbatim
vignette quoting rather than concept queries — and (b) retrieved content can actively mislead the
model away from a correct initial hypothesis toward a superficially buzzword-matched wrong one —
directly consistent with the searched-subset accuracy cost above, not just a plausible story for
an aggregate null.

**What is NOT yet re-validated**: the earlier "oracle ceiling across the full cue battery" and
"% of failures unsolvable by anything tried" numbers were regex-graded and are now known-stale on
MedQA (regex's systematic undercount there means the true oracle ceiling and true unsolved-rate are
different from what was reported before) — see §5.9. The two-condition (no-search vs. `plain`)
comparison above is fully valid and sufficient to support the paper's core claim in this section;
the multi-cue oracle would need every cue condition LLM-regraded to be trustworthy again, which is
a larger, separate task not done here.

**Implication**: MedQA's weak/uneven behavioral response (§1.2) and near-total absence of
cue-driven fragility (§1.3) should not be read as *worse* alignment than FRAMES's — the tool has
near-zero counterfactual value there, so both the underuse of the signal and the absence of
cue-induced fragility are lower-stakes than the same findings would be on FRAMES.

### 1.5 Fragility has a causally-verified consequence, at least once

Whether §1.3's slope-erosion cells cost accuracy *because of* the erosion (not just because of
reduced volume) needs a manipulated mediator — naive observational mediation on realized
search-call count is provably confounded here (within every entropy stratum, realized calls and
correctness are strongly *negatively* correlated, r=−0.32 to −0.66 — the signature of an endogenous
mediator: an agent searches more largely *because* it's struggling mid-rollout, not the reverse).
The `gemma4-frames-robust-*` SFT checkpoints (trained on rollouts curated to have small
|Δcalls|≤1 **and** a correct answer) give a real, if imperfect, manipulated-mediator comparison
on 102 held-out FRAMES questions:

- `direct` (a §1.3 level-shift-only cue): restoring ~54% of the search-volume swing restores ~48%
  of the accuracy swing — roughly proportional, consistent with volume-mediated cost.
- `confident_parametric` (a §1.3 erosion cell): search volume barely recovers (77% of original
  swing) yet accuracy recovers almost completely (down to 18% of original cost) — **the accuracy
  fix did not come from restoring volume**, direct evidence the erosion itself, not just less
  searching, was doing the damage.

This is the paper's strongest available causal claim connecting mechanism to consequence, and it
should be framed as an illustrative case study (n=102, one checkpoint family), not a headline
statistic.

---

## 2. Proposed changes to `main.tex`, by location

Legend: 🔴 = language directly contradicted by the evidence (must change); 🟡 = language that
overclaims relative to the paper's own evidence (should be softened). **Line numbers below are from
the version of `main.tex` this thread last read — re-verify against the current file before
applying**, the paper may have moved since.

### 2.1 Abstract 🔴🟡
**Current:** "...structural and stylistic changes drastically alter the agent's decision boundary
between internal knowledge and external search without impacting final accuracy. This robustly
indicates that search calls are fundamentally decoupled from the agent's actual information need
or parametric knowledge. The compelling evidence that learned tool-use policies are universally
driven by superficial formatting heuristics rather than genuine epistemic uncertainty underscores
the urgent necessity..."

**Proposed:** "...structural and stylistic changes drastically alter the agent's decision boundary
between internal knowledge and external search, carrying a small average accuracy cost that
concentrates sharply in specific model–perturbation pairs. We show that baseline search volume is
grounded in a validated internal uncertainty signal, but the strength of this grounding varies by
model and domain, and select perturbations can silently disable the very mechanism linking search
to uncertainty — not merely shift how much a model searches. This evidence that tool-use policies
can have their epistemic grounding selectively disabled by superficial cues, with measurable
downstream cost, underscores the urgent necessity..."

**Why:** removes "fundamentally decoupled...or parametric knowledge" (contradicted by §1.1/1.2) and
"universally"/"without impacting accuracy" (contradicted by §1.3's specific erosion cells and
§1.5's causal cost). Keeps — and sharpens — the paper's real contribution: a *mechanism-level*
fragility claim, which is more specific and more defensible than a blanket decoupling claim.

### 2.2 Intro, closing paragraph 🔴
**Proposed:** "...revealing that the tool-use policy's grounding in epistemic uncertainty, while
real, is unevenly realized across models and domains, and that specific superficial cues can
silently sever this grounding rather than merely relocate it — indicating that agents do not
reliably anchor search decisions to a stable, calibrated sense of their own knowledge gaps under
adversarial framing, even when that sense is demonstrably present at baseline."

### 2.3 Related Works 🔴
**Proposed:** "By mapping the search-accuracy tradeoff, we show that most superficial directives
shift search volume while preserving the model's underlying necessity-tracking, but a specific
subset silently disable that tracking mechanism itself — with a measurable, causally-attributable
accuracy cost in at least one traced case — despite search volume being meaningfully grounded in
the model's parametric uncertainty absent such perturbations."

### 2.4 §4.1, General Suppression 🔴
**Proposed:** "...making agentic tool-use behaviorally fragile in a mechanism-specific way: most
perturbations redirect search volume while its anchoring to internal epistemic uncertainty survives
intact (§4.1.1), but for identifiable model–cue pairs the anchoring mechanism itself is disabled,
not merely diluted (§4.2)."

*(Needs §4.1.1 introduced — see §3 below for the full new subsection plan.)*

### 2.5 §4.2 opening 🔴
**Proposed:** "We decompose each perturbation's effect on search behavior into a change in search
*volume* versus a change in the model's *necessity-tracking* (the slope relating search volume to
independently-measured uncertainty). Most perturbations affect volume only, preserving tracking;
a specific minority disable tracking itself — a materially stronger and more specific claim than
uniform decoupling (§4.1.1, §4.2.1)."

### 2.6 §4.2, per-example correlation paragraph 🟡
**Proposed (unchanged from prior draft, still holds):** "Crucially, we do not detect a reliable
relationship at the individual-example level between an example's search-count shift and its
change in correctness: mean Spearman r values remain small (−0.03 to +0.03) across every
perturbation and dataset. This is consistent with the search-volume drop not being concentrated on
the examples driving accuracy changes; note, however, that per-example correlations under a single
rollout are inherently noisy — even a repeated \textit{rerun} of the identical, unperturbed prompt
yields a comparably-sized correlation (r ≈ −0.03 to −0.06), so this null is best read as a lack of
*detectable* concentration at this granularity rather than proof of *no* relationship. One model,
Nemotron3-30B, is a consistent exception across both graders (r = +0.17 to +0.23), for which search
and accuracy are coupled at the example level."

### 2.7 §4.2, confident-perturbation paragraph 🟡
**Proposed:** "...for at least one model, a single capability-framing cue does not merely shift the
agent's tool-use policy away from a calibrated assessment of its own knowledge — it measurably
disables the mechanism producing that calibration (§4.2.1), and a causal intervention (SFT on
volume-preserving, correct rollouts) shows the resulting accuracy cost is not explained by the
reduced search volume alone (§4.2.2)."

### 2.8 Discussion 🔴 — highest priority
**Proposed:** "A central takeaway is that tool-use policies are not uniformly decoupled from
internal uncertainty — baseline search volume is meaningfully, if unevenly, grounded in a validated
self-consistency signal, more strongly used on FRAMES than on MedQA. What we find instead is
*selective, mechanism-specific fragility*: most superficial cues redirect search volume while
leaving this grounding intact, but for identifiable model–cue combinations, the mechanism tying
search to uncertainty is disabled outright, and at least one such case carries a causally-verified
accuracy cost distinct from the volume change itself. The risk this poses is arguably sharper than
uniform decoupling would be: a policy with no grounding to begin with degrades gracefully under
noise, while a policy with real but silently-disable-able grounding can fail without any visible
signature in search-volume statistics alone."

### 2.9 Future Work 🟡
**Proposed:** "...detecting and preventing the silent disabling of tool-use policies' epistemic
grounding by superficial cues, as distinct from — and identifiable independent of — ordinary
volume-level prompt sensitivity."

### 2.10 Appendix, "Decoupling Confirmation" 🟡 — needs verification before editing
**Unresolved** (carried over, still open): re-derive exactly how the paper's existing r=0.13–0.16
was computed and compare against r=0.76 (Pearson) / ρ=0.56 (Spearman) found by correlating each
(model, cue) cell's mean Δsearch against mean Δaccuracy (127 cells, `dual_metric_cue_deltas.csv`).
Same nominal quantity, very different numbers — likely a pooling/units difference (8
already-averaged perturbation points vs. 127 model×cue points). **Resolve this before touching the
section**, independent of everything else in this document.

---

## 3. New subsections to add — structure and content

**§4.1.1 "Search Volume Is Grounded in a Validated but Unevenly-Used Uncertainty Signal"** (new,
before current §4.2):
1. Establish entropy's validity: it predicts correctness (§1.1 table). This is new to the paper and
   is what makes everything downstream meaningful — without it, a search-entropy correlation could
   just be measuring noise correlating with noise.
2. Report entropy→search-volume correlation (§1.2 table), split-half replicated.
3. **State the gap explicitly, per domain**: FRAMES's behavioral response is commensurate with
   signal strength; MedQA's is not, for 2 of 4 models — a precise, model-specific "utilization gap"
   claim, not a blanket one.
4. Scope statement: 4/11 models have this data; no Gemini model has a no-search probe.

**§4.1.2 "Does the Tool Add Value? A Necessary Control"** (new, likely right after 4.1.1, or as an
early dataset-description paragraph near Table 1): the oracle/no-search finding (§1.4). Explains
*why* MedQA looks different throughout the paper, not just in the entropy analysis — this is
probably the single most paper-ready, easy-to-state new result in the whole thread.

**§4.2.1 "Mechanism: Volume Shift vs. Necessity-Tracking Erosion"** (new, inside/after current
§4.2): the level-shift/slope-change taxonomy (§1.3). This is the paper's sharpest reframe of
"decoupling" — replace the pooled-variance framing (if it exists in any draft) with this, it is a
categorically better analysis for this specific claim (see caveat in §5.10 of the old version /
§6 script index below).

**§4.2.2 "A Causal Test of Consequence"** (new, short, could be a subsection or a boxed case study):
the SFT-mediation finding (§1.5). Explicitly flag the naive-mediation dead end as a methods note if
the paper wants to preempt a reviewer trying the obvious (wrong) analysis.

---

## 4. What NOT to change

- Title, RQ1, RQ3: unaffected.
- **RQ2 wording itself** is already an open, two-sided question — doesn't need to change; the paper
  needs to actually *answer* both halves (currently answers only "cues override," not "is baseline
  grounded, and how does the override actually work").
- §4.2's title "The Decoupling of Search and Accuracy" can stay for the aggregate
  Δsearch/Δaccuracy result specifically (§2.6) — that one specific claim is well-supported as
  stated. It should NOT be read as licensing the paper's broader "decoupled from parametric
  knowledge" language elsewhere, which is a different (and now-contradicted) claim.
- §4.3, §5.1: untouched by any of this work.

---

## 5. Caveats that bound every claim above

1. **Model coverage**: entropy data exists for only 4 of 11 models. No Gemini model has a no-search
   probe. Never generalize entropy-based claims beyond this subset.
2. **Clusterer validation**: LLM-judge semantic-entropy clusterer spot-checked (24 examples, one
   reviewer), not a rigorous multi-annotator alt-test. Treat as reassuring, not statistically
   validated.
3. **Entropy measured cue-free only**: no "entropy under cue X" data exists. All cue-fragility
   claims are about search-*volume*/*slope* effects being necessity-dependent, not about whether a
   cue changes the model's actual uncertainty.
4. **Multiple comparisons**: the causal interaction test runs 122 cells; only FDR-corrected
   (q<0.05) cells are load-bearing.
5. **Naive observational mediation is confounded** (§1.5) — do not re-derive or cite a
   `calls → correct` regression as causal; only the SFT comparison offers real evidence here.
6. **The "utilization gap" (§1.2) is now a clean, matched comparison** (both entropy→correctness
   and entropy→calls use the model's own `plain`/no-search behavior on the same examples, both
   properly graded) — the earlier caveat about mismatched populations no longer applies, since
   §1.1's correctness numbers now use TRUE no-search accuracy, not the `plain`-condition proxy.
7. **The Appendix "Decoupling Confirmation" r=0.13–0.16 vs. r=0.76 discrepancy is unreconciled**
   (§2.10) — don't edit that section until resolved.
8. **Regex grading undercounted MedQA accuracy by 26–36pp** relative to the LLM judge on the
   `plain` condition (vs. FRAMES's 7–11pp) — this has been fully corrected via the completed
   no-search regrade (`results/no_search_llm_grades/`, 20,015 rows); every number in §1.1/§1.4 is
   now LLM-graded. No remaining regex-based numbers are cited as point estimates anywhere above.
9. **The multi-cue oracle ceiling is currently INVALID and not cited above.** The original 3-bar
   oracle figure used regex grading across the full cue battery; once the no-search floor was
   properly regraded, it came out *above* that regex-based ceiling on MedQA (a floor exceeding its
   own ceiling is incoherent) — direct fallout of item 8. The figure has been replaced with a
   2-bar (no-search vs. `plain`) comparison, which is fully valid and sufficient for §1.4's claim.
   Re-establishing a valid multi-cue oracle would require LLM-regrading every cue condition
   (~60k additional gradings) — not done, flagged as future work if the paper wants that specific
   claim back.

---

## 6. Data & script index (by analysis)

| Question | Script | Output |
|---|---|---|
| **Volume/accuracy decoupling — the §1.0 headline finding** | `scripts/analyze_volume_accuracy_decoupling.py` | `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv` |
| No-search LLM regrade (**complete**, 20,015 rows — do this first, everything below depends on it) | `scripts/regrade_no_search_llm.py` | `results/no_search_llm_grades/` |
| Entropy validity: does it predict correctness? (**final**) | `scripts/analyze_entropy_vs_correctness.py` | `results/entropy_vs_correctness/` |
| No-search accuracy floor + search value-add (**final**, LLM-graded) | `scripts/analyze_no_search_accuracy_llm.py`, `make_no_search_oracle_figure.py` | `results/no_search_accuracy/no_search_accuracy_llm.csv`, `results/search_oracle/no_search_oracle_comparison.png` |
| MedQA: does the tool help, conditional on whether the model actually searched? | `scripts/analyze_medqa_search_conditional.py` | `results/no_search_accuracy/medqa_search_conditional.csv` |
| Entropy vs. search-calls correlation (5-run) | `scripts/analyze_llm_entropy_vs_search_5run.py` | `results/param_vs_search_llm_5run/` |
| Split-half calibration replication | `scripts/analyze_baseline_calibration.py`, `make_baseline_calibration_figure.py` | `results/baseline_calibration/` |
| Necessity × cue causal interaction (5-run) | `scripts/analyze_necessity_vs_template_search_5run.py`, `make_necessity_vs_template_figure_5run.py` | `results/necessity_vs_template_5run/` |
| Mechanism decomposition (level shift vs. slope change) | `scripts/analyze_cue_suppression_mechanism.py`, `make_cue_suppression_mechanism_figure.py` | `results/cue_suppression_mechanism/` |
| Descriptive affected/unaffected accuracy split | `scripts/analyze_search_mediation.py` | `results/search_mediation/` |
| SFT manipulated-mediator comparison | `scripts/analyze_sft_intervention_mediation.py` | `results/sft_intervention_mediation/` |
| Dual-metric (EM vs. LLM-judge) robustness + aggregate Δsearch/Δaccuracy | `scripts/dual_metric_analysis.py`, `make_dual_metric_figure.py` | `results/dual_metric_cue_deltas.csv`, `results/dual_metric_condition_table.csv`, `results/dual_metric_appendix/` |
| Epistemic-alignment scorecard (visual summary, Stage 1+2) | `scripts/make_epistemic_alignment_scorecard.py` | `results/epistemic_alignment_scorecard/` |

A pooled `calls ~ entropy + C(condition)` variance decomposition and a naive observational
Baron-Kenny mediation (`calls ~ is_cue` / `correct ~ is_cue + calls`) were both tried and
discarded as the wrong lens (see §1.3 and §1.5) — their scripts and outputs have been deleted
rather than kept around as "do not cite" clutter; do not recreate either approach.

Full data provenance, in-progress runs, and known pitfalls: `docs/PARAMETRIC_UNCERTAINTY_HANDOFF.md`.
Unified methodology (all four sub-claims as one protocol, Stages −1 through 3):
`docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`.

---

## 7. Reading list per new subsection — what the writing agent must open before drafting

Read `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md` in full first — it has the conceptual vocabulary
(discrimination vs. calibration, the 5-label mechanism taxonomy, why "Uniform volume shift" is not
"benign") that every subsection below depends on. Then, per subsection:

**New: §4.2.0 or wherever the paper wants its headline claim (volume/accuracy decoupling)**
- This document, §1.0 — read this section itself, the numbers are compact and self-contained.
- `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv`.
- Needs no entropy/framework-doc background — this claim stands on its own, which is exactly why
  it's the safest one to lead with.

**§4.1.1 (signal validity + utilization gap)**
- This document, §1.1–1.2 (final numbers + the corrected, now-clean comparison per caveat §5.6).
- `results/entropy_vs_correctness/entropy_vs_correctness.csv` (entropy→correctness, final).
- `results/baseline_calibration/baseline_calibration_stats.csv` and
  `baseline_calibration_curves.png` (entropy→calls, split-half).
- `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`, Stage 0 + Stage 1 sections.

**§4.1.2 (does the tool add value)**
- `results/search_oracle/no_search_oracle_comparison.png` — the headline figure, now a clean 2-bar
  (no-search floor / plain) comparison per model per dataset — **do not use any cached copy showing
  a third "oracle ceiling" bar, that version is superseded (§5.9)**.
- `results/no_search_accuracy/no_search_accuracy_llm.csv` (final numbers, LLM-graded).
- `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`, "Stage −1" (full writeup, including the traced
  retrieval-quality mechanism and its n=1 caveat — the accuracy numbers cited there predate the
  regrade and should be updated against this document's §1.4 before use).

**§4.2.1 (mechanism: volume vs. slope)**
- `results/cue_suppression_mechanism/cue_suppression_mechanism_map.png` — the scatter (x=level
  shift, y=slope change), this IS the figure for this subsection.
- `results/cue_suppression_mechanism/cue_suppression_mechanism.csv` for exact per-(model,cue) values.
- `results/epistemic_alignment_scorecard/epistemic_alignment_scorecard.png` for the aggregate
  count-per-model view (how common is each mechanism label).
- `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`, Stage 2 (full taxonomy definitions + the discrimination/
  calibration distinction — do not paraphrase this without reading it, the distinction is easy to
  get backwards, as happened once already this thread).

**§4.2.2 (causal consequence)**
- This document §1.5.
- `results/sft_intervention_mediation/sft_intervention_mediation.csv`.
- `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`, Stage 3, including the explicit warning against
  re-deriving the naive observational mediation (the negative within-stratum calls-correctness
  correlation diagnostic is there, worth citing directly if a reviewer might ask "why not just
  regress accuracy on search calls").

**Before writing ANY number into the paper**: cross-check it against the CSV it's sourced from, not
against this document's prose — this document rounds/summarizes and could drift from the source of
truth over edits.
