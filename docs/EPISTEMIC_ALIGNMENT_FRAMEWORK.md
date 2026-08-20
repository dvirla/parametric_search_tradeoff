# A protocol for measuring epistemic-behavioral alignment and its fragility

Status: proposal / synthesis document. Everything below is a reframing of analyses already run
this session (see `accuracy_revision.md` for the paper-specific findings and edits) as a general,
reusable four-stage protocol — not new data collection. Written so it can be lifted into a paper
Methods section, or applied to a new model/dataset/cue battery without re-deriving the design from
scratch.

## The question this answers

"Is an agent's tool-use behavior grounded in its own epistemic state, and how easily does that
grounding come apart under surface-level prompt variation?" is really four separable questions,
usually conflated into one:

1. **Is there an alignment to speak of at all?** (baseline calibration)
2. **Does a given intervention change how strong that alignment is** (not just how much the agent
   searches)? (fragility, tested causally)
3. **When alignment breaks, HOW does it break** — does the agent just search a different *amount*
   while still tracking its own uncertainty, or does the uncertainty-tracking mechanism itself get
   disabled? (mechanism)
4. **When alignment breaks with a real accuracy cost, is that cost caused by the changed search
   *volume*, or by something else the intervention did to the agent's answer generation?**
   (consequence attribution)

Treating these as one question (as e.g. "decoupled from parametric knowledge" language does) is
where a lot of the overclaiming in the original paper draft came from — each stage needs its own
instrument and its own evidentiary bar, and a model can score differently on each.

## Stage -1 — Does the tool add value at all? (prerequisite to everything below)

Before asking whether search behavior is *calibrated* to uncertainty, ask whether search
*matters* for the outcome in this domain. Low search volume can reflect either miscalibration
(the model should search more and doesn't) or a **rational adaptation to a tool that doesn't
help** — these look identical from search-call counts alone and need to be told apart before any
"decoupled from epistemic state" language is applied.

**What**: for each (model, dataset), compare accuracy on the same examples under two conditions:
- **No-search floor**: mean accuracy across the Stage-0 no-search rollouts (pure parametric
  knowledge).
- **Actual (`plain`) accuracy**: the model's default search-enabled behavior.

Both **LLM-judge graded** (see Stage 0 note below on why this matters). Scripts:
`scripts/regrade_no_search_llm.py` (the regrade itself, complete, 20,015 rows),
`scripts/analyze_no_search_accuracy_llm.py`. Figure: `scripts/make_no_search_oracle_figure.py` →
`results/search_oracle/no_search_oracle_comparison.png`.

*(An earlier version of this check also computed a multi-cue "oracle ceiling" — best accuracy
pooling every rollout collected per example, including the full cue battery. That number was
regex-graded and is now invalid: once the no-search floor was properly regraded it came out
*above* the old regex-based ceiling on MedQA, which is incoherent. Re-establishing a valid oracle
would need every cue condition regraded too — not done. The two-condition comparison below doesn't
need it.)*

**Result, this project's 4-model subset, final LLM-graded numbers**:

| | FRAMES | MedQA |
|---|---|---|
| no-search floor (mean per-run) | 24.4–41.8% | 69.5–80.3% |
| plain (actual) accuracy | 49.5–69.9% | 68.0–81.8% |
| search adds (plain − floor) | **+15.2 to +35.3pp** | **−1.5 to +1.5pp** |

**On FRAMES, search is genuinely load-bearing**: it adds 15–35pp over parametric knowledge alone.

**On MedQA, search adds essentially nothing (noise-level, ±1.5pp) in every model** — and,
importantly, this is not because these models don't know medicine: the true (LLM-graded) no-search
floor is 70–80%, comparable to or higher than FRAMES's `plain`/search-enabled accuracy. These
models overwhelmingly already know MedQA from parametric knowledge.

**Which comes first — not searching, or search not helping? Both, and they compound.** Only
4–20% of MedQA examples ever receive a search call under `plain`
(`scripts/analyze_medqa_search_conditional.py`) — the aggregate "+0pp" hides that on the 80–96%
that never get searched, `plain` ≈ no-search trivially. On the small subset that IS searched,
accuracy is **consistently worse than the model's own no-search accuracy on those same examples,
in all 4 models** (−3.8pp to −9.3pp) — for two models the searched subset even had *higher*
no-search accuracy than the left-alone subset, and searching still hurt. **The precise claim is
not "the tool doesn't help" — it's "the tool is rarely invoked, and when it is, it correlates with
a real accuracy cost."** (Standard caveat: matched same-example comparison, not randomized —
can't fully rule out the searched subset was already heading toward a wrong answer for reasons
unrelated to search, same endogeneity concern as Stage 3's broken observational mediation test.)
This means **low MedQA search volume is largely a correct, not a miscalibrated, policy** — the
retrieval tool (this project's local MedQA index/BM25 corpus, or web search more generally for
clinical-vignette MCQs) simply doesn't help in this domain, plausibly because the underlying
knowledge is already densely represented in pretraining and/or the corpus doesn't retrieve well
for this question style (see `CLAUDE.md`'s MedQA index caveats, and the qualitative
retrieval-quality trace in `accuracy_revision.md` §1.4 — n=1, illustrative only, but directly
consistent with the searched-subset accuracy cost above).

**Implication for Stages 0–3 above**: this reframes, rather than contradicts, the MedQA findings
elsewhere in this doc. The weak/absent baseline calibration (Stage 1) and the near-total absence
of FDR-significant fragility cells (Stage 2) on MedQA are *consistent with* search mattering little
there — there's little for a cue to meaningfully disrupt, and little cost to any model's necessity-
tracking being weak, because acting on it wouldn't have helped anyway. Do not read MedQA's Stage 1
"not calibrated" verdicts as evidence of a *worse* alignment failure than FRAMES's — on this
evidence it's closer to a non-issue: the tool has near-zero counterfactual value in this domain, so
whether triggering it correctly tracks uncertainty is far less consequential than the same
question on FRAMES, where getting it wrong measurably costs accuracy.

## Stage 0 — Necessity instrument (shared input to everything downstream)

**What**: an independent, cue-free measure of the model's own uncertainty per example. This
project uses semantic entropy over N≥5 independent no-search rollouts, LLM-judge clustered
(gpt-oss:120b) into semantically-equivalent answer groups, Shannon entropy over the cluster-size
distribution.

**Requirements for validity**:
- Must be measured **without** the cue whose fragility you're about to test (otherwise it's not a
  pre-treatment covariate and can't license causal language in Stage 2).
- N≥5 rollouts recommended — N=3 only yields 3 discrete entropy levels, too coarse to detect a
  dose-response relationship; this project validated that findings replicate going from N=3 to
  N=5 (`accuracy_revision.md` §1), which is itself a useful robustness check to run once.
- The clusterer/judge should be validated beyond a spot-check if the paper leans on this
  instrument — this project did **not** complete that step (an alt-test per Calderon et al. 2025
  was scaffolded but never run to the required ≥3-annotator, 50-100-item bar) and that gap should
  be disclosed wherever Stage 0 output is cited.

**Output**: `entropy[example_id]` per (model, dataset).

**Construct validity check (do this before trusting anything downstream)**: does the instrument
actually predict correctness, or is it just measuring output diversity unrelated to whether the
model is right? Script: `scripts/analyze_entropy_vs_correctness.py`. Result: **yes, strongly, and
equally on both datasets** — ρ(entropy, correctness) ≈ −0.55 to −0.71 in every one of the 8
(model, dataset) cells, using the true no-search accuracy (5 rollouts, LLM-judge graded). Accuracy
at entropy=0 is 73–83% (FRAMES) / 82–87% (MedQA); it drops to 15–21% / 43–52% at entropy>0. This
is the single most important number for defending the whole framework's premise: it means Stage 1's
entropy→search-behavior correlation is being compared against a *validated* signal, not noise, and
it means the two datasets have comparably good signals — any asymmetry found downstream (Stage 1,
Stage 2) is about behavior, not about the signal itself being worse on one dataset than the other.

## Stage 1 — Baseline alignment (calibration)

**What**: correlate the Stage 0 necessity measure against search-triggering behavior (search-call
count, or P(search)) in the **unperturbed** condition only.

**Requirement for a citable "yes"**: replicate via **split-half** — run the identical unperturbed
prompt twice, independently, and check the correlation holds in both. This is the single most
important methodological upgrade found this session: a one-rollout correlation cannot rule out
sampling noise producing a spurious ρ; two independent rollouts giving near-identical ρ can
(`accuracy_revision.md` §1, "Calibration, replicated").

**Output, per (model, dataset)**: `ρ_A`, `ρ_B`, and a binary `calibrated` flag (both runs
independently significant, same sign). This flag **gates** how Stages 2–4 should be interpreted:
a model with no baseline calibration has no alignment for a cue to erode in the first place, so
"fragility" language doesn't apply the same way (MedQA for 2 of 4 models in this project fell into
exactly this bucket).

**On magnitude — "calibrated" is a real but modest correlation, not a tight coupling.** In this
project: FRAMES pooled ρ=0.35 (5-run entropy), per-model range 0.22–0.44; the calibrated MedQA
models sit lower, ρ=0.12–0.32. In R² terms, necessity explains roughly 5–19% of search-call
variance even in the best-calibrated cells. The `calibrated` flag means "a real, replicated,
statistically robust relationship survives two independent rollouts" — it does not mean search
volume closely tracks uncertainty. This matters for how Stage 2's baseline reference point should
be read: departures are measured from an already-loose anchor, not a tight one.

**The "utilization gap"**: since Stage 0 now shows the underlying signal is equally strong on both
datasets (ρ≈−0.55 to −0.71 for correctness), the behavioral correlation above (ρ≈0.22–0.44
FRAMES, 0–0.32 MedQA) is best read as *how much of that signal survives translation into behavior*,
not as a standalone number. Even FRAMES's best case only reflects roughly half the signal's own
strength — a general, partial "utilization gap" present in every model — and MedQA shows the same
partial gap for 2 models plus a *near-total* gap (ρ≈0) for the other 2. This reframing — general
partial underuse, worse for specific models on MedQA — is more precise than either "FRAMES is
calibrated, MedQA isn't" or a flat decoupling verdict. Full numbers: `accuracy_revision.md` §1.2.

## Stage 2 — Fragility under intervention, decomposed into mechanism

**What**: for each candidate cue in the battery, with the SAME examples run under both `plain` and
the cue (a complete within-subject crossover — every example gets every condition, so there's no
allocation-based confound to control for), fit:

```
calls ~ entropy + is_cue + entropy:is_cue
```

OLS, cluster-robust SEs by `example_id`. Three coefficients, each with a distinct meaning:
- `is_cue` (**level shift**): how much the cue changes search volume at zero necessity — a pure
  amount-of-searching effect.
- `entropy:is_cue` (**slope change**): how much the cue changes the *slope* of calls-on-entropy,
  i.e. whether the agent still tracks its own uncertainty as well as it did under `plain`.
- `entropy` (baseline slope, from Stage 1) is the reference point both are measured against.

**Multiple comparisons**: run the full cue battery, FDR-correct (Benjamini-Hochberg) across all
(model, dataset, cue) cells before treating any single cell as established — this project ran 122
cells and found only a handful survive correction; uncorrected p<0.05 at this scale is not a
meaningful bar.

**A conceptual point that has to be stated precisely, or the taxonomy misleads**: a level shift is
*not* a calibration-neutral event. There are two distinct properties in play, borrowed from the
forecasting-calibration literature's discrimination/calibration split:
- **Discrimination** (relative): does search volume still correctly *rank* examples by necessity —
  proportionally more search per unit of entropy, matching `plain`'s slope? This is what
  `entropy:is_cue` measures.
- **Calibration** (absolute): does the *amount* of search match what's epistemically warranted at
  each necessity level? A pure level shift means the agent now systematically over- or
  under-searches at *every* necessity level — discrimination survives, but absolute calibration
  does not. This is a real departure from alignment, just a structurally different one from a
  slope change, not an absence of one.

**Classification (the fragility taxonomy)** — every (model, dataset, cue) cell gets exactly one
label, describing which KIND of departure is present, not whether alignment is intact:

| Label | Condition | Meaning |
|---|---|---|
| **Null** | neither `is_cue` nor `entropy:is_cue` significant | no detectable change to search behavior at all |
| **Uniform volume shift** | `is_cue` significant, `entropy:is_cue` not | discrimination (relative necessity-ranking) survives; absolute calibration does not — the agent now searches a systematically different amount at every necessity level |
| **Erosion** | `entropy:is_cue` significant and negative, slope retains 0–70% of its plain value | discrimination itself degrades — the agent is losing the ability to tell which examples need more search |
| **Inversion** | `entropy:is_cue` significant, slope flips sign (ratio ≤ 0) | discrimination reverses — searches *less* on higher-necessity examples |
| **Sharpening** | `entropy:is_cue` significant and positive, slope >130% of plain | discrimination improves beyond baseline |

This is the actionable core of the framework: it turns "search calls shift under cues" (already
well known, not surprising) into a labeled, falsifiable claim about *which specific property* of
alignment is being violated, per model and per cue. **Do not read "Uniform volume shift" as
low-consequence** — it is not accuracy-free by construction. In this project, `direct` on
`gemma4_31b` falls in this exact category (slope preserved) yet produced one of the largest
accuracy costs measured anywhere in the dataset (−22.6pp), and the Stage 3 test showed that cost
*is* genuinely mediated by the volume change. Consequence severity has to be measured (Stage 3),
not inferred from the Stage 2 label — Erosion/Inversion cells are the ones where the *mechanism
of alignment itself* is damaged, which is a qualitatively different and arguably more concerning
claim than a volume shift, but "more concerning mechanism" is not the same claim as "more
accuracy cost," and the two should not be conflated.

**A headline finding sits inside this same category, and it doesn't need Stage 3 to establish.**
`direct` is a real exception (a level-shift cell with a large, real accuracy cost) — but across
*all* 90 level-shift-only cells, the correlation between |Δ search volume| and |Δ accuracy| is
statistically indistinguishable from zero (ρ=+0.016, p=0.88,
`scripts/analyze_volume_accuracy_decoupling.py`). Among the 29 cells with a large volume swing
(|Δcalls|>1.0), **48% show negligible accuracy change (<3pp)** — including MedQA cases where a
cue inflates near-zero baseline search volume by 8–38× with at most ±2.8pp accuracy movement. This
is arguably the project's most self-contained, easiest-to-defend claim: it needs no entropy
instrument, no clustering, no causal-mediation apparatus — just a null correlation between two
directly observed quantities, with a reported effect size. It's the precise, falsifiable version
of what "decoupled" was gesturing at all along: not that search never matters (`direct` and the
Stage-2 erosion cells show it sometimes does), but that for most of the volume that cues move,
whether they moved it a little or a lot tells you nothing about whether accuracy moved at all.

## Stage 3 — Consequence attribution (optional, requires a manipulated mediator)

**What**: for cells classified Erosion/Inversion in Stage 2 with a measurable accuracy cost, ask
whether that cost is caused by the changed search *volume*, or is a direct effect of the cue on
answer generation independent of volume.

**Critical warning, established the hard way this session**: this canNOT be tested with a naive
observational path model (`correct ~ is_cue + calls`, treating realized per-example search-call
count as the mediator). Diagnostic: within every necessity stratum, realized search-call count and
correctness were strongly *negatively* correlated in this project's data (r=−0.32 to −0.66) —
because search-call count is generated *during* the same rollout as the answer, so a high count on
a given example is largely a symptom of within-rollout struggle, not a cause of eventual success.
Any attempt at this stage that doesn't manipulate the mediator will inherit this confound.

**What actually works**: an intervention that manipulates search-*volume* directly, independent of
the agent's own within-rollout signals — this project used SFT on rollouts curated to have small
|Δcalls| *and* correct answers, comparing base vs. SFT'd model on the same held-out questions
across the same cues. Cleaner alternatives if building this fresh: a forced search-call cap/floor
condition at inference time, which manipulates only the mediator and nothing else (the SFT
approach jointly retrains toward "small-delta and correct," which is not a pure single-variable
manipulation — a real limitation, noted below).

**Output, per Erosion/Inversion cell (where testable)**: a label — `volume-mediated` (restoring
search volume proportionally restores accuracy), `direct effect` (accuracy recovers even without
restoring volume — this project's `confident_parametric` cue on `gemma4_31b`), or `untested` (no
manipulated-mediator data available — the default for most cells; this stage is expensive to run
per (model, cue) and won't be feasible to run exhaustively).

## The scorecard: applying the protocol to this project's data

Stages 0–2 are fully instantiated for 4 models × 2 datasets × ~15 cues; Stage 3 is instantiated
for exactly one model (`gemma4_31b`) × 2 cues, illustrating both possible verdicts. See
`scripts/make_epistemic_alignment_scorecard.py` →
`results/epistemic_alignment_scorecard/epistemic_alignment_scorecard.{png,pdf}` for the visual
summary, and `results/cue_suppression_mechanism/cue_suppression_mechanism.csv` +
`results/baseline_calibration/baseline_calibration_stats.csv` for the underlying per-cell labels.

Headline pattern the scorecard makes visible at a glance: **baseline calibration (Stage 1) is
common; a uniform volume shift (Stage 2) is the dominant mechanism where a cue has any effect at
all; genuine erosion/inversion — discrimination itself breaking, not just volume — is rare and
concentrated in specific (model, cue) pairs.** `gemma4_31b` is the standout case for both kinds of
departure, and it's the one model with Stage 3 evidence on either side: its `direct` cue (a Stage-2
Uniform-volume-shift cell) carries a real accuracy cost that *is* volume-mediated, while its
`confident_parametric` cue (a Stage-2 Erosion cell) carries a real accuracy cost that is *not*
mainly volume-mediated — a reminder that Stage 2's label predicts *which property* of alignment
broke, not how large or how mediated the resulting accuracy cost turns out to be; that always
needs its own Stage 3 test.

## What this buys the paper beyond individual findings

Framed as isolated results ("Gemma is more fragile than GPT-OSS," "MedQA is less calibrated than
FRAMES"), this session's work is a set of facts about 4 specific models. Framed as a **protocol**
(Stages −1 through 3, with the taxonomy in Stage 2 as the reusable contribution), it's a
methodology other work can apply to characterize any agent's search-epistemic alignment — which is
a materially different and stronger claim to make in a paper than the sum of the individual
numbers. The paper's existing RQ2 ("are search calls tightly aligned with epistemic uncertainty, or
does perturbation manipulate the decision boundary") is *literally this protocol stated as a
question*; what's been missing is stating the stages as separable, each with its own instrument and
evidentiary bar, rather than collapsing them into one decoupled/not-decoupled verdict — and, per
Stage −1, missing the check for whether the tool has any counterfactual value in the domain before
interpreting calibration/fragility results as meaningful at all.

## Requirements to apply this to a new model or dataset

0. Reuse of the existing cue-battery + no-search infrastructure for a no-tool-value check (Stage
   −1) — cheap, since it's the same rollouts Stage 0 already needs, just graded and diffed.
1. N≥5 independent no-search rollouts per example (Stage 0).
2. Two independent rollouts of the unperturbed prompt (Stage 1's split-half requirement).
3. A cue battery, each cue run once against the full example set alongside `plain` (Stage 2) — no
   need for per-example randomization, a complete crossover is sufficient and simpler to run.
4. (Optional, expensive) a manipulated-mediator intervention for any cell flagged
   Erosion/Inversion with a real accuracy cost, if Stage 3 attribution is wanted.

Stage −1 should run first and gate interpretation of everything else — Stages 0–2 are the
load-bearing, always-worth-running core; Stage 3 is valuable but should be scoped to the small
number of cells Stage 2 actually flags as concerning, not run exhaustively.
