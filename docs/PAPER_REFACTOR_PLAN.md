# Paper refactor plan — "policies unstable, uncertainty stable"

Target file: `/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex`
(current title: *"The Behavioral Brittleness of Info-Seeking Agents: Sensitivity to Contextual
Perturbations"*). This plan is self-contained but assumes the reader will cross-check specific
numbers against `accuracy_revision.md` (the full analysis-thread synthesis) and
`docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md` (the conceptual vocabulary) before drafting text — this
document tells you *where to cite what*, not the full derivation of every number.

**Section/line numbers below are verified against the current `main.tex` as of this writing**
(unlike earlier drafts of `accuracy_revision.md`'s §2, which warned their line numbers might be
stale) — re-verify once more only if the paper has been edited since.

---

## 0. The thesis this refactor argues for

> **The agent's search-triggering *policy* is unstable under surface-level cues — but the thing
> that policy is supposed to be tracking, the model's own epistemic state, mostly is not.**
> Baseline search volume is grounded in a real, validated internal uncertainty signal (entropy
> predicts correctness, ρ≈−0.55 to −0.71, replicated). When a cue is applied, that same signal —
> measured *under the cue itself*, not just inferred — stays essentially flat (16/18 tested cells)
> and the model's own answer content redirects at a small, uniform, cue-independent background
> rate (~10.6%, same in cue-flat and cue-shifted cells alike). Yet search-call volume swings by as
> much as 40–95 percentage points under some cues, and among the cues that move volume without
> breaking the necessity-tracking mechanism, volume and accuracy show no significant relationship.
> **The instability lives specifically in the policy layer, not in what the policy is supposed to
> be responding to.** This is a more precise, more mechanistic, and more surprising claim than the
> paper's current "decoupled from parametric knowledge" — it says the decoupling is a property of
> the *trigger*, actively demonstrated to be untethered from the two most obvious candidate causes
> (uncertainty, belief), not an unexplained correlation.

This upgrades (does not contradict) the paper's existing empirical results — Table 1 and Figure 2
(the two central Results figures) already show exactly this pattern; the paper currently just
narrates it as "decoupled from parametric knowledge" without the entropy/belief instrumentation
that would let it say *specifically what* isn't explaining the shift. That instrumentation is what
this whole analysis thread adds.

---

## 1. Fix this first, independent of the thesis reframe

**Table 1's "SEARCH MULTITURN" column is very likely a conservative undercount and should be
recomputed before anything else changes.** `AgentAsSampler.acall()` counts search calls over
pydantic-ai's `all_messages()`, which includes the injected `message_history` — so raw
`sampler_search_calls` for `searchmulti`/`2`/`3` conditions is inflated by exactly 1/2/3 **fake**
calls from the mocked conversation history itself (documented in full in `accuracy_revision.md`
§1.0's correction note; the fix is already applied in this repo's analysis scripts). Table 1's
"zero-search frequency" metric (`sampler_search_calls == 0`) is *directly and specifically*
distorted by this in one direction: any example where the model made zero **live** search calls
would still show a raw count of 1–3 (because of the injected fake call), so it does **not** get
counted as "zero search." **This means the true zero-search-frequency increase for SEARCH
MULTITURN is at least as large as currently reported (+5.3pp FRAMES, +17.4pp MedQA), likely
larger.** This doesn't change the paper's qualitative conclusion for this row — if anything it
strengthens it — but the exact numbers in Table 1 need re-deriving from corrected counts before
publication regardless of whether the broader thesis reframe below is adopted. (`MULTITURN`'s
row is unaffected — that history has no tool call in it at all.)

Whoever does this: the correction logic (`MOCK_HISTORY_OFFSET = {"searchmulti": 1, "searchmulti2":
2, "searchmulti3": 3}`, subtract and clip at 0) is already implemented in
`scripts/analyze_necessity_vs_template_search_5run.py`, `dual_metric_analysis.py`, and
`analyze_volume_accuracy_decoupling.py` — whatever script originally produced Table 1 (not
identified in this repo; likely lives in the paper repo or a `results/cue_briefing/` aggregation
not covered by this analysis thread) needs the same fix applied and Table 1 regenerated.

---

## 2. The evidentiary chain, in the order a reader should encounter it

1. **The signal is valid.** Semantic entropy over 5 independent no-search rollouts predicts
   correctness, equally strongly on both datasets (ρ≈−0.55 to −0.71, all 8 model×dataset cells,
   p<1e-40 everywhere). *(accuracy_revision.md §1.1; `results/entropy_vs_correctness/`)*
2. **The behavioral response to that signal is real but partial.** Baseline (no-cue) search-call
   count correlates with entropy, split-half replicated (FRAMES ρ≈0.35–0.44 all 4 models; MedQA
   ρ≈0.12–0.32 for 2 models, null for the other 2). A logistic companion adds calibration, not just
   discrimination: AUC 0.74–0.82 on FRAMES but Brier-score improvement over a naive base-rate
   predictor is often small (as little as 4%), and MedQA's null models show AUC *below* 0.5.
   *(§1.2, §1.2b; `results/baseline_calibration/`, `results/baseline_calibration_logistic/`)*
3. **Cues move volume via a level shift, mostly leaving the entropy→search *slope* intact.**
   FRAMES: 50/62 (dataset,model,cue) cells are a pure level shift; only a specific, nameable
   minority (`confident_parametric`/`multiturn`/`searchmulti` for `gemma4_31b`; `multiturn` for
   `gpt-oss_20b`) show real slope erosion. MedQA: 0/60 cells show a significant slope change at all.
   A logistic decomposition of the same cells shows this is almost entirely an **intensive-margin**
   effect (how many searches, once triggered) — only 1/101 fittable cells shows a significant
   entropy×cue interaction on the binary *search-or-not* decision. *(§1.3, §1.3b;
   `results/cue_suppression_mechanism/`, `results/necessity_vs_template_logistic/`)*
4. **Neither uncertainty nor belief content explains the shift — this is the new, central piece.**
   Measuring entropy and the model's own canonical answer *under the cue itself* (not inferred):
   entropy stays flat in 16/18 tested cells including all 4 available `confident_parametric` cells
   (p≥0.43); the model's modal answer redirects at a uniform ~10.6% rate regardless of cue,
   equally in entropy-flat and entropy-shifted cells (12.3% vs. 11.8%) — proof the redirection is
   invisible to the entropy test by construction, not by absence. Conversation-state cues
   (`multiturn`, and the `searchmulti` family via `--history_path`) are the single largest lever on
   volume in the whole battery, larger than the explicit "no need to search" instruction. *(§1.3c —
   new; `results/entropy_under_cue/`, `results/modal_answer_shift/`, `results/cue_feature_axes/`)*
5. **Volume and accuracy show no significant relationship among mechanism-intact cues** (ρ=+0.168,
   p=0.113, n=90 level-shift-only cells) — weaker than an earlier, bugged estimate (see §1 above),
   but still not significant; median effect is small (|Δcalls|=0.19, |Δaccuracy|=3.0pp) even though
   several individual large-swing cells carry real double-digit accuracy costs. *(§1.0 — corrected;
   `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv`)*
6. **Domain moderates the stakes.** FRAMES: search adds +15 to +35pp over parametric knowledge
   alone in every model. MedQA: search is invoked on only 4–20% of examples, and on that subset
   accuracy is *worse* than the model's own no-search accuracy on the same examples (−3.8 to
   −9.3pp) in all 4 models — "rarely invoked, and costly when it is," not "doesn't matter."
   *(§1.4; `results/no_search_accuracy/`, `results/search_oracle/`)*
7. **Fragility has a causally-verified consequence, at least once.** SFT checkpoints trained on
   volume-preserving, correct rollouts show `confident_parametric`'s accuracy cost survives even
   when search volume is restored to near-baseline — the erosion itself, not just the volume loss,
   was doing the damage. *(§1.5; `results/sft_intervention_mediation/`)*

---

## 3. Section-by-section edit plan

### 3.1 Abstract (line 56)
**Current** (verbatim): *"...Notably, while epistemic markers produce negligible shifts,
structural and stylistic changes drastically alter the agent's decision boundary between internal
knowledge and external search without impacting final accuracy. This robustly indicates that
search calls are fundamentally decoupled from the agent's actual information need or parametric
knowledge. The compelling evidence that learned tool-use policies are universally driven by
superficial formatting heuristics rather than genuine epistemic uncertainty underscores the urgent
necessity of evaluating agentic robustness beyond static accuracy..."*

**Proposed**: *"...Notably, while epistemic markers produce negligible shifts, structural,
stylistic, and conversational-state changes drastically alter the agent's decision boundary
between internal knowledge and external search, carrying a small average accuracy cost that
concentrates sharply in specific model–perturbation pairs. Critically, this instability is a
property of the search-triggering policy itself, not of the agent's underlying epistemic state:
directly measuring the model's own uncertainty and its answer content under the same perturbations
shows both remain essentially unchanged even as search volume swings by up to 95 percentage
points. This evidence that a learned tool-use policy can be destabilized by superficial cues while
the epistemic signal it is meant to track stays intact underscores the urgent necessity of
evaluating agentic robustness beyond static accuracy..."*

**Cite**: §2 above, items 1 and 4 (entropy/belief flat under cue — this is the new
claim the abstract needs to earn). **Figure**: none in the abstract; this sets up Figure A below.

### 3.2 Introduction, final paragraph (line 69)
**Current**: *"Our extensive analysis confirms a severe behavioral misalignment specifically
driven by stylistic and structural changes. While epistemic perturbations generally fail to move
the needle, we demonstrate that the active process of composing search queries is deeply
manipulated by verbosity and length directives, heavily decoupling the search volume from the
underlying information need. This stark decoupling proves that agents are not always searching
based on a genuine epistemic need, but are instead over-relying on superficial heuristics and
partial signals learned during training, leaving the agent susceptible to factual errors."*

**Proposed**: *"Our extensive analysis confirms a severe behavioral misalignment specifically
driven by stylistic, structural, and conversational-state changes. While epistemic perturbations
generally fail to move the needle in the question text, we show that the active process of
composing search queries is deeply manipulated by verbosity, length directives, and even
unrelated prior conversation turns — and, using a novel within-cue measurement of the model's own
uncertainty and answer content, that this manipulation happens without a corresponding change in
what the model actually knows or believes. The policy, not the epistemic state, is where the
instability lives, leaving the agent susceptible to factual errors precisely because its tool-use
decisions are untethered from a signal that is otherwise demonstrably present and usable."*

**Cite**: §2 above, items 1 and 4.

### 3.3 Related Works, "Robustness and Brittleness of Agentic Tool Usage" (line 81)
**Current**: *"...By mapping the search-accuracy tradeoff, we uniquely prove that the suppression
of search volume is completely decoupled from the information need. Our findings demonstrate that
agents actively trade off their epistemic state to adhere to learned formatting heuristics,
identifying a fundamental structural misalignment..."*

**Proposed**: *"...By mapping the search-accuracy tradeoff and, novelly, directly instrumenting the
model's own uncertainty and answer content under the same perturbations that move its search
behavior, we show that agents do not trade off a *changed* epistemic state for formatting
compliance — their epistemic state is measurably unchanged, and the tool-use policy shifts
anyway. This is a stronger and more specific claim than prior brittleness findings: not just that
behavior is sensitive to surface form, but that the sensitivity is demonstrably untethered from
the internal state the behavior is nominally conditioned on."*

**Cite**: §2 above, item 4 (this is the paper's actual novelty claim relative to prior brittleness
literature — worth stating explicitly here, since "prior studies focus on *whether* agents fail...
our work... *why*" is exactly the rhetorical slot this fills).

### 3.4 Research Questions, RQ2 (line 88)
No wording change needed (already correctly posed as a two-sided open question, per
`accuracy_revision.md` §4) — but flag in the surrounding prose (or in Results) that RQ2 is now
**answered on both halves**: yes, calls align with uncertainty at baseline (§2 above, item 2); and yes,
specific perturbations manipulate the decision boundary — but demonstrably *not* by first
manipulating the uncertainty itself (§2 above, item 4). Currently the paper only answers the second
half.

### 3.5 Results → "The General Suppression of Search Policies" (Table 1, lines 132–160)
Keep Table 1 as the paper's primary "the policy moves a lot" evidence — it already is exactly
that, it just needs the SEARCH MULTITURN column recomputed (§1 above) before anything else. Add
one sentence after the existing paragraph connecting this table forward to the new instrumentation:
*"Section [Decoupling] shows this suppression is not because the agent's own uncertainty or belief
about the answer has changed."* This is the thread that ties Table 1 to the new §1.3c evidence
instead of leaving them as two disconnected observations.

**Figure**: Table 1 stays as-is (recomputed). No new figure needed here.

### 3.6 Results → "The Decoupling of Search and Accuracy" — the section to restructure most
**Current opening** (line 173): *"...our analysis reveals a stark decoupling of an agent's search
behavior from its accuracy. If agents queried tools based on true parametric uncertainty, severe
search reductions should proportionally degrade accuracy. However, Figure~\ref{fig:combined_search_acc}
shows that for most structural directives, massive search drops occur without significant accuracy
degradation. This proves that tool-use policies are decoupled from parametric knowledge, defaulting
to superficial formatting heuristics rather than genuine epistemic gaps."*

**Proposed**: *"...our analysis reveals a stark instability in an agent's search-triggering
behavior that is not explained by a change in its accuracy-relevant knowledge. Figure
\ref{fig:combined_search_acc} shows that for most structural directives, massive search-volume
shifts occur without significant accuracy degradation. Because this could in principle reflect a
real (if unmeasured) shift in the model's uncertainty, we directly instrumented that uncertainty:
measuring semantic entropy over repeated no-search rollouts \emph{under each perturbation itself}
shows it remains statistically indistinguishable from the unperturbed baseline in 16 of 18 tested
cases (Figure \ref{fig:policy_instability}, panel A), including every tested case of the
capability-framing perturbation that produces the single largest search-volume swing in our whole
grid. The model's own canonical answer is similarly stable: it changes to a different, non-
equivalent response for only $\sim$10.6\% of questions on average, at a rate that does not differ
between perturbations that move search volume and those that don't (panel B). This is the
strongest available evidence that the instability is a property of the tool-use \emph{policy}
itself, not a downstream consequence of a real change in the model's knowledge or beliefs."*

**Then keep, essentially unchanged** (already well-hedged, `accuracy_revision.md` §4 says don't
touch): the per-example Spearman r (−0.03 to +0.03) paragraph (line 175, first two sentences) and
the `direct`/EM-artifact paragraph (line 177).

**Rewrite the `confident` paragraph** (line 179): *"When explicitly told it has the knowledge (the
\textit{confident} perturbation), search drops dramatically — and, critically, this drop happens
without any detectable change in the model's own self-consistency (entropy flat, $p \geq 0.43$ in
all 4 tested cells) or in what it would have answered anyway (modal-answer redirection at 11.2\%,
squarely in the middle of the range for every perturbation tested, not an outlier). On MedQA, this
occurs without a corresponding accuracy drop; on the complex FRAMES dataset, it causes a drastic
average accuracy drop that a causal (SFT) intervention traces specifically to erosion of the
search-to-necessity mapping itself, not to the reduced volume (Appendix \ref{app:causal_mediation}).
Ultimately, whether over-relying on tools or confidently hallucinating, the agent's tool-use policy
shifts independent of any change in a calibrated assessment of its own knowledge — the assessment
itself does not change; only the policy responding to it does."*

**Figure**: replace/supplement `Figure~\ref{fig:combined_search_acc}` (`brief_aggregate_search_acc_mean_{FRAMES,MedQA}.png`,
kept as-is) with a **new** figure right after it:
`results/policy_instability_summary/policy_instability_summary.png`
(script: `scripts/make_policy_instability_figure.py`) — the 3-panel figure built for this refactor:
panel A = entropy shift per cue (flat), panel B = modal-answer redirection rate per cue (uniform),
panel C = search-call level-shift magnitude per cue (large, cue-specific). This is the figure that
makes the "policy unstable, epistemic state stable" contrast visually, side by side, in one glance
— it should be the paper's new central figure for this claim, immediately after Figure 2.
**Caveat to carry into the caption**: coverage is partial (`gpt-oss_120b`/`gpt-oss_20b` have
multi-cue data; `gemma4_31b`/`nemotron-3-nano_30b` currently only `elaborate`) — say so in the
caption, don't imply full 11-model coverage.

### 3.7 Results → "Interaction Between Perturbation and Phrasing" (lines 181–183)
No changes needed — orthogonal finding, unaffected by anything in this analysis thread.

### 3.8 Discussion (lines 187–193) — second-highest priority after §3.6
**Current** (line 191): *"A central takeaway is that the tool-use policies of these agents are
heavily decoupled from their true internal uncertainty. Rather than searching when they internally
lack the parametric knowledge required for a query, models over-rely on structural formatting
heuristics..."*

**Proposed**: *"A central takeaway is that the tool-use policies of these agents are not
uniformly decoupled from internal uncertainty at baseline — search volume is meaningfully, if
unevenly, grounded in a validated self-consistency signal (Section [Results], RQ2). What we find
instead is a policy that can be destabilized by superficial cues \emph{without any corresponding
change in the state it is meant to track}: the same cues that swing search volume by tens of
percentage points leave the model's measured uncertainty and canonical answer essentially
untouched. This is a sharper and, we argue, more concerning failure mode than uniform decoupling
would be — a policy with no grounding to begin with degrades gracefully under noise, while a policy
with real but silently-destabilizable grounding can fail with no visible signature in the
epistemic state itself, only in the resulting tool-use behavior. Rather than searching when they
internally lack the parametric knowledge required for a query, models over-rely on structural
formatting and conversational-state heuristics that are demonstrably orthogonal to any change in
what they actually know."*

**Cite**: §2 above, items 3, 4, and 5.

### 3.9 Future Work (line 199) — turn a forward-looking sentence into a citation of already-run work
**Current**: *"...We are currently exploring supervised fine-tuning (SFT) and Reinforcement
Learning (RL) as active interventions. By explicitly aligning a model's true epistemic uncertainty
with its execution policy, we hope to calibrate agentic behavior..."*

**Proposed**: keep the RL sentence as genuine future work, but split out and move the SFT part into
Results/Discussion as **already-collected evidence**, not future work — see §3.6's rewritten
`confident` paragraph above and the new Appendix subsection in §3.11. Future Work should instead
read: *"...Having shown that a manipulated-mediator (SFT) intervention can decouple a cue's
volume effect from its accuracy cost for at least one perturbation (Appendix
\ref{app:causal_mediation}), a natural next step is extending this causal test to more
perturbation–model pairs, and exploring Reinforcement Learning as a complementary intervention
that directly rewards policy stability under paraphrase-invariant perturbations..."*

### 3.10 Appendix → "Dual-Metric Robustness Check" (`app:dual_metric`, lines 280–290)
No text changes — this section's "$r=0.13$/$r=0.16$" numbers are the ones flagged as unreconciled
against a separately-computed $r=0.76$/$\rho=0.56$ from the full 127-cell run
(`accuracy_revision.md` caveat §5.7 / §2.10). **Do not touch this section until that discrepancy is
resolved** — re-derive exactly how the paper's number was computed and compare pooling levels
before drafting any change here.

### 3.11 Appendix → new subsection: causal mediation case study
Add a new appendix subsection (suggested label `app:causal_mediation`), referenced from the
rewritten `confident` paragraph (§3.6) and Future Work (§3.9):

*"To test whether a perturbation's accuracy cost is caused by its search-volume reduction or by
something else, we compared base \texttt{gemma4:31b} against an SFT checkpoint trained on rollouts
curated to preserve search volume within $\pm1$ call of baseline while remaining correct, on 102
held-out FRAMES questions across 8 perturbations. For \textit{direct} (a perturbation that leaves
the necessity-tracking mechanism intact), restoring $\sim$54\% of the volume swing restored
$\sim$48\% of the accuracy swing — roughly proportional, consistent with a volume-mediated cost.
For \textit{confident}, search volume barely recovered (77\% of the original swing persisted) yet
accuracy recovered almost completely (down to 18\% of the original cost) — the fix did not come
from restoring volume, direct evidence that this perturbation damages the mechanism linking search
to necessity itself, not merely the amount of searching. We flag this as a single case study
(n=102, one checkpoint family), not a general result."*

**Cite**: §1.5; `results/sft_intervention_mediation/sft_intervention_mediation.csv`.

### 3.12 Appendix → "Conversational State Templates" (already exists, `app:conversational_templates`)
No changes to the existing template listing — it already documents the exact `multiturn`/
`searchmulti` history injected. Add one paragraph after the existing template dump, pointing
forward to the new finding: *"Section [Decoupling] shows these conversational-state perturbations
are the single largest lever on search volume in our entire grid — larger even than the explicit
capability-framing instruction — while leaving the model's own uncertainty and answer content
essentially unchanged (Appendix, cue-feature analysis, below)."* Optionally add a compact table of
the per-cue feature coding used in the cue-feature-axes analysis (12 rows,
`results/cue_feature_axes/cue_feature_axes_summary.csv`) if the paper wants to show its work on
"what textual property predicts suppression magnitude" — this is exploratory/descriptive (n=12
cues, say so) but concretely shows conversation-state cues are the outlier.

---

## 4. Figure inventory

### Existing figures — keep as-is
| File | Used in | Status |
|---|---|---|
| `Figures/brief_aggregate_search_acc_mean_{FRAMES,MedQA}.png` | Decoupling section, Fig. 2 | Keep — pair with the new figure below, don't replace |
| `Figures/brief_suppression_primary.png` | Effort Reallocation appendix | Keep, unaffected |
| `Figures/brief_zero_search_primary.png`, `brief_combined_search_acc_primary.png` | Not currently `\includegraphics`'d in the sections read — verify usage before touching | Leave alone unless found unused, in which case flag for removal |

### New figure to add — the paper's new centerpiece for this claim
| File | Script | Where | What it shows |
|---|---|---|---|
| `results/policy_instability_summary/policy_instability_summary.png` | `scripts/make_policy_instability_figure.py` | Immediately after Fig. 2 in the Decoupling section | 3 panels, same cues/questions: (A) entropy shift ≈ flat, (B) modal-answer redirection ≈ uniform ~10.6%, (C) search-call level-shift magnitude — large and cue-specific. This is THE figure that visually makes the paper's new thesis in one glance. |

### Supporting figures — reference in appendix or supplementary material, not main body
| File | Script | What it shows | Suggested use |
|---|---|---|---|
| `results/baseline_calibration/baseline_calibration_curves.png` | `make_baseline_calibration_figure.py` | Entropy→calls, split-half replicated, per model/dataset | Appendix, backs §2 above, item 2's baseline-grounding claim |
| `results/baseline_calibration_logistic/baseline_calibration_logistic_reliability.png` | `make_baseline_calibration_logistic_figure.py` | Reliability diagram: observed vs. fitted P(search), AUC + Brier | Appendix, if the paper wants to show discrimination ≠ calibration explicitly |
| `results/cue_suppression_mechanism/cue_suppression_mechanism_map.png` | `make_cue_suppression_mechanism_figure.py` | Scatter: level-shift vs. slope-change per cue, 5-label taxonomy | Appendix or main body if the paper wants the full mechanism taxonomy, not just the level-shift-only/erosion split narrated in prose |
| `results/search_oracle/no_search_oracle_comparison.png` | `make_no_search_oracle_figure.py` | No-search floor vs. `plain` accuracy, per model/dataset | Appendix, backs §2 above, item 6 (domain moderates stakes) |
| `results/epistemic_alignment_scorecard/epistemic_alignment_scorecard.png` | `make_epistemic_alignment_scorecard.py` | Aggregate mechanism-label counts per model × dataset | Appendix, if the paper wants a compact "how common is each failure mode" summary |

### Gaps — no figure exists yet, flagged for a future session if wanted
- A dose-response/round-count figure for `searchmulti`/`2`/`3` specifically, now that the counting
  bug is fixed and the corrected data shows no reliable dose-response (n=3, non-monotonic) — could
  be a small inset or explicitly stated as null in prose instead of a figure.
- A joint entropy-and-modal-answer scatter at the per-example level (not just per-cue aggregates)
  — would let a reader see the (rare) cases where entropy shifts *and* the modal answer changes
  together, vs. the common case of neither.

---

## 5. Full experiment/result citation table

| Claim | Script | Output file | Key number(s) |
|---|---|---|---|
| Entropy predicts correctness | `analyze_entropy_vs_correctness.py` | `results/entropy_vs_correctness/entropy_vs_correctness.csv` | ρ=−0.55 to −0.71, all 8 cells, p<1e-40 |
| Baseline calibration, split-half | `analyze_baseline_calibration.py` | `results/baseline_calibration/baseline_calibration_stats.csv` | FRAMES ρ=0.35–0.44 (4/4 models); MedQA ρ=0.12–0.32 (2/4), null (2/4) |
| Baseline calibration, binary/AUC/Brier | `analyze_baseline_calibration_logistic.py` | `results/baseline_calibration_logistic/baseline_calibration_logistic_stats.csv` | AUC 0.74–0.82 (FRAMES); Brier improvement over null as low as 4%; MedQA AUC <0.5 for 2 models |
| Mechanism decomposition (level shift vs. slope) | `analyze_cue_suppression_mechanism.py` | `results/cue_suppression_mechanism/cue_suppression_mechanism.csv` | FRAMES 50/62 level-shift-only; MedQA 60/60; 12 erosion/inversion/sharpening cells |
| Same, binary/extensive-margin | `analyze_necessity_vs_template_search_logistic.py` | `results/necessity_vs_template_logistic/necessity_vs_template_logistic_interaction.csv` | 1/101 fittable cells FDR-significant |
| **Entropy under cue (new)** | `cluster_cues_llm_judge.py` (producer) + `analyze_entropy_under_cue.py` | `results/entropy_under_cue/entropy_under_cue.csv` | 16/18 cells flat (sign test p≥0.43 for `confident_parametric`); `direct` the one exception, ±0.05 bits, not a length artifact |
| **Modal-answer redirection (new)** | `compare_modal_plain_vs_cue.py` (producer) + `analyze_modal_answer_shift_judged.py` | `results/modal_answer_shift/modal_answer_shift_judged.csv` | ~10.6% pooled rate, 8.4–11.5% per cue; 12.3% in flat-entropy cells vs. 11.8% in shifted cells |
| **Cue-feature characterization (new)** | `analyze_cue_feature_axes.py` | `results/cue_feature_axes/cue_feature_axes_summary.csv` | Conversation-state cues largest lever (`multiturn` −0.851, `confident_parametric` −0.982 signed level-shift); no reliable dose-response post-correction |
| Volume/accuracy relationship | `analyze_volume_accuracy_decoupling.py` | `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv` | ρ=+0.168 (p=0.113, n=90) — corrected, see §1.0's correction note |
| No-search value-add by domain | `analyze_no_search_accuracy_llm.py`, `analyze_medqa_search_conditional.py` | `results/no_search_accuracy/` | FRAMES +15 to +35pp; MedQA searched-subset −3.8 to −9.3pp vs. own no-search accuracy |
| Causal mediation (SFT) | `analyze_sft_intervention_mediation.py` | `results/sft_intervention_mediation/sft_intervention_mediation.csv` | `direct`: 54%/48% volume/accuracy recovery (proportional); `confident_parametric`: 77% volume swing persists, only 18% of accuracy cost persists |
| **Mocked-history-call counting bug (methods correction)** | fixed in `analyze_necessity_vs_template_search_5run.py`/`_logistic.py`, `dual_metric_analysis.py`, `analyze_volume_accuracy_decoupling.py` | — | Affects any raw `sampler_search_calls` statistic for `multiturn`/`searchmulti`/`2`/`3` computed before this fix, **including likely Table 1's SEARCH MULTITURN row** — see §1 above |

---

## 6. What NOT to change

- Title, RQ1, RQ3 — unaffected by any of this.
- The per-example Spearman r paragraph (−0.03 to +0.03) and the `direct`/EM-artifact paragraph in
  the Decoupling section — already correctly hedged, don't touch.
- The Appendix "Dual-Metric Robustness Check" / "Decoupling Confirmation" r=0.13/0.16 numbers —
  unreconciled against a separately-computed r=0.76/ρ=0.56 (`accuracy_revision.md` §2.10/§5.7).
  **Do not edit this section until that discrepancy is resolved**, independent of everything else.
- §"Interaction Between Perturbation and Phrasing" — orthogonal, untouched.
- §Limitations — untouched by this thread (though see §7 below for a caveat worth adding there).

---

## 7. Open items before this goes to a writing session

1. **Table 1's SEARCH MULTITURN column needs recomputation** (§1 above) — this is a correctness
   fix, not a framing choice, and should happen regardless of whether the rest of this plan is
   adopted.
2. **Partial model coverage for the new §1.3c evidence**: `gpt-oss_120b`/`gpt-oss_20b` have
   multi-cue entropy-under-cue data; `gemma4_31b`/`nemotron-3-nano_30b` currently only have
   `elaborate`; no model has `polite`/`natural`/`query`/`epi_strong_boost`/`epi_strong_hedge`
   coverage yet. State this explicitly in whatever figure/table caption cites this data — do not
   imply full-grid coverage.
3. **Single-reviewer cue-feature coding** (`results/cue_feature_axes/`) — hand-labeled by one
   person, not adjudicated; treat the "what predicts suppression magnitude" claims as descriptive
   (n=12 cues), not statistically confirmed.
4. **Entropy/modal-answer judge validation**: both the entropy clusterer and the modal-answer
   equivalence judge are the same model (`gpt-oss:120b`), spot-checked on only 24 examples by one
   reviewer — reassuring, not rigorously validated (a proper alt-test would need ≥3 annotators,
   50–100 items).
5. **The Appendix §2.10 discrepancy remains unresolved** — flagged again here because it sits right
   next to material this refactor touches; don't let proximity tempt an edit before it's resolved.
6. If reviewers are expected to ask "why not regress accuracy on search calls directly instead of
   the level-shift/slope decomposition" — the answer is in `accuracy_revision.md` §1.5: realized
   search-call count and correctness are strongly *negatively* correlated within every entropy
   stratum (r=−0.32 to −0.66), the signature of an endogenous mediator (an agent searches more
   *because* it's struggling, not the reverse) — cite this directly if it comes up.

---

## 8. Suggested next experiments, if there's time before submission (not required for this refactor)

1. Complete entropy-under-cue coverage for `gemma4_31b`/`nemotron-3-nano_30b` and the remaining
   cues, to turn §1.3c from a 2-model into a 4-model, full-grid claim — this is currently the
   single biggest scope limitation on the paper's strongest new claim.
2. Extend the SFT causal-mediation test (§1.5) beyond `gemma4_31b`/`direct`+`confident_parametric`
   to more model/cue pairs, to move it from "illustrative case study" to a broader result.
3. A proper multi-annotator alt-test for the entropy/modal-answer judge, if reviewers push on
   instrument validity.
