# PLAN — Results-section revision: per-dataset FDR families + the search/accuracy decoupling reframe

**Audience:** the agent making the edits to the paper.
**Target file:** `/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex`
(at revision `15495c1`, with uncommitted changes to `main.tex` already present — line numbers
below are against the *working-tree* version, verify with `grep -n` before editing).
**Analysis repo:** `/home/dvirla/projects/parametric_search_tradeoff`, branch `hotpotqa-cue-pilot`.
**Written:** 2026-09-10. **Finalised:** 2026-09-11, after the HotpotQA uncertainty grid closed.
**Uncertainty axis authority:** [`cross_dataset_uncertainty_findings.md`](cross_dataset_uncertainty_findings.md)
(72 cells: 3 datasets x 6 models x 4 cues). This plan owns the paper edits; that document owns the
derivations behind §4.

**Scope.** Four commissioned changes (§2 Change A, §3 Change B, §4 Change C, §4A Change E) plus four
consistency fixes found during the review (§5). Changes are lettered, not numbered in order:
A = FDR families (§2), B = decoupling (§3), C = epistemic state (§4), E = the SFT section (§4A). Everything is numeric-substitution or prose-replacement; **no new
experiments are required**. All numbers below were recomputed and verified during the review —
provenance for each is given so you can re-derive rather than trust.

**Out of scope — do not do:** re-run any eval, re-grade any dataset, change any figure's underlying
data, or touch §`sec:sft_interventions` / the HotpotQA transfer result. Those are correct as
written.

---

## 0. TL;DR — the change list

| # | Where | Change | Risk |
|---|---|---|---|
| A1 | `tab:zero_search` body | FRAMES MULTITURN `+15.0` → `+15.0*`, SHORT `+4.9` → `+4.9*` | none, mechanical |
| A2 | `tab:zero_search` caption | Drop the HotpotQA-specific FDR sentence; state the per-dataset rule once | none |
| A3 | `fig:combined_search_acc` caption | State the per-dataset family rule | none |
| A4 | `Figures/` | Replace 3 PNGs (FRAMES mean, MedQA-llm6 mean, + median variants) | none |
| A5 | §`The General Suppression of Search Policies` | **Optional new claim**: MedQA ELABORATE search bar becomes significant | judgement call — see §2.5 |
| B1 | §`sec:decoupling` ¶1 (line 235) | Replace the `ρ=+0.168, n=90` pooled statistic. With HotpotQA now mechanism-classified, the pooled 3-dataset statistic is **+0.275, p=.0006 — significant**; there is no aggregation level at which the null holds | **high — this is the substantive change** |
| B2 | §`sec:decoupling` ¶2 (line 237) | Keep the example-level claim, strengthen it with the noise-floor correction | medium |
| B3 | §`sec:decoupling` title + framing | Rename; the current title asserts what the data contradicts | medium |
| B4 | `app:dual_metric` "Decoupling Confirmation" (line 375) | Same pooling artifact, worse — must be rewritten | **high** |
| C1 | `tab:entropy_validity_correctness` | Row mixes an LLM-judge ρ with EM accuracies, and the ρ is a stale 5-model mean (−0.603 → **−0.526**) | **high — wrong number in a table** |
| C2 | §`sec:policy_not_uncertainty` | mean \|Δ\| entropy `0.042–0.044` → **`0.039–0.042`** (72-cell grid) | none, mechanical |
| C3 | §`sec:policy_not_uncertainty` | Replace "entropy stays flat" with the in-domain result: **1 of 12 tests significant**, and it is the length-confounded one | medium |
| C4 | §`sec:policy_not_uncertainty` | **New paragraph**: the null is *bounded* (power, bootstrap floor, non-saturation). Per-dataset stays the claim; cross-dataset is appendix-only and **model-level, q=.050** — the naive 18-cell pooling is pseudo-replicated | **high — stop the unqualified "belief does not move"** |
| C5 | §`sec:policy_not_uncertainty` | Canonical-answer leg is FRAMES+MedQA only; scope it or run HotpotQA | medium |
| E1 | §`sec:sft_interventions` + abstract + intro | **Swap the whole SFT result to the resolved checkpoint** — the reported one has a known defect. FRAMES 1.30 (6 of 9) → **0.77 (3 of 9)**; HotpotQA 0.69 (7 of 8) → **0.33 (3 of 8)** | **high — whole-section rewrite** |
| E2 | §`sec:sft_interventions` | Delete "no meaningful accuracy cost" on FRAMES — it is unresolvable at n=102; make the accuracy claim on HotpotQA | **high — unsupported claim** |
| E3 | §`sec:sft_interventions` | **Add the belief-invariance result** (entropy 1.034 vs 1.033; 7 of 8 arm differences n.s.) — the payoff the earlier checkpoint could not support | high — new argument |
| D1–D4 | various | Stale cross-references and a terminology collision (§5) | low |

---

## 1. The organizing finding behind Changes A and B

Three separate "no relationship" claims in this paper are computed by **pooling (model, cue) cells
across FRAMES and MedQA**. FRAMES cells sit at 4.2–6.4 baseline search calls; MedQA cells sit at
**0.09–0.25**. Pooling a dataset with a live search-call axis against one whose search-call axis is
a floor produces a null that describes neither dataset.

One instance of this was already found and fixed (entropy-vs-search pooled ρ; the paper now reports
mean per-model ρ in `tab:entropy_validity_search`, per
[`cross_dataset_uncertainty_findings.md`](cross_dataset_uncertainty_findings.md) §2). **The other
two are still in the paper** and are Change B below.

Concretely, for the 90 level-shift-only cells behind §`sec:decoupling`:

| | n | median \|Δcalls\| | as % of that cell's baseline | median \|Δacc\| | baseline calls |
|---|---|---|---|---|---|
| FRAMES | 50 | 0.87 | 18.2% | 3.7pp | 4.19 – 6.40 |
| MedQA | 40 | 0.11 | **82.1%** | 1.9pp | **0.09 – 0.25** |
| pooled (as reported) | 90 | 0.19 | 30.8% | 3.0pp | 0.09 – 6.40 |

The paper's "typical case is small movement in both (median $|\Delta\text{calls}|=0.19$)" is true of
no dataset: FRAMES's typical cell moves 0.87 calls, MedQA's moves 0.11 calls but that *is* 82% of
its entire baseline. Provenance: `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv`.

---

## 2. Change A — one FDR family per dataset

### 2.1 Why

`scripts/make_aggregate_cue_tradeoff_figure.py:576-593` defines the BH family as *whatever panels
are loaded in one invocation*. Today that is: FRAMES + MedQA-llm6 + MedQA-regex5 in one family
(60 figure tests / 30 table tests), and HotpotQA alone in a second family — a split adopted purely
to avoid perturbing already-published FRAMES/MedQA stars, not on principle.

The rationale to state in the paper is **not** "different graders need different families" — a BH
family is the set of hypotheses you want joint error control over, not the set sharing a metric.
The defensible rationale is: **each dataset is a separate replication of the same experiment, and
each is reported as its own panel supporting its own claim.** Under that rule HotpotQA's current
carve-out stops being an exception and becomes the general policy, which is what makes the caveat
sentence droppable.

One structural note: **MedQA's two panels (llm6 + regex5) stay in one family** — same dataset, same
questions, split only by grader and roster. The script cannot separate them further without a code
change, and it should not: that would be double-counting one dataset as two replications.

Resulting family sizes — FRAMES 20 figure / 10 table tests; MedQA 40 / 20; HotpotQA 20 / 10.

### 2.2 What was run

```bash
cd /home/dvirla/projects/parametric_search_tradeoff
uv run python scripts/make_aggregate_cue_tradeoff_figure.py \
    --datasets FRAMES --output-dir results/cue_briefing_per_dataset/frames
uv run python scripts/make_aggregate_cue_tradeoff_figure.py \
    --datasets MedQA  --output-dir results/cue_briefing_per_dataset/medqa
# HotpotQA is unchanged — it is already its own family:
#   results/hotpotqa_cue_briefing/
```

`results/cue_briefing/` (the current paper family) is left in place for comparison. Do not delete
it until the edit is verified.

### 2.3 Every number that changes

Exhaustive — I diffed all 120 table cells and all 60 figure bars between the joint and per-dataset
families. **Six cells move, every one of them upward** (a smaller family is less conservative):

**Tables (2 of 120 cells):**

| table | panel | row | joint family | per-dataset |
|---|---|---|---|---|
| Zero-search suppression | FRAMES | MULTITURN | `+15.0pp` | `+15.0pp*` |
| Zero-search suppression | FRAMES | SHORT | `+4.9pp` | `+4.9pp*` |

**Figures (4 of 60 bars):**

| panel | metric | bar | joint | per-dataset |
|---|---|---|---|---|
| FRAMES | accuracy | DIRECT | `**` | `***` |
| FRAMES | search | CONFIDENT | `**` | `***` |
| MedQA-llm6 | search | ELABORATE | *(n.s.)* | `*` |
| MedQA-llm6 | search | CONFIDENT | `**` | `***` |

Unchanged: every HotpotQA number (already its own family); every MedQA zero-search star; every
example-level correlation star on FRAMES and MedQA; the MedQA-regex5 panel entirely; all point
estimates everywhere.

### 2.4 Edits

**A1 — `tab:zero_search` body** (~line 133 and ~line 136, verify with `grep -n "MULTITURN" main.tex`):

```latex
    MULTITURN           & Conversation State    & +15.0*  & +26.7* & +18.2   \\
    ...
    SHORT               & Directives            & +4.9*   & +14.6* & +5.4    \\
```
(the FRAMES column only — MedQA and HotpotQA columns are unchanged).

**A2 — `tab:zero_search` caption** (line ~142). Delete:

> Note: The HotpotQA significance levels are corrected over their own separate FDR family.

Replace with:

> Benjamini-Hochberg FDR correction is applied within each dataset separately, treating each dataset
> as an independent replication (* $q<.05$, ** $q<.01$).

**A3 — `fig:combined_search_acc` caption** (line ~231). Append after the existing significance
sentence:

> FDR correction is applied within each dataset's own family of tests.

**A4 — figure files.** Copy the regenerated PNGs into the paper:

```bash
P=/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/Figures
R=/home/dvirla/projects/parametric_search_tradeoff/results/cue_briefing_per_dataset
cp $R/frames/brief_aggregate_search_acc_mean_FRAMES.png       $P/
cp $R/medqa/brief_aggregate_search_acc_mean_MedQA_llm6.png    $P/
# (MedQA-regex5 is byte-identical under both families -- md5 a1cbc159...; copying is a no-op)
# median variants, if the appendix uses them:
cp $R/frames/brief_aggregate_search_acc_median_FRAMES.png     $P/
cp $R/medqa/brief_aggregate_search_acc_median_MedQA_llm6.png  $P/
```
Verified md5s: FRAMES mean `c759b9a0` -> `e0c4e7dd`, MedQA-llm6 mean `8bf1314c` -> `227ab3d3`.
Do **not** re-copy the HotpotQA PNGs — they are unchanged and already in `Figures/` (untracked,
per `git status`).

### 2.5 A5 — the one judgement call

Per-dataset correction makes **MedQA's ELABORATE search bar significant (−45%, `*`)** where it was
not before. This is a claim the paper currently does not make. Two options:

- **Report it.** It fits the existing argument (ELABORATE suppresses search on MedQA as it does
  elsewhere) and it is simply what the stated correction rule yields. One clause in the
  §`General Suppression` prose is enough.
- **Say nothing.** The prose does not currently enumerate per-bar significance for MedQA, so
  leaving it as a figure-only star is internally consistent.

Prefer reporting it. Silently gaining a significant effect from a correction change and not
mentioning it is the version a reviewer would object to.

### 2.6 Verification

```bash
diff <(cat results/cue_briefing/brief_aggregate_tables.md) \
     <(cat results/cue_briefing_per_dataset/frames/brief_aggregate_tables.md)   # FRAMES rows only
```
Expect exactly the two zero-search star additions listed in §2.3. If more cells move, stop — the
inputs have changed since this plan was written.

---

## 3. Change B — reframe §`sec:decoupling`

### 3.1 Audit of the three claims currently made

The section makes three statistically distinct claims. **They do not all survive, and they are not
all the same claim** — the failure to distinguish them is what produced the error.

| # | Claim | Level of aggregation | Status |
|---|---|---|---|
| 1 | "no reliable relationship" between \|Δsearch\| and \|Δaccuracy\|, ρ=+0.168, n=90 | (model × cue) cells, **pooled across datasets** | **FALSE as stated** — pooling artifact |
| 2 | no example-level relationship, Spearman −0.03…+0.03 | individual examples within a (model, cue) | **TRUE, and stronger than stated** |
| 3 | dual-metric confirmation, Pearson 0.13–0.16 (`app:dual_metric`) | (model × condition) delta cells, **pooled** | **FALSE as stated** — same artifact, worse |

### 3.2 Claim 1 — the n=90 statistic, split by dataset

Same cells, same estimator, same level-shift-only restriction as the paper; only the pooling is
removed. Provenance: `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv`.

**Regenerated 2026-09-11.** All three datasets are now mechanism-restricted to level-shift-only
cells — HotpotQA's classification was run (§6.1), so its row is finally like-for-like. The
FRAMES/MedQA n's also grew because the mechanism CSV in the repo was stale relative to its own
upstream (see §6.1); the roster is now 5 models on FRAMES/MedQA and 6 on HotpotQA.

| cells (level-shift-only) | n | paper's metric: ρ(\|Δcalls\|, \|Δacc\|) | signed ρ(Δcalls, Δacc) | median \|Δcalls\| | base calls |
|---|---|---|---|---|---|
| **FRAMES** | 58 | **+0.264, p=.045** | **+0.642, p<.001** | 0.94 | 4.19–10.74 |
| **MedQA** | 44 | −0.201, p=.190 | +0.049, p=.75 | 0.12 | **0.09–5.71** |
| **HotpotQA** | 50 | **+0.521, p=.0001** | **+0.716, p<.001** | 0.44 | 1.50–4.89 |
| pooled FRAMES+MedQA — *the paper's ρ=+0.168 stat* | 102 | +0.143, p=.152 | +0.513, p<.001 | 0.21 | 0.09–10.74 |
| **pooled all three** | **152** | **+0.275, p=.0006** | **+0.594, p<.001** | 0.30 | 0.09–10.74 |

Two things to take from this. First, **the paper's pooled null does not survive the third dataset**:
pooling FRAMES+MedQA still gives a non-significant +0.143 (the published +0.168 recomputed on the
current roster), but adding HotpotQA makes the pooled statistic **significant at +0.275**. There is
no longer any aggregation level at which "no reliable relationship" holds. Second, HotpotQA is now
the *strongest* coupling of the three (+0.521), not the weakest — restricting to level-shift-only
cells strengthened it, which is the opposite of what a confound would do.

¹ *(Former caveat, now resolved: this row used to be unrestricted 9-model x 8-cue cells because
no mechanism classification existed for HotpotQA. It has been run — §6.1 — so the row above is
restricted exactly as the other two are.)*

**FRAMES on its own contradicts the paper's own claim, in the paper's own metric, at the paper's own
threshold.** The +0.168 null is the average of a significant positive effect and a near-significant
negative one.

Additional supporting statistic for the new text — the HotpotQA cue-aggregate relationship is robust
to dropping the extreme cues, so it is not a two-point artifact:

| HotpotQA, cross-cue Spearman(mean Δsearch%, mean Δacc pp) | n | ρ | p |
|---|---|---|---|
| all 8 cues | 8 | +0.83 | .010 |
| minus DIRECT (length-confounded) | 7 | +0.96 | <.001 |
| minus DIRECT + CONFIDENT | 6 | +0.94 | .005 |
| minus DIRECT + CONFIDENT + MULTITURN | 5 | +0.90 | .037 |

### 3.3 Claim 2 — the example-level result survives, and can be stated more strongly

Current text hedges: *"On HotpotQA, this coupling is weakly positive (+0.06 to +0.17) and sometimes
significant."* That concedes more than the data requires.

The significance test (`make_aggregate_cue_tradeoff_figure.py:474-489`) is a one-sample test of the
per-model ρ's **against zero**. On HotpotQA zero is the wrong null: its own RERUN (noise-floor)
correlation is **+0.075** (mean; +0.05 median), and it is wildly model-dependent (−0.167 for
Nemotron-Cascade-2 to +0.294 for Nemotron3). Re-testing each cue paired against **that model's own
rerun ρ** instead of against 0:

| cue | mean ρ | p vs 0 | ρ − own rerun floor | paired p |
|---|---|---|---|---|
| RERUN (floor) | +0.075 | .135 | — | — |
| MULTITURN | +0.174 | .003 | +0.099 | .010 |
| CONFIDENT | +0.134 | .0001 | +0.059 | .199 |
| QUERY | +0.128 | .004 | +0.052 | .161 |
| SEARCH MULTITURN | +0.122 | .039 | +0.047 | .213 |
| ELABORATE / SHORT / DIRECT / POLITE | +0.06 … +0.10 | .03–.17 | −0.01 … +0.02 | .52–.83 |

Only MULTITURN clears its own floor, and it does not survive BH over the 8 tests. So the
example-level decoupling **does hold on HotpotQA**; the raised band reflects a noisier baseline
coupling on that dataset, not cue-induced coupling.

Reproduce: the script is preserved at
`results/cue_briefing_per_dataset/floor_relative_example_correlation.py` (see §7); it reads
`results/hotpotqa_cue_grid_regex/per_row.csv` and reproduces the published table to ±0.005.

Also worth correcting in the same sentence: the stated FRAMES/MedQA range "−0.03 to +0.03" is
FRAMES's range. MedQA-llm6 runs −0.07…−0.01 and MedQA-regex5 +0.00…+0.05
(`results/cue_briefing/brief_aggregate_tables.md`). Either widen the range to −0.07…+0.05 or scope
the quoted range to FRAMES.

### 3.4 Claim 3 — `app:dual_metric` is the same artifact, and worse

Provenance: `results/dual_metric_cue_deltas.csv`, 219 (dataset, model, condition) delta cells.

| cells | n | Pearson(Δsearch, Δ**LLM**) | Pearson(Δsearch, Δ**EM**) | Spearman(Δsearch, ΔLLM) | Spearman(Δsearch, ΔEM) |
|---|---|---|---|---|---|
| **FRAMES** | 113 | **+0.354, p<.001** | +0.175, p=.063 | +0.557, p<.001 | +0.503, p<.001 |
| **MedQA** | 106 | **−0.453, p<.001** | +0.137, p=.161 | −0.258, p=.008 | +0.363, p<.001 |
| pooled — *the appendix's r* | 219 | +0.133, p=.050 | +0.156, p=.021 | **+0.187, p=.005** | **+0.412, p<.001** |

Two problems, not one:

1. The pooled "very weak" Pearson averages **two significant, opposite-signed** effects
   (+0.354 and −0.453 under the LLM judge).
2. Even pooled, the **rank** correlation is significant under both metrics (+0.187 and +0.412). The
   appendix reports only Pearson, and the relationship is not linear.

MedQA's negative coupling is not anomalous — it is the same phenomenon already reported in
`app:oracle_control_detail` (on the small searched subset, MedQA accuracy runs 3.8–9.3pp *below*
the model's own no-search accuracy, an endogenous-selection effect: the agent searches when a
rollout is struggling). The appendix should cross-reference that rather than treat it as noise.

### 3.5 What the rewritten section should say

The defensible three-part claim:

1. **Volume and accuracy are coupled at the (model, cue) level wherever a non-degenerate search
   policy exists** — FRAMES signed ρ=+0.723, HotpotQA +0.659. Suppressing search costs accuracy.
2. **MedQA is the exception, and it is a floor artifact, not evidence of general decoupling.** Its
   search axis spans 0.09–0.25 calls; there is nothing for a volume-accuracy relationship to be
   measured over. This is the same explanation the paper already gives for MedQA in
   §`sec:baseline_grounding` and in the SFT transfer paragraph — it should be the same explanation
   here, which makes the paper *more* internally consistent, not less.
3. **What is genuinely decoupled everywhere is the example-level attribution** (§3.3). Between cues,
   bigger suppression → bigger accuracy loss; within a cue, an individual example's search delta
   does not predict its own correctness delta. Both are true simultaneously; this is an aggregation
   effect, not a contradiction, and it is worth saying so explicitly in one sentence so a reader
   does not read it as one.

**Two guardrails the new text must keep:**

- **Correlation ≠ mediation.** Cue identity is a common cause of both axes. The paper's own traced
  result (line 271: *restoring search volume alone does not recover the accuracy lost, but restoring
  the mechanism does*) argues the coupling is **not** causal through volume. Cite it in the new
  section — it is the counterweight that keeps the reframe honest.
- **DIRECT is length-confounded** on EM (median 2 words vs. plain's 63) and must stay flagged. Note
  that removing DIRECT *strengthens* the coupling (+0.359 → +0.446 on HotpotQA), so the confound is
  not what produces the result.

**Why this strengthens the paper.** The current framing ("massive search shifts, no accuracy cost")
partly undercuts the thesis: brittleness that costs nothing is a curiosity. The corrected framing —
perturbation-driven search suppression costs real accuracy wherever retrieval is load-bearing, and
appears free only where the tool was barely used — is the more alarming and more publishable
finding, and it is what the data says.

### 3.6 B3 — the title, and a terminology collision

`\subsection{The Decoupling of Search and Accuracy}` asserts the thing being retracted. Also, the
paper uses "decoupling" for **two different relationships**:

- epistemic signal ↔ tool-use policy (abstract line 56, discussion line 271, future work line 279)
  — **this one is intact and is the paper's central claim**;
- search volume ↔ accuracy (§`sec:decoupling`, `app:dual_metric`) — this one is being reframed.

Reusing the word invites a reader to think the retraction touches the headline claim. Suggested
retitle: **"Does Search Volume Predict Accuracy? A Domain-Dependent Coupling"** (or similar), and
reserve the bare word "decoupling" for the signal-vs-policy sense throughout.

---

## 4. Change C — the epistemic-state sections, now that HotpotQA uncertainty is complete

**Status.** The HotpotQA uncertainty grid closed **2026-09-11**: 3 datasets x 6 models x 4 cues =
**72 cells**, all 5run/5run, `confident_parametric` included (verified — every non-SFT model under
`results/hotpotqa_parametric/` carries 5 cue cluster files).
**[`cross_dataset_uncertainty_findings.md`](cross_dataset_uncertainty_findings.md) is the authority
on this axis.** This section states only what the *paper* must change and defers every derivation
to that document; its §5 "Paper-ready claims" and "Claims that must NOT be made" are the contract.

### 4.1 C1 — `tab:entropy_validity_correctness` mixes metrics and carries a stale roster

Three defects in one three-row table (verified against
`results/entropy_vs_correctness/entropy_vs_correctness.csv`, 6 models per dataset):

| paper cell | what it actually is |
|---|---|
| FRAMES −0.488, range, 0.617/0.180 | **EM throughout** — correct and internally consistent |
| HotpotQA −0.552, range, 0.699/0.261 | **EM throughout** — correct (EM is the only option) |
| MedQA ρ = **−0.603** | **a stale 5-model judge mean.** The 6-model judge mean is **−0.526**; −0.603 is the mean with `qwen3.5:122b` excluded, whose MedQA judge ρ is −0.141 against −0.556…−0.654 for the other five |
| MedQA range = "—" | simply absent; available as EM −0.101…−0.272, judge −0.141…−0.654 |
| MedQA 0.451 / 0.268 | **EM accuracies**, sitting in a row whose ρ is labelled LLM-judge |

So the MedQA row is internally mixed — judge ρ beside EM accuracies — and its ρ comes from a
superseded roster. The table as a whole then invites exactly the comparison that
`cross_dataset_uncertainty_findings.md` §1.1 calls *"the single most important methodological point
in this document"*: **EM attenuates this correlation** (≈0.14 on FRAMES, ≈0.35 on MedQA), so an EM
row and a judge row cannot be read against each other.

*(The stale numbers came from the synthesis doc's own §1.1, whose judge column was 5-model while its
EM column was 6-model. That was corrected at source on 2026-09-11 — the doc now carries a visible
correction note. If you see −0.649 / −0.603 anywhere, it is pre-correction text.)*

**Fix — one EM table with a judge column, all 6 models:**

| Dataset | ρ (EM) | range (EM) | ρ (LLM judge) | acc @ H=0 | acc @ H>0 |
|---|---:|---|---:|---:|---:|
| FRAMES | −0.488 | −0.409 … −0.537 | −0.627 | 0.617 | 0.180 |
| HotpotQA | −0.552 | −0.408 … −0.645 | — (no judge exists) | 0.699 | 0.261 |
| MedQA | −0.174 | −0.101 … −0.272 | −0.526 | 0.451 | 0.268 |

Caption must add: MedQA's low EM ρ is EM-on-option-text, not instrument failure — its judge ρ
(−0.526) is in line with the other two datasets.

⚠️ **Do not "fix" this by dropping the EM column and quoting judge numbers.** HotpotQA has no judge
at all, so EM is the only column in which all three datasets are comparable. The EM column is the
one that carries the cross-dataset claim; the judge column is the disambiguator for MedQA.

### 4.2 C2 — `sec:policy_not_uncertainty`: the entropy numbers moved

Current text: *"the mean $|\Delta|$ entropy is remarkably stable at $0.042-0.044$ bits across all
three datasets."*

Completed 72-cell grid: **0.039–0.042 bits** per dataset (FRAMES 0.041, MedQA 0.042, HotpotQA
0.039), **0.041 across all 72 cells**. Substitute the range; the rhetorical point is unchanged and
the grid is now complete rather than partial.

The adjacent *"up to $+60$pp for \textit{confident} on HotpotQA"* is still correct (+59.9pp).

### 4.3 C3 — adopt the in-domain framing: one cell of twelve, not "flat everywhere"

The completed grid supports a **sharper and more defensible** claim than the current prose, which
reads as an undifferentiated "entropy stays flat":

> Of 12 (dataset x cue) in-domain tests — a one-sample t over that dataset's 6 per-model estimates,
> BH-FDR over all 12 — **exactly one is significant: `direct` on MedQA (+0.105 bits, q=.018, 6/6
> models positive)**, and that is the single cell independently flagged as length-confounded
> (MedQA's |dH|~|dLen| correlation is 0.513, the highest of any dataset, and `direct` is the most
> extreme length cue). Nothing else moves belief in-domain — not `confident_parametric`, not
> `elaborate`, on any dataset.

Two rules that must ship with it, both from the synthesis doc's "Claims that must NOT be made":

- **Never quote `direct`'s pooled row.** It is significantly heterogeneous across datasets
  (Friedman χ²=9.00, p=.011) and its sign *flips*: −0.053 / +0.105 / −0.047.
- **The pooled cue table is not an in-domain result.** Pooling the 18 model x dataset cells makes
  `confident_parametric` (+0.0186, q=.023) and `elaborate` (+0.0236, q=.045) nominally significant,
  but only by combining individually underpowered, same-signed estimates. It belongs in an appendix
  as a sensitivity analysis with its homogeneity precondition (the Friedman test) stated.

### 4.4 C4 — the null is *bounded*, not merely unrejected

**Read this together with C3, not against it.** C3 (per-dataset) is the claim that goes in the
paper. C4 does **not** overturn it — it says what "no detectable shift" is allowed to mean, and
stops one specific sentence.

#### The two tests, and which governs

| test | question it answers | `confident_parametric` | verdict |
|---|---|---|---|
| **In-domain** (primary): one-sample t over that dataset's 6 per-model estimates, BH over 12 | does belief move *within* a dataset? | FRAMES q=.217, MedQA q=.353, HotpotQA q=.228 — **none significant** | **this is the paper's claim** |
| **Cross-dataset** (appendix): does a consistent small shift recur across datasets? | is the in-domain null a true zero? | **+0.0186 bits, q=.050, 5/6 models positive** | small, marginal, one cue only |

They are not in conflict. The in-domain tests are **underpowered by construction** — minimum
detectable effect is 0.076–0.104 bits per model, and the effect is ≈0.02. A test that cannot
resolve 0.02 bits returning "not significant" is not evidence that the value is 0. The
cross-dataset test has the power to resolve it, and finds it.

#### ⚠️ The pooled number in the synthesis doc is pseudo-replicated — use the model-level one

`analyze_entropy_under_cue_stats.py`'s pooled table treats the **18 (model x dataset) cells as
independent**. They are not: the same 6 models appear in all 3 datasets, so each model is counted
three times. Collapsing each model to one number (its mean across the 3 datasets) and testing over
n=6 models changes the picture materially:

| cue | model-level (n=6) mean | 95% CI | q (BH/4) | naive 18-cell q |
|---|---:|---|---:|---:|
| `confident_parametric` | +0.0186 | [+0.0061, +0.0311] | **.050** | .023 |
| `elaborate` | +0.0236 | [−0.0100, +0.0572] | **.261** | **.045** |
| `multiturn` | −0.0031 | [−0.0444, +0.0383] | .920 | .934 |
| `direct` | +0.0017 | — | .920 | .934 | *(never aggregate — Friedman p=.011)* |

So **`elaborate`'s pooled significance is an artifact of pseudo-replication** and must not be
reported. `confident_parametric` survives, but at **q=.050 — exactly on the threshold**, from a
single cue, with 5 of 6 models positive. Reproduce:
`uv run python results/entropy_under_cue/model_level_cross_dataset_test.py`.
Recommend folding this into `analyze_entropy_under_cue_stats.py` so the producer emits it directly.

#### What C4 actually requires of the writer

1. **Add the boundedness evidence** — this is new and a reviewer will ask for it:
   - *The instrument is not saturated*: between `plain` and `confident_parametric`, **41–56% of
     examples change entropy level** (gemma4:31b: 159/300 flat, 80 up, 61 down). Entropy moves a
     great deal per example; it does not move *systematically*.
   - *Power must be conceded*: per-example SD of the paired delta is 0.47–0.65 bits; MDE at 80%
     power is **0.076–0.104 bits per model**.
   - *Bootstrap noise floor*: resampling each example's `plain` cluster proportions puts the null SD
     of a 5-run entropy difference at **0.021–0.028 bits**. Of the six HotpotQA models only
     gemma4:31b (+0.061) falls outside its own null band.

2. **Do not write an unqualified "belief does not move" / "entropy is unchanged."** Write *no
   detectable shift within any dataset*, and let the bound carry the weight: any shift is below
   ≈0.1 bits per model, ≈0.02 observed, against policy shifts of up to **+60 percentage points**.
   That is the claim the evidence supports, and it is the stronger one — a bounded effect is more
   informative than an unrejected null.

3. **Keep the cross-dataset result in the appendix**, reported at model level (q=.050), with its
   homogeneity precondition (Friedman) stated and `elaborate` **not** listed as significant. Do not
   promote it to the main text; it is one marginal cue and it cannot bear a headline.

### 4.5 C5 — the canonical-answer leg is still two datasets, in a subsection that now says three

§`sec:policy_not_uncertainty` has two legs. The entropy leg now covers three datasets; the
canonical-answer leg (*"Across 7,087 eligible examples… $\sim$10.6\%"*) does **not** —
`results/modal_answer_shift/modal_answer_shift_judged.csv` contains `frames` and `medqa` only (44
rows, 6 models x 4 cues), and both producers hardcode the two datasets
(`analyze_modal_answer_shift.py:55-57`, `analyze_modal_answer_shift_judged.py:61-63`).

As written the subsection reads as though both legs are three-dataset. Two options:

- **Cheap, do this now:** scope the sentence — "across FRAMES and MedQA" — and say HotpotQA's
  canonical-answer leg is not measured.
- **Better, if there is time:** run it on HotpotQA. Now feasible, because the cluster files exist;
  it needs the same plain-naming accommodation `analyze_entropy_under_cue.py:79-86` already carries
  and documents (HotpotQA names every run `<cond>_run_<r>`, so a bare cluster glob matches four
  files per model and would silently use a *cue's* entropy as the cue-free baseline). Judge pass is
  local gpt-oss:120b, same as the clustering — no cloud calls.

### 4.6 How Change C interacts with Change B — read this before writing either

Change B establishes that search volume and accuracy *are* coupled wherever a real search policy
exists. A careless reading of that could be taken to undercut the paper's thesis. Change C is what
prevents it, and the two should be written together:

On HotpotQA, `confident_parametric` moves the zero-search rate **+59.9pp**, costs **−21.4pp
accuracy**, and moves belief **+0.022 bits with 0 of 6 models significant**. That is the complete
causal picture in one cell: *the accuracy loss is real, and it is not accompanied by any meaningful
change in what the model knows or believes.* The damage is done by the policy shift, not by a
knowledge shift — which is precisely the paper's central claim, now with a measured accuracy cost
attached instead of a null one.

State it that way and the two changes reinforce each other. Keep them in separate sections written
by different hands and they will read as a contradiction.

## 4A. Change E — replace the SFT section with the resolved checkpoint

**Decision taken 2026-09-12:** the paper's §`sec:sft_interventions` is **stale**. It reports
`gemma4-frames-robust-q4km`, the checkpoint that cannot answer when no tool is offered (75-98% of
its no-search responses truncate mid-reasoning, root cause in `docs/resolved_sft_handoff.md` §1).
Report **`gemma4-frames-resolved-q4km`** instead. Its search-arm result is equivalent, its anchor is
much closer to baseline, and it is the only checkpoint on which the belief measurement is valid --
which is what lets this section close the loop on the paper's own thesis (§4A.4).

**Metric:** the paper's own — **mean |Δ| search calls vs. the model's own `plain`, plus the count of
perturbations significant** (paired Wilcoxon per perturbation), exactly as §`sec:sft_interventions`
and `app:sft_full_results` already report it. %Δ is given alongside only where the paper already
uses it (the HotpotQA transfer paragraph).

Producer: **`scripts/report_resolved_sft_paper_metric.py`** (new). It reproduces the paper's
existing baseline numbers exactly — FRAMES 1.30 / 6 of 9, HotpotQA 0.69 / 7 of 8 — which is the
check that it is measuring the same thing the paper measures.

### 4A.1 The numbers

| | FRAMES base | **FRAMES SFT** | HotpotQA base | **HotpotQA SFT** |
|---|---:|---:|---:|---:|
| `plain` search level | 4.85 calls | **5.50** | 2.14 calls | **2.17** |
| `plain` zero-search | 12.7% | **2.0%** | 6.0% | **2.7%** |
| `plain` accuracy | 54.9% | 51.0% | 81.0% | 80.0% |
| **mean \|Δ\| calls, all perturbations** | **1.30 (6 of 9)** | **0.77 (3 of 9)** | **0.69 (7 of 8)** | **0.33 (3 of 8)** |
| — trained perturbations | 0.80 (3 of 6) | **0.32 (1 of 6)** | 0.57 (4 of 5) | **0.14 (1 of 5)** |
| — held-out perturbations | 2.29 (3 of 3) | 1.66 (2 of 3) | 0.90 (3 of 3) | 0.64 (2 of 3) |
| run-to-run floor \|Δ\| | 0.22 (p=.267) | 0.18 (p=.112) | 0.03 (p=.745) | 0.06 (p=.378) |

In the %Δ unit the HotpotQA transfer paragraph already uses: trained perturbations **26.5% → 6.4%**
(4 of 5 → 1 of 5), held-out **42.3% → 29.7%**.

### 4A.2 Three things that get *better* than the current text

1. **The anchor objection largely goes away.** The paper currently concedes "invariance, not
   restoration: 4.85 → 6.08 calls on `plain`". The resolved checkpoint lands at **5.50 on FRAMES
   (+13%, versus robust's +25%) and 2.17 on HotpotQA (+1.4%, versus robust's +13%)** — out of
   domain it is essentially indistinguishable from the baseline's own level. The caveat should be
   softened accordingly, not deleted: FRAMES is still anchored high.
2. **Zero-search collapses**: 12.7% → 2.0% (FRAMES), 6.0% → 2.7% (HotpotQA). The intervention
   removes the "skip search entirely" failure mode, which is the behaviour Table~\ref{tab:zero_search}
   is about.
3. **Residual instability is ~2x the arm's own floor, and now measured on both datasets**: trained
   perturbations leave 0.32 calls against a 0.18-call floor (FRAMES, 1.8x) and 0.14 against 0.06
   (HotpotQA, 2.3x). The paper's "~1.6x its own run-to-run floor" phrasing survives, with new
   numbers, and now has an in-domain twin.

### 4A.3 Two things that get *worse*, and must be stated

1. **`#significant` moves the wrong way on FRAMES**: robust was 2 of 9, resolved is 3 of 9. In
   mean |Δ| the resolved checkpoint is better (0.77 vs 0.87); it just leaves one more perturbation
   individually detectable. Report both numbers and do not cherry-pick the metric.
2. **FRAMES accuracy is unresolvable, not "no meaningful cost."** 54.9% → 51.0% is −3.9pp, and the
   SFT's own accuracy floor on the same split is ±4.9pp — the gap sits inside its own run-to-run
   variation at n=102. **Delete the current "at no meaningful accuracy cost (54.9% → 53.9%)" claim
   and make the accuracy claim on HotpotQA instead** (81.0% → 80.0%, n=300). Say explicitly that
   n=102 cannot resolve it and that a larger held-out split — not more repeat runs — is what would.

### 4A.4 The new argument this unlocks — add it, it is the payoff

The robust checkpoint could not support any belief measurement. The resolved one can, and the
result is the cleanest statement of the paper's thesis available anywhere in the manuscript:

> The intervention changes the policy and leaves belief untouched. `plain` semantic entropy is
> **1.034 bits (SFT) vs 1.033 (baseline)** on FRAMES and **0.748 vs 0.756** on HotpotQA, and of the
> 8 (dataset x perturbation) cue-wise arm differences, **7 are not separable from zero** (the
> exception, FRAMES `multiturn` +0.103 bits, is isolated — the same cue is null on HotpotQA at
> −0.008 — and is best read as multiple-comparison noise).

This is the mirror image of §`sec:policy_not_uncertainty`: there, perturbations move policy without
moving belief; here, a *training intervention* moves policy back without moving belief either. The
policy layer is separable in both directions. Source: `analyze_resolved_sft.py`, ARM 2b.

### 4A.5 Edits required

- **§`sec:sft_interventions` ¶1** (line ~245): swap 1.30 (6 of 9) → **0.77 (3 of 9)**; 4.85 → 6.08
  becomes 4.85 → **5.50**; delete the accuracy-parity claim per §4A.3(2). The sentence about which
  held-out perturbations resist still holds — `confident` (−3.63 calls) and `multiturn` (−1.06) are
  the two significant ones, and `searchmulti` is now n.s. (−0.29, p=.060), matching the earlier text.
- **§`sec:sft_interventions` ¶ out-of-domain** (line ~258): 26.5% → **6.4%** (4 of 5 → 1 of 5) or, in
  calls, 0.57 → **0.14**; untrained 42.3% → **29.7%**; plain 2.14 → **2.17** with zero-search
  6.0% → 2.7%. Keep the MedQA degenerate-case paragraph unchanged — it is about the earlier
  checkpoint and the user has confirmed MedQA is not being re-run for this one; say which checkpoint
  each result belongs to.
- **Intro paragraph** (line 69) and **abstract**: same substitutions.
- **`app:sft_full_results`** (line ~450): this is robust-line detail. Keep it, relabel it explicitly
  as the earlier checkpoint, and add the resolved numbers as the primary table.
- **The 8- and 10-perturbation `confident`-exposure variants** (line ~263) are robust-line
  checkpoints. They remain a valid ablation about direct exposure to `confident`, but the text must
  say they belong to the earlier checkpoint family, or a reader will assume they are resolved-line.
- **Figure `fig:gemma_sft`**: both panels are robust-line. Regenerate from the resolved arm
  (`make_gemma_cue_figure.py` now carries the per-file offset test) or label the existing figure as
  the earlier checkpoint.

### 4A.6 Caveats that ship with it

1. **Training truncation at `SEQ=16384` is kept as-is** (decided 2026-09-12). It biases the training
   set's search level down (kept-example mean 1.95 vs a true 2.52). State it as a limitation; do not
   claim the checkpoint is trained on the full distribution.
2. **MedQA was not run on this checkpoint.** Correctly so — its baseline does zero search on 95.8%
   of examples at `plain`, so cue suppression is unmeasurable there. The existing MedQA
   degenerate-case argument stands on the robust checkpoint; attribute it.
3. **`confident_parametric` FRAMES rows are 486-499 of 501** (ollama returns a non-retryable 400 on
   a handful); backfills do not recover them.
4. **FRAMES `searchmulti` needed a per-file offset correction** on the *baseline* side (4.95 → 3.95).
   Any recomputation must decide this per file, never per condition — the two arms differ within the
   same dataset. See §6.3.

### 4A.7 The mediation analysis also moves to resolved — and gets sharper

`analyze_sft_intervention_mediation.py` (the producer behind the Discussion's *"restoring search
volume alone does not recover the accuracy lost"*) was robust-line and read `searchmulti` raw. Both
are fixed: it now takes `--checkpoint {resolved,robust}` (default **resolved**; the two share the
identical 102 held-out ids, verified 102/102 overlap) and applies the per-file mocked-history test.

The offset fix changes only the `searchmulti` row, and flips its sign: `d_calls_base` +0.10 →
**−0.90**, `d_calls_sft` +0.31 → **−0.69**, ratio 3.20 → 0.76. Every other row is unchanged.

**The resolved checkpoint makes the Discussion's claim much stronger.** On `direct`, search volume
is now almost perfectly restored while the accuracy cost is not:

| checkpoint | `direct` Δcalls base → SFT | calls ratio | Δacc base → SFT | acc ratio |
|---|---|---:|---|---:|
| robust (currently cited) | −1.73 → −0.78 | 0.46 | −22.6pp → −11.8pp (p=.004) | 0.52 |
| **resolved** | −1.73 → **−0.04** | **0.02** | −22.6pp → **−10.8pp** (p=.007) | 0.48 |

With robust, volume was only half-restored, so "volume alone doesn't recover accuracy" was a weak
inference. With resolved the volume effect is **eliminated** (ratio 0.02) and half the accuracy cost
still survives, significantly — that is the clean natural experiment the sentence wants.

⚠️ **But check the attribution while you are there.** The Discussion says the traced case is *"a
perturbation that erodes the necessity-tracking mechanism"*. The cue that actually demonstrates it
is `direct`, which for this model is classified **level-shift-only (calibration intact)**
(`results/cue_suppression_mechanism/cue_suppression_mechanism.csv`). The mechanism-eroding cue on
gemma4:31b/FRAMES is `confident_parametric`, and there volume is *not* restored (ratio 0.88), so it
cannot carry the claim. Either re-attribute the sentence to `direct` and drop the erosion clause, or
say explicitly which cue is meant.

**Reproduce:** `uv run python scripts/report_resolved_sft_paper_metric.py` (paper metric + floors)
and `uv run python scripts/analyze_resolved_sft.py` (%Δ, arm-vs-arm bootstrap, parametric arm,
entropy arm).

## 5. Consistency fixes found during the review

| id | Location | Issue | Fix |
|---|---|---|---|
| D1 | line 271 (Discussion) | *"predicts true correctness equally well on both datasets we test"* — stale; `tab:entropy_validity_correctness` now has three datasets | "on all three datasets we test" |
| D2 | Limitations (line 276) | *"the corresponding uncertainty evaluations and handling for HotpotQA remain ongoing and have not yet been fully analyzed"* — **now genuinely stale** (it was accurate when this plan was first written; the grid closed 2026-09-11, §6.2). It contradicts a Results section that reports HotpotQA entropy in two tables and §`sec:policy_not_uncertainty` | Replace with what is actually unavailable on HotpotQA: no LLM judge (EM only), no thinking-token/suppression data (no Logfire traces), no `cue_suppression_mechanism` rows, no no-search value control, no live-web replication — and, per §4.5, no canonical-answer leg |
| D3 | §`sec:decoupling` title | see §3.6 | retitle |
| D4 | §`sec:mechanism` (line 199) | *"50 of 62 on FRAMES, and all 60 on MedQA"* is **stale** (computed from a mechanism CSV that lagged its own upstream), and HotpotQA is missing | Use **58 of 70** (FRAMES), **67 of 73** (MedQA — no longer "all": 6 eroded cells), **50 of 54** (HotpotQA). Numbers and caveats in §6.1 |

---

## 6. Deliberately not done — and what it would cost

### 6.1 HotpotQA mechanism classification — DONE 2026-09-11

**Status: run, not pending.** This section previously said "blocked", on the mistaken reasoning
that the analysis needs entropy-under-cue (only 4 cues probed). It does not: the necessity proxy is
the **cue-free** entropy probe, a pre-treatment covariate
(`analyze_necessity_vs_template_search_5run.py:5-9`). HotpotQA had every input already.

#### Results — HotpotQA reproduces the mechanism finding

| dataset | cells | level-shift-only (calibration intact) | mechanism-breaking |
|---|---:|---:|---|
| FRAMES | 70 | **58 (83%)** | 7 sharpened, 3 eroded, 2 marginal |
| MedQA | 73 | **67 (92%)** | 6 eroded (all `nemotron-cascade-2`) |
| **HotpotQA** | **54** | **50 (93%)** | 4 "inverted" (all `nemotron-3-nano`) |

**Five of six HotpotQA models are 100% level-shift-only** (9 of 9 cues each): `gemma4:31b`,
`gpt-oss:120b`, `gpt-oss:20b`, `nemotron-cascade-2:30b`, `qwen3.5:122b`. Every breaking cell
belongs to `nemotron-3-nano:30b` — the same model that inverts direction on FRAMES and is this
project's standing non-conformer.

⚠️ **Do not describe those 4 cells as "calibration eroded."** They are labelled *inverted* because
`nemotron-3-nano`'s HotpotQA **plain slope is already slightly negative** (−0.128), so the
classifier's slope-ratio flips sign on any positive change. What actually happens is the cue moves
the slope *up*, to +0.40…+0.97 — i.e. it **improves** necessity-tracking from a near-zero base
rather than destroying it. With a near-zero denominator the "inverted" label is a sign artifact.
Say so if the cells are mentioned at all.

**Paper-ready sentence:** the necessity-tracking mechanism survives the perturbation in 83% / 92% /
93% of cells on FRAMES / MedQA / HotpotQA; only the level shifts.

#### ⚠️ The paper's current §`sec:mechanism` numbers are stale

It says *"50 of 62 on FRAMES, and all 60 on MedQA."* Those come from a `cue_suppression_mechanism.csv`
that was **122 rows while its own upstream interaction CSV already had 143** — it had not been
re-run after the roster widened. Current values are **58 of 70** and **67 of 73**, and the MedQA
claim changes qualitatively: it is **no longer "all"** — 6 cells are calibration-eroded, all of them
`nemotron-cascade-2`. Update all three numbers, and drop "all".

#### What was changed in the code

1. `analyze_necessity_vs_template_search_5run.py` — a `hotpotqa` entry in `DATASETS`, a
   `hotpotqa_cond()` parser (cue list matched longest-first so `confident_parametric` is not
   shadowed and `plain_rep2` is not read as `plain`), `hotpotqa_plain_cond_for()` returning bare
   `"plain"`, and a **plain-specific entropy glob** (a bare wildcard matches five cluster files per
   model and would silently use a *cue's* entropy as the cue-free baseline).
2. **The roster is per dataset**, via a `models` key. FRAMES/MedQA keep their original 5 models;
   HotpotQA gets 6 (adds `qwen3.5:122b`). Widening globally would have changed the FRAMES/MedQA
   cell counts the paper quotes.
3. **The BH family is now per dataset too.** It was global: adding HotpotQA shifted all 143
   pre-existing FRAMES/MedQA q-values. This is the same failure Change A documents, in a different
   script. With per-dataset families, **0 of the shared FRAMES/MedQA cells change classification**
   (verified by diff against the pre-change outputs).
4. `analyze_volume_accuracy_decoupling.py` — HotpotQA paths, the two extra model tags, and
   exclusion of the 14 yes/no golds from *accuracy only* (kept for search volume), matching
   `grade_hotpotqa_regex.py`.

Reproduce:

```bash
uv run python scripts/analyze_necessity_vs_template_search_5run.py
uv run python scripts/analyze_cue_suppression_mechanism.py
uv run python scripts/analyze_volume_accuracy_decoupling.py
```

### 6.2 `confident_parametric` entropy on HotpotQA — CLOSED 2026-09-11

**No longer an open item.** Rollouts were collected 2026-09-08 and clustering completed 2026-09-11:
all 6 baseline models carry 5 cue cluster files under `results/hotpotqa_parametric/`. Result —
**+0.022 bits, 0 of 6 models significant**, against a +59.9pp zero-search shift. The grid is now
72 cells. Everything that follows from it is **§4 (Change C)**; do not treat it as missing.

Two related arms are genuinely still partial and must **not** be folded into the 6-model roster —
they are trained to be cue-invariant by construction: `gemma4-frames-robust-q4km` (2 of 5 cue
cluster files) and `gemma4-frames-resolved-q4km` (5 of 5, but it belongs to the sibling
`docs/resolved_sft_handoff.md` arm).

⚠️ **Provenance trap, learned the hard way twice on this item.** Which remote is authoritative
differs per dataset: FRAMES/MedQA parametric lives on **srv3**
(`/data/home/dvirla/parametric_search_tradeoff`; Athena holds partials from an earlier pass),
HotpotQA parametric on **Athena** for five models and an **srv3 worktree** for `qwen3.5:122b`.
Pulling from the wrong one has produced a false "data gap" claim in these docs more than once.
Check both, and check the filesystem rather than a doc's status line, before declaring anything
incomplete.

### 6.3 `searchmulti` counter offset — narrower than this plan first stated

**Corrected 2026-09-11 against the audit in
[`cross_dataset_uncertainty_findings.md`](cross_dataset_uncertainty_findings.md) §6.1.** Earlier
revisions of this plan (and `hotpotqa_paper_integration.md` §4 caveat 5) implied the paper's
FRAMES/MedQA `searchmulti` numbers were wrong and needed re-collection. **They are not.** The raw
JSONs do carry inflated `sampler_search_calls` (+1 per mocked history call; +2/+3 for the 2- and
3-round variants), but the main pipeline corrects them **at read time**, so every published number
is sound.

What is actually true is narrower and still worth knowing: the correction is applied by some
scripts and not others. `make_aggregate_cue_tradeoff_figure.py`, `make_cue_briefing_figures.py`,
`make_gemma_cue_figure.py`, `analyze_volume_accuracy_decoupling.py`, `dual_metric_analysis.py` and
`grade_hotpotqa_regex.py` **do** correct — which covers every producer this plan relies on, so
§1–§3's numbers are unaffected. `analyze_hotpotqa_transfer.py`, `compare_searchmulti_rounds.py`,
`summarize_frames_cues_grid.py`, `regrade_regex.py` and `analyze_thinking_tokens.py` **do not**.
The synthesis doc carries the full two-column list.

**Rule for the writer: never quote a `searchmulti` search-volume number without checking which
script produced it.**

#### ⚠️ A live over-correction on HotpotQA, found 2026-09-11

`grade_hotpotqa_regex.py:161-166` resolves the offset as *"use the row's own
`history_search_calls` if present, else subtract the per-condition constant."* No HotpotQA row
carries that field, so the constant always wins — but the files are **not** uniformly pre-fix:

| `searchmulti` file | min calls | zero rows | state |
|---|---:|---:|---|
| `gemma4-frames-resolved-q4km` | **0** | **17/300** | **post-fix — already correct** |
| the other 10 models | 1 (robust: 2) | 0/300 | pre-fix — needs −1 |

So a re-grade would silently subtract 1 from all 300 `searchmulti` rows of the resolved SFT
checkpoint, which are already correct. **No published number is affected yet** — that checkpoint
landed after the current `per_row.csv` was written and is absent from it — but the next
`grade_hotpotqa_regex.py` run will corrupt it.

Fix before re-grading, and before adding HotpotQA to the mechanism script (§6.1): make the fallback
conditional on the file actually looking pre-fix (`min(sampler_search_calls) >= offset` and a 0%
zero-search rate — the same empirical test the script's own comment at lines 70-72 describes but
does not enforce), rather than applying the constant unconditionally. The inflation is large enough to flip a sign — uncorrected, FRAMES
`searchmulti` shows a −4.9pp *decrease* in zero-search vs. plain; corrected it is an *increase*.

## 7. Reproduction inventory

Every number in this plan, and where it comes from:

| Numbers | Source | Command |
|---|---|---|
| §2.3 star diffs | `results/cue_briefing_per_dataset/{frames,medqa}/brief_aggregate_tables.md` vs `results/cue_briefing/` | see §2.2 |
| §1, §3.2 cell-level ρ (FRAMES/MedQA) | `results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv` | `uv run python scripts/analyze_volume_accuracy_decoupling.py` |
| §3.2 HotpotQA cell-level ρ, §3.2 cue-aggregate | `results/hotpotqa_cue_grid_regex/per_row.csv` | `results/cue_briefing_per_dataset/hotpotqa_cell_level.py` |
| §3.3 floor-relative example correlations | `results/hotpotqa_cue_grid_regex/per_row.csv` | `results/cue_briefing_per_dataset/floor_relative_example_correlation.py` |
| §3.4 dual-metric per-dataset split | `results/dual_metric_cue_deltas.csv` | `uv run python scripts/dual_metric_analysis.py` |
| Published HotpotQA aggregates | `results/hotpotqa_cue_briefing/brief_aggregate_tables.md` | unchanged |
| §4.1 entropy-vs-correctness (EM + judge, 6 models) | `results/entropy_vs_correctness/entropy_vs_correctness.csv` | `uv run python scripts/analyze_entropy_vs_correctness.py` |
| §4.2–4.4 entropy under cue, in-domain + pooled families | `results/entropy_under_cue/entropy_under_cue{,_indomain,_pooled}.csv` | `uv run python scripts/analyze_entropy_under_cue.py` then `analyze_entropy_under_cue_stats.py` |
| §4.5 canonical-answer leg (FRAMES+MedQA only) | `results/modal_answer_shift/modal_answer_shift_judged.csv` | `uv run python scripts/analyze_modal_answer_shift_judged.py` |

Both helper scripts use repo-relative paths — run them from the repo root with `uv run python`.

Note `results/` is gitignored — these outputs live only on this machine.

---

## 8. Checklist

- [ ] A1 two star additions in `tab:zero_search` FRAMES column
- [ ] A2 caption: HotpotQA-specific FDR sentence deleted, per-dataset rule stated
- [ ] A3 `fig:combined_search_acc` caption states the family rule
- [ ] A4 five PNGs copied from `results/cue_briefing_per_dataset/`
- [ ] A5 decision made on MedQA ELABORATE (report / don't report), and recorded
- [ ] B1 §`sec:decoupling` ¶1 rewritten — `ρ=+0.168, n=90` no longer presented as evidence of no relationship
- [ ] B2 example-level claim retained, HotpotQA hedge replaced with the floor-relative result, FRAMES/MedQA range corrected
- [ ] B3 section retitled; "decoupling" reserved for the signal↔policy sense
- [ ] B4 `app:dual_metric` "Decoupling Confirmation" rewritten with the per-dataset split + Spearman
- [ ] B: mediation guardrail present (cite line 271's restoration result)
- [ ] B: DIRECT length confound still flagged; note removing it strengthens the coupling
- [ ] C1 `tab:entropy_validity_correctness` rebuilt as EM + judge columns, MedQA ρ = −0.526 not −0.603, MedQA range filled
- [ ] C1 caption says MedQA's low EM ρ is EM-on-option-text, not instrument failure; EM column kept (HotpotQA has no judge)
- [ ] C2 mean |Δ| entropy reads 0.039–0.042 bits over 72 cells
- [ ] C3 in-domain framing adopted (1 of 12, `direct`/MedQA, length-confounded); `direct` never pooled; pooled table demoted to appendix with its Friedman precondition
- [ ] C4 boundedness paragraph added; no sentence claims an unqualified "belief does not move" — the wording is "no detectable shift within any dataset", with the ≈0.1-bit bound
- [ ] C4 cross-dataset result, if reported at all, is appendix-only, **model-level (q=.050)**, and does **not** list `elaborate` as significant
- [ ] C5 canonical-answer sentence scoped to FRAMES+MedQA (or HotpotQA run and folded in)
- [ ] C/B written together per §4.6 — the confident_parametric cell states cost AND belief-stability in one place
- [ ] E1 SFT section reports the **resolved** checkpoint throughout; every robust-line number is relabelled as the earlier checkpoint (incl. `fig:gemma_sft`, `app:sft_full_results`, the 8/10-perturbation variants, and the MedQA paragraph)
- [ ] E1 metric is mean |Δ| calls + # significant, the paper's own — %Δ only where the paper already used it
- [ ] E2 no "no meaningful accuracy cost" claim on FRAMES; n=102 stated as unable to resolve ±4.9pp
- [ ] E3 belief-invariance paragraph added, with the isolated FRAMES `multiturn` cell flagged as noise
- [ ] E: truncation (`SEQ=16384`) stated as a limitation; MedQA-not-run attributed to the earlier checkpoint
- [ ] D1–D4 consistency fixes
- [ ] D4 §`sec:mechanism` updated to 58/70, 67/73, 50/54 — MedQA is no longer "all", and HotpotQA is reported (§6.1)
- [ ] the 4 HotpotQA "inverted" cells, if mentioned, are described as a near-zero-denominator sign artifact, not erosion
- [ ] §6.3 `searchmulti` fallback made conditional before any re-grade (the resolved checkpoint is already-corrected data)
- [ ] Paper recompiles; every `\ref` still resolves
