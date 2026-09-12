# Uncertainty across FRAMES, MedQA and HotpotQA — cross-dataset synthesis

**Written:** 2026-09-08. **Branch:** `hotpotqa-cue-pilot`.
**Scope:** everything downstream of semantic entropy, on all three datasets. The search-volume
and SFT-transfer results live in [`hotpotqa_paper_integration.md`](hotpotqa_paper_integration.md);
this document is the *uncertainty* axis and supersedes that file's §5 "Pending" row.

The three questions, in the order the paper needs them:

1. **Is entropy a valid instrument?** (entropy → correctness)
2. **Does uncertainty drive search?** (entropy → search calls)
3. **Do perturbations move belief, or only policy?** (entropy under cue)

Roster throughout: the **6 models** with a cue-free 5-run `no_search` probe on all three datasets —
`gemma4:31b`, `gpt-oss:20b`, `gpt-oss:120b`, `nemotron-3-nano:30b`, `nemotron-cascade-2:30b`,
`qwen3.5:122b`. Entropy = Shannon entropy over gpt-oss:120b clusterings of 5 independent no-search
rollouts, identical judge and prompt on every dataset.

---

## 1. Entropy is a valid instrument, and equally so on HotpotQA

`results/entropy_vs_correctness/entropy_vs_correctness.csv` (`scripts/analyze_entropy_vs_correctness.py`)

Spearman ρ between an example's entropy and its mean correctness over the same 5 rollouts.

| Dataset | ρ (EM), mean | range | acc @ H=0 | acc @ H>0 | separation |
|---|---:|---|---:|---:|---:|
| FRAMES | **−0.488** | −0.409 … −0.537 | 0.617 | 0.180 | 3.4× |
| HotpotQA | **−0.552** | −0.408 … −0.645 | 0.699 | 0.261 | 2.7× |
| MedQA | −0.174 | −0.101 … −0.272 | 0.451 | 0.268 | 1.7× |

**HotpotQA validates the instrument at least as strongly as FRAMES** — every model negative, all
six inside −0.41…−0.65, mean slightly *stronger* than FRAMES, and consistent in sign for 5 of 6
models. A model that answers the same way 5 times is right ~2.7× as often as one that does not.

### 1.1 The grader is not neutral here — quote EM against EM

This is the single most important methodological point in this document.

The paper grades FRAMES with EM and MedQA with an LLM judge. For *accuracy levels* that is a
documented, defensible split. For *this correlation* it is not interchangeable, because EM's
undercount is **correlated with entropy** rather than uniform: high-entropy examples produce
hedged, verbose, reformulated answers that EM misses disproportionately. So EM does not shift the
level, it **attenuates the correlation**:

| Dataset | ρ (LLM judge) | ρ (EM) | attenuation |
|---|---:|---:|---:|
| FRAMES | −0.627 | −0.488 | ~0.14 |
| MedQA | −0.526 | −0.174 | **~0.35 — near-vanishing** |

> **Corrected 2026-09-11.** This column previously read −0.649 / −0.603. Those were **5-model**
> means computed before the roster was widened; the EM column beside them was already 6-model, so
> the two columns were not comparable and the attenuation was overstated. The 6-model judge means
> are −0.627 (FRAMES) and −0.526 (MedQA), verified against
> `results/entropy_vs_correctness/entropy_vs_correctness.csv`. The change is driven almost entirely
> by `qwen3.5:122b`, whose MedQA judge ρ is −0.141 against −0.556…−0.654 for the other five — it is
> the one model whose MedQA judge correlation is weak, and dropping it inflated the column. The
> qualitative point is unchanged: EM attenuates this ρ, mildly on FRAMES and severely on MedQA.

HotpotQA has **no LLM judge at all** (every row `--no_grader`), so its only available number is EM.
**Therefore: compare HotpotQA's ρ to the other datasets' EM column, never to their judge column.**
`analyze_entropy_vs_correctness.py` now emits `rho_em` on every row for exactly this reason; a
single `grading` tag previously hid the issue and made an EM number look comparable to a judge one.

MedQA's EM row (−0.174) should not be read as "the instrument fails on MedQA" — its judge row
(−0.603) is in line with the other two datasets. It is a statement about EM on multiple-choice
option text, which the paper already documents as a 26–36pp undercount.

---

## 2. Uncertainty drives search — strongly on FRAMES, weakly on HotpotQA, and MedQA's "null" is a pooling artifact

`results/param_vs_search_llm_5run/llm_entropy_vs_search_stats.csv`
(`scripts/analyze_llm_entropy_vs_search_5run.py`)

Spearman ρ between an example's cue-free entropy and the search calls the agent spends on that
same example in the `plain` search condition.

| Dataset | pooled ρ | **mean per-model ρ** | median | models with sig. positive ρ | calls @ H=0 (range) |
|---|---:|---:|---:|---:|---|
| FRAMES | +0.300 (p=2e-63) | **+0.318** | +0.319 | **6/6** | 2.01 – 6.57 |
| HotpotQA | +0.146 (p=5e-10) | **+0.129** | +0.117 | 4/6 | 1.34 – 4.03 |
| MedQA | +0.014 (p=.44, n.s.) | **+0.129** | +0.112 | 3/6 | **0.04 – 4.87** |

### 2.1 Revising the "MedQA null"

The project's standing note records MedQA as a null for this relationship. With the roster widened
to 6 models that reading no longer holds as stated. MedQA's *pooled* ρ is still null (+0.014), but
its *mean per-model* ρ is **+0.129 — identical to HotpotQA's**. The two models added here are the
only MedQA models that meaningfully search, and both couple clearly:

| Model | MedQA ρ | p | calls @ H=0 |
|---|---:|---|---:|
| `nemotron-cascade-2:30b` | **+0.350** | 7.5e-16 | 4.87 |
| `qwen3.5:122b` | **+0.179** | 5.7e-05 | 1.64 |
| the other four | −0.03 … +0.22 | mostly n.s. | 0.04 – 0.22 |

Pooling raw (entropy, calls) pairs across models whose baseline search levels differ by ~100×
destroys the within-model signal. **The correct statement is not "uncertainty does not drive search
on MedQA" but "four of six MedQA models sit on a zero-search floor where there is nothing to
correlate; the two that do search behave like FRAMES models."** This is the same floor that makes
MedQA a degenerate venue for the SFT transfer result, so it is one explanation, not two.

Recommend reporting **mean per-model ρ** as the headline and pooled ρ only as a footnote, on all
three datasets.

### 2.2 The genuine cross-dataset gap

Even corrected, FRAMES' coupling (+0.318) is ~2.5× HotpotQA's (+0.129). Entropy predicts
*correctness* about equally well on both (§1), but predicts *search behaviour* far better on
FRAMES. So the gap is not instrument quality — it is that HotpotQA agents' search decisions track
their own uncertainty less. Consistent with HotpotQA's shorter multi-hop chains and its
bounded 67k-passage corpus, where a fixed 1–2 searches often suffices regardless of confidence.

`nemotron-3-nano:30b` is the one consistent non-conformer: slightly negative on both HotpotQA
(−0.085) and MedQA (−0.033), weakest on FRAMES (+0.165).

---

## 3. Perturbations move policy, not belief — now on three datasets

`results/entropy_under_cue/entropy_under_cue.csv` (`scripts/analyze_entropy_under_cue.py`)

Same cue, applied to a **no-search** rollout, paired per example against that model's own cue-free
`plain` probe. If entropy moves, the cue changed what the model believes; if it does not while
search volume moves, the cue changed only the trigger.

| Dataset | cells | mean Δ | mean \|Δ\| | range | sign-test p<.05 *(uncorrected)* | ρ(plain,cue) |
|---|---:|---:|---:|---|---:|---|
| FRAMES | 24 | −0.002 | 0.041 | −0.131 … +0.083 | 4/24 | 0.77 |
| MedQA | 24 | +0.029 | 0.042 | −0.049 … +0.164 | 8/24 | 0.57 |
| HotpotQA | 24 | +0.003 | 0.039 | −0.133 … +0.100 | 3/24 | 0.75 |

**Mean |Δ| is 0.039–0.042 bits on all three datasets** (0.041 over all 72 cells) — remarkably stable, and negligible against
between-model spread (HotpotQA plain entropy runs 0.745 → 1.224 across the roster). Meanwhile the
same cues move the zero-search rate by **+5 to +60 percentage points**:

| Cue | Δ zero-search (FRAMES / MedQA / HotpotQA) | Δ entropy (FRAMES / MedQA / HotpotQA) |
|---|---|---|
| RERUN (noise floor) | +0.3 / +1.7 / +0.2 pp | — |
| CONFIDENT | **+44.1 / +24.1 / +59.9 pp** | +0.020 / +0.012 / **+0.022** |
| DIRECT | +10.0 / +13.8 / +11.0 pp | −0.053 / **+0.105** / −0.046 |
| MULTITURN | +15.0 / +25.1 / +18.2 pp | −0.007 / +0.007 / −0.006 |
| ELABORATE | +4.4 / +7.0 / +5.4 pp | +0.024 / +0.005 / +0.042 |

This is the **pure policy shortcut** reading, now replicated on a third dataset: the decision
threshold moves by tens of percentage points while the underlying self-consistency moves by
hundredths of a bit.

### 3.0 CONFIDENT closes the strongest case (added 2026-09-10)

`confident_parametric` is the cue with the largest search effect on every dataset, and the one
whose belief-vs-policy status matters most. It is now measured on all three:

| Dataset | Δ entropy (mean over models) | range | models with sign-test p<.05 |
|---|---:|---|---:|
| FRAMES | +0.026 | −0.014 … +0.060 | 0/6 |
| MedQA | +0.008 | −0.013 … +0.025 | 0/6 |
| **HotpotQA** | **+0.022** | −0.013 … +0.061 | **0/6** |

> **Corrected 2026-09-11 (roster).** The FRAMES and MedQA rows previously read +0.020 (1/5) and
> +0.012 (0/5) — 5-model means predating the provenance fix that brought both to 6. All three rows
> are now 6-model and match `results/entropy_under_cue/entropy_under_cue_indomain.csv`. Note the
> FRAMES sign-test count drops to 0/6: **no model on any dataset reaches a per-cell sign test.**

On HotpotQA the cue moves the zero-search rate by **+59.9pp** — the largest single behavioural
effect anywhere in this project — while moving entropy by **+0.022 bits, with not one of the six
models reaching significance**. The three datasets agree to within 0.02 bits despite differing
wildly in domain, retrieval corpus and baseline search level.

#### Is +0.022 real, or an artifact of 5-run resolution? (checked 2026-09-10)

With 5 samples, per-example entropy is quantised — the only attainable values are
0, 0.722, 0.971, 1.371, 1.522, 1.922, 2.322 bits — so a mean shift of 0.022 is a small fraction
of one quantum. Three checks:

**(a) The instrument is not saturated.** Between `plain` and `confident_parametric`, **41–56% of
examples change entropy level** (e.g. gemma4:31b: 159/300 flat, 80 up, 61 down). It moves a great
deal per example; it simply does not move systematically. So the small mean is not a floor effect.

**(b) Single-dataset tests are underpowered.** Per-example SD of the paired delta is 0.47–0.65
bits, giving SE ≈ 0.027–0.037 on a 300-example mean. The minimum detectable effect at 80% power is
**0.076–0.104 bits per model** — five times the effect we are trying to resolve. Every per-model
95% CI includes zero, and per-dataset t-tests over models are n=6. **"Not significant" here does
not mean "zero".**

**(c) A parametric bootstrap gives the noise floor.** Treating each example's `plain` cluster
proportions as the true distribution and drawing two independent 5-samples, the null mean delta has
**SD ≈ 0.021–0.028 bits** (95% band ≈ ±0.04–0.05) — closely matching the observed SEs. Of the six
HotpotQA models, only gemma4:31b (+0.061) falls outside its own null band; the rest sit inside it.

**(d) Pooling the 18 model×dataset estimates, the effect is small but REAL:**

| Aggregation | n | mean Δ | 95% CI | p |
|---|---:|---:|---|---:|
| FRAMES (models) | 6 | +0.0255 | [−0.0058, +0.0568] | .091 |
| MedQA (models) | 6 | +0.0078 | [−0.0071, +0.0228] | .235 |
| HotpotQA (models) | 6 | +0.0223 | [−0.0077, +0.0524] | .114 |
| **All 18** | **18** | **+0.0186** | **[+0.0061, +0.0310]** | **.0059** |

> **Corrected 2026-09-11 (roster).** This table was 16-cell (FRAMES/MedQA at n=5). The 18-cell
> figures come straight from `results/entropy_under_cue/entropy_under_cue_pooled.csv`. The
> conclusion is unchanged to three decimal places.

So `confident_parametric` **does** raise entropy slightly and consistently — 12 of 18 estimates are
positive — but the effect is bounded above by **+0.031 bits, which is 6.5% of the 0.479-bit
between-model spread** in plain entropy.

**Revised claim.** Not "belief does not move." Rather: *the cue moves belief a little and policy
enormously.* Telling a model it already knows the answer raises its answer-inconsistency by
≈0.02 bits (≈6% of between-model variation) while raising its zero-search rate by up to 60
percentage points. The policy-shortcut reading survives, but as a statement about **relative
magnitude**, not about a null belief effect — and any single-dataset test is too underpowered to
carry it alone.

#### The same test, applied to every cue — in-domain first

**Pooling across datasets is the wrong default here.** The three datasets differ in domain,
corpus and baseline search level, and at least one cue's effect demonstrably flips sign between
them. The primary analysis is therefore **per dataset**: a one-sample t over that dataset's 6
per-model estimates, BH-FDR across all 12 (dataset × cue) tests.

| Cue | FRAMES | MedQA | HotpotQA |
|---|---|---|---|
| `confident_parametric` | +0.026 (q=.22) | +0.008 (q=.35) | +0.022 (q=.23) |
| `elaborate` | +0.024 (q=.35) | +0.005 (q=.91) | +0.042 (q=.19) |
| `multiturn` | −0.003 (q=.96) | +0.000 (q=.99) | −0.006 (q=.91) |
| **`direct`** | −0.053 (q=.19) | **+0.105 (q=.018) ✱** | −0.047 (q=.19) |

All corrected numbers in this section come from **`scripts/analyze_entropy_under_cue_stats.py`**
(outputs: `results/entropy_under_cue/entropy_under_cue_{indomain,pooled}.csv`), not from ad hoc
computation. Its BH implementation is verified identical to
`statsmodels.stats.multitest.multipletests(method="fdr_bh")` to 1.1e-16.

**Exactly one of twelve cells shows a detectable in-domain belief shift: `direct` on MedQA**
(+0.105 bits, 6/6 models positive, q=.018). That is the cell §3.1 independently flags as
measurement-suspect — MedQA's |dH|~|dLen| correlation is 0.513, and `direct` is the most extreme
length cue (2–3 words vs 31–97 for plain) against gold answers that are long option strings. So
the single surviving effect is the one we have the strongest reason to distrust.

**Nothing else moves belief in-domain.** Not `confident_parametric`, not `elaborate`, on any
dataset.

##### When is pooling legitimate?

Only when the datasets are exchangeable for that cue. A Friedman test across the three datasets
(repeated measures over the same 6 models) answers this per cue:

| Cue | χ² | p | pooling |
|---|---:|---:|---|
| `direct` | 9.00 | **.011** | **INVALID — sign flips: −0.053 / +0.105 / −0.047** |
| `elaborate` | 4.00 | .135 | permissible |
| `confident_parametric` | 0.33 | .847 | permissible |
| `multiturn` | 0.00 | 1.000 | permissible |

For the three homogeneous cues, pooling the 18 model×dataset cells gives `elaborate` +0.0236
(q=.045) and `confident_parametric` +0.0186 (q=.023) as nominally significant, `multiturn` null.

> **⚠️ Corrected 2026-09-11 — this pooled test is pseudo-replicated.** The 18 cells are 6 models x
> 3 datasets; each model is counted three times, so they are not independent. Collapsing each model
> to one number (its mean across the 3 datasets) and testing over n=6 models, BH over the same 4
> cues, gives:
>
> | cue | model-level mean | 95% CI | q (BH/4) | naive 18-cell q |
> |---|---:|---|---:|---:|
> | `confident_parametric` | +0.0186 | [+0.0061, +0.0311] | **.050** | .023 |
> | `elaborate` | +0.0236 | [−0.0100, +0.0572] | **.261** | .045 |
> | `multiturn` | −0.0031 | [−0.0444, +0.0383] | .920 | .934 |
>
> **`elaborate`'s pooled significance does not survive** — it was an artifact of counting each model
> three times. `confident_parametric` does survive, but lands exactly on q=.050 with 5/6 models
> positive. Reproduce: `uv run python results/entropy_under_cue/model_level_cross_dataset_test.py`.
> Worth folding into `analyze_entropy_under_cue_stats.py` so the producer emits it directly.
> This strengthens rather than weakens §5's recommendation: the in-domain statement is the one to
> publish, and the pooled row is a marginal, single-cue sensitivity analysis.
But read that for what it is: **a cross-domain claim that a small positive shift recurs, not
evidence of an in-domain effect** — it reaches significance only by combining three individually
underpowered, consistently-signed estimates. `direct` must never be pooled.

##### Is the verdict sensitive to the family choice?

No. `direct`/MedQA is the only survivor under every partition of the tests:

| Family | m | significant |
|---|---:|---|
| all 12 (dataset × cue) — **used** | 12 | `direct`/MedQA |
| within each dataset (4 cues) | 4 | `direct`/MedQA |
| within each cue (3 datasets) | 3 | `direct`/MedQA |
| **uncorrected** | — | `direct`/MedQA **and** `direct`/FRAMES |

Uncorrected, `direct` on FRAMES also clears .05 (p=.046) — with the **opposite sign** (−0.053 vs
+0.105). Under any correction it drops out. That is further evidence `direct` is behaving
inconsistently rather than showing a coherent effect, consistent with its Friedman p=.011.

Note the two families in play, which are deliberately different: the in-domain table corrects over
its own 12 tests; the pooled sensitivity table over its own 4. The per-cell sign tests in §3's
summary table are **uncorrected** and marked as such — they are a per-cell diagnostic, not a
result, and must not be read alongside the q-columns as if comparable.

**What to put in the paper.** The conservative, in-domain statement is the defensible one:
*no perturbation produces a detectable shift in model belief within any dataset, with the single
exception of `direct` on MedQA, which is confounded with response length.* The pooled result
belongs in an appendix as a sensitivity analysis, with its homogeneity precondition stated.

Either way the magnitude comparison is unchanged and is the actual point: bounded belief shifts of
≈0.02–0.03 bits against policy shifts of up to **+60 percentage points**.

#### `n` and a provenance trap worth knowing

`n` is the number of **models** contributing a 5-run cluster file for that (dataset, cue). It is
now **6 everywhere** — 72 cells total.

It briefly read 5 for FRAMES/MedQA `confident_parametric` and `multiturn`, and that was a
**provenance error on my part, not a data gap**. The FRAMES/MedQA parametric trees exist on BOTH
remotes, and they are not equivalent:

| | `confident_parametric` | `multiturn` |
|---|---|---|
| Athena `~/parametric_search_tradeoff` | FRAMES 3/5, MedQA 2/5 | FRAMES 3/5, MedQA 2/5 |
| **srv3 `/data/home/dvirla/parametric_search_tradeoff`** | **5/5 both** | **5/5 both** |

I had pulled from Athena and concluded the runs were missing. **For the FRAMES and MedQA
parametric arms, srv3's main checkout is the source of truth**; Athena holds partials from an
earlier pass. (HotpotQA is the reverse for five of six models — see the integration doc.) Always
check both before declaring a gap. One cell (`medqa`/`qwen3.5_122b`/`confident_parametric`) had
complete runs but no cluster file; it has since been clustered.

### 3.1 The one systematic exception: DIRECT on MedQA

DIRECT is the only cue whose entropy effect is both large and consistent — and only on MedQA,
where it raises entropy in **5 of 6 models** (mean +0.105, up to +0.164). On FRAMES and HotpotQA
the same cue *lowers* entropy slightly (−0.053, −0.046). DIRECT is also the cue with the most
extreme length effect (median 2–3 words vs. 31–97 for plain), so on MedQA — whose gold answers are
long option strings — a two-word answer plausibly clusters differently for reasons of form rather
than belief. MedQA's |dH|~|dLen| correlation reaches 0.513, the highest of any dataset, versus
0.254 (FRAMES) and 0.338 (HotpotQA). **Treat DIRECT-on-MedQA as measurement-suspect**, and note
that this is the same cue whose accuracy bar the paper already flags as length-confounded.

### 3.2 Length is not driving the other cues

ELABORATE adds ~1,000–1,200 chars on both FRAMES and HotpotQA, yet |dH|~|dLen| correlations stay
near zero (max 0.254 / 0.338). So the clustering-artifact worry does not bear out for the
non-DIRECT cues — the small entropy shifts are small, not hidden by verbosity.

---

## 4. What HotpotQA cannot say

1. ~~**No `confident_parametric` entropy.**~~ **CLOSED 2026-09-10.** The 4-cell run was
   executed (6 models x 5 runs x 300 = 9,000 rollouts, Athena for five models + srv3 for
   qwen3.5:122b, each on the machine that produced its other cues) and clustered. Result in
   §3.0: +0.022 bits, 0/6 models significant. The gap this document opened is closed.
2. **No LLM judge**, hence §1.1's EM-only constraint.
3. **No thinking-token / suppression data** — no Logfire traces were downloaded (see the
   integration doc's §5).
4. ~~**`mech=?` throughout**~~ **CLOSED 2026-09-11.** The mechanism classification has now been
   run for HotpotQA (it needed no new data — the regression uses the *cue-free* entropy probe as a
   pre-treatment covariate, not entropy-under-cue). Result: **50 of 54 cells are level-shift-only
   (93%)**, against 58/70 on FRAMES and 67/73 on MedQA. Five of six models are 100% level-shift;
   all 4 breaking cells are `nemotron-3-nano:30b`, and they are labelled "inverted" only because
   that model's HotpotQA plain slope is already −0.128 — the cue moves it *up*, so the label is a
   sign artifact of a near-zero denominator. Details in `PLAN_results_revision.md` §6.1.

5. **Significance is uncorrected** in §3's sign tests. The 3 HotpotQA and 4 FRAMES starred cells
   would not survive FDR across their own families; treat them as descriptive.

---

## 5. Paper-ready claims

Ordered by how much each rests on new evidence. Numbers regenerated 2026-09-10 from the committed
scripts in §6 — **the grid is 72 cells: 3 datasets × 6 models × 4 cues, all 5run/5run.**

1. **Semantic entropy is a valid uncertainty instrument on all three datasets.**
   ρ(entropy, correctness) = −0.488 (FRAMES), −0.552 (HotpotQA), −0.174 (MedQA) on matched EM
   grading over 6 models; −0.627 / −0.526 under the LLM judge where one exists (6-model means —
   see the correction note in §1.1; do **not** use the older −0.649 / −0.603, which were 5-model). Accuracy at H=0
   vs H>0: 0.617/0.180, 0.699/0.261, 0.451/0.268. **Quote EM against EM** — the grader is not
   neutral for this statistic (§1.1); MedQA's low EM row is an artefact of EM on option text, not
   instrument failure.

2. **Perturbations move search policy without a detectable in-domain shift in belief.**
   Of 12 (dataset × cue) in-domain tests — one-sample t over 6 per-model estimates, BH-FDR over
   all 12 — **exactly one is significant: `direct` on MedQA (+0.105 bits, q=.018, 6/6 models)**,
   and that is the cell independently flagged as length-confounded (§3.1). Nothing else moves
   belief in-domain, including `confident_parametric` and `elaborate`. Meanwhile the same cues
   move zero-search rates by **+5 to +60pp**. Mean |Δ entropy| is 0.039–0.042 bits per dataset.

3. **The null is bounded, not merely unrejected.** Per-model minimum detectable effect at 80%
   power is 0.076–0.104 bits; a parametric bootstrap puts the noise floor of a 5-run entropy
   difference at 0.021–0.028 bits (§3.0). The instrument is not saturated — 41–56% of examples
   change entropy level between conditions. So "belief does not move" means "any shift is below
   ≈0.1 bits per model", against policy shifts of up to 60 percentage points.

4. **Uncertainty-driven search is real but dataset-dependent in strength.**
   Mean per-model ρ(entropy, search calls) = **+0.318 FRAMES, +0.129 HotpotQA, +0.129 MedQA**.
   Entropy predicts correctness about equally well on FRAMES and HotpotQA but predicts *search*
   2.5× better on FRAMES — the gap is in the policy, not the instrument.

5. **The MedQA "null" for entropy-vs-search is a pooling artefact, not an absence of coupling.**
   Pooled ρ = +0.014 (n.s.), but mean per-model ρ = +0.129, identical to HotpotQA. The two MedQA
   models that actually search couple clearly (`nemotron-cascade-2` +0.350 p=7.5e-16;
   `qwen3.5:122b` +0.179 p=5.7e-05); the other four sit at 0.04–0.22 calls with nothing to
   correlate. **Report mean per-model ρ, pooled only as a footnote.** Same zero-search floor that
   makes MedQA degenerate for the SFT transfer result — one mechanism, not two.

### Claims that must NOT be made

* **Do not quote `direct`'s pooled row.** It is significantly heterogeneous across datasets
  (Friedman χ²=9.00, p=.011) and its sign flips: −0.053 / +0.105 / −0.047. Report it per dataset.
* **Do not read the pooled cue table as an in-domain result, and do not quote `elaborate` from it.**
  Its q=.045 is pseudo-replication (see the correction box in §3.0); at model level it is q=.261.
  Only `confident_parametric` survives a model-level test, at q=.050 — one cue, exactly on the
  threshold, reached by combining three individually underpowered, same-signed estimates. It is a *cross-domain recurrence* claim, and
  belongs in an appendix with its homogeneity precondition stated.
* **Do not compare HotpotQA's ρ(entropy, correctness) to the FRAMES/MedQA judge column.**
  HotpotQA is EM-only; EM attenuates this ρ by ~0.16 (FRAMES) to ~0.43 (MedQA).
* **Do not read §3's per-cell sign tests as corrected.** They are uncorrected per-cell
  diagnostics, labelled as such.

## 6. Handoff notes — traps a writer must not step on

### 6.1 The mocked-history search offset IS corrected, but only in some scripts

`searchmulti` (and `searchmulti2`/`3`) prepend a conversation whose assistant turn makes its own
`search` tool call. `AgentAsSampler.acall` counted those, inflating `sampler_search_calls` by
exactly the number of mocked calls. This was fixed at source for new runs
(`agent_sampler.py` now subtracts and records `history_search_calls`), but **the FRAMES and MedQA
raw JSONs predate the fix and still carry inflated counts** — verified: every `searchmulti` file
has `min=1` and `0.0%` zero-search rows, with no `history_search_calls` field.

They are corrected **at read time** by the main pipeline, which is why the published numbers are
sound. But the correction is not universal:

| | scripts |
|---|---|
| **Correct the offset** | `make_aggregate_cue_tradeoff_figure.py`, `make_cue_briefing_figures.py`, `make_gemma_cue_figure.py`, `make_gemma_cue_figure_10cond.py`, `analyze_necessity_vs_template_search_5run.py`, `analyze_necessity_vs_template_search_logistic.py`, `analyze_volume_accuracy_decoupling.py`, `dual_metric_analysis.py`, `grade_hotpotqa_regex.py`, `analyze_hotpotqa_transfer.py` (via the graded `per_row.csv`), `analyze_thinking_tokens.py`, `compare_searchmulti_rounds.py`, `analyze_resolved_sft.py`, `build_resolved_frames_sft.py` |
| **Do NOT correct** (read `searchmulti` raw) | `analyze_sft_intervention_mediation.py`, `compare_local_vs_brave_cues.py`, `summarize_frames_cues_grid.py`, `analyze_cue_feature_axes.py`, `regrade_regex.py` |

> **Corrected 2026-09-11 (audit re-verified).** The second column previously listed nine scripts;
> **four of them do correct** and have been moved: `analyze_hotpotqa_transfer.py` (reads the graded
> `per_row.csv`, which corrects — so the paper's HotpotQA transfer numbers were never at risk),
> `analyze_thinking_tokens.py` (`_SEARCHMULTI_ROUND_OFFSET`, line 68), `compare_searchmulti_rounds.py`
> (the script where the bug was first found — the earlier note calling it "the most exposed" was
> backwards), and `make_gemma_cue_figure_10cond.py`. Verified by grepping each for an actual
> subtraction, not by comment. `analyze_sft_intervention_mediation.py` is the one that matters of
> the remainder: it carries `searchmulti` in its `CUES` list and reads raw JSONs, and it is the
> producer behind the paper's Discussion claim about restoring search volume.

> **A second, subtler form of this bug** (found 2026-09-11): applying the constant to files that are
> already correct. Files collected after the `agent_sampler.py` fix exclude the mocked call, but
> carry no `history_search_calls` field to prove it, so a blanket subtraction undercounts them by 1.
> This had already corrupted `gemma4-frames-resolved-q4km`'s 300 HotpotQA `searchmulti` rows
> (mean 1.14 vs a true 2.08) and reversed a conclusion in `resolved_sft_handoff.md`. Both
> `grade_hotpotqa_regex.py` and `analyze_resolved_sft.py` now decide **per file**, using the pre-fix
> signature (every row >= offset, 0% zero-search). Any new consumer must do the same.

**Never quote a `searchmulti` search-volume number produced by a script in the second row without
checking it.** The inflation is +1 per row on FRAMES/MedQA/HotpotQA (one mocked call per
conversation in `data/frames_cues/search_multi_turn.json`), +2/+3 for the 2- and 3-round variants.
It is large enough to flip a sign: uncorrected, FRAMES `searchmulti` shows a −4.9pp *decrease* in
zero-search vs plain; corrected it is an *increase*. `compare_searchmulti_rounds.py` is the most
exposed, since its entire subject is the round-count ablation.

### 6.2 Which remote is authoritative differs per dataset

| Arm | Source of truth |
|---|---|
| FRAMES / MedQA parametric (`*_parametric/`) | **srv3** `/data/home/dvirla/parametric_search_tradeoff` (Athena holds partials from an earlier pass) |
| HotpotQA parametric | **Athena** for 5 models; **srv3 worktree** `..._hpqcue` for `qwen3.5:122b` |
| HotpotQA cue grid | both; identical after the 2026-09-09 sync |

Pulling from the wrong one produced a false "data gap" claim in this document once (qwen3.5:122b's
`confident_parametric`/`multiturn` looked like 3/5 and 2/5 runs; srv3 had the full 5/5).
**Check both before declaring anything incomplete.**

### 6.3 Known-stale text elsewhere

`hotpotqa_paper_integration.md` §5 currently says HotpotQA `confident_parametric` clustering is
"in progress (0 of 6 clustered as of 2026-09-10)". That was true when written; **all 6 are
clustered** as of 2026-09-11 (verified: every non-SFT model has 5/5 cue cluster files). The same
file's §4 caveat 5 says the FRAMES/MedQA `searchmulti` rows "need the same treatment" — narrow
that to §6.1 above: the main pipeline already corrects them, the listed scripts do not.

### 6.4 What is NOT available, and should not be asked for

* **No LLM judge on HotpotQA** — every row `--no_grader`. EM only, hence §1.1.
* **No `confident_parametric` entropy for the SFT checkpoints** — `gemma4-frames-robust-q4km` has
  2 of 5 cue cluster files; `gemma4-frames-resolved-q4km` has 5 of 5 but belongs to a sibling
  session's arm. Do not fold SFT checkpoints into the 6-model roster; they are trained to be
  cue-invariant by construction.
* **No thinking-token / suppression data for HotpotQA** — no Logfire traces were downloaded.
* **No `cue_suppression_mechanism` rows for HotpotQA** — the `mech=?` column is empty because the
  regression has not been run on HotpotQA, **not** because the data is missing (see §4 item 4). It
  is a small code change away; do not ask for new rollouts, and do not tell a writer it is
  unavailable.

## 7. Reproducing

```bash
cd /home/dvirla/projects/parametric_search_tradeoff
uv run python scripts/analyze_entropy_vs_correctness.py       # -> results/entropy_vs_correctness/
uv run python scripts/analyze_llm_entropy_vs_search_5run.py   # -> results/param_vs_search_llm_5run/
uv run python scripts/analyze_entropy_under_cue.py            # -> results/entropy_under_cue/
uv run python scripts/analyze_entropy_under_cue_stats.py     # -> the corrected in-domain/pooled tables
```

All three read the clusterer outputs `results/{frames,medqa,hotpotqa}_parametric/<model>/
*_llm_clusters_5run.json`, produced by `scripts/cluster_cues_llm_judge.py --n-runs 5`
(judge: gpt-oss:120b, unchanged across datasets — this is what makes the entropies comparable).

**HotpotQA naming traps** these scripts had to accommodate, worth knowing before adding a fourth
dataset: its driver names every run `<cond>_run_<r>`, so (a) the plain baseline is a *named* cue
file rather than the bare no-infix file FRAMES/MedQA use, and (b) a bare `*_llm_clusters_5run.json`
glob matches **four** files per model and would silently use a CUE's entropy as the cue-free
baseline. Both are now plain-specific in all three scripts.
