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
| FRAMES | −0.649 | −0.488 | ~0.16 |
| MedQA | −0.603 | −0.174 | **~0.43 — near-vanishing** |

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

| Dataset | cells | mean Δ | mean \|Δ\| | range | sign-test p<.05 | ρ(plain,cue) |
|---|---:|---:|---:|---|---:|---|
| FRAMES | 22 | −0.005 | 0.042 | −0.131 … +0.083 | 4/22 | 0.77 |
| MedQA | 22 | +0.034 | 0.044 | −0.049 … +0.164 | 8/22 | 0.58 |
| HotpotQA | 24 | +0.003 | 0.039 | −0.133 … +0.100 | 3/24 | 0.75 |

**Mean |Δ| is 0.042–0.044 bits on all three datasets** — remarkably stable, and negligible against
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
| FRAMES | +0.020 | −0.014 … +0.060 | 1/5 |
| MedQA | +0.012 | −0.004 … +0.025 | 0/5 |
| **HotpotQA** | **+0.022** | −0.013 … +0.061 | **0/6** |

On HotpotQA the cue moves the zero-search rate by **+59.9pp** — the largest single behavioural
effect anywhere in this project — while moving entropy by **+0.022 bits, with not one of the six
models reaching significance**. The three datasets agree to within 0.01 bits despite differing
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
95% CI includes zero, and per-dataset t-tests over models are n=5–6. **"Not significant" here does
not mean "zero".**

**(c) A parametric bootstrap gives the noise floor.** Treating each example's `plain` cluster
proportions as the true distribution and drawing two independent 5-samples, the null mean delta has
**SD ≈ 0.021–0.028 bits** (95% band ≈ ±0.04–0.05) — closely matching the observed SEs. Of the six
HotpotQA models, only gemma4:31b (+0.061) falls outside its own null band; the rest sit inside it.

**(d) Pooling the 16 model×dataset estimates, the effect is small but REAL:**

| Aggregation | n | mean Δ | 95% CI | p |
|---|---:|---:|---|---:|
| FRAMES (models) | 5 | +0.0202 | [−0.017, +0.058] | .207 |
| MedQA (models) | 5 | +0.0120 | [−0.002, +0.026] | .077 |
| HotpotQA (models) | 6 | +0.0223 | [−0.008, +0.052] | .114 |
| **All 16** | **16** | **+0.0184** | **[+0.0057, +0.0311]** | **.007** |

So `confident_parametric` **does** raise entropy slightly and consistently — 13 of 16 estimates are
positive — but the effect is bounded above by **+0.031 bits, which is 6.5% of the 0.479-bit
between-model spread** in plain entropy.

**Revised claim.** Not "belief does not move." Rather: *the cue moves belief a little and policy
enormously.* Telling a model it already knows the answer raises its answer-inconsistency by
≈0.02 bits (≈6% of between-model variation) while raising its zero-search rate by up to 60
percentage points. The policy-shortcut reading survives, but as a statement about **relative
magnitude**, not about a null belief effect — and any single-dataset test is too underpowered to
carry it alone.

#### The same test, applied to every cue

The pooling argument above is not specific to `confident_parametric`. Applied to all four cues
(one estimate per model×dataset cell, one-sample t vs 0, BH-FDR over the four tests):

| Cue | cells | mean Δ | 95% CI | p | q (FDR) | positive |
|---|---:|---:|---|---:|---:|---:|
| `elaborate` | 18 | **+0.0236** | [+0.0037, +0.0435] | .023 | **.045** | 13/18 |
| `confident_parametric` | 18 | **+0.0186** | [+0.0061, +0.0310] | .006 | **.023** | 12/18 |
| `multiturn` | 18 | −0.0031 | [−0.023, +0.017] | .75 | .93 | 11/18 |
| `direct` | 18 | +0.0017 | [−0.041, +0.045] | .93 | .93 | 7/18 |

The grid is complete and balanced: **72 cells = 3 datasets × 6 models × 4 cues**, all 5run/5run.

**Two cues have a small but real positive belief effect** — `elaborate` as well as
`confident_parametric`, both surviving FDR. `multiturn` is genuinely null.

**`direct`'s pooled null is an artifact of cancellation, not evidence of no effect.** Its sign
flips by dataset: −0.053 (FRAMES), −0.047 (HotpotQA), **+0.105 (MedQA)**. Pooling averages a
real negative against a real positive to ≈0. This is the same cell §3.1 flags as
measurement-suspect (MedQA |dH|~|dLen| = 0.513, and `direct` is the most extreme length cue at
2–3 words). **Do not quote `direct`'s pooled row** — report it per dataset.

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
4. **`mech=?` throughout** — no `cue_suppression_mechanism` rows exist for HotpotQA, so the
   level-shift vs. calibration-erosion classification is unavailable.
5. **Significance is uncorrected** in §3's sign tests. The 3 HotpotQA and 4 FRAMES starred cells
   would not survive FDR across their own families; treat them as descriptive.

---

## 5. Paper-ready claims

Ordered by how much each rests on new evidence.

1. *Semantic entropy is a valid uncertainty instrument across three datasets and six open-weight
   models* (ρ = −0.49 FRAMES, −0.55 HotpotQA on matched EM grading; −0.60/−0.65 under the LLM
   judge where one exists). **Report EM against EM.**
2. *Perturbations move search policy without moving belief.* Mean |Δ entropy| is 0.042–0.044 bits
   on all three datasets while zero-search rates move +5 to +60pp. Third-dataset replication of the
   paper's core claim.
3. *Uncertainty-driven search is real but dataset-dependent in strength* — mean per-model ρ = +0.32
   (FRAMES) vs +0.13 (HotpotQA and MedQA alike).
4. *The MedQA "null" should be restated as a floor effect*, not an absence of coupling — and it is
   the same floor that makes MedQA a degenerate venue for the SFT transfer result.

## 6. Reproducing

```bash
cd /home/dvirla/projects/parametric_search_tradeoff
uv run python scripts/analyze_entropy_vs_correctness.py       # -> results/entropy_vs_correctness/
uv run python scripts/analyze_llm_entropy_vs_search_5run.py   # -> results/param_vs_search_llm_5run/
uv run python scripts/analyze_entropy_under_cue.py            # -> results/entropy_under_cue/
```

All three read the clusterer outputs `results/{frames,medqa,hotpotqa}_parametric/<model>/
*_llm_clusters_5run.json`, produced by `scripts/cluster_cues_llm_judge.py --n-runs 5`
(judge: gpt-oss:120b, unchanged across datasets — this is what makes the entropies comparable).

**HotpotQA naming traps** these scripts had to accommodate, worth knowing before adding a fourth
dataset: its driver names every run `<cond>_run_<r>`, so (a) the plain baseline is a *named* cue
file rather than the bare no-infix file FRAMES/MedQA use, and (b) a bare `*_llm_clusters_5run.json`
glob matches **four** files per model and would silently use a CUE's entropy as the cue-free
baseline. Both are now plain-specific in all three scripts.
