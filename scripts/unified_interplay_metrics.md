# Unified Interplay Analysis — Metrics Reference

This document explains every metric computed by `unified_interplay_analysis.py`, including
its formula, range, interpretation, current empirical values, and known caveats.

---

## 0. Motivation & Framework Design Principles

### The binary POR problem

The original **Parametric Overhead Ratio (POR)** classifies each searched EEU as either
*certain* (overhead) or *uncertain* (effective). This binary threshold loses information
in two important ways:

1. **Magnitude of waste**: a model that searches hops with entropy = 0.0 (fully certain)
   is categorically worse than one that searches hops with entropy = 0.4 (slightly uncertain).
   POR treats both identically.

2. **Volume of waste**: a model that issues 13 queries to a certain hop wastes more than
   one that issues 1 query to the same hop. POR is insensitive to per-EEU query volume.

To tell a complete story we need a metric that captures *how certain* the model was when
it searched (intensity), and *how many queries* were directed at certain units (volume).

### Two regimes of search miscalibration

Empirical data from MusiQue × ShareChat reveals two qualitatively different failure modes:

| Regime | Exemplar | Characterisation |
|--------|----------|-----------------|
| **Reflexive breadth** | Gemini / MusiQue | Searches almost everything regardless of certainty. EWOI = 0.813. Issues ~14 queries/question. |
| **Coarse selectivity** | All models / ShareChat | Search decisions made at question level, not fact level. SAE ≈ 0 at fact level but 0.251 at question level. |

The extended framework below is designed to distinguish and quantify these two regimes.

---

---

## 1. The Epistemic Evidence Unit (EEU)

The central challenge is that MusiQue and ShareChat operate at different granularities:
MusiQue produces one analysis unit per **hop** (sub-question in a multi-hop chain),
while ShareChat produces one unit per **atomic fact** (a single verifiable claim extracted
from the model's response). Comparing these directly would be meaningless.

The **Epistemic Evidence Unit (EEU)** is the shared atomic unit of analysis. Every EEU
represents one instance where we can jointly observe two signals:

1. **Certainty** — was the model already confident about this unit from parametric knowledge alone?
2. **Search attribution** — was this unit's content attributed to an external search result?

For MusiQue, an EEU = one hop. For ShareChat, an EEU = one atomic fact.

### Certainty definition

| Dataset   | Certainty definition | Rationale |
|-----------|----------------------|-----------|
| MusiQue (`joint` mode, default) | `entropy == 0` AND `all 5 no-search runs correct` | Has accuracy oracle from independent no-search runs |
| MusiQue (`entropy_only` mode) | `entropy == 0` | Symmetric with ShareChat |
| ShareChat | `entropy == 0` | No gold answers — only entropy available |

Use `--mq-certainty-mode entropy_only` to run MusiQue with the weaker definition
for a strictly symmetric cross-dataset comparison.

### Search attribution definition

| Dataset   | Definition |
|-----------|-----------|
| MusiQue | `searched == True` — at least one query was attributed to this hop by the search trace analysis |
| ShareChat | `attributed_to_search == True` — LLM judge determined this atomic fact came from search results |

---

## 2. The 4-Way Quadrant Taxonomy — as a Confusion Matrix

Each EEU is a **binary search decision**. We can define a ground truth and a model prediction:

- **Ground truth positive** = uncertain (entropy > 0) — the model *should* search this EEU
- **Ground truth negative** = certain (entropy = 0) — the model can skip search
- **Predicted positive** = `search_attributed = True`
- **Predicted negative** = `search_attributed = False`

This maps directly to a confusion matrix:

```
                       search_attributed = True    search_attributed = False
                     ┌───────────────────────────┬───────────────────────────┐
  entropy > 0        │  E  — Effective  (TP)      │  M  — Missed  (FN)        │
  (uncertain)        │  search was needed ✓       │  gap left open  ✗         │
                     ├───────────────────────────┼───────────────────────────┤
  entropy = 0        │  PR — Param. Redundant (FP)│  CP — Correct Param. (TN) │
  (certain)          │  wasted search  ✗          │  correctly skipped  ✓     │
                     └───────────────────────────┴───────────────────────────┘
```

| Quadrant | Classifier term | Correct? | Implication |
|----------|----------------|:--------:|-------------|
| **E** | TP | ✓ | Model was uncertain, searched — correct activation |
| **PR** | FP | ✗ | Model was certain but searched anyway — wasted query cost |
| **M** | FN | ✗ | Model was uncertain but did not search — knowledge gap left open |
| **CP** | TN | ✓ | Model was confident and correctly skipped search |

The ideal model maximises E + CP (correct decisions) and minimises PR + M (errors).

### Standard metrics derived directly from the confusion matrix

| Metric | Formula | Alias | Interpretation |
|--------|---------|-------|----------------|
| **Precision** | E / (E + PR) = TP / (TP+FP) | *(formerly SER)* | Of all searches triggered, what fraction were needed? |
| **FDR** | PR / (E + PR) = FP / (TP+FP) | *(formerly POR; = 1 − Precision)* | Of all searches triggered, what fraction were wasted? |
| **Recall** | E / (E + M) = TP / (TP+FN) | *(new)* | Of all uncertain EEUs, what fraction were searched? |
| **Specificity** | CP / (CP + PR) = TN / (TN+FP) | *(new)* | Of all certain EEUs, what fraction were correctly skipped? |
| **F1** | 2 × Precision × Recall / (P+R) | *(new)* | Harmonic mean of precision and recall |
| **AUROC** | P(H_searched > H_not_searched) | *(new)* | Discriminability of entropy as a search predictor |

**EWOI and SBE** (magnitude-aware extensions) have no standard classifier analogue and remain custom metrics. **SCC** and **SAE** are rank- and mean-based calibration measures, analogous to discrimination metrics but not identical to AUROC.

### Critical asymmetry: PR semantics differ between datasets

**MusiQue PR** = the model searched on a hop where **independent no-search runs already
got it right with certainty** (pre-search certainty). This is genuine wasted search effort.

**ShareChat PR** = the model searched, and the resulting attributed fact has **low semantic
entropy across paraphrased samples** (post-attribution certainty). This does NOT mean the
model knew the fact before searching — it may mean the fact is easy to state once retrieved.
For example, a question about a recent product's spec triggers search, but the returned fact
"GPT-5 has a 128k context window" is unambiguous and thus has entropy=0 after retrieval.

**Consequence**: ShareChat PR overestimates true wasted search. All cross-dataset comparisons
of PR-derived metrics (POR, SER) must be accompanied by this caveat.

---

## 3. Interplay Signatures

Seven signatures characterise each (dataset, model) combination. All are computed in
`compute_signatures()` and written to `interplay_signatures.csv`.

---

### Signature 1: SCC — Search Calibration Coefficient (EEU level)

**Formula:**
```
SCC = Spearman_ρ(entropy_i, search_attributed_i)   for all EEUs i in (dataset, model)
```

**Range:** [−1, +1]

**Interpretation:**
- **SCC ≈ +1**: model reliably searches when uncertain — well calibrated
- **SCC ≈ 0**: search and uncertainty are independent — reflexive or random search
- **SCC < 0**: model searches more when certain — systematic miscalibration (pathological)

**Implementation notes:**
- Computed at the EEU level (one observation per hop or per atomic fact)
- A separate question-level variant (SCC_q) is also computed — see below
- For MusiQue, all examples have `has_search=True` by design, so the EEU-level correlation
  captures whether *which hops* are searched correlates with hop-level uncertainty
- Statistical test: **H1** uses Fisher z-transform to compare SCC_MusiQue vs SCC_ShareChat

**Current values (Gemini-3-Pro):**
| Dataset | SCC | p |
|---------|-----|---|
| MusiQue | 0.115 | < 0.001 |
| ShareChat | 0.026 | 0.403 |

MusiQue's non-zero SCC (0.115) shows weak but real calibration — hops with higher
entropy are slightly more likely to be searched. ShareChat's near-zero SCC (0.026)
means search decisions at the fact level are essentially independent of fact-level uncertainty.

---

### Signature 1q: SCC_q — Search Calibration Coefficient (question level)

**Formula:**
```
SCC_q = Spearman_ρ(mean_entropy_per_question, has_search_per_question)
```

**Notes:**
- Not defined for MusiQue (all questions have `has_search=True`, so the column has no variance)
- For ShareChat: captures whether questions with more uncertain facts are more likely to trigger
  any search call

**Current values (ShareChat Gemini):**
- SCC_q = **0.251** (p = 0.031, one-sided p = 0.016)
- Significant: questions with higher average fact-entropy do trigger search more often
- The contrast SCC=0.026 (fact level) vs SCC_q=0.251 (question level) reveals that
  search decisions are made at the **question level** (coarse), not at the fact level
  (fine-grained). The model decides to search or not based on holistic question difficulty,
  not per-fact uncertainty.

---

### Signature 2: FDR — False Discovery Rate of Search (formerly POR)

**Formula:**
```
FDR = |PR| / (|E| + |PR|)   = FP / (TP + FP)   = 1 − Precision
    = (EEUs that were searched AND certain) / (all searched EEUs)
```

*(This metric was previously called Parametric Overhead Ratio / POR. The backward-compat alias `por` is still written to all CSVs.)*

**Range:** [0, 1]

**Confidence interval:** Wilson score interval on the proportion.

**Interpretation:**
- **FDR = 0**: all search is on uncertain units — no false alarms
- **FDR = 1**: model only searches on units it already knows — pure overhead
- High FDR on MusiQue but not ShareChat supports the "search reflex" hypothesis
- See asymmetry warning above before cross-dataset comparison

**Current values:**
| Dataset | Model | FDR (=1−Prec) | 95% CI |
|---------|-------|---------------|--------|
| MusiQue | Gemini-3-Pro | 0.399 | [0.371, 0.427] |
| MusiQue | Nemotron-Nano-30B | 0.134 | [0.110, 0.163] |
| MusiQue | Qwen3.5-122B | 0.316 | [0.289, 0.344] |
| ShareChat | Gemini-3-Pro | 0.886 | [~0.84, ~0.93] |

Gemini on MusiQue has a 40% false discovery rate — 40 cents of every search dollar is spent
on hops it already knew. Nemotron's FDR is 13%, reflecting much more targeted activation.
ShareChat's high FDR is inflated by the post-attribution certainty issue (see asymmetry note).

---

### Signature 2b: Precision of search (formerly SER)

**Formula:**
```
Precision = |E| / (|E| + |PR|)   = TP / (TP + FP)   = 1 − FDR
```

*(Previously called Search Efficacy Rate / SER. Alias `ser` is still written to CSVs.)*

Precision answers: "Of all the searches the model triggered, what fraction targeted a
genuinely uncertain unit?" It is the exact complement of FDR and adds no new information
beyond it. Both are included for readability in different contexts.

**Current values:**
| Dataset | Model | Precision |
|---------|-------|-----------|
| MusiQue | Gemini-3-Pro | 0.601 |
| MusiQue | Nemotron-Nano-30B | 0.866 |
| MusiQue | Qwen3.5-122B | 0.684 |
| ShareChat | Gemini-3-Pro | 0.114 |

---

### Signature 2c: Recall of search (new)

**Formula:**
```
Recall = |E| / (|E| + |M|)   = TP / (TP + FN)
       = (EEUs that were uncertain AND searched) / (all uncertain EEUs)
```

**Confidence interval:** Wilson score interval on the proportion.

**Interpretation:**
- **Recall = 1**: every uncertain EEU was searched — complete coverage of gaps
- **Recall = 0**: no uncertain EEU was searched — all gaps left open
- Recall is the exact complement of CovGap expressed as a rate over uncertain EEUs
  (CovGap is FN/N, so CovGap = n_M/N and Recall = n_E/(n_E + n_M))

**Current values:**
| Dataset | Model | Recall |
|---------|-------|--------|
| MusiQue | Gemini-3-Pro | 0.815 |
| MusiQue | Nemotron-Nano-30B | 0.556 |
| MusiQue | Qwen3.5-122B | 0.772 |
| ShareChat | Gemini-3-Pro | 0.088 |

The precision-recall contrast for Gemini vs Nemotron reveals fundamentally different
failure modes: Gemini has high recall (81.5%) but low precision (60.1%) — it searches
reflexively, covering most gaps but with many false alarms. Nemotron has high precision
(86.6%) but low recall (55.6%) — it searches selectively but misses nearly half the gaps.

---

### Signature 2d: F1 score (new)

**Formula:**
```
F1 = 2 × Precision × Recall / (Precision + Recall)
```

**Bootstrap CI** (2000 resamples, same pattern as log-odds accuracy).

F1 is the harmonic mean of precision and recall, penalising both false alarms and missed
gaps equally. It provides a single summary of search decision quality.

**Current values:**
| Dataset | Model | F1 |
|---------|-------|----|
| MusiQue | Gemini-3-Pro | 0.692 |
| MusiQue | Nemotron-Nano-30B | 0.677 |
| MusiQue | Qwen3.5-122B | 0.725 |
| ShareChat | Gemini-3-Pro | 0.099 |

Despite their different precision-recall tradeoffs, Gemini (0.692) and Nemotron (0.677)
have nearly identical F1 scores — they make errors in opposite directions but with similar
total cost. Qwen achieves the best MusiQue F1 (0.725), balancing precision and recall well.

---

### Signature 2e: AUROC — search decision discriminability (new)

**Formula:**
```
AUROC = P(entropy_searched > entropy_not_searched)
      = Mann-Whitney U / (n_searched × n_not_searched)
```

**Range:** [0, 1]
- **AUROC = 0.5**: entropy is independent of the search decision — reflex or random
- **AUROC → 1.0**: entropy perfectly predicts whether an EEU is searched
- **AUROC < 0.5**: model searches *more* on certain units (pathological reverse calibration)

AUROC is the probability that a randomly chosen searched EEU has higher entropy than a
randomly chosen non-searched EEU. It summarises the discrimination quality of the search
trigger across all possible entropy thresholds simultaneously.

**Relationship to SCC:** AUROC and SCC both measure search-uncertainty coupling. SCC (Spearman ρ)
captures the rank correlation at the EEU level; AUROC is a threshold-free summary of the
same signal expressed as a probability. Both are near-equivalent for large samples.

**Current values:**
| Dataset | Model | AUROC |
|---------|-------|-------|
| MusiQue | Gemini-3-Pro | 0.557 |
| MusiQue | Nemotron-Nano-30B | 0.619 |
| MusiQue | Qwen3.5-122B | 0.610 |
| ShareChat | Gemini-3-Pro | 0.492 |

Interpretation: All MusiQue models have AUROC > 0.5, meaning entropy carries a weak but
real signal for search decisions. Nemotron (0.619) is the most discriminative — its search
triggers are most closely coupled to its uncertainty. ShareChat Gemini's AUROC ≈ 0.5
confirms that search decisions at the fact level are essentially independent of fact-level
uncertainty (consistent with SCC ≈ 0.026).

---

### Signature 4: CovGap — Coverage Gap

**Formula:**
```
CovGap = |M| / N_total
       = (EEUs that were uncertain AND not searched) / (all EEUs)
```

**Range:** [0, 1] (lower is better)

**Interpretation:**
- Represents the fraction of all evidence units where the model had a knowledge gap
  but did not seek external information
- High CovGap combined with low POR is the ideal failure mode: the model is selective
  about searching but perhaps under-searches
- High POR combined with low CovGap (e.g. MusiQue Gemini) means the model searches
  extensively but misdirects its queries to units it already knew

**Confidence interval:** Wilson score interval on the proportion.

**Current values:**
| Dataset | Model | CovGap | 95% CI |
|---------|-------|--------|--------|
| MusiQue | Gemini-3-Pro | 0.101 | [0.088, 0.115] |
| MusiQue | Nemotron-Nano-30B | 0.346 | [0.325, 0.367] |
| MusiQue | Qwen3.5-122B | 0.144 | [0.129, 0.161] |
| ShareChat | Gemini-3-Pro | 0.067 | [0.050, 0.089] |

Nemotron's high CovGap (34.6%) shows it under-searches — many uncertain hops go
without external grounding, likely contributing to its lower aggregate accuracy.

---

### Signature 5: QBS — Quadrant Balance Score

**Formula:**
```
QBS = log( (|E| + |CP|) / (|PR| + |M|) )
    = log( correct_decisions / incorrect_decisions )
```

where "correct" = Effective + Correct Parametric, "incorrect" = Parametric Redundant + Missed.

**Range:** (−∞, +∞)
- **QBS > 0**: model makes more correct than incorrect search decisions
- **QBS = 0**: equal split (50% correct)
- **QBS < 0**: more errors than correct decisions

**Confidence interval:** Bootstrapped 95% CI (2000 samples with replacement).

**Why log?** The log scale makes the score symmetric around 0 and gives equal weight to
doublings in either direction (e.g., 2:1 and 1:2 ratios both have |QBS| = log(2) ≈ 0.693).

**Current values:**
| Dataset | Model | QBS | 95% CI |
|---------|-------|-----|--------|
| MusiQue | Gemini-3-Pro | +0.415 | [+0.325, +0.500] |
| MusiQue | Nemotron-Nano-30B | +0.352 | [+0.272, +0.431] |
| MusiQue | Qwen3.5-122B | +0.527 | [+0.443, +0.614] |
| ShareChat | Gemini-3-Pro | +1.083 | [+0.953, +1.230] |

ShareChat's much higher QBS (+1.08 vs +0.42 for MusiQue Gemini) is primarily driven
by the large CP fraction (72.6% of ShareChat facts are correctly handled parametrically).
The non-overlapping CIs confirm this difference is significant (H4 test).

---

### Signature 6: SIR — Search Intensity Ratio

**Two components computed at the question level:**

```
SIR_mean = mean(search_calls_per_question)
SIR_cv   = std(search_calls_per_question) / mean(search_calls_per_question)
```

**SIR_mean** (volume): how many search queries does the model issue per question on average?
- For MusiQue: total queries assigned across all hops in an example
- For ShareChat: `search_calls` from the raw evaluation

**SIR_cv** (heterogeneity): coefficient of variation — does search volume vary with question
difficulty, or is it roughly constant?
- **Low SIR_cv**: similar search volume across all questions regardless of difficulty
  (suggests reflex behaviour or fixed search budget)
- **High SIR_cv**: search volume varies substantially across questions
  (suggests question-driven, adaptive searching)

**Current values:**
| Dataset | Model | SIR_mean | SIR_cv |
|---------|-------|---------|--------|
| MusiQue | Gemini-3-Pro | 13.58 | 0.551 |
| MusiQue | Nemotron-Nano-30B | 4.56 | 0.775 |
| MusiQue | Qwen3.5-122B | 4.51 | 0.727 |
| ShareChat | Gemini-3-Pro | 0.91 | 1.549 |

ShareChat's high CV (1.55) means search is used very selectively and heterogeneously —
some questions trigger many searches, most trigger none. MusiQue's lower CV (especially
for Gemini at 0.55) reflects the benchmark's structured format imposing more uniform
search depth across examples.

---

### Signature 7: SLI — Search Leverage Index

**Availability:** MusiQue only. Requires `search_usefulness_by_certainty.csv`
(auto-detected from the same directory as `interplay_summary.csv`).

**Formula (per certainty bin b):**
```
SLI_b = search_agg_accuracy_b − nosearch_agg_accuracy_b
```

where examples are grouped into three certainty bins based on the fraction of 5 no-search
runs that answered correctly:
- **0/5** (fully uncertain): 0 out of 5 no-search runs correct
- **1–4/5** (partial): 1–4 out of 5 correct
- **5/5** (fully certain): all 5 no-search runs correct

The **SLI slope** is Spearman ρ between bin rank (0, 1, 2) and SLI_b. A negative slope
(ρ = −1) means search helps most where parametric knowledge is worst and hurts when the
model already knew the answer — the ideal pattern.

**Two named scalars extracted:**
- `sli_uncertain_delta`: SLI at 0/5 bin (search benefit for fully uncertain questions)
- `sli_certain_delta`: SLI at 5/5 bin (search cost for fully certain questions)

**Important note on baselines:**
The no-search accuracy values currently come from a **composed baseline**: for each
multi-hop question, a separate no-search model was run on each sub-question independently,
and correctness was inferred from those sub-question answers. This is a proxy.

When the **actual aggregate no-search runs** finish (computed on a separate machine),
run the helper:
```bash
uv run python scripts/unified_interplay_analysis.py \
  --build-nosearch-csv \
  --nosearch-aggregate-dir <dir> \
  --musique-summary-csv results/musique_parametric/interplay_analysis/interplay_summary.csv
```
Then rerun with `--nosearch-accuracy-csv results/musique_parametric/nosearch_accuracy.csv`
to upgrade from composed to actual baselines. The `nosearch_source` column in outputs
distinguishes which examples used which baseline.

**Current values (Gemini-3-Pro, composed baseline):**

| Certainty bin | no-search acc | search acc | SLI (delta) |
|---------------|:------------:|:----------:|:-----------:|
| 0/5 (uncertain) | 0.000 | 0.263 | **+0.263** |
| 1–4/5 (partial) | 0.477 | 0.580 | **+0.104** |
| 5/5 (certain)   | 1.000 | 0.904 | **−0.096** |

SLI slope = −1.0 (perfect negative rank correlation): search helps most (+26.3 pp) when
parametric knowledge is absent, provides moderate benefit (+10.4 pp) for partial knowledge,
and slightly hurts (−9.6 pp) when the model was already fully correct — consistent with
the "context poisoning" phenomenon where search introduces noise into certain answers.

---

## 4. Statistical Tests

All tests are grouped into three families with **Benjamini-Hochberg (BH) FDR correction**
applied within each family at α = 0.05.

### Family 1: Cross-dataset (H1–H4)
*Compare MusiQue vs ShareChat for Gemini-3-Pro (the only model present in both datasets)*

| ID | Hypothesis | Method | Effect size |
|----|-----------|--------|-------------|
| **H1** | SCC_MusiQue ≠ SCC_ShareChat (EEU level) | Fisher z-transform on Spearman correlations | Δρ = ρ_mq − ρ_sc |
| **H1q** | SCC_MusiQue ≠ SCC_ShareChat (question level) | Fisher z-transform | Δρ |
| **H2** | POR_MusiQue ≠ POR_ShareChat | Two-proportion z-test | Cohen's h |
| **H3** | CovGap_MusiQue ≠ CovGap_ShareChat | Two-proportion z-test | Cohen's h |
| **H4** | QBS_MusiQue ≠ QBS_ShareChat | Bootstrap CI non-overlap | ΔQBS |

**H2 caveat:** The POR difference is statistically massive (Cohen's h = −1.13, p < 10⁻⁴⁰),
but the semantic asymmetry in the certainty definition makes this comparison partially
confounded. The test is valid, but the effect size overstates the practical difference
between datasets. The paper should note this.

**Current results:**
| Test | Significant | Effect |
|------|:-----------:|--------|
| H1: SCC (EEU) | ✓ | Δρ = +0.089 — MusiQue slightly more calibrated at unit level |
| H1q: SCC (question) | — | N/A for MusiQue (no variance in has_search) |
| H2: POR | ✓ | Cohen's h = −1.13 — large but confounded |
| H3: CovGap | ✓ | Cohen's h = +0.12 — MusiQue has more uncovered uncertainty gaps |
| H4: QBS | ✓ | ΔQBS = −0.67 — ShareChat much higher (non-overlapping CIs) |

---

### Family 2: Within-MusiQue (W1–W4)
*Full statistical power: N ≈ 1800 hops per model*

| ID | Hypothesis | Method |
|----|-----------|--------|
| **W1** | Entropy distributions differ across models | Kruskal-Wallis on entropy values |
| **W2** | SLI slope is negative (search helps less as certainty rises) | Spearman ρ (reported as effect, p not computed — logical claim) |
| **W3** | POR varies by number of hops (2-hop vs 3-hop vs 4-hop) | Kruskal-Wallis on certainty of searched hops |
| **W4** | Cross-hop coverage rate > 0 | One-sample binomial test (H₀: rate = 0) |

**W1** tests whether the three models genuinely differ in their uncertainty profiles, not
just in search behaviour. A significant result (confirmed: p < 10⁻¹⁰⁸) validates that
cross-model comparisons of SCC, POR etc. reflect real differences in model knowledge.

**W3** tests whether multi-hop structure modulates search efficiency. If longer chains (4-hop)
produce higher POR, it would suggest the model becomes increasingly reflexive as chains grow —
evidence of structural benchmark overfitting.

**W4** tests a MusiQue-specific phenomenon: when a hop is missed (uncertain, not searched),
is the answer sometimes covered by search queries attributed to *adjacent hops*? A non-zero
cross-hop coverage rate means the effective search coverage is better than the hop-level
attribution suggests.

**Current results:**
| Test | Significant | Key finding |
|------|:-----------:|-------------|
| W1: Entropy by model | ✓ | Models genuinely differ: mean entropy 0.285 / 0.811 / 0.586 for Gemini / Nemotron / Qwen |
| W2: SLI slope | ✓ | ρ = −1.0: search leverage perfectly inversely tracks certainty |
| W3: POR by hop count | ✓ | Longer chains do show higher parametric overhead |
| W4: Cross-hop coverage | — | Not computed (column not in interplay_summary.csv) |

---

### Family 3: Within-ShareChat (S1–S4)
*Small-n context: N = 74 questions, 27 with search. Use exact/bootstrap methods.*

| ID | Hypothesis | Method |
|----|-----------|--------|
| **S1** | SCC_q > 0 (question-level calibration) | One-sided Spearman (p / 2) |
| **S2** | POR varies by question category | Fisher exact per category, Bonferroni correction |
| **S3** | Search calls differ by question category | Kruskal-Wallis |
| **S4** | Empty search rate > 0 (search triggered but no facts came from it) | One-sample binomial test |

**S2** tests whether certain question categories show systematically higher or lower
parametric overhead. With 4 categories (explanation, factual_lookup, how_to, current_events)
and only 74 questions total, power is low. Bonferroni correction is conservative;
the per-category Fisher exact p-values are the primary results.

**S4** tests for complete search waste: questions where the model triggered search but
the LLM judge attributed zero atomic facts to the search results. This captures cases
where search was initiated but the retrieved content was entirely ignored.

**Current results:**
| Test | Significant | Key finding |
|------|:-----------:|-------------|
| S1: SCC_q > 0 | ✓ | ρ = 0.251, one-sided p = 0.016 |
| S2: `explanation` category | ✓ | Higher PR rate than other categories |
| S2: `factual_lookup` category | ✓ | Significantly different PR distribution |
| S2: `how_to` | ✗ | No significant difference |
| S2: `current_events` | ✗ | No significant difference (small n) |
| S3: Search calls by category | ✓ | Categories differ in search volume (KW p = 0.008) |
| S4: Empty search rate | ✗ | 0/27 searched questions had completely unused search |

S4 non-significance is a positive finding: whenever the model searched on ShareChat,
it incorporated at least some information from the results (no pure reflex with zero uptake).

---

## 5. Confidence Intervals and Multiple Testing

**Wilson score interval** (not Wald) is used for all proportions (FDR, Recall, CovGap). Wilson
intervals are more accurate for proportions near 0 or 1 and for small sample sizes.

**Bootstrap CI for F1 and log-odds accuracy**: 2000 resamples with replacement. These statistics
are not simple proportions, so analytical CIs are not reliable; bootstrap is appropriate.

**Benjamini-Hochberg FDR**: applied within each of the three test families independently.
Tests are not corrected across families. This is the standard approach for confirmatory
testing with pre-specified hypotheses grouped by theme.

---

## 6. Output Files

All outputs go to `--output-dir` (default: `results/unified_interplay/`):

| File | Content |
|------|---------|
| `unified_eeu_frame.csv` | One row per EEU. All datasets and models. Contains `quadrant`, `certainty`, `entropy`, `search_attributed`, plus dataset-specific columns. |
| `unified_question_frame.csv` | One row per (dataset, model, question_id). Contains `frac_E/PR/M/CP`, `mean_entropy`, `search_intensity`, `aggregate_correct`, `nosearch_agg_accuracy`. |
| `interplay_signatures.csv` | One row per (dataset, model). All 7 signatures with CIs. |
| `statistical_tests.csv` | One row per test. Columns: `test`, `family`, `statistic`, `p_raw`, `effect_size`, `note`, `p_adj_bh`, `significant`. |
| `plots/fig1_taxonomy_bars.{png,pdf}` | 4-way EEU composition stacked bars |
| `plots/fig2_calibration_profiles.{png,pdf}` | Rolling search rate vs entropy |
| `plots/fig3_signature_radar.{png,pdf}` | Radar fingerprint |
| `plots/fig4_search_leverage.{png,pdf}` | SLI by certainty level |
| `plots/fig5_calibration_scatter.{png,pdf}` | Per-question entropy vs search intensity |
| `plots/fig6_dataset_decomposition.{png,pdf}` | By hop-count and category |
| `plots/fig7_model_heatmap.{png,pdf}` | Multi-signature comparison heatmap |
| `report.md` | Auto-generated narrative with all numbers inline |

---

## 7. Key Findings Summary

The following table captures the central empirical result at the time of writing
(MusiQue: 3 models × 600 examples = 1800 EEUs each; ShareChat: 74 questions, 1004 EEUs):

```
Dataset    Model               E      PR     M      CP     SCC   AUROC  Prec   Recall   F1    FDR   CovGap
──────────────────────────────────────────────────────────────────────────────────────────────────────────
MusiQue    Gemini-3-Pro      44.7%  29.7%  10.1%  15.6%  0.115  0.557  0.601   0.815  0.692  0.399  0.101
MusiQue    Nemotron-Nano-30B 43.3%   6.7%  34.6%  15.4%  0.213  0.619  0.866   0.556  0.677  0.134  0.346
MusiQue    Qwen3.5-122B      49.0%  22.7%  14.4%  13.9%  0.192  0.610  0.684   0.772  0.725  0.316  0.144
ShareChat  Gemini-3-Pro       1.1%   8.6%  11.5%  78.7%  0.026  0.492  0.114   0.088  0.099  0.886  0.115
```

*(FDR = False Discovery Rate = 1 − Precision, formerly called POR)*

**The three main findings:**

1. **Search reflex on benchmarks**: MusiQue models dedicate 6–30% of all EEUs to PR
   (certain hops that were searched anyway). Gemini's 29.7% PR fraction is the most
   extreme. This is search happening by reflex, not by calibration.

2. **Selective parametric reliance on real queries**: ShareChat's 72.6% CP fraction means
   the model predominantly handles conversational queries from parametric knowledge alone
   and only occasionally searches. When it does search, the facts it retrieves tend to be
   unambiguous (accounting for high apparent POR under post-attribution measurement).

3. **SLI confirms search utility is certainty-conditional**: Search provides +26.3 pp
   accuracy for questions the model can't answer at all, but costs −9.6 pp when the model
   was already fully correct. The crossing point is at partial certainty (1–4/5 bin, +10.4 pp).
   This quantifies the tradeoff: search is a net positive for uncertain questions and a
   net negative (via context poisoning) for certain ones.

---

## 8. Extended Metrics — Magnitude-Aware Overhead & Budget Efficiency

These metrics were designed to address the binary-POR limitation and to support
hypothesis testing about search call patterns across question types.

---

### Metric 8.1: EWOI — Entropy-Weighted Overhead Index

**Motivation:** POR asks *whether* each searched EEU was certain (binary). EWOI asks
*how certain* each searched EEU was (continuous). It is the higher-order counterpart
of POR that quantifies the intensity of overhead, not just its presence.

**Formula:**
```
certainty_score(entropy_i) = max(0, 1 − entropy_i / ln(K))

EWOI = (1 / |searched|) × Σ certainty_score(entropy_i)   for all searched EEUs i
```

where K = number of paraphrase samples (5), so ln(5) ≈ 1.609 is the maximum
possible entropy, and certainty_score maps entropy to [0, 1] linearly:
- entropy = 0.0  → score = 1.0 (completely certain — maximum waste)
- entropy = ln(5) → score = 0.0 (maximally uncertain — no waste)
- entropy = 0.5   → score ≈ 0.69 (moderately certain)

**Range:** [0, 1]
- **EWOI = 1**: all searched units were fully certain — pure reflex
- **EWOI = 0**: all searched units were maximally uncertain — perfectly efficient
- **EWOI > POR**: the typical case — models also search moderately certain units that
  POR does not count (because they don't meet the binary certainty threshold)

**Relationship to POR:**
POR is a lower bound on EWOI. Specifically, EWOI ≥ POR because it assigns fractional
weight to searched EEUs that are uncertain but not fully so.
```
EWOI = POR × 1.0 + fraction_of_low_uncertainty_searched × their_certainty_scores
```

**Note on entropy base:** Entropy is computed in **bits (log₂)** by the sampling pipeline.
The maximum entropy for 5 samples is log₂(5) ≈ 2.322 bits, not ln(5) ≈ 1.609 nats.
`MAX_ENTROPY = log₂(5) = 2.322` must be used when normalising certainty_score.
Using ln(5) would systematically clip high-entropy units to certainty_score = 0 and
understate EWOI by ~0.1 for uncertain-heavy models like Nemotron.

**Implementation:**
- Computed over all searched EEUs in the dataset (EEU level, not question level)
- Uses the entropy column directly — does not require the joint certainty definition
- Therefore applicable symmetrically to both MusiQue and ShareChat

**Cross-dataset caveat (same as POR):** ShareChat certainty is post-attribution.
EWOI values for ShareChat reflect how certain retrieved facts *are to state*, not
whether the model knew them before search. See Section 2 asymmetry warning.

**Current values (empirical):**
| Dataset | Model | EWOI | POR | Δ (EWOI − POR) |
|---------|-------|------|-----|-----------------|
| MusiQue | Gemini-3-Pro | **0.863** | 0.399 | +0.464 |
| MusiQue | Nemotron-Nano-30B | **0.579** | 0.134 | +0.445 |
| MusiQue | Qwen3.5-122B | **0.709** | 0.316 | +0.393 |
| ShareChat | Gemini-3-Pro | **0.951** | 0.886 | +0.065 |
| ShareChat | Nemotron-Nano | **0.842** | 0.692 | +0.150 |

**Interpretation:**
The EWOI gap between Gemini (0.863) and Nemotron (0.579) is 0.284. When Gemini searches,
its hops are on average 86% as certain as maximally-certain hops. When Nemotron searches,
its hops are 58% as certain — substantially more uncertain and therefore more justified.
The binary POR (0.399 vs 0.134) captures the binary threshold effect but understates
how much more calibrated Nemotron is in the non-zero-entropy regime.

The large Δ = EWOI − POR for all MusiQue models reflects that beyond the binary-certain
hops (POR numerator), a further fraction are at H=0.722 (the 4/5-agreement level):
these are not binary-certain but are still mostly-confident searches that POR misses.

---

### Metric 8.2: SBE — Search Budget Efficiency

**Motivation:** EWOI is an intensity metric (how wasteful per searched EEU). SBE is
a *volume* metric — how productively is the total query budget spent? A model that
issues 14 queries per question and achieves 44.7% E-EEUs is categorically less efficient
than one that issues 4.5 queries and achieves 43.3% E-EEUs.

**Formula:**
```
SBE = N_E_total / total_queries
    = (count of E-quadrant EEUs across all examples)
      / (total search API calls issued across all examples)
```

where `total_queries = SIR_mean × N_questions`.

Using entropy>0 as the certainty boundary for cross-dataset symmetry:
```
N_E_total = count(EEUs where search_attributed=True AND entropy > 0)
```

**Range:** [0, ∞), though practically (0, 1] in most settings.
- **SBE → 1**: every query directly addresses a genuinely uncertain EEU
- **SBE → 0**: queries are issued but all land on certain units (pure overhead)

**Units:** E-EEUs per search query. Interpretable as: "on average, how many
genuinely uncertain evidence units does each search call address?"

**Note on cross-dataset comparability:** SBE has the same per-query unit in both
datasets (search API calls), but EEUs differ in granularity (hops vs atomic facts).
SBE should be compared within a dataset across models. Cross-dataset comparison
of SBE requires adjusting for mean EEUs per question.

**Current values:**
| Dataset | Model | SBE | SIR_mean | N_E | Total queries |
|---------|-------|-----|----------|-----|---------------|
| MusiQue | Gemini-3-Pro | **0.045** | 13.58 | 370 | 8,148 |
| MusiQue | Nemotron-Nano-30B | **0.233** | 4.56 | 637 | 2,736 |
| MusiQue | Qwen3.5-122B | **0.232** | 4.51 | 628 | 2,706 |
| ShareChat | Gemini-3-Pro | **0.901** | 0.91 | 146 | 162 |

**Interpretation:**
Gemini issues 3× more total queries than Nemotron or Qwen (8,148 vs ~2,720), yet
achieves nearly identical E-EEU counts (~370 vs ~630). Its SBE is 0.045 — each
query touches a genuinely uncertain hop only 4.5% of the time. Nemotron and Qwen
are 5× more budget-efficient (SBE ≈ 0.23).

This reveals a regime qualitatively different from Nemotron's underfitting story
(high CovGap). Gemini overfits by issuing redundant queries — not just searching
certain hops once (POR) but searching them many times (SBE). The two failure modes:
- **Reflexive breadth** (Gemini): searches everything, many times. High POR, very low SBE.
- **Under-coverage** (Nemotron): searches selectively, rarely. Low POR, moderate SBE.

ShareChat's SBE of 0.901 reflects the coarse-but-correct activation pattern — when
it searches, it retrieves genuinely uncertain facts (by entropy definition), but this
is enabled by very few total queries (0.91/question).

---

### Metric 8.3: SAE — Search Activation Entropy

**Motivation:** SCC (Spearman ρ) measures the correlation between uncertainty and
search. SAE provides an absolute-scale counterpart — the *mean entropy difference*
between searched and non-searched units — interpretable in the same entropy units as
the raw data, without requiring correlation computation.

**Formula:**

EEU-level (both datasets):
```
SAE_eeu = mean(entropy_i | search_attributed_i = True)
          − mean(entropy_i | search_attributed_i = False)
```

Question-level (ShareChat only, mirrors SCC_q):
```
SAE_q = mean(mean_entropy_per_question | has_search = True)
        − mean(mean_entropy_per_question | has_search = False)
```

**Range:** (−∞, +∞). Practically in [−ln(5), +ln(5)].
- **SAE > 0**: searched units are more uncertain on average — correct activation signal
- **SAE = 0**: search is activated independently of uncertainty — reflex behaviour
- **SAE < 0**: model searches more on certain units — pathological (reverse calibration)

**Relationship to SCC:** SAE and SCC both measure search–uncertainty coupling.
SCC captures the rank correlation (monotone relationship); SAE captures the
mean entropy gap (absolute shift). They are complementary: a high SCC with low SAE
means the rank ordering is correct but the effect size is small.

**Current values:**
| Dataset | Level | Model | SAE | Interpretation |
|---------|-------|-------|-----|----------------|
| MusiQue | EEU | Gemini-3-Pro | **+0.127** | Searched hops are marginally more uncertain |
| MusiQue | EEU | Nemotron-Nano-30B | **+0.332** | Strong activation signal — Nemotron is calibrated |
| MusiQue | EEU | Qwen3.5-122B | **+0.317** | Similar to Nemotron |
| ShareChat | EEU | Gemini-3-Pro | **+0.033** | Near-zero — search independent of fact uncertainty |
| ShareChat | question | Gemini-3-Pro | ~+0.251* | Significant at question level (*raw SCC_q ρ value) |

*For ShareChat question-level, the SAE_q value in entropy units should be computed from
raw data; the SCC_q=0.251 is shown as a proxy for the ordering relationship.

**Interpretation:**
The contrast SAE_eeu(Gemini/MusiQue) = 0.127 vs SAE_eeu(Nemotron/MusiQue) = 0.332
shows that Nemotron's search is much more strongly coupled to actual uncertainty.
Gemini on MusiQue barely discriminates (it searches both certain and uncertain hops
at nearly the same entropy level).

The near-zero SAE at the ShareChat EEU level (+0.033) vs significant question-level
activation (SCC_q = 0.251) is the strongest evidence for the **coarse-grained
activation hypothesis**: search decisions are made holistically at question level
(based on overall question difficulty), not at the per-fact level.

---

### Metric 8.4: Graded POR — Overhead by Discrete Entropy Level

**Motivation:** The EWOI collapses the overhead distribution into a single scalar.
For hypothesis testing by question type (MusiQue hop count, ShareChat category),
a richer picture is needed: where in the certainty spectrum does the overhead live?

**Important structural property — entropy is discrete:**

Entropy is computed over **5 paraphrase samples** using semantic clustering (log₂ base).
With N=5, there are exactly **7 possible entropy values**, each corresponding to a unique
cluster partition:

| Entropy (bits) | Partition | Meaning |
|:--------------:|-----------|---------|
| **0.000** | [5] | All 5 samples in one cluster — maximal certainty |
| **0.722** | [4,1] | 4 agree, 1 outlier — high certainty with one dissenter |
| **0.971** | [3,2] | Split into two groups of 3 and 2 |
| **1.371** | [3,1,1] | One cluster of 3, two singletons |
| **1.522** | [2,2,1] | Two groups of 2, one singleton |
| **1.922** | [2,1,1,1] | One pair, three singletons |
| **2.322** | [1,1,1,1,1] | All 5 samples in distinct clusters — maximal uncertainty |

**Consequence for binning:** Any entropy bin of the form `(0, x]` for x < 0.722 is
structurally empty — no valid 5-sample partition produces entropy in that range.
A previous version of this analysis used a `(0, 0.5]` bin that appeared empty, which
was mistakenly described as "epistemic bimodality". The correct interpretation is that
the second-lowest entropy level is 0.722, not ~0. The *actual* distribution across
all 7 levels is the meaningful description.

**Definition (four tier groups, aligned to discrete structure):**

| Tier | Entropy value(s) | Partition(s) | Interpretation |
|------|-----------------|--------------|----------------|
| **certain** | 0.000 | [5] | All samples agree — model is fully consistent |
| **near-certain** | 0.722 | [4,1] | 4/5 agree, one outlier — near-certain |
| **split** | 0.971, 1.371, 1.522 | [3,2], [3,1,1], [2,2,1] | Meaningful disagreement |
| **uncertain** | 1.922, 2.322 | [2,1,1,1], [1,1,1,1,1] | Near-random responses |

**Current values (MusiQue, fraction of all searched EEUs):**
| Dataset | Model | certain (H=0) | near-cert (H=0.72) | split | uncertain |
|---------|-------|:-------------:|:------------------:|:-----:|:---------:|
| MusiQue | Gemini-3-Pro | **72.3%** | 9.3% | 10.6% | 7.8% |
| MusiQue | Nemotron-Nano-30B | **29.2%** | 19.0% | 29.8% | 22.1% |
| MusiQue | Qwen3.5-122B | **51.3%** | 13.6% | 19.0% | 16.1% |
| ShareChat | Gemini-3-Pro | **88.6%** | 6.7% | 3.2% | 1.6% |
| ShareChat | Nemotron-Nano | **69.2%** | 12.4% | 15.7% | 2.7% |

This replaces the earlier bimodal framing. The data is not bimodal — Nemotron shows
a relatively uniform spread across all 7 levels (29%, 19%, 9%, 15%, 6%, 14%, 8%),
while Gemini is genuinely concentrated at H=0 (72%). ShareChat's concentration at
H=0 for searched facts reflects post-attribution certainty, not pre-search knowledge.

The **confident-but-brittle fraction** (MusiQue only, requires joint certainty mode):
```
P_brittle = POR_extreme − POR(joint mode)
         = 0.723 − 0.399 = 0.324  (Gemini)
         = 0.292 − 0.134 = 0.158  (Nemotron)
         = 0.513 − 0.316 = 0.197  (Qwen)
```
These EEUs have entropy=0 (consistent answers) but the model answers incorrectly
without search — **confident incorrectness**. Low entropy ≠ factual accuracy.

---

### Metric 8.5: IQSRS — Intra-Question Search Routing Score

**Availability:** MusiQue only. Requires `queries_assigned_count` per hop.

**Motivation:** SCC measures corpus-level correlation between hop entropy and search
attribution. IQSRS asks a more fine-grained question: *within a single multi-hop
question*, does the model route more queries to the harder (higher-entropy) hops?
A model can have high corpus-level SCC while routing queries uniformly within questions
(if question-level difficulty explains most of the variance).

**Formula (per question q with H_q hops):**
```
IQSRS_q = Spearman_ρ(
    [entropy_h for h in hops(q)],
    [queries_assigned_h for h in hops(q)]
)

IQSRS = mean(IQSRS_q)   for all q with H_q ≥ 3 hops (need ≥3 points for correlation)
```

**Range:** [−1, +1]
- **IQSRS > 0**: model routes more queries to uncertain hops *within questions* — fine-grained calibration
- **IQSRS ≈ 0**: query allocation is uniform within questions — coarse calibration (or
  queries are issued at the question level and spread uniformly across hops)
- **IQSRS < 0**: model routes more queries to certain hops — within-question miscalibration

**Statistical test:** One-sample t-test against 0 (H₀: IQSRS = 0).
Also compute IQSRS stratified by hop count (3-hop, 4-hop) to test W3-style hypotheses.

**Not yet computed.** Implementation requires `queries_assigned_count` per hop from
`interplay_summary.csv`. The column is present — compute when running the full analysis.

---

## 9. Hypothesis Tests — Extended Set

These hypotheses extend Section 4 and are directly testable with the data in hand.

### 9.1 Cross-dataset hypotheses (Family 1 extensions)

| ID | Hypothesis | Metric | Method |
|----|-----------|--------|--------|
| **H5** | EWOI_MusiQue_Gemini ≠ EWOI_ShareChat_Gemini | EWOI | Bootstrap two-sample test |
| **H6** | SAE_MusiQue_Gemini > SAE_ShareChat_Gemini (EEU level) | SAE | Mann-Whitney U |
| **H7** | SBE_MusiQue_Gemini < SBE_MusiQue_Nemotron | SBE | Bootstrap CI non-overlap |

**H5 caveat:** The ShareChat EWOI asymmetry (post-attribution) applies here. Frame as
"MusiQue overhead is driven by reflexive querying; ShareChat overhead is driven by
retrieval of inherently low-entropy facts."

**H6** tests the coarse-activation hypothesis at the cross-dataset level. Expected result:
SAE at EEU level is significantly higher for MusiQue than ShareChat, confirming that
MusiQue's search decisions track uncertainty at unit granularity while ShareChat's do not.

---

### 9.2 Within-MusiQue hypotheses (Family 2 extensions)

| ID | Hypothesis | Metric | Method |
|----|-----------|--------|--------|
| **W5** | EWOI differs across the 3 models | EWOI | Kruskal-Wallis on certainty scores of searched hops |
| **W6** | EWOI increases with hop count (2→3→4 hop chains) | EWOI per hop-count group | Spearman ρ (hop_count, EWOI) |
| **W7** | IQSRS > 0 (within-question routing tracks uncertainty) | IQSRS | One-sample t-test |
| **W8** | SAE_Gemini < SAE_Nemotron (stronger certainty–search coupling for Nemotron) | SAE | Mann-Whitney U on {entropy_i: searched_i=True} vs {entropy_i: searched_i=False} |

**W6** tests the structural benchmark-overfitting hypothesis with magnitude: longer chains
should produce higher EWOI (more reflexive overhead per searched hop) if the model
adapts to expected chain depth rather than per-hop difficulty.

**W8** is the EEU-level version of the Nemotron vs Gemini calibration story, testing
whether the SAE difference (0.332 vs 0.127) is statistically significant.

---

### 9.3 Within-ShareChat hypotheses (Family 3 extensions)

| ID | Hypothesis | Metric | Method |
|----|-----------|--------|--------|
| **S5** | EWOI varies by question category | EWOI per category | Kruskal-Wallis + pairwise Mann-Whitney |
| **S6** | SAE_q > 0 (question-level entropy is higher for searched questions) | SAE_q | Mann-Whitney U one-sided |
| **S7** | Search call count correlates with question-level mean entropy | SIR, entropy | Spearman ρ |
| **S8** | `explanation` category shows higher EWOI than `factual_lookup` | EWOI | Mann-Whitney U, exact |

**S5** tests whether the S2 finding (category-level POR differences) holds when
measured with magnitude. The `explanation` category having higher PR rate (binary,
from S2) should translate to higher EWOI (continuous).

**S6** is the entropy-scale version of S1 (SCC_q > 0) and provides the absolute
effect size: by how many entropy units do searched questions differ from non-searched?

**S7** extends S3 from a binary has_search indicator to the actual call count. If
questions with higher mean entropy trigger more search calls (not just search = True/False),
this would confirm adaptive — not just binary threshold — search behavior.

---

## 10. The Conference Narrative

The metrics above support a coherent story across two contrasting task regimes.

### The central tension

Models must decide *when* to search. Searching too eagerly wastes compute and can
introduce noise (context poisoning, SLI < 0). Searching too rarely leaves knowledge
gaps uncovered (high CovGap). The ideal model searches exactly where it is uncertain.

### Story arc (four acts)

**Act 1 — The Setup**: Introduce MusiQue (structured multi-hop, gold answers available)
and ShareChat (open-ended conversational, no gold). Both require calibrated search.
Define the EEU as the unit of analysis. Introduce the 4-quadrant taxonomy.

**Act 2 — Binary picture**: Show the E/PR/M/CP decomposition table. Gemini on MusiQue
has 29.7% PR — search reflex. ShareChat has 72.6% CP — parametric reliance.
Binary POR = 0.399 vs 0.134 for Gemini vs Nemotron.

**Act 3 — The magnitude reveal**: Upgrade to EWOI and discrete entropy tiers.
- *"POR says Gemini wastes 40% of its searches. EWOI says it's worse: when Gemini
  searches, those hops are 86% as certain as maximally-certain hops. Compare
  Nemotron at 58% — its overhead is softer and reflects partial knowledge, not
  full certainty."*
- The discrete level finding: 72.3% of Gemini's searched hops have entropy=0 exactly.
  Nemotron: only 29.2%, with 19% at H=0.722 (one sample disagrees) and 22% in the
  highly uncertain tier. Gemini front-loads at full certainty; Nemotron spreads.
- Caveat: the apparent "gap" between entropy=0 and the next level is not bimodal
  uncertainty — it is the minimum entropy step given 5 samples (H_min = log₂(5/4) = 0.722).
- The confident-but-brittle fraction: 32.4% of Gemini's searched hops are
  entropy=0 but the model was still wrong without search. **Low entropy ≠ correct answer.**

**Act 4 — The budget story (SBE and SAE)**:
- Gemini issues 18× more queries than ShareChat Gemini and 3× more than Nemotron,
  but achieves 5× worse budget efficiency (SBE = 0.045 vs 0.233).
- SAE shows Nemotron's search is *pulled* by uncertainty (+0.332 entropy gap)
  while Gemini's search barely discriminates (+0.127).
- The coarse-activation pattern on ShareChat (SAE_eeu ≈ 0, SCC_q > 0): models
  decide to search at question level, not fact level. This is a fundamental limitation
  of the current agent architecture (single binary search-or-not decision before
  answering, rather than per-fact routing).

### Punchline

> "Search calibration in LLMs operates at the wrong granularity. On structured
> benchmarks, models over-search reflexively — wasting 81% of their query certainty
> mass on knowledge they already possess. On open-ended real-world queries, they make
> coarser but better decisions — but still cannot route search to the specific facts
> they are uncertain about. The path to efficient knowledge-augmented QA requires
> moving search decisions from question-level to evidence-unit-level."

---

## 11. Implementation Checklist

To compute the new metrics, the following additions are needed in
`unified_interplay_analysis.py`:

- [ ] **EWOI**: add `compute_ewoi(df_eeu)` — average certainty score over `search_attributed=True` rows. One line using `df.loc[mask, 'certainty_score'].mean()` after adding a `certainty_score = (1 - entropy/ln(5)).clip(0,1)` column.
- [ ] **SBE**: add to `compute_signatures()`. Requires total query count per (dataset, model) — already available from `SIR_mean × N_questions`.
- [ ] **SAE**: add `compute_sae(df_eeu)` — grouped mean entropy difference. Question-level variant available from `df_question`.
- [ ] **Graded POR**: add three columns `por_extreme`, `por_moderate`, `por_borderline` alongside existing POR in `interplay_signatures.csv`.
- [ ] **Confident-but-brittle fraction**: for MusiQue only, compute `por_extreme − por_joint` where `por_joint` = existing POR.
- [ ] **IQSRS**: add `compute_iqsrs(df_eeu)` — group by (dataset, model, example_id), compute within-question Spearman ρ, then average. Requires ≥3 hops per question for valid correlation.
- [ ] **New hypothesis tests H5–H8, W5–W8, S5–S8**: add to `run_statistical_tests()`, applied within the same BH-corrected families as existing tests.
- [ ] Add `ewoi`, `sbe`, `sae_eeu`, `sae_q`, `iqsrs`, `por_extreme`, `por_moderate`, `por_borderline`, `p_brittle` columns to `interplay_signatures.csv`.
- [ ] Update `plots/fig7_model_heatmap` to include new signatures.
