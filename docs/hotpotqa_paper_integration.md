# HotpotQA — analysis summary and paper-integration guide

**Target paper:** `/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex`
**Analysis repo:** `/home/dvirla/projects/parametric_search_tradeoff`, branch `hotpotqa-cue-pilot`
(commits `9d94297`, `79ce09a`, `7bc32a0`).
**Written:** 2026-09-06.

HotpotQA is a **third dataset** for the perturbation grid, and — more importantly — the venue where
the SFT intervention's **out-of-domain transfer actually succeeds**, which the paper currently
reports as a negative result on MedQA.

---

## 1. What was run

| | |
|---|---|
| Questions | `hotpotqa-300` — 300 questions, type-stratified (240 bridge / 60 comparison), seed 0, from `hotpotqa/hotpot_qa` **distractor** config, validation split. Nested subsets at 50/300/500. Manifest: `data/hotpotqa_subset_manifest.json`, questions: `data/hotpotqa_300.jsonl`. |
| Retrieval | Local BM25 over `data/hotpotqa_index` — 67,432 passages pooled from the distractor config's own per-question paragraphs (2 gold + 8 distractor each), ≤1500 chars. Gold-title **recall@10 = 0.935 vs 0.000 random** (`scripts/validate_hotpotqa_index.py`, n=200, seed 0). Deterministic; no live web. |
| Phrasings | **One** (the dataset's own question text). No terse rewrite — see §5. |
| Perturbations | 9: `plain`, `polite`, `multiturn`, `searchmulti`, `natural`, `elaborate`, `query`, `direct`, `confident_parametric` — the same templates/histories as FRAMES/MedQA. Plus `plain_rep2`, a full second `plain` run per model = the run-to-run noise floor. |
| Models | **9 open-weights** (the paper's roster minus the two Gemini models, which were not run here): Qwen3.5 122B/35B/4B, Gemma4 31B/E4B, Nemotron3 30B, Nemotron-Cascade2 30B, GPT-OSS 120B/20B. Plus the FRAMES-SFT checkpoint as a separate transfer arm. |
| Rows | 30,000 in the search grid (10 result sets × 10 conditions × 300), complete. |
| Grading | **Exact match only.** Every row was collected with `--no_grader`, so `sampler_correct` is `None` everywhere and no LLM judge exists for this dataset. Verdicts from `scripts/grade_hotpotqa_regex.py`, which imports `regrade_regex.py`'s match functions — so HotpotQA EM is *the same grader* as the paper's FRAMES EM. |

Result files: `results/hotpotqa_cue_grid/<model_slug>/hotpotqa-300_baseline_<model>_<cond>.json`
Graded rows: `results/hotpotqa_cue_grid_regex/{per_row.csv, by_condition.csv, summary.md}`

---

## 2. Figures — paths, and where each one goes in the paper

All under `/home/dvirla/projects/parametric_search_tradeoff/`.

| Figure path | What it shows | Paper destination |
|---|---|---|
| **`results/hotpotqa_cue_briefing/brief_aggregate_search_acc_mean_HotpotQA.png`** | The Δsearch / Δaccuracy aggregate — direct twin of the paper's `brief_aggregate_search_acc_mean_{FRAMES,MedQA_llm6}.png` | **Fig. `fig:combined_search_acc`** (§\ref{sec:decoupling}) — add as a third panel |
| **`results/hotpotqa_cue_briefing/gemma_cue_robustness_hotpotqa.png`** | Baseline vs. FRAMES-SFT on HotpotQA, Δsearch + Δaccuracy per perturbation, shared y-axis | **Fig. `fig:gemma_sft`'s companion** (§\ref{sec:sft_interventions}) — this is the new headline transfer result |
| `results/hotpotqa_cue_briefing/brief_aggregate_search_acc_median_HotpotQA.png` | Median-estimator version of the above | Appendix / robustness |
| `results/hotpotqa_cue_briefing/brief_zero_search_dotplot.png` | Δ zero-search rate per perturbation | Supports Table `tab:zero_search` |
| `results/hotpotqa_cue_briefing/brief_aggregate_tables.md` | All numbers behind the above, incl. the example-level Spearman table | Source for the table column in §3.2 below |
| `results/hotpotqa_cue_briefing/brief_combined_search_acc_{primary,secondary}.png` | Per-model grids (3 + 6 models) | Appendix, if per-model detail is wanted |
| `results/hotpotqa_cue_briefing/brief_{search_bars,search_counts,accuracy,zero_search}_{primary,secondary}.png` | Per-model Δsearch %, raw call counts w/ CIs, Δaccuracy, Δzero-search | Appendix |
| `results/hotpotqa_cue_briefing/BRIEFING_HOTPOTQA.md` | Prose write-up of the whole reproduction | Reading reference, not for the paper |

**To copy the two headline figures into the paper repo:**

```bash
P=/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/Figures
cp results/hotpotqa_cue_briefing/brief_aggregate_search_acc_mean_HotpotQA.png  $P/
cp results/hotpotqa_cue_briefing/gemma_cue_robustness_hotpotqa.png            $P/
# optional appendix material
cp results/hotpotqa_cue_briefing/brief_aggregate_search_acc_median_HotpotQA.png $P/
cp results/hotpotqa_cue_briefing/brief_combined_search_acc_secondary.png       $P/hotpotqa_combined_search_acc_secondary.png
```

---

## 3. Section-by-section integration

### 3.1 §Dataset and Retrieval Setup

Currently: *"We utilize 2,000 distinct evaluation questions: 500 from FRAMES ... and 500 from MedQA
... Each question is expressed in two base phrasings."*

Needs to become 2,000 + 300 = **2,300**, with HotpotQA carrying **one** phrasing, not two. Note
`yang_hotpotqa_2018` is already in the bibliography (currently cited only as a descriptor of
FRAMES's multi-hop character) — it now also cites an evaluated dataset. Suggested addition:

> ...and 300 from HotpotQA \citep{yang_hotpotqa_2018}, a multi-hop dataset retrieved over a local
> BM25 index built from its own distractor-config paragraph pool (67,432 passages; gold-title
> recall@10 $=0.935$ against a $0.000$ random baseline). HotpotQA is evaluated in a single
> phrasing and on the nine open-weight models only.

### 3.2 §The General Suppression of Search Policies — Table `tab:zero_search`

Drop-in third column (Δ zero-search rate in pp, mean across the 9-model HotpotQA roster; `—` for
TERSE, which does not exist on this dataset):

```latex
    \begin{tabular}{llrrr}
    \toprule
    \textbf{Perturbation} & \textbf{Group} & \textbf{FRAMES} & \textbf{MedQA} & \textbf{HotpotQA} \\
    \midrule
    RERUN (noise floor) & Variation Measurement & +0.3    & +1.4   & +0.2    \\
    TERSE               & Style                 & -0.9    & -2.0   & ---     \\
    POLITE              & Style                 & +0.6    & -6.0   & +0.8    \\
    MULTITURN           & Conversation State    & +15.0*  & +26.7* & +18.2   \\
    SEARCH MULTITURN    & Conversation State    & +25.6** & +30.1* & +6.5    \\
    SHORT               & Directives            & +4.9*   & +14.6* & +5.4    \\
    ELABORATE           & Directives            & +4.4    & +9.9*  & +5.4    \\
    DIRECT              & Directives            & +10.0*  & +16.6* & +11.0*  \\
    QUERY               & Directives            & +4.3    & +10.1  & +3.6    \\
    CONFIDENT           & Directives            & +44.1** & +26.8* & +59.9** \\
    \bottomrule
    \end{tabular}
```

⚠️ **The HotpotQA stars come from a separate BH-FDR family** (its own 10 bars, 9 models), not the
FRAMES/MedQA 3-panel family. Either say so in the caption, or re-run all four panels in one
invocation to get a single family (§6). **Measured effect of doing so** (I ran both):

* The FRAMES and MedQA **aggregate figures are byte-identical** either way — no bar in
  `brief_aggregate_search_acc_mean_{FRAMES,MedQA_llm6}.png` crosses a threshold. So adding the
  HotpotQA panel does **not** disturb the paper's existing Figure `fig:combined_search_acc`.
* Three stars in the **markdown tables** do move: MedQA (LLM-judge panel) example-level
  correlations for MULTITURN ($-0.07$) and SEARCH MULTITURN ($-0.06$) *gain* a `*`, and HotpotQA's
  zero-search DIRECT (+11.0pp) *loses* its `*`. Both directions are expected — BH-FDR power depends
  on the whole p-value distribution.

Given that, the simplest defensible choice is the separate family plus a caption sentence; the
combined family is available if you prefer one correction over the whole analysis.

Two things worth a sentence in this subsection:

* **HotpotQA has no zero-search floor.** Its `plain` zero-search rate is 0–17% per model, versus
  MedQA's ~96% for Gemma 4 31B. This is why the suppression effects here are measuring real policy
  movement rather than a model that had already stopped searching, and it makes HotpotQA the
  cleaner venue for the Table `tab:zero_search` claim than MedQA.
* **`confident` is the largest zero-search lever on all three datasets** (+59.9pp here), which
  strengthens the existing "critical suppressed baseline" argument.

### 3.3 §The Decoupling of Search and Accuracy — Fig. `fig:combined_search_acc`

Add the HotpotQA panel. Headline numbers (mean across 9 models, paired vs. each model's own
`plain`; EM):

| Perturbation | Δsearch HotpotQA | Δsearch FRAMES | Δsearch MedQA | ΔEM HotpotQA | ΔEM FRAMES |
|---|---|---|---|---|---|
| RERUN (floor) | +1% | +1% | −7% | +0.2 | −0.7 |
| POLITE | +3% | −1% | +22% | +0.8 | −0.2 |
| MULTITURN | **−31%\*\*** | −34%\*\*\* | −90%\*\*\* | **−6.0\*** | −3.1 |
| SEARCH MULTITURN | **−20%\*\*** | −37%\*\* | −86%\*\*\* | −2.0 | −1.1 |
| SHORT | −11% | −16% | −54%\*\* | −1.2 | −2.7 |
| ELABORATE | +2% | −9% | −45% | +0.9 | +0.5 |
| DIRECT | −5% | −9% | −63%\*\* | −10.1\*\* | −7.0\*\* |
| QUERY | +24% | +9% | −13% | +1.5 | +1.1 |
| CONFIDENT | **−64%\*\*\*** | −52%\*\* | −88%\*\* | **−21.4\*\*** | −8.3\*\* |

**What HotpotQA adds to this section's argument.** The paper currently contrasts MedQA
(`confident` suppresses search with *no* accuracy cost ⇒ over-reliance on tools) against FRAMES
(`confident` causes a drastic drop ⇒ confident hallucination). HotpotQA turns that two-point
contrast into a **three-point gradient ordered by how substitutable parametric knowledge is for
retrieval**: MedQA ≈0pp → FRAMES −8.3pp → HotpotQA **−21.4pp**. HotpotQA is multi-hop over a
corpus the model genuinely cannot answer from memory, and it pays the largest price. This is a
stronger version of the same claim, and it makes the "whether over-relying on tools or confidently
hallucinating" sentence land on an ordered axis rather than two poles.

**Median vs. mean.** On the two *increase* perturbations the cross-model mean is carried almost
entirely by Nemotron3 30B (QUERY +109%, ELABORATE +62%, DIRECT +48% — the same model that inverts
on FRAMES). QUERY is +24% on the mean but +8% on the median. Quote the median for those two, or
say the mean is one-model-driven.

⚠️ **One claim in this section does NOT reproduce cleanly on HotpotQA — do not extend it silently.**
The paper states the example-level Spearman between Δsearch and Δcorrectness ranges *"from −0.03 to
+0.03 across all perturbations for both FRAMES and MedQA"*. On HotpotQA the same statistic runs
**+0.06 to +0.17**, and is significant for MULTITURN (+0.17\*), QUERY (+0.13\*) and CONFIDENT
(+0.13\*\*). The sign is consistent with the paper's story (searching *less* on an example goes with
getting it *less* right), but the magnitude is no longer "zero". Options: (a) restrict the
zero-correlation sentence to FRAMES/MedQA and report HotpotQA's weak-but-nonzero coupling as the
expected behaviour on a task where retrieval is genuinely load-bearing; or (b) reframe the claim as
"weak at best, and only detectable where the corpus is not substitutable". Option (a) is the
smaller edit and is more defensible. Numbers: `results/hotpotqa_cue_briefing/brief_aggregate_tables.md`,
§ Search-Accuracy Example-Level Correlation.

### 3.4 §Interaction Between Perturbation and Phrasing

**Not applicable to HotpotQA** — single phrasing, so there is no interaction to model. Say so in one
clause if the section otherwise reads as covering all datasets.

### 3.5 §Mitigating Search-Policy Instability via Fine-Tuning — the biggest change

The paper's current out-of-domain paragraph reads: *"Out-of-domain, on MedQA (never seen in
training), search propensity transfers (0.09 → 2.35 plain calls) but perturbation-invariance does
not (still 6 of 6 perturbations significant), with accuracy flat."*

**HotpotQA overturns that negative.** Same 7-perturbation Gemma 4 31B checkpoint, never trained on
HotpotQA, all 300 questions held out. Of HotpotQA's 8 non-plain perturbations, **5 were trained on**
(`polite`, `short`, `elaborate`, `query`, `direct`) and **3 were not** (`multiturn`,
`search multiturn`, `confident`) — so one run reads new-dataset *and* new-perturbation
generalization at once.

In the paper's own metric (mean $|\Delta|$ search calls vs. own `plain`, # significant of 8):

| | baseline Gemma 4 31B | SFT frames-robust |
|---|---|---|
| `plain` search level | 2.14 calls | 2.41 calls |
| `plain` zero-search rate | 6.0% | 0.0% |
| `plain` EM | 0.804 | 0.804 |
| mean $\|\Delta\text{calls}\|$ / #sig (all 8) | 0.69 / **7 of 8** | **0.25 / 4 of 8** |
| **SEEN** perturbations, mean $\|\Delta\%\|$ / #sig | 26.5% / **4 of 5** | **2.8% / 1 of 5** |
| **UNSEEN** perturbations | 42.3% / 3 of 3 | 22.6% / 3 of 3 |
| run-to-run floor (search) | −1.4%, n.s. | **+1.8%, n.s.** |
| run-to-run floor (EM) | −2.8pp, n.s. | −0.7pp, n.s. |

1. **Trained perturbations transfer almost completely.** ELABORATE −43.1\*\*\* → −0.1% n.s.,
   DIRECT −36.8\*\*\* → −3.0% n.s., POLITE −26.5\*\*\* → +2.8% n.s. The 2.8% mean residual is
   ~1.6× the SFT's **own** run-to-run floor (+1.8%, n.s.). Only SHORT survives (−3.5%, $p=.015$),
   ~2× the floor.
2. **Untrained perturbations transfer partially** — roughly halved (42.3% → 22.6%) but all three
   still significant, `confident` still −41%. So this is perturbation-robustness generalizing, not
   a blunt "always search" reflex the model acquired.
3. **Perturbation-induced accuracy damage roughly halves**, at zero cost to `plain` accuracy
   (0.804 both arms): DIRECT −17.8pp → −8.0pp, CONFIDENT −24.1pp → −9.8pp, ELABORATE −7.0pp →
   +4.9pp — at matched median response length, so the EM verbosity bias does not drive the
   between-arm contrast.
4. **Same caveat as FRAMES: invariance, not restoration.** The SFT is flat across perturbations but
   anchored at a *shifted* `plain` level (2.14 → 2.41 calls; 6.0% → 0.0% zero-search), exactly the
   pattern the paper already reports in-domain (4.85 → 6.08 on FRAMES).
5. **The two perturbations that resist are the same two as in-domain** — `multiturn` and
   `confident` — reinforcing the existing sentence that these are "the largest levers on search
   volume in the whole grid".

**Suggested reframing of the out-of-domain paragraph:** MedQA and HotpotQA are not contradictory
results, they are a dose-response. MedQA's baseline sits at a ~96% zero-search floor for this model
(0.09 `plain` calls), so there is almost no search policy left to be invariant *about*; the
intervention there can only move propensity. HotpotQA's baseline searches 2.14 calls with a 6%
zero-search rate — a real policy — and there invariance transfers. The honest claim is
**"perturbation-invariance transfers out of domain wherever a non-degenerate search policy exists
to transfer"**, with MedQA as the degenerate case rather than a failure. §\ref{app:sft_full_results}
already carries the MedQA detail; HotpotQA belongs in the main text beside it.

---

## 4. Caveats that must ship with any HotpotQA number

1. **EM only.** No LLM judge exists for HotpotQA (all rows `--no_grader`). The paper's own
   Appendix `app:dual_metric` argument — that EM is a pessimistic lower bound that amplifies but
   does not fabricate the `direct` effect — applies here with no correction available.
2. **The `direct` accuracy bar is length-confounded.** Median response words across the roster:
   DIRECT **2**, PLAIN **63**, ELABORATE **252**. DIRECT's −10.1pp is an upper bound and is
   probably mostly the artifact the paper already documents (regex −10..−13pp vs. judge −0..−6pp on
   FRAMES). ELABORATE's +0.9pp is inflated for the mirror-image reason. `confident` (51 words),
   `multiturn` (70), `short` (54), `searchmulti` (57) and `query` (71) *are* length-comparable to
   `plain` — those bars are clean, including the −21.4pp `confident` headline.
3. **Yes/no golds excluded from accuracy.** 14 of 300 have gold literally "yes"/"no", where
   substring matching is meaningless; they are dropped from EM but kept for search volume. This is
   why the transfer figure's accuracy bars sit <1pp off `scripts/analyze_hotpotqa_transfer.py`,
   which grades all 300.
4. **One salvaged row excluded.** A single `UsageLimitExceeded` row (qwen3.5:4b `natural`) with
   `search_calls=100` is dropped from every aggregate — on its own it moved that cell by 0.33
   calls, more than most models' entire run-to-run floor.
5. **`searchmulti` counts needed correcting.** The injected conversation history's own mocked
   `search` tool call was being counted as a search the model chose to make, inflating
   `sampler_search_calls` by exactly +1. Fixed at source (commit `f0e71ec`) and corrected for
   pre-fix rows in `grade_hotpotqa_regex.py`. **The paper's FRAMES and MedQA `searchmulti` rows
   were collected with the same uncorrected counter and need the same treatment** — this may be why
   `searchmulti` looked noisy. Worth checking before the camera-ready.
6. **Separate FDR family.** HotpotQA's stars are corrected over its own 10 bars, not the paper's
   3-panel family. Point estimates are directly comparable; significance is not, bar for bar.
7. **SFT transfer runtime confound.** The baseline Gemma 4 31B HotpotQA run used srv3 / ollama
   0.22.0; the SFT arm used Athena / ollama 0.32.5. Identical `plain` EM (0.804 vs 0.804) bounds
   this loosely. One 300-rollout SFT-`plain` run on srv3 would settle it. This is the only open gap
   in the transfer result.
8. **9 models, not 11.** The two Gemini models were never run on HotpotQA, so this panel is
   open-weights only.

---

## 5. What is not available for HotpotQA

| Paper element | Status |
|---|---|
| §`app:effort_reallocation` — suppression (search vs. thinking) | **Not possible.** No Logfire traces were downloaded for the HotpotQA grid, and the eval rows carry no thinking-token field. `make_cue_briefing_figures.py` skips this figure for HotpotQA and prints why. |
| §Interaction Between Perturbation and Phrasing | **Not applicable.** Single phrasing; the TERSE slot is empty in every HotpotQA figure by construction. |
| §`sec:epistemic_instrumentation`, §`sec:policy_not_uncertainty` — semantic entropy under perturbation | **Pending.** The `no_search` probe (`results/hotpotqa_parametric/`, 6 models × 4 perturbations × 5 runs) is 4/6 models complete (Gemma4 31B, GPT-OSS 120B/20B, Nemotron3 30B); Nemotron-Cascade2 30B is at 4.7/5 runs and Qwen3.5 122B ~40%. **The semantic-entropy clustering run has not been done at all**, so nothing downstream of entropy — entropy-vs-correctness validation, entropy-vs-search correlation, modal-answer shift — can be plotted yet. |
| §`sec:oracle_control` — no-search value control | Not run on HotpotQA. |
| Live-web (Brave) replication | Not run on HotpotQA. |

---

## 6. Reproducing everything

```bash
cd /home/dvirla/projects/parametric_search_tradeoff

# 1. Offline grading (regenerates results/hotpotqa_cue_grid_regex/)
uv run python scripts/grade_hotpotqa_regex.py --results-root results/hotpotqa_cue_grid

# 2. Aggregate figure + tables (own FDR family, own output dir)
uv run python scripts/make_aggregate_cue_tradeoff_figure.py \
    --datasets HotpotQA --output-dir results/hotpotqa_cue_briefing

# 3. Per-model grids
uv run python scripts/make_cue_briefing_figures.py \
    --datasets HotpotQA --output-dir results/hotpotqa_cue_briefing

# 4. SFT transfer figure
uv run python scripts/make_gemma_cue_figure.py --dataset hotpotqa

# 5. Numeric transfer table (seen/unseen breakdown)
uv run python scripts/analyze_hotpotqa_transfer.py
```

`--datasets` defaults to `FRAMES,MedQA` on both briefing scripts, and that default reproduces every
existing paper figure byte-identically (verified by md5). **Run HotpotQA in its own invocation and
output directory**: the BH-FDR correction is defined over whichever panels are loaded, so folding
HotpotQA into the FRAMES/MedQA call would silently change the FRAMES/MedQA significance stars in
the paper. To deliberately put all four panels in one family:

```bash
uv run python scripts/make_aggregate_cue_tradeoff_figure.py \
    --datasets FRAMES,MedQA,HotpotQA --output-dir results/cue_briefing_3ds
```

That output already exists in `results/cue_briefing_3ds/` for comparison; see §3.2 for exactly which
stars differ between the two families (the FRAMES/MedQA figures do not).

The transfer figure shares a y-axis across its two panels by default; `--no-sharey` restores
per-panel scaling (with independent axes the SFT's near-flat bars are drawn as tall as the
baseline's −40..−80% ones, which visually erases the effect).

---

## 7. One-paragraph summary for the abstract/intro

> Adding HotpotQA as a third evaluation dataset reproduces the paper's central finding — that
> semantically irrelevant prompt perturbations move search volume by −64% to +24% against a +1%
> run-to-run floor — on a corpus where retrieval is genuinely load-bearing. It also sharpens two
> claims. First, the accuracy consequence of capability-framing (`confident`) orders cleanly by how
> substitutable parametric knowledge is for retrieval: ≈0pp on MedQA, −8.3pp on FRAMES, −21.4pp on
> HotpotQA. Second, the on-policy SFT intervention's perturbation-invariance **does** transfer out
> of domain: the Gemma 4 31B checkpoint, never trained on HotpotQA, cuts mean $|\Delta|$ search
> across trained perturbations from 26.5% (4 of 5 significant) to 2.8% (1 of 5) — ~1.6× its own
> run-to-run floor — while untrained perturbations transfer partially (42.3% → 22.6%) and
> perturbation-induced accuracy damage roughly halves, at no cost to plain accuracy. The earlier
> MedQA non-transfer is a degenerate case, not a counterexample: that baseline sits at a ~96%
> zero-search floor, leaving almost no search policy to be invariant about.
