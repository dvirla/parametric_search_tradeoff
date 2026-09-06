# FRAMES Cue-Robustness SFT

On-policy rejection-sampling SFT to make a model **robust to prompt cues** on FRAMES: to answer
and *search* the same way under a surface cue as it does on the plain-original question. First model:
**gpt-oss:20b**. Pipeline scripts: `scripts/create_frames_sft_data.py` (collect),
`scripts/curate_frames_sft_data.py` (curate + split), `scripts/check_sft_tokenization.py`,
`scripts/athena_frames_sft.job`.

## Stage 1 — Rollout collection

K=5 on-policy rollouts of the baseline search agent per **question × condition**, over the free
offline FRAMES BM25 index (`data/frames_index`), graded by deterministic regex (`heuristic_match`
from `scripts/regrade_regex.py`, no LLM judge). Conditions: the plain-original reference
(`verbose_plain`) plus 6 cues — `verbose_polite`, `terse_plain`, `verbose_natural`,
`verbose_elaborate`, `verbose_query`, `verbose_direct`.

- Source questions: `data/frames_cues/neutral_audited.jsonl` (501 audited FRAMES questions).
- **17,532 rollouts** (501 × 7 × ~5; 3 lost to retry-exhaustion), gpt-oss:20b.
- Started on nlp-srv3 (1,152/3,507 units), migrated mid-run to Athena L40S via
  `rsync data/sft/frames/{rollouts.jsonl,progress.json}` + `--resume` (skips done
  `example_id::condition`, no redo). Output: `data/sft/frames/rollouts.jsonl`.

Reproduce the summary: `uv run python scripts/curate_frames_sft_data.py
--rollouts data/sft/frames/rollouts.jsonl --require-correct-plain-ref --stats`.

## Rollout statistics (all 17,532)

Overall correct: **7,089 / 17,532 (40.4%)**.

| Condition | n | %correct | search median | search mean | % zero-search |
|---|---|---|---|---|---|
| verbose_plain (ref) | 2505 | 40.4% | **3** | 5.25 | 18.0% |
| verbose_polite | 2505 | 42.3% | 4 | 5.75 | 12.9% |
| terse_plain | 2504 | 39.9% | 4 | 5.61 | 15.2% |
| verbose_natural | 2505 | 39.2% | **2** | 4.09 | 26.6% |
| verbose_elaborate | 2505 | 43.5% | 3 | 5.19 | 16.7% |
| verbose_query | 2503 | 43.1% | 3 | 5.16 | 13.9% |
| **verbose_direct** | 2505 | **34.7%** | **2** | 4.21 | **31.1%** |

Search-call distribution (all rollouts): 0→19.2%, 1→13.2%, 2→12.1%, 3→9.5%, 4→6.9%, 5→5.0%,
6+→34.1% (heavy-tailed: the agent tends to either not search or search a lot).

### Key finding — the cue effect is present in the raw data
The two "commit-to-an-answer" cues shift behavior exactly as hypothesized:
- **`verbose_direct`** ("Just answer the question directly — final answer only") — **fewest searches**
  (median 2, 31.1% zero-search) **and lowest accuracy (34.7%)** vs the 40–43% band. The cue suppresses
  search and costs accuracy.
- **`verbose_natural`** ("Please answer in 2–4 sentences") — also suppresses search (median 2, 26.6%
  zero-search).
- `polite` / `terse_plain` search slightly *more* than plain (median 4).

This is the training signal: the SFT should teach the model to keep searching (and stay accurate)
under `direct`/`natural` rather than truncating.

## Stage 2 — Curation config (chosen)

`uv run python scripts/curate_frames_sft_data.py --rollouts data/sft/frames/rollouts.jsonl
--require-correct-plain-ref --threshold 1 --output-dir data/sft/frames`

- **Keep a cue rollout iff** it is correct **and** `|search_calls − plain_ref| ≤ 1`
  (`--threshold 1`), where `plain_ref` = median search_calls over that question's **correct**
  `verbose_plain` rollouts.
- **`--require-correct-plain-ref`**: drop any question with no correct plain rollout (no all-plain
  fallback). Rationale: for 43.5% of questions there is no correct plain reference; matching cue
  search behavior to an *incorrect* plain trace is not a meaningful target. Cleaner signal, modestly
  smaller set (threshold 1: 3,644 → 3,527 examples).
- Correct `verbose_plain` rollouts are always kept as a **neutral anchor** (preserve baseline
  behavior on the un-cued case).
- **20% held-out question split** (deterministic md5(example_id) % 100 < 20), for the robustness
  eval — never trained on.

### Curation result
- 501 questions → 399 train / 102 test. Of the 399 train questions, **180 dropped** (no correct plain
  ref) → **219 usable train questions**.
- **SFT set: 3,527 examples** (`data/sft/frames/procedure1_onpolicy_sft_rewired.jsonl`, ChatML with
  tool calls). Composition: verbose_plain 773 (anchor), verbose_elaborate 500, verbose_query 478,
  verbose_natural 467, verbose_polite 460, terse_plain 445, verbose_direct 404.
- Held-out: 102 test questions (`data/sft/frames/test_ids.json`).

## Stage 2.5 — Tokenizer verification & gpt-oss harmony conversion

Verified the SFT data against the gpt-oss HF tokenizer (`openai/gpt-oss-20b`,
`check_sft_tokenization.py` + manual probes). Two findings:

1. **No native assistant masking.** gpt-oss's chat template has no `{% generation %}` markers, so
   `return_assistant_tokens_mask` returns an all-zero mask. `train_sft.py` correctly falls through to
   its **prefix-retokenization** masking fallback (the Nemotron path).
2. **Our `<think>`-in-content ChatML is not harmony-compatible.** gpt-oss wants reasoning in a
   `thinking` field (→ `analysis` channel), not `<think>` tags in `content` (→ `final` channel). Fed
   as-is, the template **drops reasoning on tool-call turns** and renders final-turn reasoning as
   literal `<think>` tags in the answer. Also, harmony keeps the `analysis` channel of only the LAST
   assistant turn — intermediate CoT is ephemeral by design.

Fix: **`scripts/harmonize_sft_chatml.py`** rewrites each assistant message — `<think>…</think>` →
`thinking` field, remaining text → `content` — and **strips `thinking` from all but the final
assistant turn** (required for the prefix-mask to stay monotonic; harmony drops it anyway).

```bash
uv run python scripts/harmonize_sft_chatml.py \
    --in  data/sft/frames/procedure1_onpolicy_sft_rewired.jsonl \
    --out data/sft/frames_gptoss/procedure1_onpolicy_sft_rewired.jsonl
```

**Verified end-to-end** on the harmony file: all 3,527 final answers preserved (0 empty), 3,522 keep
final reasoning, and the prefix-retokenization mask unmasks **exactly** the assistant spans — the
search tool-calls, the final analysis, and the final answer — with **no user/tool/system leakage**
(~10% of tokens trained). So training on gpt-oss teaches the search behavior (the cue-robustness
signal) plus the final reasoning+answer; intermediate CoT is dropped per harmony design.

Training dir: **`data/sft/frames_gptoss/`** (`procedure1_onpolicy_sft_rewired.jsonl` + `test_ids.json`).

## Next steps
1. LoRA SFT: `scripts/archive/train_sft.py --model-name openai/gpt-oss-20b
   --data-dir data/sft/frames_gptoss` (fallback masking auto-triggers). On Athena, submit inside the
   apptainer container via a slurm job (`UV_CACHE_DIR=/workspace/.uv_cache`).
2. Merge, re-quantize the merged checkpoint to gpt-oss's ollama format (**MXFP4**), and eval
   cue-sensitivity on the **64 usable** held-out test questions (of the 102; those with a correct
   plain reference) vs the vanilla baseline (`run_frames_grid_experiment.sh` restricted to test ids,
   `NO_GRADER=1` + `regrade_regex.py`).

---

## RESULTS (2026-07-27) — eval + Q4 vanilla control

Training + serving done (serving path: see
[frames_gptoss_serving_attempts.md](frames_gptoss_serving_attempts.md), "RESOLVED"). Because MXFP4
quantize is impossible, the SFT model is served as **Q4_K_M / Q4_K_S**, so we also built a **Q4 vanilla
control** (un-fine-tuned base, identical recipe minus the merge) to separate quantization from
fine-tuning. All three variants evaluated on the same 102 held-out test questions × 7 conditions,
graded with `regrade_regex.heuristic_match` (runs used `NO_GRADER=1`, so the LLM-judge column is 0 —
regex is the real signal).

**Analysis code:** `scripts/make_sft_control_figure.py` → figure
`results/frames_cue_eval_test_regrade/brief_combined_sft_control.png`. Eval outputs under
`results/frames_cue_eval_test/{gpt-oss_20b, gpt-oss-vanilla-q4km, gpt-oss-frames-robust-q4km,
-q4ks}/`. **usable-64** = held-out questions with a correct plain reference in the collection rollouts,
derived to `data/sft/frames/usable_test_ids.json` (from `rollouts.jsonl`, `verbose_plain` correct).

### Search-call robustness — mean |Δ vs own PLAIN| across the 6 cues (headline)

| set | MXFP4 base | Q4 base | **Q4 base+LoRA (SFT)** |
|---|---|---|---|
| whole-102 · mean\|Δ\| / #sig-cues | 0.946 / 3 | 0.580 / 2 | **0.229 / 0** |
| usable-64 · mean\|Δ\| / #sig-cues | 0.938 / 2 | 0.570 / 2 | **0.268 / 0** |
| plain search level (calls) | ~5.5 | ~3.7 | ~3.5 |

- **Quantization alone** lowers the search level (~5.5→3.7) and partly flattens cues, but Q4 base
  still has **2 significant** cue effects.
- **Fine-tuning (Q4 base → Q4 SFT, quant held fixed)** cuts mean|Δsearch| a further ~55–60% and
  **eliminates every significant cue effect (2 → 0)**. This is the clean, attributable SFT win.

### Accuracy — the drop is quantization, NOT fine-tuning

Regex-strict accuracy, mean over 7 conditions (plain in parens):

| set | MXFP4 base | Q4 base | Q4 base+LoRA (SFT) |
|---|---|---|---|
| whole-102 | 0.468 (0.480) | 0.381 (0.333) | **0.382** (0.392) |
| usable-64 | 0.739 (0.750) | 0.596 (0.531) | **0.607** (0.609) |

- MXFP4→Q4 quantization costs ~9 pp (whole) / ~14 pp (usable).
- **SFT at matched quant costs nothing**: Q4 base → Q4 SFT is flat (0.381→0.382) / mildly up
  (0.596→0.607). The earlier "SFT dropped accuracy ~12 pp" reading was a **quantization artifact**;
  the control reattributes it entirely to MXFP4→Q4.

### Bottom line
gpt-oss:20b SFT delivered **cue-robust search behavior** (attributable to fine-tuning, not quant —
q4km eliminates all significant cue effects) **at zero accuracy cost** at matched quantization. q4km is
the stronger variant; q4ks helps less (still bends to natural/elaborate). The Q4 vanilla control was
the load-bearing run — it turned the story from "worked but ~12 pp accuracy cost" into "worked, for
free."

### Reproduce
```bash
# (models already served on Athena: gpt-oss-{vanilla,frames-robust}-q4km/q4ks + gpt-oss:20b)
uv run python scripts/regrade_regex.py --dataset frames \
    --grid-dir results/frames_cue_eval_test --output-dir results/frames_cue_eval_test_regrade
uv run python scripts/make_sft_control_figure.py   # -> brief_combined_sft_control.png
```

---

## RESULTS — gemma-4-31B (2026-08-04): second model, replicates gpt-oss

Full pipeline done for gemma-4-31B (`google/gemma-4-31B-it`): collect (17,535 rollouts, 50% correct)
→ curate (`--require-correct-plain-ref --threshold 1` → 5,105 SFT examples, 58 usable held-out of 102)
→ **gemmify** the ChatML (`scripts/gemmify_sft_chatml.py`: `<think>`-in-content → `reasoning` field so
gemma-4's canonical template renders it in the thought channel before tool_calls) → LoRA SFT
(`scripts/athena_frames_gemma_sft.job`, transformers upgraded 5.2→5.14.1 for gemma4 modeling; loss
0.22) → convert/quantize to **Q4_K_M** (`scripts/athena_gemma_merge_serve.job`) → register in ollama
**0.32.5** (gemma4 renderer, tools work with a bare `FROM`) → eval on the 102 held-out test ids
(sharded 7 ways, `scripts/athena_frames_cue_eval.job` with `CONDS=<one>` + `12h_4g` QoS).

**Key advantage over gpt-oss:** the baseline `gemma4:31b` is already a normal **Q4_K_M** quant, so
fine-tune-Q4_K_M vs baseline-Q4_K_M is directly comparable — **no MXFP4→Q4 confound, no quant control
needed.** Baseline = existing `results/frames_cues_full/gemma4_31b` (local-BM25, all cue conditions),
restricted to the 102 test ids. Both use local BM25 (comparable search-call scale).

### Curated training-data composition (paper-relevant curation statistics)

The curated JSONL (`data/sft/frames_gemma4/procedure1_onpolicy_sft_rewired.jsonl`) only stores
`{"messages": [...]}` — condition/search_calls/is_correct are dropped during curation. The table
below re-derives which raw rollouts were kept (identical selection logic to
`curate_frames_sft_data.py`, re-joined against the raw metadata) so it reflects **exactly** what
went into the training file, not an approximation.

Reproduce:
```bash
uv run python scripts/summarize_sft_curation_stats.py \
    --rollouts data/sft/frames_gemma4/rollouts.jsonl \
    --require-correct-plain-ref --threshold 1
```

Because curation keeps only rollouts that are **correct**, every kept trajectory is correct by
construction — the columns below characterize *search behavior*, not accuracy, among the correct
trajectories the model actually produced.

**Curation yield:** 25,050 raw rollouts (10 conditions, extended 2026-08-09/11 with
`verbose_confident_parametric`, `verbose_multiturn`, `verbose_searchmulti` on top of the original
7 — branch `frames-sft-history-cue-conditions` / commit `02de594`, merged to `master`) → 501
questions (399 train / 102 test) → 164 train questions dropped (no correct `verbose_plain`
reference, 41.1% of train) → 235 usable train questions → **6,905 kept rollouts (27.6% of raw)**,
up from 5,105 in the original 7-condition arm.

| condition | n | % of set | search median | search mean | contains search | zero search | fewer than plain | same as plain | more than plain |
|---|---|---|---|---|---|---|---|---|---|
| verbose_plain (anchor) | 1,037 | 15.0% | 2 | 3.45 | 90.8% | 9.2% | — | — | — |
| verbose_polite | 730 | 10.6% | 2 | 2.42 | 80.5% | 19.5% | 23.6% | 63.3% | 13.2% |
| terse_plain | 768 | 11.1% | 2 | 2.73 | 91.9% | 8.1% | 14.7% | 64.7% | 20.6% |
| verbose_natural | 673 | 9.7% | 2 | 2.24 | 78.5% | 21.5% | 28.8% | 62.9% | 8.3% |
| verbose_elaborate | 647 | 9.4% | 2 | 2.10 | 72.8% | 27.2% | 32.3% | 60.1% | 7.6% |
| verbose_query | 725 | 10.5% | 2 | 2.80 | 83.6% | 16.4% | 17.9% | 62.9% | 19.2% |
| verbose_direct | 525 | 7.6% | 2 | 2.25 | 73.7% | 26.3% | 26.1% | 61.9% | 12.0% |
| verbose_confident_parametric | 368 | 5.3% | **0** | **1.03** | **48.4%** | **51.6%** | **45.7%** | 50.3% | 4.1% |
| verbose_multiturn | 630 | 9.1% | 1 | 1.80 | 76.2% | 23.8% | 31.6% | 61.4% | 7.0% |
| verbose_searchmulti | 802 | 11.6% | 2 | 2.57 | 90.8% | 9.2% | 16.8% | 65.2% | 18.0% |

**Aggregate across the 9 cue conditions (n=5,868, excludes the plain anchor):**
- **79.6%** of kept cue trajectories still contain at least one real search call; **20.4%** are
  pure-parametric (zero search) — i.e. the majority of the training signal is "search, just keep
  doing it under the cue," not "learn to stop searching."
- Relative to each question's own correct-plain reference: **62.2%** search the exact same number
  of times, **24.8%** search fewer times, **13.0%** search more (all within the `±1`-call
  curation threshold by construction).
- For context, the plain anchor itself already contains search in 90.8% of kept rollouts — so
  "contains search" is not, by itself, evidence of cue-robustness; the closeness-to-reference
  columns are the load-bearing statistic for the paper's robustness claim, since they show the
  *training signal* enforces matching the un-cued search count rather than merely searching often.

**`verbose_confident_parametric` is a qualitatively different cue.** Every other condition shows
zero-search rates of 5–27% and "fewer than plain" rates of 15–32% among *kept* (correct + close)
rollouts. `verbose_confident_parametric` — the explicit "you already know this, no need to search"
instruction — is roughly **2x stronger on both axes** (51.6% zero-search, 45.7% fewer-than-plain)
even after the correctness+closeness filter, and its raw (pre-curation) zero-search rate is 67.5%
vs 5–33% for the others (see per-condition raw table below). It's also the only condition whose
raw accuracy (39.2%) sits below `verbose_direct`'s (41.0%), the previous low mark — consistent with
the cue sometimes convincing the model to skip search on a question it actually needed to look up.
This makes it the sharpest test case in the set for whether the SFT can teach the model to keep
searching *even when explicitly told not to*, as opposed to just resisting milder phrasing/framing
cues.

`verbose_multiturn` (unrelated chit-chat history prefix) shows a mild but real suppression effect
of its own (median 1 vs 2, 31.6% fewer-than-plain) despite carrying no explicit instruction about
search at all — the mere presence of prior conversational turns nudges search down slightly.
`verbose_searchmulti` (mocked-search history prefix), by contrast, looks almost identical to the
original un-cued conditions (median 2, 9.2% zero-search, 90.8% contains-search) — a prior *fake*
search exchange does not meaningfully suppress or inflate genuine search behavior on the real
question, consistent with the same near-null finding from the separate (non-SFT) eval-grid study of
this cue.

**Caveat for the paper:** this table reflects the training data that will be used for a
**re-trained** `gemma4-frames-robust` checkpoint — it is *not* the same data that produced the
currently-deployed `gemma4-frames-robust-q4km` checkpoint reported in the results below, which was
trained on the original 7-condition, 5,105-example set. Retraining/re-serving/re-evaluating on this
10-condition, 6,905-example set is a separate follow-up task.

### Search-call robustness — Δ vs each model's own plain (`*` = paired Wilcoxon p<0.05)

| | baseline gemma4:31b | SFT frames-robust |
|---|---|---|
| Whole-101 · mean\|Δsearch\| / #sig-cues | 0.81 / **3** (natural, elaborate, direct) | **0.46 / 0** |
| Usable-58 · mean\|Δsearch\| / #sig-cues | 0.44 / 3 | 0.36 / 2 |

→ SFT cuts cue-sensitivity ~43% and **eliminates all 3 significant cue effects** on the whole set —
including the load-bearing search-*suppression* cues (natural −1.54, direct −1.69, elaborate).

### Accuracy — the guardrail (regex-strict, mean over 7 conds)

| | baseline | SFT |
|---|---|---|
| Whole-101 | 0.519 | **0.529** |
| Usable-58 | 0.889 | **0.889** |

→ **Zero accuracy cost** (equal or slightly up). Directly visible here because both sides are Q4_K_M.

### IMPORTANT nuance — SFT-under-cue vs the ORIGINAL baseline plain

Within-model the SFT is flat, but it does **not** restore the *original baseline-plain* behavior — it
anchors to a **new, higher, uniform search level (~+1 call above baseline plain)** and flattens around
it. Every cue's `SFT_cue − baseline_plain` is positive (+0.4 to +1.7), and that residual (mean\|Δ\|
0.83 whole / 0.97 usable) is **as large as / larger than the original cue effect** (0.81 / 0.44). So
the recipe delivers cue-*invariance*, not restoration-to-plain. Cross-model: gpt-oss anchored *lower*
(~3.7 vs 5.5, partly quant), gemma-4 anchored *higher* (~5.9 vs 4.8, pure fine-tuning). Accuracy under
cues ≈ baseline plain either way.

### Two-model conclusion
On-policy rejection-sampling SFT makes search behavior **robust to prompt cues at no accuracy cost**,
replicated across gpt-oss:20b and gemma-4-31B. The achievement is cue-*consistency* (flat across cues);
the absolute search level shifts from the original baseline and is not guaranteed to match plain.

Reproduce: `scripts/make_gemma_cue_figure.py` (below) + the inline analysis in this session.

---

## TRANSFER — does gemma-4 FRAMES-SFT cue-robustness carry to MedQA? (2026-08-04, **REVISED 2026-09-02**)

Tested the gemma-4 SFT (trained ONLY on FRAMES cues) on the MedQA cue grid — a clean out-of-domain
transfer test (never trained on MedQA, all 500 questions valid, no held-out concept). Conditions mirror
FRAMES: orig_plain (ref) + orig_{polite,natural,elaborate,query,direct} + terse_plain. Both models
Q4_K_M + local MedQA BM25. Baseline = `results/medqa_grid/gemma4_31b`, SFT =
`results/medqa_grid/gemma4-frames-robust-q4km_latest`. Figure:
`scripts/make_medqa_transfer_figure.py` → `results/medqa_regex_regrade/medqa_cue_transfer.png`
(search axis in ABSOLUTE calls — baseline plain ~0.1 makes %-of-plain explode and mislead). Only the
7-condition arm was ever run on MedQA; the 10-cond and 8-cond arms have no MedQA data.

| | baseline gemma4:31b | SFT frames-robust |
|---|---|---|
| plain search | 0.09 calls | 2.35 calls |
| **zero-search at plain** | **95.8% of examples** | 5.8% |
| mean\|Δsearch\| (calls) / #sig cues | 0.06 / 4 | 0.33 / 6 |
| mean\|Δsearch\| as % of own plain | **71.2%** | **13.9%** |
| mean\|rank-biserial r\| (scale-free) | **0.659** | **0.328** |
| plain regex accuracy | 0.438 | 0.436 |

**What holds:** search *propensity* transfers out of domain — the SFT searches ~2.35 calls on MedQA
plain vs the baseline's ~0.09 (26×) — at no accuracy cost and no accuracy gain (0.438 → 0.436).

**REVISED verdict — the original "cue-invariance does NOT transfer" was a metric artifact.** That
conclusion rested on two confounded statistics:

1. **Absolute Δcalls and #sig cues are not comparable across these two arms.** The MedQA baseline does
   **zero search on 95.8% of examples at plain**. Cue-*suppression* is unmeasurable on a model already
   at the floor — you cannot suppress below zero — so its mean\|Δ\| of 0.06 calls measures the floor,
   not invariance. Its effects are nevertheless real, not noise: a plain↔plain rerun
   (`results/medqa_grid_rerun/gemma4_31b`) moves baseline search by only **+0.014 calls (p=0.74, ns)**,
   ~4× smaller than the smallest cue effect. And `#sig` is power-, not effect-, driven: at n=500 the
   SFT's larger search variance turns a −5.6% shift (TERSE) significant while the baseline's −93%
   ELABORATE shift rides on 21 examples.
2. **On a scale-free metric the ranking inverts.** Mean \|matched-pairs rank-biserial r\| over the 6
   cues is **uncorrelated with a model's search level** across the 10 non-SFT MedQA arms (Spearman
   −0.055, p=0.88), whereas mean\|Δcalls\| is perfectly rank-correlated with it (Spearman +1.000).
   On that metric the **baseline is the most cue-sensitive of all 12 MedQA arms (r=0.659)** and the SFT
   is less than half as sensitive (0.328).

**Valid comparison 1 — against MedQA arms that actually search** (zero-search@plain < 15%, the only
arms where suppression is measurable at all):

| arm | plain calls | mean\|Δ\|/plain | mean\|r\| |
|---|---|---|---|
| gemini-3.5-flash | 10.47 | 23.1% | 0.554 |
| nemotron-cascade-2_30b | 5.71 | 26.8% | 0.459 |
| gemini-3.1-pro-preview | 3.49 | 25.1% | 0.379 |
| qwen3.5_122b (n=84) | 1.83 | 38.3% | 0.552 |
| **SFT frames-robust** | **2.35** | **13.9%** | **0.328** |

The SFT is the flattest of the set on both metrics — so its residual MedQA cue sensitivity is *not*
just the generic "models that search have something to modulate" effect.

**Valid comparison 2 — the SFT against itself, in vs out of domain** (same metric, matched ids):

| | mean\|Δ\|/plain | mean\|r\| | #sig/6 |
|---|---|---|---|
| SFT on FRAMES (102 held-out) | 7.5% | 0.150 | 0 |
| SFT on MedQA (500) | 13.9% | 0.328 | 6 |
| *baseline* on FRAMES (102, matched) | 16.5% | 0.333 | 3 |

**Roughly half the robustness transfers.** Out of domain the SFT's cue sensitivity regresses to almost
exactly the level an *untrained* model shows *in* domain (0.328 vs 0.333) — better than the MedQA
baseline on any scale-free reading, materially worse than its own in-domain flatness.

The residual MedQA effects are the canonical suppression signature, concentrated in the two cues the
SFT fully defeated on FRAMES: ELABORATE −25.0% (r=.599), SHORT −19.4% (r=.488), then QUERY +15.8%,
DIRECT −9.5%, POLITE −8.2%, TERSE −5.6%. DIRECT is also the only cue that moves accuracy for either
arm (SFT −7.2pp, baseline −10.6pp, both p<.001).

**Caveat on what the transferred behavior is worth:** search buys nothing on MedQA. 26× more search
leaves accuracy flat (0.438→0.436), and within the SFT accuracy *falls* with search volume (0.574 at
1 call, 0.358 at 2–3, 0.347 at 4+ — confounded by question difficulty, but there is no positive
signal). So MedQA cue-robustness here is robustness of a behavior that does not pay off on the task.

**Open gaps:** (1) no plain↔plain rerun for the SFT on MedQA, so the SFT has no measured noise floor
— every SFT p-value above is uncorrected for run-to-run variance, which for the baseline was ns but
could be larger at 2.35 calls; (2) the 10-cond arm was never evaluated on MedQA; (3) accuracy is
regex-graded on free-form answers to a multiple-choice task.

Reproduce: `uv run python scripts/analyze_medqa_cue_transfer.py` (the full revised analysis —
rank-biserial r, zero-search share, plain↔plain rerun, cross-arm table, in-vs-out-of-domain, and the
search-vs-accuracy breakdown). The original absolute-calls figure is unchanged at
`scripts/make_medqa_transfer_figure.py`; it plots only the confounded metric, so it should not be
used on its own to support a transfer claim.

---

## TRANSFER — HotpotQA (2026-09-06): the out-of-domain result the paper should lead with

MedQA could not carry a transfer claim (95.8% zero-search floor, search accuracy-inert; see the
revised section above). HotpotQA can: the gemma4:31b baseline searches **2.14 calls** at plain with
only **6.0% zero-search**, and search is load-bearing — cue-induced suppression costs up to 23 pp of
accuracy. Both arms: `hotpotqa-300` (type-stratified, nested), local BM25 over `data/hotpotqa_index`,
9 conditions, `--no_grader` + offline regex grading (`scripts/grade_hotpotqa_regex.py`).
Baseline = `results/hotpotqa_cue_grid/gemma4_31b` (srv3, ollama 0.22.0);
SFT = `results/hotpotqa_cue_grid/gemma4-frames-robust-q4km_latest` (Athena 0.32.5, jobs 141147-155),
the **7-condition** FRAMES SFT, never trained on HotpotQA.

**The 2x2.** The 7-cond SFT trained on polite/terse/natural/elaborate/query/direct. Of HotpotQA's 8
cues, **5 are SEEN** (natural, elaborate, polite, direct, query) and **3 are UNSEEN**
(confident_parametric, multiturn, searchmulti). One run therefore measures both generalization axes:
new dataset, and cues never trained on.

| Δ search vs own plain | baseline gemma4:31b | SFT frames-robust |
|---|---|---|
| plain search level | 2.14 calls | 2.41 calls |
| zero-search at plain | 6.0% | 0.0% |
| **SEEN cues** — mean\|Δ\| / #sig | 0.57 calls (26.5%) / **4 of 5** | **0.07 calls (2.8%) / 1 of 5** |
| **UNSEEN cues** — mean\|Δ\| / #sig | 0.90 calls (42.3%) / 3 of 3 | 0.54 calls (22.6%) / 3 of 3 |
| all 8 cues | 0.69 calls (32.4%) / 7 of 8 | 0.25 calls (10.2%) / 4 of 8 |
| **run-to-run floor** (plain vs plain_rep2) | **−0.03 calls (−1.4%), p=0.75 ns** | **+0.04 calls (+1.8%), p=0.63 ns** |
| plain accuracy (strict regex) | 0.810 | 0.807 |

**Headline 1 — robustness to TRAINED cues transfers to a new dataset almost completely.** On the five
seen cues the mean effect collapses from 26.5% to **2.8% of plain**, against a measured run-to-run
floor of **1.8% measured on the SFT itself** (job 142221). The seen-cue effect is therefore 1.6x the
model's own run-to-run noise: trained cues become nearly indistinguishable from re-running the
identical prompt. Only `natural` remains significant (−3.5%, p=.015), ~2x the floor. Per-cue:
ELABORATE −43.1%\*\*\* → **−0.1% ns**, DIRECT −36.8%\*\*\* → −3.0% ns, POLITE −26.5%\*\*\* →
+2.8% ns, NATURAL −25.6%\*\*\* → −3.5%\*, QUERY +0.6% ns → +4.6% ns.

**Headline 2 — robustness to UNSEEN cues transfers only partially.** confident_parametric −77.7% →
−41.3%, multiturn −38.5% → −17.6%, searchmulti −10.8% → −9.0%: roughly halved, all still significant.
The cue-suppression ordering is preserved, so what the SFT learned is not a generic "always search
k times" reflex — it is specific to the cue family it saw.

**Headline 3 — the accuracy damage from cues is roughly halved, at matched response length.** This is
new relative to FRAMES, where accuracy was merely flat. Median response words are near-identical
between the two arms in every condition (elaborate 203 vs 209, direct 2 vs 2, plain 29 vs 33), so the
grader's known verbosity bias does not drive the *between-arm* contrast. Read accuracy deltas
against each arm's accuracy floor: baseline −2.7 pp (p=.077), SFT −0.3 pp (p=1.0).

| strict regex accuracy | baseline | SFT | median words (base / SFT) |
|---|---|---|---|
| plain | 0.810 | 0.807 | 33 / 29 |
| confident_parametric | 0.580 | **0.717** | 29 / 43 |
| direct | 0.643 | **0.737** | 2 / 2 |
| elaborate | 0.740 | **0.857** | 209 / 203 |
| multiturn | 0.753 | 0.813 | 39 / 31 |
| natural | 0.753 | 0.820 | 39 / 38 |
| polite | 0.770 | 0.817 | 38.5 / 31 |

Baseline loses 16.7 pp under DIRECT and 23.0 pp under confident_parametric; the SFT loses 7.0 and
9.0. Read accuracy deltas against the accuracy floor: plain vs plain_rep2 is −2.7 pp (p=.077), so
differences below ~3 pp are run noise.

**Caveats.** (1) The baseline ran on srv3/ollama 0.22.0 and the SFT on Athena/0.32.5, so a runtime
difference rides along with the fine-tuning; the near-identical plain levels (2.14 vs 2.41) and
identical plain accuracy bound it loosely, but a 300-rollout SFT-plain run on srv3 would settle it.
(2) RESOLVED 2026-09-06 — the SFT's own `plain_rep2` now exists (+1.8% search, p=0.63; −0.3 pp
accuracy, p=1.0), so its "1 of 5 significant" is tested against its own null, not a proxy. The SFT is
also more run-to-run stable in accuracy than the baseline (−0.3 pp vs −2.7 pp). (3) Grading is regex/EM; an LLM
judge is the fix, and cross-*condition* accuracy comparisons remain verbosity-confounded even though
the between-arm ones are not. (4) searchmulti counts are corrected for the mocked history's own tool
call (`HISTORY_SEARCH_OFFSET`, commit f0e71ec) — uncorrected rows show a spurious increase.

Reproduce: `uv run python scripts/grade_hotpotqa_regex.py --results-root results/hotpotqa_cue_grid`,
then the transfer analysis over `results/hotpotqa_cue_grid_regex/per_row.csv`.

---

## RESULTS — gemma-4-31B 10-condition arm (2026-08-13)

Retrained gemma-4-31B on the extended 10-condition dataset (6,905 examples — the original 7 cues
plus `verbose_confident_parametric`, `verbose_multiturn`, `verbose_searchmulti`; see "Curated
training-data composition" above), registered as **`gemma4-frames-robust-10cond-q4km`**, alongside
(not replacing) the original 7-condition `gemma4-frames-robust-q4km`. Same recipe, hyperparameters,
and Q4_K_M quantization as the 7-condition arm — only the training data differs. Final training loss
**0.204** (vs 0.22 for the 7-condition arm). Evaluated all three arms (baseline, SFT 7-cond, SFT
10-cond) on the **full** 102 held-out test questions × 9 cues + a PLAIN↔PLAIN noise-floor rerun —
**n=102/102 for every condition of every arm**, no examples dropped.

**Data-completeness note:** a client/ollama `invalid message content type: <nil>` edge case
(triggered after a tool-call round-trip, more likely on longer/multi-hop questions needing more
search steps) intermittently drops 1-2 examples per condition on both SFT checkpoints — not observed
on baseline. It is stochastic, not a deterministic per-example failure: the built-in retry loop in
`run_frames_grid_experiment.sh` (`MAX_PASSES=4`, using `--resume` to reattempt only missing rows)
resolves most instances automatically; the handful that survived 4 automatic passes (2 for the
10-cond arm, 3 for the 7-cond arm, all traced to the same couple of multi-hop questions — e.g. example
745, which needs multiple director-birth-year lookups) succeeded on a subsequent manual resubmission
with the same `--resume` mechanism. **Earlier draft of this section used an accidentally-reduced
n=99** (intersecting all three models' *first-pass* results, which silently pulled the baseline and
7-cond arms' own numbers away from their independently-complete 102-example values and flipped
QUERY/TERSE significance for the 7-cond arm) — the table below supersedes it.

Reproduce: `uv run python scripts/make_gemma_cue_figure_10cond.py` →
`results/frames_cue_eval_test_regrade/gemma_cue_robustness_10cond_compare.png`.

### Search-call robustness — Δ vs each model's own plain (mean|Δsearch| in calls, #significant cues / 9)

| | baseline gemma4:31b | SFT 7-cond | SFT 10-cond |
|---|---|---|---|
| mean\|Δsearch\| (calls) / #sig cues | 1.30 / **6** | 0.87 / **2** | **0.72 / 2** |
| plain search level (calls) | 4.85 | 6.08 | 5.58 |
| plain accuracy | 0.549 | 0.539 | 0.520 |

Both SFT arms land at the **same** significant-cue count (2/9) — the 10-condition arm's advantage is
a smaller mean effect size, not a different set of resolved cues. Accuracy at plain is within noise
across all three (0.52–0.55).

### Per-cue detail (search Δ%, `*/**/***` = Wilcoxon p<.05/.01/.001)

| cue | baseline | SFT 7-cond | SFT 10-cond |
|---|---|---|---|
| POLITE | −2.2% | −6.5% | −4.7% |
| TERSE (PLAIN) | +10.1% | +0.5% | +0.4% |
| SHORT | −14.3%* | −8.2% | −7.2% |
| ELABORATE | −31.7%*** | −13.1% | −3.5% |
| QUERY | +5.1% | +3.9% | −1.4% |
| DIRECT | −35.6%*** | −12.9% | −4.7% |
| MULTITURN | −38.4%*** | −20.3%** | −19.2%** |
| SEARCHMULTI | −18.6%** | −11.3% | −9.7% |
| NO-SEARCH-NEEDED | −84.8%*** | −52.3%*** | **−65.4%*** |

Both SFT arms fully resolve SEARCHMULTI, SHORT, ELABORATE, DIRECT, POLITE, TERSE, and QUERY to
non-significance (all were significant or borderline at baseline for SHORT/ELABORATE/DIRECT). The
**same two cues resist both arms**: MULTITURN and NO-SEARCH-NEEDED.

**MULTITURN** is essentially flat between the two SFT arms (−20.3%**→−19.2%**, still significant in
both) despite being in the 10-condition training data (630 kept examples, 9.1% of the curated set) —
direct training exposure bought negligible additional robustness here.

**NO-SEARCH-NEEDED is the standout exception, and it goes the wrong way.** Despite
`verbose_confident_parametric` being directly in the 10-condition training data (368 kept examples,
5.3% of the set), search suppression under this cue got **worse**, not better (−52.3%*** →
−65.4%***, still the single largest and most significant effect in either SFT arm). Likely
explanation, tying back to the curation statistics above: this cue's raw rollouts were 67.5%
zero-search-and-correct — far higher than any other condition — so the correctness+closeness
curation filter disproportionately kept zero-search-correct examples for this cue specifically
(51.6% zero-search even after filtering, vs 5–27% for every other cue). The training signal the model
actually received for this condition was predominantly "under this instruction, answer without
searching," which is what a plain SFT objective will learn — the *closeness-to-plain* selection
criterion doesn't override that when the zero-search pool for a given cue is this dominant. Directly
training on a cue does not automatically buy invariance to it if the correct+curated examples for
that cue are not actually close to plain behavior on average; the curation filter's threshold
controls per-kept-example closeness, not the *mix* of conditions each cue contributes.

### Three-arm conclusion

Folding confident_parametric/multiturn/searchmulti into the SFT data shrinks the *magnitude* of
already-non-significant cue effects (mean|Δsearch| 0.87→0.72 calls) but does **not** shrink the *set*
of cues that remain significant — MULTITURN and NO-SEARCH-NEEDED resist both arms, and
NO-SEARCH-NEEDED specifically gets worse with direct training exposure. This is a more modest
(and more accurate) result than "10-condition training fixes what 7-condition training missed" — the
honest read is that on-policy rejection-sampling SFT resolves the *milder* phrasing/framing cues
robustly, but the two cues with the strongest raw effect (an explicit history-based distraction and an
explicit "don't search" instruction) need a different curation strategy, not just more data from the
same recipe. A follow-up worth considering: down-weighting or excluding the zero-search-correct
examples specifically for `verbose_confident_parametric` during curation (or raising its
closeness-to-plain bar independent of the shared `--threshold`), so training data for that cue better
represents "search the same as plain," not "correctly recall without searching."

---

## RESULTS — gemma-4-31B 8-condition arm, isolating confident_parametric (2026-08-14)

Direct test of the hypothesis above: is `verbose_confident_parametric`'s regression (7-cond → 10-cond)
intrinsic to its own training data, or an interaction effect with `verbose_multiturn`/
`verbose_searchmulti` being trained at the same time? Curated an **8-condition** set — the original 7
cues plus `verbose_confident_parametric` **only**, excluding multiturn/searchmulti — from the same
superset `rollouts.jsonl` already collected for the 10-condition arm (new `--conditions` filter on
`curate_frames_sft_data.py`). **5,473 examples** (the 7-condition arm's 5,105 + confident_parametric's
368 — identical composition to what confident_parametric contributed in the 10-condition set). Same
recipe/hyperparameters/quantization throughout. Final training loss **0.228**. Registered as
`gemma4-frames-robust-8cond-q4km`, alongside (not replacing) the other two checkpoints. Evaluated on
the same full 102/102 test set × 9 cues + PLAIN↔PLAIN rerun.

Reproduce: `uv run python scripts/make_gemma_cue_figure_10cond.py` (now a 4-panel figure) →
`results/frames_cue_eval_test_regrade/gemma_cue_robustness_10cond_compare.png`.

### NO-SEARCH-NEEDED search Δ% across all four arms (the cue under test)

| baseline | SFT 7-cond (no exposure) | **SFT 8-cond (isolated exposure)** | SFT 10-cond (exposure + multiturn/searchmulti) |
|---|---|---|---|
| −84.8%*** | −52.3%*** | **−55.3%*** | −65.4%*** |

**Result: training on confident_parametric in isolation does not help — it's flat-to-slightly-worse
than not training on it at all** (−55.3% vs −52.3%, both p<.0001, a 3pp gap well within normal
run-to-run LoRA variance). This confirms the hypothesis from the 10-condition writeup: the
regression isn't a multiturn/searchmulti interaction effect, it's intrinsic to confident_parametric's
own curated training data (67.5% raw zero-search-and-correct → 51.6% zero-search even after the
closeness filter — see "Curated training-data composition" above). Adding more of this cue's data
without changing the curation criterion doesn't buy invariance to it; it just adds more of the same
skewed signal.

**A second, less expected finding: the 10-condition arm's confident_parametric result (−65.4%) is
worse than BOTH the 7-cond and 8-cond arms**, despite 8-cond and 10-cond sharing the *exact same* 368
confident_parametric training examples. Since the only difference between 8-cond and 10-cond is the
presence of multiturn/searchmulti training data, this points to a real negative-transfer interaction
— training on multiple search-suppression-flavored cues together seems to compound into a stronger
general "trust your own judgment over searching" tendency than any one cue teaches alone. This is a
hypothesis, not yet isolated further (would need e.g. a 9-condition arm with multiturn+searchmulti but
NOT confident_parametric to test it directly).

### Other cues (context, not the primary test)

MULTITURN and SEARCHMULTI also shift somewhat between 7-cond and 8-cond (MULTITURN −20.3%→−16.2%,
SEARCHMULTI −11.3%(n.s.)→−15.7%**) despite **neither arm training on either condition** — this is
run-to-run training variance (different LoRA stochastic dynamics), not a systematic effect of adding
confident_parametric data, and is a useful reminder not to over-read every individual
significance-star flip in these tables; the NO-SEARCH-NEEDED result above is the one with a clean,
controlled, repeated (7-cond vs 8-cond vs 10-cond, same underlying data subset each time) comparison
behind it.
