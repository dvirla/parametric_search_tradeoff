# Resolved gemma-4 FRAMES SFT — handoff

**Status:** complete on FRAMES and HotpotQA, both arms. **MedQA was not run** (see Gaps).
**Checkpoint:** `models/gemma-4-31b-frames-resolved` (Athena) → ollama tag `gemma4-frames-resolved-q4km`.
**Regenerate every number below:** `uv run python scripts/analyze_resolved_sft.py`.
Tables here are pasted from that script's stdout; re-run it rather than trusting the paste.

---

## 1. Why this retrain exists

The previous gemma-4 FRAMES SFT (`gemma4-frames-robust-q4km`, "7-cond") achieved cue-robust
search but **could not answer at all when no tool was offered**: 75–98% of its `no_search`
responses ended mid-reasoning at a `<channel|>` marker with no answer. That made every
parametric/uncertainty measurement on it meaningless.

Three hypotheses were tested and **all failed** (do not retry): retyping mis-converted GGUF
control tokens; auditing training-data serialisation; repackaging through ollama's own
converter. Details in `[[project_gemma_sft_gguf_eog_defect]]`.

**Actual cause — a train/inference prompt mismatch.** Collection registered a real
`Tool(search_service.search)` (`create_frames_sft_data.py:362`), but `train_sft.py` called
`apply_chat_template` with **no `tools=`**, so no training prompt ever rendered a tool schema.
The system prompt is empty in all four places (collection, training, search eval, no_search
eval), which makes the training prompt **byte-identical to the no_search inference prompt** —
and in exactly that prompt form, 83% of training decision turns emitted a search call. The
model was not forgetting how to answer; it was doing precisely what it had been trained to do.
The prompt form it never saw was the *search* one (`<|tool>declaration:search{…}<tool|>`, +75
tokens, rendered by `chat_template.jinja:207`).

Note the widely-repeated claim "the training set had no no-search examples" is **false**:
17–19% of every arm was already zero-search. The missing ingredient was the tool schema, not
the examples.

## 2. The fix

Builder: **`scripts/build_resolved_frames_sft.py`** → `data/sft/frames_gemma4_resolved/`.

Three example types, distinguished by an explicit per-record `tools_available` flag. The flag
**cannot be derived** from `messages`: types B and C have identical message content and differ
only in whether the prompt renders the tool schema — that contrast is the entire fix.

| type | prompt | trajectory | source | n |
|---|---|---|---|---|
| A | tools rendered | searches | `rollouts.jsonl` | 4,083 |
| B | tools rendered | answers directly | `rollouts.jsonl` zero-search | 875 |
| C | **no tools** | answers directly | no_search probe's Logfire traces | 612 (11.0%) |
| | | | **total** | **5,570** |

Two further corrections baked into the builder:

* **Tool-absent trajectories come from the probe's own traces, not from `rollouts.jsonl`.**
  A zero-search rollout came from a run where the tool *was* offered and declined — a
  different condition from "no tool exists". Using it would pair a label and a trajectory
  measuring different events. The traces carry `thinking` and the answer as separate parts,
  which is exactly the `reasoning`+`content` split gemma-4's template wants.
* **`--clip-plain`**: the old curation left the plain anchor unfiltered while clipping every
  cue to `|search−plain_ref| ≤ 1`, so plain kept a fatter tail (21.3% of its examples at ≥6
  calls vs 5.6–16.2% for cues). That trains a residual "cue → search less" gradient — the very
  thing the SFT removes. Clipping plain too drops its p90 from 8.0 to 6.0.

Labels for type C use a parametric solve rate over the probe's runs (`--solve-rate-min 0.8`),
capped at `--absent-cap-per-cue 3` so the cues' own accuracy differences don't skew the mix.
Only *correct* trajectories are emitted, so the model is never taught a confident wrong answer.

**Trained on 7 conditions** (plain, polite, terse_plain, natural, elaborate, query, direct).
`confident_parametric`, `multiturn`, `searchmulti` were **deliberately held out** so the
10-condition evaluation measures unseen-cue generalisation.

### Code changes (all in the working tree, uncommitted)

| file | change |
|---|---|
| `scripts/build_resolved_frames_sft.py` | new — builds the dataset + `search_tool_schema.json` sidecar |
| `scripts/archive/train_sft.py` | `tools_available` in the forced Features schema; `load_tool_schema()`; `tools=` threaded through both `apply_chat_template` paths; `--drop-over-length`; `--resume-from-checkpoint` |
| `scripts/gemmify_sft_chatml.py` | preserves extra top-level keys (without this, `tools_available` is silently dropped and the whole fix vanishes) |
| `scripts/download_traces.py` | `--no-dedup` (opt-in) |
| `scripts/regrade_regex.py` | `strip_reasoning_channel()` |
| `scripts/athena_frames_gemma_sft.job` | `SEQ` 8192→16384, `--drop-over-length`, `--requeue`, merge made opt-in (`MERGE=1`), dropped `a100-public` |
| `scripts/athena_frames_parametric.job` | `RUN_FROM`/`RUN_TO` |
| `scripts/cluster_cues_llm_judge.py` | `SLUG_TO_TAG` entry for the new model |
| `scripts/analyze_resolved_sft.py` | new — regenerates all results below |

The tool schema is generated from the same `Tool(LocalIndexSearchService.search)` code path the
eval uses, **including a pydantic-ai docstring-parser wart** (`<type>A list of {"title"</type>`,
from splitting on the colon inside `{"title": str}`). Training must reproduce it verbatim or
the prompts differ.

## 3. Training

LoRA r16 `all-linear`, 3 epochs, bf16, batch 1 × grad-accum 16, `SEQ=16384`, `--drop-over-length`.
Final running loss ~0.20 (comparable to 7-cond 0.2216 / 10-cond 0.204). Peak VRAM 68 GB of 96 GB
on an RTX PRO 6000 Blackwell.

**`--drop-over-length` matters.** Truncating an agentic trace does not merely lose the tail — it
ends the trajectory after a search call with no answer turn, i.e. it *teaches the defect*. The
old `SEQ=8192` default (inherited from a MusiQue/Nemotron calibration) truncated **25%** of the
7-cond set. Dropping is the right trade, but it is not neutral: length correlates with search
count, so a low cap biases the search level down. At 16,384 it drops 364 of 5,570 (7.3%), all on
the search side — **zero** tool-absent examples are lost (they top out at 14,396 tokens).

> **Reported `train_loss = 0.05278` in `training_info.json` is an artifact, not a result.**
> The run resumed from checkpoint-700, and HF averages accumulated loss over all 978 steps while
> only 278 ran (0.20 × 278/978 ≈ 0.057). Do **not** compare it to other arms' losses.

## 4. Results — Arm 1: search (tools present)

Primary metric: **% deviation in mean search calls from the same arm's own plain condition.**
Compare arms by comparing deviations, not absolute counts — the arms sit at different plain
levels, so equal absolute drops mean different robustness.

### FRAMES (n=102 held-out test questions)

| condition | SFT sc | SFT Δ% | SFT acc | base sc | base Δ% | base acc |
|---|---|---|---|---|---|---|
| verbose_plain | 5.50 | +0.0% | 51.0% | 4.85 | +0.0% | 54.9% |
| verbose_polite | 5.25 | −4.6% | 51.0% | 4.75 | −2.2% | 52.0% |
| terse_plain | 5.82 | +5.9% | 50.0% | 5.34 | +10.1% | 51.0% |
| verbose_natural | 5.09 | −7.5% | 52.0% | 4.16 | −14.3% | 52.9% |
| verbose_elaborate | 4.85 | −11.8% | 54.9% | 3.31 | −31.7% | 52.9% |
| verbose_query | 5.76 | +4.8% | 58.8% | 5.10 | +5.1% | 54.9% |
| verbose_direct | 5.46 | −0.7% | 50.0% | 3.13 | −35.6% | 41.2% |
| *confident_parametric* | 1.87 | −66.0% | 44.1% | 0.74 | −84.8% | 41.2% |
| *multiturn* | 4.44 | −19.3% | 52.0% | 2.99 | −38.4% | 50.0% |
| *searchmulti* | 5.21 | −5.3% | 52.9% | 4.95 | +2.0% | 54.9% |

**SEEN mean|Δ| 5.9% (SFT) vs 16.5% (base) · UNSEEN 30.2% vs 41.8%**

### HotpotQA (n=300, out of domain — the SFT never trained on it)

| condition | SFT sc | SFT Δ% | SFT acc | base sc | base Δ% | base acc |
|---|---|---|---|---|---|---|
| plain | 2.17 | +0.0% | 80.0% | 2.14 | +0.0% | 81.0% |
| natural | 2.01 | −7.7% | 77.7% | 1.59 | −25.6% | 75.3% |
| elaborate | 2.06 | −5.4% | 81.3% | 1.22 | −43.1% | 74.0% |
| polite | 2.16 | −0.6% | 77.3% | 1.57 | −26.5% | 77.0% |
| direct | 2.13 | −2.1% | 71.7% | 1.35 | −36.8% | 64.3% |
| query | 2.52 | +16.1% | 82.3% | 2.15 | +0.6% | 81.3% |
| *confident_parametric* | 0.86 | −60.4% | 63.7% | 0.48 | −77.7% | 58.0% |
| *multiturn* | 1.65 | −24.2% | 76.0% | 1.31 | −38.5% | 75.3% |
| *searchmulti* | 1.14 | −47.7% | 78.7% | 1.91 | −10.8% | 78.3% |

**SEEN mean|Δ| 6.4% vs 26.5% · UNSEEN 44.1% vs 42.3%**

HotpotQA counts come from `results/hotpotqa_cue_grid_regex/per_row.csv`, which applies
`HISTORY_SEARCH_OFFSET` — `searchmulti`'s mocked history contains its own tool call and the raw
`sampler_search_calls` counter is wrong for it.

### Paired arm-vs-arm test (bootstrap over questions, 95% CI)

| dataset | cue | SFT Δ% | base Δ% | SFT − base | 95% CI | |
|---|---|---|---|---|---|---|
| FRAMES | confident_parametric | −66.0% | −84.8% | **+18.9%** | [+11.5, +26.8] | separable |
| FRAMES | multiturn | −19.3% | −38.4% | **+19.1%** | [+7.2, +30.6] | separable |
| FRAMES | searchmulti | −5.3% | +2.0% | −7.4% | [−23.6, +5.9] | not sep. |
| HotpotQA | confident_parametric | −60.4% | −77.7% | **+17.3%** | [+8.3, +26.3] | separable |
| HotpotQA | multiturn | −24.2% | −38.5% | **+14.3%** | [+6.6, +22.1] | separable |
| HotpotQA | searchmulti | −47.7% | −10.8% | **−36.9%** | [−46.9, −27.6] | separable, **wrong direction** |

**Reading for the paper.** (1) Trained cues transfer near-completely out of domain: 6.4% vs
26.5% on HotpotQA, with the baseline's large suppressions (elaborate −43%, direct −37%) reduced
to −5% and −2%. (2) Cue-induced *accuracy* damage is substantially repaired out of domain:
direct +7.4pp, elaborate +7.3pp, confident_parametric +5.7pp over baseline. (3) Of the three
unseen cues, **two improve significantly on both datasets** and one (`searchmulti`) inverts.
Do not report the unseen-cue *average* — 44.1% vs 42.3% hides both facts. `searchmulti`
prepends a mocked history in which searches already happened, and the SFT appears to read that
as the work being done; it is the only condition where the SFT is more cue-sensitive than base.

## 5. Results — Arm 2: parametric (tools absent)

5 runs pooled. `trunc%` = responses where **nothing** follows a leaked channel marker.

### FRAMES
| condition | n | trunc% | SFT acc | base acc |
|---|---|---|---|---|
| plain | 2504 | 2.76% | 28.4% | 29.7% |
| elaborate | 2505 | 0.76% | 30.3% | 30.9% |
| direct | 2505 | 0.28% | 23.2% | 23.4% |
| multiturn | 2505 | 1.24% | 27.0% | 28.4% |
| confident_parametric | 2486 | 6.44% | 28.7% | 29.3% |

### HotpotQA
| condition | n | trunc% | SFT acc | base acc |
|---|---|---|---|---|
| plain | 1500 | 0.80% | 41.5% | 41.9% |
| elaborate | 1500 | 0.20% | 44.0% | 45.1% |
| direct | 1500 | 0.20% | 34.1% | 34.9% |
| multiturn | 1500 | 1.27% | 40.5% | 40.9% |
| confident_parametric | 1500 | 3.87% | 39.4% | 42.3% |

**The defect is fixed.** Truncation 0.2–6.4% (vs 75–98% for the old arm) and accuracy within
~1pp of the base model everywhere except HotpotQA `confident_parametric` (−2.9pp).

> **Two grading rules, both load-bearing.**
> 1. Always apply `regrade_regex.strip_reasoning_channel()` before grading. Leaked reasoning
>    inflates substring matching; every row it changes was a verified false positive (gold
>    "Verizon Center" matched reasoning that concluded "The Fillmore"). It is a no-op on the
>    base model, so apply it to both arms through one code path.
> 2. **Never** use a "<3 words after the marker" truncation test. It misreads correct terse
>    `direct` answers as dead ends — of 297 FRAMES rows it flagged, 80 were correct and only 7
>    genuinely empty. Test for *nothing* after the marker.

## 6. Results — Arm 2b: semantic entropy (belief)

Same `gpt-oss:120b` judge, prompt and formula as all existing baselines, so directly comparable.

**Plain entropy is unchanged from base: FRAMES 1.034 vs 1.033 bits; HotpotQA 0.748 vs 0.756.**

| dataset | cue | SFT Δbits | base Δbits | SFT − base | 95% CI | |
|---|---|---|---|---|---|---|
| FRAMES | elaborate | −0.083 | −0.030 | −0.052 | [−0.132, +0.031] | not sep. |
| FRAMES | direct | +0.019 | −0.053 | +0.071 | [−0.008, +0.148] | not sep. |
| FRAMES | multiturn | +0.022 | −0.081 | +0.103 | [+0.019, +0.191] | sep. |
| FRAMES | confident_parametric | −0.043 | +0.026 | −0.068 | [−0.150, +0.014] | not sep. |
| HotpotQA | elaborate | −0.061 | −0.012 | −0.049 | [−0.161, +0.064] | not sep. |
| HotpotQA | direct | +0.001 | −0.042 | +0.043 | [−0.063, +0.145] | not sep. |
| HotpotQA | multiturn | −0.087 | −0.079 | −0.008 | [−0.124, +0.112] | not sep. |
| HotpotQA | confident_parametric | +0.120 | +0.061 | +0.059 | [−0.045, +0.164] | not sep. |

**Reading for the paper.** Cues move belief by ≤0.12 bits against plain levels of 0.75–1.03,
and 7 of 8 arm-differences are indistinguishable from zero. The SFT did **not** buy its search
robustness by changing what the model believes — same plain entropy, same near-null cue
sensitivity — while its search policy shifts by a separable +17–19pp on `confident_parametric`.
This is the clean policy-vs-belief dissociation, and it reproduces the "policy moves tens of
percent, belief moves hundredths of a bit" pattern in `[[project_cross_dataset_uncertainty]]`.
The single separable cell (FRAMES multiturn, +0.103) is isolated — the same cue is null on
HotpotQA (−0.008) — and is best treated as multiple-comparison noise.

## 7. Reproduction

```bash
# 1. dataset (needs the no_search Logfire traces; see Data locations)
uv run python scripts/build_resolved_frames_sft.py \
    --traces "results/parametric_traces_raw/gemma4_31b_no_search_frames_alruns/*.json" \
    --eval-json "results/frames_parametric/gemma4_31b/frames-cues_no_search_gemma4:31b_run_1.json" \
    --clip-plain --solve-rate-min 0.8 --absent-cap-per-cue 3 \
    --output-dir data/sft/frames_gemma4_resolved
uv run python scripts/gemmify_sft_chatml.py \
    --in  data/sft/frames_gemma4_resolved/procedure1_resolved_raw.jsonl \
    --out data/sft/frames_gemma4_resolved/procedure1_onpolicy_sft_rewired.jsonl

# 2. train (Athena)
sbatch --export=ALL,MODEL=gemma4:31b,OLLAMA_VER=0.32.5,NO_LOGFIRE=1,\
DATADIR=data/sft/frames_gemma4_resolved,OUTDIR=models/gemma-4-31b-frames-resolved \
  scripts/athena_frames_gemma_sft.job

# 3. merge -> quantize/register -> eval, all chained on afterok
TRAIN_JOB=<jobid> bash scripts/chain_resolved.sh

# 4. HotpotQA arms
MODELS="gemma4-frames-resolved-q4km" bash scripts/athena_submit_hotpotqa_cue_grid.sh
MODELS="gemma4-frames-resolved-q4km" CONDITIONS="plain elaborate direct multiturn confident_parametric" \
  RUNS=5 bash scripts/athena_submit_hotpotqa_parametric.sh

# 5. grade + analyse
uv run python scripts/grade_hotpotqa_regex.py --results-root results/hotpotqa_cue_grid
uv run python scripts/analyze_resolved_sft.py

# 6. entropy clustering (srv3; REPO_ROOT is required, it defaults to the Athena path)
REPO_ROOT=/data/home/dvirla/parametric_search_tradeoff OLLAMA_BASE_URL=http://127.0.0.1:11890/v1/ \
  uv run python scripts/cluster_cues_llm_judge.py --n-runs 5 \
  --only-model gemma4-frames-resolved-q4km --only-dataset frames hotpotqa --workers 8
REPO_ROOT=/data/home/dvirla/parametric_search_tradeoff \
  uv run python scripts/cluster_plain_llm_judge.py --n-runs 5 --workers 4
```

## 8. Data locations

| what | path |
|---|---|
| SFT dataset + tool schema + test ids | `data/sft/frames_gemma4_resolved/` |
| adapter / merged | `models/gemma-4-31b-frames-resolved` (Athena) · `/work/models/…_merged` |
| FRAMES search eval | `results/frames_cue_eval_resolved/gemma4-frames-resolved-q4km/` |
| FRAMES parametric + clusters | `results/frames_parametric/gemma4-frames-resolved-q4km/` |
| HotpotQA search eval (+ graded) | `results/hotpotqa_cue_grid/…` · `results/hotpotqa_cue_grid_regex/per_row.csv` |
| HotpotQA parametric + clusters | `results/hotpotqa_parametric/gemma4-frames-resolved-q4km/` |
| **no_search source traces (only copy)** | `results/parametric_traces_raw/gemma4_31b_no_search_frames_alruns/` |
| baselines | `results/frames_cues_full/gemma4_31b/`, `results/{frames,hotpotqa}_parametric/gemma4_31b/` |

`results/` and `data/` are gitignored. The trace pull is **irreplaceable** — Logfire has a hard
30-day retention wall and the runs it came from have since expired. Back it up.

## 9. Gaps and caveats

* **MedQA was not run on this checkpoint** — neither arm. Only FRAMES and HotpotQA exist. MedQA
  was superseded as a transfer venue because its baseline does zero search on 95.8% of examples
  at plain, making cue suppression unmeasurable there. If the paper needs a third dataset, this
  must be run; the claim cannot be made from existing data.
* **FRAMES plain accuracy is 51.0% vs the baseline's 54.9%** (−3.9pp). The baseline's own
  run-to-run floor is ~2.7pp, so this is near noise but is the lowest of the four arms. There
  is **no plain↔plain rerun for the resolved SFT**, so it has no measured noise floor. One
  300-rollout repeat would settle whether "zero accuracy cost" is defensible.
* **Search level is anchored high on FRAMES** (5.50 vs 4.85, +13%) though not on HotpotQA (2.17
  vs 2.14). The cue-invariance claim is about *flatness*, not about matching the baseline's
  absolute level — the same caveat as all earlier arms.
* **Truncation at `SEQ=16384` biases the search level down** (kept-example search mean 1.95 vs
  a true 2.52). Peak VRAM was only 68/96 GB, so `SEQ=24576` (128 dropped instead of 364) or
  `liger-kernel` fused cross-entropy would remove most of this. Worth doing before final numbers.
* **`confident_parametric` FRAMES rows are 486–499 of 501**, not complete: ollama returns a
  non-retryable 400 (`invalid message content type: <nil>`) on a handful of examples. Backfill
  reruns do not recover them.
* **Old-arm comparisons in `docs/frames_cue_robustness_sft.md` use absolute Δcalls**, not the
  percentage metric used here. The two are not directly comparable.
* Nothing in section 2's code-change table is committed.
