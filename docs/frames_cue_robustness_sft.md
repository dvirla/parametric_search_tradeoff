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
