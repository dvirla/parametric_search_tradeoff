---
name: project_frames_cue_robustness_sft
description: "Cue-robustness SFT on FRAMES — DONE for BOTH gpt-oss:20b AND gemma-4-31B: on-policy rejection-sampling SFT gives cue-robust (cue-invariant) search at zero accuracy cost; caveat: anchors to a shifted absolute search level, not the original baseline plain"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3953a199-64f7-4231-bfd8-fe897e862da5
  modified: 2026-08-07T03:16:07.225Z
---

On-policy rejection-sampling SFT to make models robust to FRAMES **cues** (the cue conditions that
shift `sampler_search_calls` / final-response text). Start with **gpt-oss:20b**, then **gemma4:31b**.
Approved plan: `~/.claude/plans/let-s-plan-running-supervised-peppy-cocoa.md`.

Design: generate K=5 rollouts per question per condition over the **6 templates**
(polite, terse_plain, natural/short, elaborate, query, direct) + **plain original** (verbose_plain)
reference, on the free local BM25 index (`data/frames_index`, `LocalIndexSearchService`). Keep
rollouts that are correct AND whose search-call count is close to the same question's plain-original
reference. Hold out ~20% of questions (question-level split) for the robustness eval; include correct
plain rollouts as a neutral anchor. Full audited 501 set (`data/frames_cues/neutral_audited.jsonl`).

New scripts: `scripts/create_frames_sft_data.py` (collector, modeled on archived
`scripts/archive/create_musique_sft_data.py`), `scripts/curate_frames_sft_data.py` (curation+split).
Reuse `scripts/archive/train_sft.py` (LoRA+merge; curated file must be named
`procedure1_onpolicy_sft_rewired.jsonl` to match its `_ARM_FILES`). Eval via
`run_frames_grid_experiment.sh` (restricted to test ids) + `summarize_frames_cues_grid.py`.

**Two locked conventions (user corrections):**
1. **Grade with deterministic regex, NOT LLM judge** — use `heuristic_match(gold, response)` from
   `scripts/regrade_regex.py` (strict SQuAD substring; `relaxed_match` = word-subset fallback). Free,
   offline, matches existing FRAMES regrade workflow. Eval runs grid with `NO_GRADER=1` then regrades.
2. **Quant-match before eval** — train from HF base weights, but re-quantize the merged checkpoint to
   the **exact GGUF format the original ollama model uses** (check via `ollama show --modelfile`;
   gpt-oss = MXFP4, not q8_0) before serving, or the robustness comparison is confounded.

**MOVED to Athena 2026-07-24:** after 1152/3507 units (5759 rollouts) on nlp-srv3, the collection was
stopped and **migrated to Athena** (~2.4x faster: ~105 vs ~44 units/hr). Migration recipe that worked:
stop collector → verify every rollouts.jsonl line parses → rsync `data/sft/frames/{rollouts.jsonl,
progress.json}` to the target → sbatch the job with the SAME `--output-dir` + `--resume`. Because
`--resume` keys on `example_id::condition` from progress.json, the target skips done units exactly
("2355 units to run (1152 already done)") and appends to the same file — no split/merge needed, no
questions redone. Athena job 125453. NOTE: `pgrep -f create_frames_sft_data` self-matches the ssh
command line — use `pgrep -f "[c]reate_frames_sft_data"` or the process count will look ALIVE forever.

**Run status (2026-07-23):** rollout collection was first run on **nlp-srv3** (`ssh nlp-srv3`, repo
`/data/home/dvirla/parametric_search_tradeoff`), detached via nohup → `frames_sft_full.log`,
gpt-oss:20b, full 501×7 conditions×K=5 = 3507 units (~17.5k rollouts), output
`data/sft/frames/rollouts.jsonl`, `--resume`-safe. ~2.5 rollouts/min (OLLAMA_NUM_PARALLEL=2) →
~4–5 days. nlp-srv3 gotcha: Ollama's GPU discovery times out and crashes the runner when its visible
GPUs (CUDA_VISIBLE_DEVICES=0,1,2) are all saturated by other jobs — not a model/code bug; wait for a
free GPU. Athena L40S is ~3.5x faster (NUM_PARALLEL=6) if we later parallelize. Job files:
`scripts/athena_frames_sft.job` (self-contained ollama on a dedicated GPU; sets LOGFIRE + UV_CACHE_DIR),
`scripts/athena_frames_setup.job` (venv repair), `scripts/athena_viability.job`. See Athena
container-eval memory for the venv-wipe pitfall hit during this work.

**Git (2026-07-23):** pipeline committed on branch **`frames-cue-robustness-sft`** (commit 0e39198,
off master), pushed to origin (github.com/dvirla/parametric_search_tradeoff). Both nlp-srv3 and Athena
were `git checkout -f`'d onto it (they were on sensitivity_to_prompt_cues, a few commits behind master
with identical tree; only run_frames_grid_experiment.sh was locally modified = same change now
committed, so nothing lost). Scripts previously reached the remotes via scp (untracked); now tracked.
Left untouched for the user: `scripts/compare_local_vs_brave_cues.py` (untracked, not mine),
`frames_brave_gemini.log`.

**Stage 1 DONE + Stage 2 curated (2026-07-26):** collection complete — 17,532 rollouts (501 Q × 7
cond × ~5), 40.4% correct. **Headline finding (raw data confirms the premise):** verbose_direct
("answer directly, final answer only") suppresses search most (median 2, 31% zero-search) AND drops
accuracy to 34.7% vs the 40–43% band; verbose_natural also suppresses (median 2). Full analysis:
**docs/frames_cue_robustness_sft.md**. Chosen curation: `--require-correct-plain-ref --threshold 1`
(drop the 43.5% of questions with no correct plain rollout rather than use an all-plain fallback ref;
keep cue rollout iff correct AND |search−plain_median|≤1) → **3,527 SFT examples** from 219 usable
train questions, 102 held-out test questions (data/sft/frames/{procedure1_onpolicy_sft_rewired.jsonl,
test_ids.json}; test_ids committed). Analysis was run LOCALLY after `rsync`-ing rollouts.jsonl down —
Athena's venv python only works inside the apptainer container via a slurm job (login-node python is
pre-3.10; the script now has `from __future__ import annotations`). **Next: check_sft_tokenization.py
vs gpt-oss HF tokenizer (harmony/channel template + assistant-mask) BEFORE training.**

**gpt-oss harmony gotcha (2026-07-26):** the rollout ChatML stores reasoning Qwen-style as
`<think>...</think>` in assistant `content`. gpt-oss's chat template is INCOMPATIBLE with this — it
wants reasoning in a **`thinking`** field (→ analysis channel; `reasoning`/`<think>`-in-content do
NOT work), renders `content` as the final channel, DROPS reasoning on tool-call turns, and keeps the
analysis channel of only the LAST assistant turn (intermediate CoT is ephemeral by design). Also
gpt-oss has NO `{% generation %}` markers, so native assistant-masking is all-zero → train_sft.py
uses its prefix-retokenization fallback. Fix committed: **scripts/harmonize_sft_chatml.py** (extract
`<think>`→`thinking`, answer→content, STRIP thinking from all but the final turn — required so the
prefix-mask stays monotonic). Verified: unmasks exactly search tool-calls + final analysis + answer,
no leakage. gpt-oss training data at **data/sft/frames_gptoss/** (3527 examples). Commit 0458940.
Next: LoRA SFT via train_sft.py --model-name openai/gpt-oss-20b --data-dir data/sft/frames_gptoss
(on Athena: apptainer slurm job, UV_CACHE_DIR=/workspace/.uv_cache); then merge → MXFP4 quant → eval
on the 64 usable held-out test questions.

**SERVING RESOLVED + FULL RESULTS (2026-07-27):** the 3 failed routes (Ollama runtime LoRA; transformers
bf16 save `revert_weight_conversion`; vLLM cu13 vs n315 driver 12.8) were bypassed by the **winning
path**: merge LoRA into `unsloth/gpt-oss-20b-BF16` (clean bf16 base, NO quantizer → no revert crash) →
`convert_hf_to_gguf --outtype auto` → **`llama-quantize` to Q4_K_M/Q4_K_S** (MXFP4 is NOT a valid
quantize target: `invalid ftype 'MXFP4'` in both ollama and llama-quantize) → register in ollama with
the base `gpt-oss:20b` TEMPLATE (`ollama show --modelfile`; bare `FROM` gives "does not support tools")
+ `PARAMETER stop "<|call|>"`/`"<|return|>"` (converted GGUF lacks harmony EOG → malformed tool calls).
Full detail: **docs/frames_gptoss_serving_attempts.md** ("RESOLVED" + "Bugs solved 2026-07-27").

Because Q4≠baseline MXFP4, built a **Q4 vanilla control** (`athena_gptoss_vanilla_q4.job`, un-fine-tuned
base, same recipe minus merge). Evaluated MXFP4 base + Q4 base + Q4 SFT (q4km/q4ks) on 102 held-out Q ×
7 cues; regex-graded (`NO_GRADER=1`→regex is the signal). **usable-64** = held-out Q with a correct
plain reference → `data/sft/frames/usable_test_ids.json`. Figure:
`scripts/make_sft_control_figure.py` → `results/frames_cue_eval_test_regrade/brief_combined_sft_control.png`.

**HEADLINE (both hold on whole-102 and usable-64):** (1) SFT achieves **cue-robust search** — mean
|Δsearch vs own plain| across 6 cues collapses 0.94→0.27 (usable) / 0.95→0.23 (whole), and #significant
cue effects go 2–3 → **0** for q4km (q4ks weaker). (2) The apparent ~12pp accuracy drop is **entirely
MXFP4→Q4 quantization, NOT fine-tuning** — Q4 base→Q4 SFT is flat/mildly up (0.596→0.607 usable). SFT =
robustness gain at **zero accuracy cost** at matched quant. Quantization itself also lowers search level
(~5.5→3.7) and partly flattens cues, but leaves 2 sig effects; SFT removes the rest.

**Deployment/eval bugs fixed this session (see serving doc):** ollama **port collision** on same node
(host networking, both on 11434) → unique `PORT=11500+SLURM_JOB_ID%4000`; **.venv torch broken** by vLLM
experiments (libcudnn/libcusparseLt) → `uv sync --reinstall`; **logfire wedge** (missing certifi → OTLP
exporter hangs workers) → disable via `LOGFIRE_API_KEY=''`, re-enabled after venv restore; **incomplete
baseline** (missing verbose_direct + half verbose_query) → same job `--resume` fills gaps; **Q4-vanilla
convert** failed (base HF snapshot has NO tokenizer) → stage base weights + tokenizer copied from merged
dir; `set -e` missed it (pipe-to-grep) → explicit `[ -f $GGUF ]` gates; **disk quota** `OSError: 0
written` = personal /home quota 300G soft/330G hard (check `quota -s`, not `df`); `/work`→`/rg/reichart_prj`
group share; freed by deleting reproducible BF16 intermediates (regenerable from the 769M adapter).

**GEMMA4:31B collection — MOVED nlp-srv3 → ATHENA (2026-07-28):** 2nd-model full comparison. Started on
nlp-srv3 (reached 125/3507 units, ~1.67 rollouts/min via the shared ollama pid 6013 with
`OLLAMA_NUM_PARALLEL=2` → ~7 days; NOT actually stalled, just slow), then **transferred to Athena** for a
dedicated GPU. Transfer recipe: rsync `data/sft/frames_gemma4/{rollouts.jsonl,progress.json}`
nlp-srv3→local→Athena, stop nlp-srv3 collector, launch Athena job with `--resume` (skips the 125 done).
Athena job (126782): `sbatch --export=ALL,MODEL=gemma4:31b,OLLAMA_VER=0.32.5,NO_LOGFIRE=1,OUTDIR=data/sft/frames_gemma4
scripts/athena_frames_sft.job` — validated clean start (0.32.5 binary, logfire disabled, gemma4 warmup
OK, collecting with 6 workers). Output `data/sft/frames_gemma4/` on Athena repo.

**Key gemma4-on-Athena facts (see [[project_athena_storage_and_ollama]] + docs/athena_container_eval.md):**
gemma4:31b needs ollama **≥0.20** (renderer/parser/tools compiled into the binary; container's 0.18.2
has zero gemma4 support). Use **0.32.5** at `~/work/opt/ollama/0.32.5` (validated, job 126672/126761),
via `OLLAMA_VER=0.32.5` which prepends it to PATH inside the apptainer. `athena_frames_sft.job` now has a
**`NO_LOGFIRE=1`** toggle. **48GB L40S is fine — DON'T cap context, DON'T request big GPUs:** ollama
auto-fits gemma4's context to VRAM (nlp-srv3's 98GB GPU loaded the full 262144 → 73GB, which is what made
me wrongly think it needed a huge GPU). Path trap: `/work` only exists INSIDE the container; from the
login node use `~/work` (→ `/rg/reichart_prj/dvirla`). gemma4 needs NO MXFP4 gymnastics — quantizes to a
normal GGUF, so serving can quant-match its ollama baseline directly (unlike gpt-oss). Next after
collection: curate (`--require-correct-plain-ref --threshold 1`) → train → serve → eval + control.

**GEMMA-4-31B ARM COMPLETE (2026-08-04) — two-model study done.** Full pipeline finished:
collect(17,535)→curate(5,105 SFT, 58 usable/102)→**gemmify** ChatML (`gemmify_sft_chatml.py`: <think>→
`reasoning` field for gemma-4's canonical template)→LoRA SFT (needed transformers 5.2→**5.14.1** for
gemma4 modeling; and drop `device_map=auto` — tied embed/lm_head land on `meta` → backward crash; loss
0.22)→convert/quant **Q4_K_M**→register in ollama **0.32.5** (gemma4 renderer auto, tools work, bare FROM;
the register trap: `export OLLAMA_HOST` so create/show use the serve port, NOT default 11434)→eval sharded
7-ways (one CONDS per job, `12h_4g` QoS + 6h + light 8cpu/32G footprint so they backfill; 4d_1g got stuck
0-running). Baseline = existing local `results/frames_cues_full/gemma4_31b` (all cues, local-BM25),
restricted to the 102 test ids. Both Q4_K_M+local → **directly comparable, NO quant control needed**.

**RESULT (replicates gpt-oss):** SFT gives **cue-robust search at ZERO accuracy cost**. Whole-101:
mean|Δsearch vs own plain| 0.81→0.46, **#sig cues 3→0** (kills the suppression cues natural/elaborate/
direct); accuracy 0.519→0.529. Docs: **docs/frames_cue_robustness_sft.md**; figure:
`scripts/make_gemma_cue_figure.py` → `results/frames_cue_eval_test_regrade/gemma_cue_robustness.png`.
**Key nuance (both models):** SFT delivers cue-INVARIANCE (flat across cues) but does NOT restore the
ORIGINAL baseline-plain behavior — it anchors to a SHIFTED absolute search level (gpt-oss LOWER ~3.7 vs
5.5 partly quant; gemma-4 HIGHER ~5.9 vs 4.8 pure fine-tuning). SFT_cue−baseline_plain is +0.4..+1.7
everywhere (residual ≥ the original cue effect). So: consistency achieved, plain-level match not.

**HOTPOTQA TRANSFER — DONE 2026-09-06, and this is the transfer result the paper should lead with.**
7-cond gemma-4 FRAMES-SFT (never trained on HotpotQA) vs gemma4:31b baseline, hotpotqa-300, 9 conds,
local BM25, offline regex grading. Athena jobs 141147-155, all COMPLETED 300/300, ~1-1.6 h each.
Repro: `scripts/grade_hotpotqa_regex.py` then `scripts/analyze_hotpotqa_transfer.py`. Commit a5c916e.
FIGURE (the FRAMES `gemma_cue_robustness.png` twin): `scripts/make_gemma_cue_figure.py --dataset hotpotqa`
-> `results/hotpotqa_cue_briefing/gemma_cue_robustness_hotpotqa.png`. **Panels SHARE a y-axis by
default** — with per-panel scaling matplotlib stretches the SFT's flat bars to the baseline's
-40..-80% height and the figure visually ERASES the effect; `--no-sharey` restores it. That figure
drops the 14 yes/no golds from EM (search still over all 300), so its accuracy bars sit <1pp off
`analyze_hotpotqa_transfer.py`, which grades all 300.

**The 2x2 that makes it publishable:** of HotpotQA's 8 cues, **5 were in the SFT's training**
(natural, elaborate, polite, direct, query) and **3 were NOT** (confident_parametric, multiturn,
searchmulti) — so one run measures new-dataset AND new-cue generalization.

| Δsearch vs own plain | baseline | SFT |
|---|---|---|
| plain / zero-search | 2.14 calls / 6.0% | 2.41 / 0.0% |
| SEEN cues mean\|Δ\| / #sig | 26.5% / 4 of 5 | **2.8% / 1 of 5** |
| UNSEEN cues | 42.3% / 3 of 3 | 22.6% / 3 of 3 |
| FLOOR (plain vs plain_rep2) | −1.4%, p=0.75 ns | **+1.8%, p=0.63 ns** (job 142221) |
| plain accuracy | 0.810 | 0.807 |

(1) Trained cues transfer ~completely: 2.8% vs a 1.4% floor; ELABORATE −43.1%\*\*\* → −0.1% ns,
DIRECT −36.8%\*\*\* → −3.0% ns, POLITE −26.5%\*\*\* → +2.8% ns; only NATURAL survives (−3.5%, p=.015).
(2) Unseen cues transfer PARTIALLY (~halved, all still sig) — so it is not a generic "always search"
reflex. (3) **NEW vs FRAMES: cue-induced ACCURACY damage roughly halves** — DIRECT −16.7pp → −7.0pp,
confident_parametric −23.0pp → −9.0pp, ELABORATE −7.0 → +5.0 — **at matched median response length**
(elaborate 209 vs 203 words, direct 2 vs 2), so the grader's verbosity bias does not drive the
between-arm contrast. Accuracy floor is −2.7pp, so sub-3pp deltas are noise.

SFT FLOOR DONE 2026-09-06 (job 142221): +1.8% search p=0.63, −0.3pp accuracy p=1.0. So the seen-cue
residual (2.8%) is 1.6x the model's OWN noise, and the SFT is more run-to-run stable in accuracy than
the baseline (−0.3pp vs −2.7pp). Open: (b) baseline ran srv3/ollama 0.22.0 vs SFT Athena/0.32.5, a runtime confound bounded only loosely by
the near-identical plain level and accuracy — one 300-rollout SFT-plain on srv3 settles it;
(c) regex/EM grading, LLM judge still pending; (d) 10-cond arm never run on HotpotQA.
**Pitfall: searchmulti search counts need `HISTORY_SEARCH_OFFSET` correction** (commit f0e71ec) —
the mocked history's own tool call inflated `sampler_search_calls`, flipping its sign. FRAMES and
MedQA searchmulti rows carry the same uncorrected counter and need the same treatment.
`analyze_hotpotqa_cue_pilot.py` reads RAW counts and is wrong for searchmulti; the transfer script
reads the graded per_row.csv instead.

**MedQA TRANSFER — REVISED 2026-09-02: PARTIAL transfer (the earlier "does NOT transfer" was a
metric artifact).** gemma-4 FRAMES-SFT (7-cond, never trained on MedQA) on the MedQA cue grid,
`results/medqa_grid/gemma4-frames-robust-q4km_latest` vs baseline `.../gemma4_31b`, all 500 q.
Reproduce: **`scripts/analyze_medqa_cue_transfer.py`**; write-up in docs/frames_cue_robustness_sft.md.

What still holds: search **propensity** transfers — SFT 2.35 calls on MedQA plain vs baseline 0.09
(26x) — at zero accuracy change (0.438 -> 0.436).

Why the old verdict was wrong: it compared **absolute Δcalls + #sig cues**, both confounded here.
(a) The MedQA baseline does **zero search on 95.8% of examples at plain**, so cue-suppression is
unmeasurable on it — its mean|Δ|=0.06 calls measures the FLOOR, not invariance (its cue effects are
real though: a plain<->plain rerun, `results/medqa_grid_rerun/gemma4_31b`, moves search only +0.014
calls, p=0.74 ns). (b) #sig is power-driven at n=500. On the scale-free **mean |matched-pairs
rank-biserial r|** (verified uncorrelated with an arm's search level: Spearman -0.055 p=0.88 across
the 10 non-SFT arms; absolute Δcalls is +1.000) the ranking **inverts**: baseline r=0.659 = the MOST
cue-sensitive of all 12 MedQA arms; SFT r=0.328.

Valid comparisons: (1) among the arms that actually search on MedQA (zero@plain<15%) the SFT is the
FLATTEST — SFT 13.9%/r=.328 vs gemini-3.5-flash 23.1%/.554, nemotron-cascade 26.8%/.459,
gemini-3.1-pro 25.1%/.379, qwen122b 38.3%/.552. (2) SFT vs itself: FRAMES 7.5%/r=.150/0-of-6-sig ->
MedQA 13.9%/r=.328/6-of-6 — **~half the robustness carries**, landing at about the level the
untrained baseline shows IN domain (FRAMES baseline matched-102 r=.333). Residual MedQA effects are
the canonical suppression signature, worst on the two cues it fully beat on FRAMES: ELABORATE -25.0%
(r=.599), SHORT -19.4% (r=.488). DIRECT is the only accuracy-moving cue (SFT -7.2pp, base -10.6pp).

Caveat worth remembering: on MedQA search buys nothing (26x search, accuracy flat; SFT acc 0.574 at
1 call -> 0.347 at 4+, difficulty-confounded), so this is cue-robustness of an inert behavior.
Gaps: no plain<->plain rerun for the SFT (no SFT noise floor); the 10-cond/8-cond arms were never
evaluated on MedQA; accuracy is regex-graded on free-form answers to a multiple-choice task.

**SFT PARAMETRIC (no_search) ARM launched 2026-09-06** — 5 runs x {plain, elaborate, direct,
multiturn} on **FRAMES and HotpotQA**, to measure the SFT's uncertainty the same way as the
open-weights baselines. Athena jobs **142372-142375 (FRAMES)** and **142376-142379 (HotpotQA)**.
Submitters: `athena_submit_frames_parametric.sh` (NEW) and `athena_submit_hotpotqa_parametric.sh`.
Commit 2135b7b.

**Capability gap that had to be built: FRAMES had no parametric sweep on Athena.** HotpotQA already
had the (model, condition, RUNS) job pair, but `run_frames_grid_experiment.sh` hardcodes
`--agent_type baseline` and a single run — it has no AGENT_TYPE/RUNS knobs like
`run_hotpotqa_cue_experiment.sh` got. The FRAMES parametric sweep instead goes through
`scripts/cues_single_ranged.sh`, which already owns the condition -> (template, history) map that
produced the existing `results/frames_parametric/` baselines. Generalized it with `CUES` /
`DATASETS` env knobs (defaults unchanged, so old invocations still work) plus a `plain` condition,
then added `scripts/athena_frames_parametric.job` + submitter mirroring the HotpotQA pair.

**FRAMES plain naming is ASYMMETRIC and must not be "fixed":** plain writes
`frames-cues_no_search_<model>_run_<i>.json` with NO condition token (that is what
`run_parametric_probe_experiment.sh` produced for the baselines), while every cue writes
`..._<cue>_run_<i>.json`. cues_single_ranged.sh special-cases this. Get it wrong and nothing pairs
with the baselines by filename.

**Same OLLAMA_VER trap, third occurrence:** BOTH parametric submitters matched only `gemma4:*`, so
`gemma4-frames-robust-*-q4km:latest` got no OLLAMA_VER, would run under the container's 0.18.2, and
every example would be skipped as "does not support tools" — a job completing with zero work. Both
now use `gemma4:*|gemma4-*`. Check this pattern in ANY new submitter that takes a model tag.


See [[project_athena_storage_and_ollama]], [[project_cue_search_truncation_smoke.md]], [[project_cue_final_response_axes.md]].


**RESOLVED SFT — THE DEFECT IS FIXED, BOTH GOALS HOLD AT ONCE (2026-09-10).** Retrained the
gemma-4 7-cond arm with the two fixes from [[project_gemma_sft_gguf_eog_defect]]: the search tool
schema rendered into the training prompt (`tools=` in `apply_chat_template`, per-example
`tools_available` flag) plus 612 genuine tool-ABSENT examples taken from the no_search probe's own
Logfire traces. Builder: **`scripts/build_resolved_frames_sft.py`** -> `data/sft/frames_gemma4_resolved/`
(5,570 examples: A=4,083 tools+searches, B=875 tools+answered, C=612 no-tools+answered = 11.0%);
checkpoint `models/gemma-4-31b-frames-resolved`, ollama tag **`gemma4-frames-resolved-q4km`**.

| arm | resolved SFT | baseline gemma4:31b |
|---|---|---|
| FRAMES search, SEEN-cue mean\|Δ\| | **5.9%** | 16.5% |
| FRAMES search, UNSEEN-cue mean\|Δ\| | 30.2% | 41.8% |
| FRAMES plain search level / acc | 5.50 / 51.0% | 4.85 / 54.9% |
| FRAMES parametric acc (5 runs) | 23.2-30.3% | 23.4-30.9% |
| HotpotQA parametric acc (5 runs) | 34.1-44.0% | 34.9-45.1% |

**Parametric ability is fully retained** -- every no_search cell within ~1pp of base (worst:
HotpotQA confident_parametric -2.9pp), vs the old 7-cond arm's 75-98% truncation. True truncation
is now **~0.3%**. Cue-robustness was NOT sacrificed: 5.9% seen-cue invariance sits between the old
7-cond (7.5%) and 10-cond (3.7%) arms.

**confident_parametric, on the metric that matters (does it search at all):** FRAMES zero-search
75.5% -> 52.0%, 25 questions flip to searching vs 1 the other way, **McNemar p=8.1e-07**. The
*volume* DiD is NOT separable at n=102 (+0.49 calls, CI [-0.15,+1.16], Wilcoxon p=0.334) -- but
HotpotQA at n=300 separates on both (+0.67 calls, CI [+0.41,+0.96], p=1.7e-07), so FRAMES is
underpowered, not different. Report the decision metric, not the volume.

**Grading pitfall, now fixed in code:** leaked reasoning inflates substring grading. Added
**`regrade_regex.strip_reasoning_channel()`** -- split on the last `<channel|>`/`<|channel>` and
keep the tail. Every row it changes was a verified FALSE POSITIVE (gold "Verizon Center" matched
reasoning that concluded "The Fillmore"). Apply to BOTH arms via one code path; it is a no-op on
the base model, which never emits the markers.

**Do NOT use a `<3 words after marker` truncation test.** It misreads correct terse `direct`
answers as dead ends: of 297 FRAMES rows it flagged, 80 were CORRECT and only 7 genuinely empty.
