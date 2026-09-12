---
name: project_gemma_sft_gguf_eog_defect
description: "gemma-4 FRAMES-SFT cannot answer without a tool. NOT a packaging bug (3 fixes failed) and NOT an absence of no-search examples (17-19% of every arm is zero-search) -- training strips both the tool schema and the system prompt, so tool-available and tool-absent prompts are indistinguishable and the cue-free prior is search."
metadata:
  node_type: memory
  type: project
---

**SUPERSEDED CLAIM (do not repeat): "the SFT has no trained path to answering without a tool call /
every training trajectory continued with a search call."** A dataset audit (2026-09-07, same day)
DISPROVES this. Every gemma-4 arm contains well-formed reason-then-answer-with-no-tool trajectories:
**7-cond 877/5105 = 17.2%, 8-cond 1067/5473 = 19.5%, 10-cond 1291/6905 = 18.7%** zero-search, all
carrying a populated `reasoning` field AND non-empty answer `content` (0 degenerate turns, 0 leftover
literal `<think>`). Reproduce with `scripts/summarize_sft_curation_stats.py --rollouts
data/sft/frames_gemma4/rollouts.jsonl --require-correct-plain-ref --threshold 1`.

**ACTUAL DATASET-LEVEL ROOT CAUSE: training erases every cue that distinguishes "tools available"
from "tools absent", then teaches search as the cue-free default.** Collection registered a real
`Tool(search_service.search)` (`create_frames_sft_data.py:362`), but `train_sft.py` (a) calls
`apply_chat_template` with **no `tools=`** (line ~247), so no training prompt ever renders a tool
schema. In that tool-free prompt form **81-83% of decision turns target a search call**. At inference
the search agent DOES pass a tool schema (gemma-4 renders it as a `<|tool>declaration:search{...}
<tool|>` block in the first system turn, `chat_template.jinja:207`, +75 tokens) and no_search does
not -- a distinction the model never saw.

**Sharpest form: the training prompt is byte-identical to the no_search inference prompt.** The
system prompt is empty in ALL FOUR places -- collection (`create_frames_sft_data.py` passes no
`system_prompt`, so `base_agent.py:117` defaults to `''`; every rollout confirms `''`), training,
search eval and no_search eval (no FRAMES driver passes `--baseline_sys_prompt_path`, and the
`no_search` branch of `run_qa_eval_experiment.py` accepts none at all). Both render
`<bos><|turn>system\n<turn|>`. So `train_sft.py`'s system-blanking is a **no-op** here, inherited
from the MusiQue pipeline where a system prompt did exist -- do NOT "fix" it, there is nothing to
blank. The model is not forgetting at no_search time; it is doing exactly what it was trained to do
under that exact prompt. The prompt form it never saw is the SEARCH one.

Two secondary weaknesses of the no-search signal that survives: it is **concentrated** (only
46/48/49 of 235 train questions have any no-search example; the top 10% of questions hold ~75% of
them) and it is **correctness-filtered**, teaching "answer when you happen to know" rather than
"commit to an answer with no tool".

In gemma-4's channel format the TEMPLATE emits the channel opener; the model writes reasoning, emits
`<channel|>` to close the thinking channel, then opens a new channel for its answer or a tool call.
The SFT does the first half and **stops**. With no tool registered the
continuation it learned does not exist, generation dies at the channel boundary, the parser is left
with an unterminated structure, `thinking` stays empty and the raw closer lands in `content`.

Decisive evidence: **300/300 no_search responses have `<|channel>` opener count 0 and `<channel|>`
closer count 1.** Traces end mid-plan ("1. Search for X. 2. Identify the company name from the search
results." then the closer).

**Blast radius** -- confined to the no-tool-call path, which is why a month of search-mode evals never
surfaced it: leak on **89.2% of zero-search rows vs 0.1% of rows with a search**. no_search arms
(FRAMES+HotpotQA, all 4 cues) 74-97% truncated. Search grids: 0.0% in 6 of 9 cells; only
`confident_parametric` is hit (HotpotQA 34.3%, FRAMES Aug-2026 30-36% across all three arms) because
it is the cue that talks the model out of searching. Base gemma4:31b in no-tools mode is CLEAN (0/20),
so "no tools" alone does not cause it -- it is specific to the fine-tune.

**THREE FIXES TESTED, ALL FAILED (do not retry these):**
1. **Retype the 6 mis-converted control tokens** (`<|tool_call>`,`<tool_call|>`,`<|tool_response>`,
   `<tool_response|>`,`<|channel>`,`<channel|>`: USER_DEFINED -> CONTROL, 24 bytes in place via
   `scripts/patch_gguf_token_types.py`). No change: 20/20 leaked. Also verified on the search path --
   confident_parametric leak 97.2% -> 93.9% (noise), search levels unchanged.
2. **Training-data serialization.** Scanned all 5,105 examples: ZERO literal channel markers in
   content, 17,500 messages carry a proper structured `reasoning` field. Data is clean. (Confirmed
   again by the 2026-09-07 audit -- the *content* is clean; what is missing is the tool schema and
   system prompt around it, see root cause above.)
3. **Repackage through ollama's own converter** (`ollama create --quantize q4_K_M` FROM the merged
   bf16 dir, job 143139, tag `gemma4-frames-robust-q4km-oconv`). Metadata became **byte-equivalent to
   base** -- `tokenizer.ggml.model='llama'`, `pre='gemma4'`, `eos_token_ids=[1,106,50]`, all 10
   watched tokens CONTROL, plus base's top_k 64/top_p 0.95. **Behaviour identical: 20/20 leaked, 18
   truncated, 0/20 thinking.**

**The packaging defect is REAL but NOT the cause.** llama.cpp's `convert_hf_to_gguf.py` classifies
specials from `tokenizer_config.json:added_tokens_decoder`, which gemma-4 ships EMPTY (its 24 specials
are flagged in `tokenizer.json:added_tokens` instead -- llama.cpp issue #5838), and gguf-py defines
only the singular `EOS_ID` key so gemma-4's `[1,106,50]` cannot be written at all. Acceptance test:
**`scripts/verify_gguf_special_tokens.py <tag-or-file> ...`**. Use ollama's converter for gemma-4,
not llama.cpp -- but expect no behavioural change from it.

**SALVAGE IS NOT VIABLE.** Splitting on the marker recovers answers only where text follows it.
Examples with all 5 runs usable: HotpotQA plain 2/300, elaborate 3/300, direct 26/300, multiturn
12/300; FRAMES plain 1/501, elaborate 5/501, direct 49/500, multiturn 17/496. Zero usable runs for
93.7/84.0/47.7/43.0% of examples. Worse, survivors are selected on *not searching* -- i.e. on
parametric confidence, the very quantity entropy would estimate -> biased by construction.
(For rows with nothing after the marker, the gold answer does appear in the pre-marker reasoning
17.5-20.9% on HotpotQA / 7-10% on FRAMES -- the model often knows but never commits.)

**Interpretation / how to apply:** the headline is narrower than first written. Search-behaviour SFT
that generalises out of domain (see [[project_frames_cue_robustness_sft]]) removed the ability to
answer parametrically **as configured** -- not inherently. Adding no-search trajectories is NOT the
missing ingredient (they were already there at 17-19%); the fix is to make tool availability
*visible* to the model: render `tools=` into the training prompt so tool-available and tool-absent
prompts differ, and add tool-free-prompt examples whose target is an answer. Headroom without any
new rollouts, on train questions: 1,463 correct zero-search rollouts over 109 questions (7-cond
conditions) / 2,313 over 143 (10-cond), vs 877 / 1,291 actually used -- the closeness filter
(`|search_calls - plain_ref| <= 1`) admits zero-search only where the plain reference is <=1, which
is just 63/235 usable train questions. Counting incorrect ones too (for teaching *commitment*
rather than only correctness): 2,724 over 210 questions / 4,626 over 341.

**The 10-cond checkpoint was never probed for this leak** -- `diagnose_gemma_sft_channel.py` defaults
`SFT_TAGS` to `gemma4-frames-robust-q4km:latest` (the 7-cond arm) only. At 18.7% vs 17.2% zero-search
it is very unlikely to differ, but it is untested.

**Process lesson: row counts are not integrity.** Nothing crashed (sacct COMPLETED 0:0, full
300/501-row files); the content is degenerate. Audit response TEXT -- control tokens, median words vs
a sibling arm, empty rate -- before calling a cell complete. I reported `elaborate` "5/5 complete"
from row counts alone and another session caught it.

Artifacts: 7-cond adapter + merged bf16 are intact at `models/gemma-4-31b-frames-robust{,_merged}`
(`training_info.json`: dataset_size 5105, loss 0.2216 -- both match the 7-cond arm), so no retraining
is needed for repackaging. Probe: `scripts/diagnose_gemma_sft_channel.py` (SFT_TAGS, NO_VARIANTS=1).


**Parametric labels for a resolved SFT already exist on disk (2026-09-07).** Which FRAMES train
questions gemma-4 can answer with no tool is recoverable from
`results/frames_parametric/gemma4_31b/frames-cues_no_search_gemma4:31b_run_[1-5].json` (base model,
plain, no_search, 5 runs x 501). **TRAP: those runs were produced with `NO_GRADER=1`, so
`sampler_correct`/`metrics.correct` are PLACEHOLDERS** -- taken at face value they read 2.0/0/0/0/0%
correct and imply gemma-4 solves ~nothing parametrically. Regrade offline with
`scripts.regrade_regex.heuristic_match(correct_answer, sampler_response)` and the real numbers are
**30.5/28.5/29.7/30.5/29.1%**. Applies to every `NO_GRADER=1` FRAMES arm.

Regraded, over the 235 curation-usable train questions: **85 solved 5/5, 107 at >=3/5, 101 at 0/5.**
The current closeness filter sources no-search examples from only **46** questions, so a
solve-rate label at >=3/5 more than doubles coverage (40 overlap, 67 new) using data already held.
Of the 107, **85 already have >=1 in-format correct zero-search trajectory** in `rollouts.jsonl`
(1,333 trajectories, `<think>` reasoning intact); only 22 would need new rollouts. Conversely **79
trajectories over 16 questions** currently teach "answer without searching" on questions scoring
<3/5 -- i.e. they teach unfounded confidence and should be dropped. NOTE the parametric run files
store only `sampler_response` (final answer, reasoning stripped), so they are usable as LABELS but
NOT as training targets in gemma-4's reasoning+answer format -- take trajectories from
`rollouts.jsonl`.

**Tool-absent trajectories pulled from Logfire (2026-09-07) -- the right source, replacing the
rollouts.jsonl plan.** Using zero-search rollouts from `rollouts.jsonl` as the no-tool training
target is WRONG: those come from runs where the tool WAS offered and the model declined it, so the
label ("answerable with no tool") and the trajectory measure different conditions. The no_search
probe's own Logfire traces give both from one event: `message_trace` carries `thinking` and `text`
as separate parts (-> gemma-4's `reasoning` + `content`) and the system message is `''`, so they
render into the exact prompt form the SFT trains on.

**PITFALL that silently ruins this: `download_traces.py` dedups on the cleaned problem keyed by
`agent_name`, and `run_qa_eval_experiment.py:77` HARDCODES `agent_name="no_search_agent"` for every
run** (it ignores `run_name`). So a default pull keeps ONE trace per (question, cue) and discards
every other run -- 2,308 of 36,743 records -- with no warning, and any solve rate over it is wrong.
It also keeps the NEWEST run, so the survivors came from a different sweep (Aug 25-27) than the
result JSONs on disk (Aug 15-20). Fixed by adding an opt-in **`--no-dedup`** flag (default behaviour
unchanged, so `download_cue_traces.py` is unaffected). Retention was NEVER the problem: all 36,743
records carry `all_messages` back to 2026-08-15.

Pull command (FRAMES plain only -- `clean_problem()` does not strip cue suffixes, so `--eval-json`
matches the plain form and silently excludes cued traces; use `download_cue_traces.py`'s
suffix-stripping for those):
    uv run python scripts/download_traces.py --agent-name no_search_agent --model-name "gemma4:31b" \
        --limit 0 --lookback-days 30 --no-dedup \
        --eval-json results/frames_parametric/gemma4_31b/frames-cues_no_search_gemma4:31b_run_1.json \
        --output-dir results/parametric_traces_raw/gemma4_31b_no_search_frames_alruns

**Yield -- CORRECTED. The 6,015 traces are NOT 12 plain runs; they are three pooled conditions:
plain 1,003 (2/question), direct 2,507 (5/question), elaborate 2,505 (5/question).**

Why the pooling happened: **`clean_problem()` (download_traces.py:78-84) DOES strip the elaborate,
polite and direct suffixes**, so `--eval-json` matching happens on the stripped text and cued traces
match the plain problem set. (`download_cue_traces.py`'s docstring says clean_problem does NOT strip
the elaborate suffix -- that is now STALE; the suffix was added to the separators list since.) To
segment, recover the condition from the RAW first user message inside `message_trace`, not from the
trace's `problem` field, which is already cleaned. Traces with >1 user message = the history-prefix
conditions (multiturn/searchmulti); none survived here, since their first user message is chit-chat
and does not match any FRAMES problem.

**Only 2 of the 5 plain runs are recoverable as traces, and the other 3 are GONE FOR GOOD --
Logfire retention, not a logfire-off run.** Surviving plain traces: Aug 15 (818) + Aug 16 (185) =
result JSON run_4 and run_5. **The entire Logfire store starts at 2026-08-08** -- `SELECT
min(start_timestamp) FROM records` with a 60/120/365-day interval all return the same
2026-08-08T05:46 and the same 890,773 rows, i.e. a hard 30-day wall, no older data for ANY agent.
Plain runs 1-3 were generated before that wall; their identical Aug-20 16:06 mtime is a batch
rewrite (a regrade), not their generation time -- corroborated by the answer-text pairing, where
477 traces pair to run_5 and 471 to run_4 but essentially none to runs 1-3.

**So there IS a real clock on trace-derived work, just not the one to panic about:** the surviving
Aug 15-16 plain traces expire ~Sep 14-15; the direct/elaborate traces (Aug 20-27) expire ~Sep 19-26.
Pull traces promptly, and never assume a result JSON on disk implies a retrievable trace -- an eval
JSON survives indefinitely, its trace does not.

**Traces pair EXACTLY to result-JSON rows by answer text** (477 to run_5, 471 to run_4, remainder
are ties where several runs gave an identical answer, 1 unmatched of 1,003). So the traces and the
result JSONs are the same events in different serializations -- which makes "label from the 5 result
JSONs + reasoning from the traces" legitimate rather than the cross-source coupling it looks like.
Verified, not assumed.

**Corrected plain-only yield** on the 235 curation-usable train questions (label = regex-graded
solve rate over all 5 result JSONs; trajectory = correct + reasoning-bearing traced plain run):
| label | questions | with >=1 traced trajectory | trajectories |
|---|---|---|---|
| >=5/5 | 85 | 85 | 170 |
| >=3/5 | 107 | 107 | 200 |
| >=2/5 | 122 | 117 | 211 |
Total correct plain traced trajectories at any label: 299 over 169 questions.

**So the tool-absent side is EXAMPLE-poor, not question-poor:** 200 trajectories over 107 questions
vs the current 877 no-search examples over just 46 questions -- 2.3x the question coverage but ~4.4x
fewer examples, i.e. ~4% of a ~5,000-example base against Hammer's 12.5% reference ratio. Closing
that needs either the cued no_search traces (direct/elaborate are fully traced, 5 runs each, and
cue-invariant no-tool answering is arguably on-objective) or fresh plain no_search rollouts on the
235 train questions -- which cost nothing but GPU time (no search calls, no Brave spend) and stay
inside the train split.
