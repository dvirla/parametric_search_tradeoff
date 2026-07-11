# Plan: Cue → Search-Truncation Mechanism Analysis (facts_open + MedQA)

**Context.** PLAIN vs ELABORATE cue runs (branch `sensitivity_to_prompt_cues`) show large search-call
reductions with flat-or-up aggregate accuracy, but per-question analysis reveals correctness churn
coupled to search reduction (weak models) or selective, harmless pruning (Gemini). We now want the
*mechanism*, using **existing rollouts only** — no new agent evals. Three phases: download traces
from Logfire, run a commitment-locus probe on 4 cells, and build a Streamlit flip-forensics
inspector.

**Existing eval JSONs (per-example: `problem`, `correct_answer`, `sampler_response`,
`sampler_correct`, `sampler_search_calls`, `stop_reason`, `example_id`):**

| cell | PLAIN | ELABORATE |
|---|---|---|
| facts_open × Gemini | `results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_plain.json` | `..._elaborate.json` |
| facts_open × Qwen | `results/facts_qwen100/facts-open_baseline_qwen3.5:122b_plain.json` | `..._elaborate.json` |
| facts_open × Gemma | `results/facts_gemma100/facts-open_baseline_gemma4:31b_plain.json` | `..._elaborate.json` |
| MedQA × Gemini | `results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_plain.json` | `..._elaborate.json` |
| MedQA × Qwen | `results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_plain.json` | `..._elaborate.json` |
| MedQA × Gemma | `results/cue_smoke_medqa/medqa_baseline_gemma4:31b_plain.json` | `..._elaborate.json` |

Analysis conventions (match `scripts/summarize_frames_cues_grid.py` / project memory): pair
conditions on `example_id`; drop rows with `stop_reason != None` (UsageLimitExceeded = runaway);
`sampler_search_calls` counts tool-call *decisions*, not HTTP requests.

---

## Phase 0 — Download traces from Logfire (URGENT: retention window)

Runs executed 2026-07-07/08. Do this first, before anything else, and check
`--lookback-days` covers the run dates (default is 5 — bump as needed).

For each of the 6 cells × 2 cues (12 downloads):

```bash
uv run python scripts/download_traces.py \
    --agent-name baseline_agent \
    --model-name <model_name> \
    --eval-json <eval json for that cell+cue> \
    --limit 0 \
    --lookback-days 7 \
    --output-dir results/cue_traces
```

- `--eval-json` filters Logfire traces to problems present in that eval file. **This is what
  separates PLAIN from ELABORATE traces** (same agent, same model): the `problem` field includes
  the cue suffix (`"...\n\nPlease answer with a detailed explanation - at least 8-10 sentences"`),
  so exact problem matching disambiguates the conditions. Verify this actually holds — check that
  the eval JSON `problem` strings for ELABORATE contain the suffix and that the downloader's
  problem matching is exact (read `get_agent_traces_from_logfire` in
  `scripts/download_traces.py`); if it's fuzzy/LIKE-based, post-filter downloaded traces by exact
  `problem` match against the eval JSON.
- Rename/organize outputs to a deterministic convention:
  `results/cue_traces/{dataset}_{model_slug}_{cue}_traces.json`
  (model_slug: `gemini-3.1-pro-preview`, `qwen3.5_122b`, `gemma4_31b`; dataset: `facts-open`, `medqa`).
- Trace JSON schema (verified on `results/curated_sharechat/*_traces.json`): list of
  `{problem, agent_name, start_timestamp, end_timestamp, result, message_trace, metadata}`;
  `message_trace` is a list of `{role, timestamp, finish_reason, parts}` where parts contain
  `{'type': 'text'|'thinking'|'tool_call'|'tool_call_response', ...}` with search queries in tool
  calls and retrieved `[{title, content/url, ...}]` in tool responses. **Caution:** confirm
  whether `parts` deserializes as a list or a stringified repr; if string, parse with
  `ast.literal_eval` fallback.
- **Risk to check first:** Qwen and Gemma runs executed remotely on nlp-srv3 — confirm
  `LOGFIRE_API_KEY` was set in that box's `.env` at run time (i.e., their traces exist in Logfire
  at all). If a cell's traces are missing, note it in the run report and proceed with available
  cells; do NOT re-run evals.

**Acceptance:** per cell+cue, #downloaded traces ≈ #eval rows (report exact match rate of trace
`problem` → eval `problem`; ≥90% required, investigate below that). Write a small manifest CSV
`results/cue_traces/manifest.csv` (cell, cue, n_eval, n_traces, n_matched).

## Phase 1 — Commitment-locus probe (4 cells only)

**Cells:** facts_open × {Gemma, Qwen}; MedQA × {Qwen, Gemini}. Both cues each → 8 trace sets,
~600 trajectories total.

**Question answered:** at which point in the trajectory does the model first commit to its final
answer, and are the searches the cue prunes pre- or post-commitment? Hypothesis: ELABORATE mostly
cancels *post-commitment verification* (harmless, MedQA) but in weak models also *pre-commitment
evidence gathering* (harmful, facts_open).

**Implementation:** new script `scripts/probe_cue_commitment_locus.py`, adapted from
`scripts/archive/probe_commitment_locus.py` (reuse its judge prompt, per-step replay logic,
`--max-search-result-chars` truncation, shard/resume machinery). Changes:

- I/O: instead of MusiQue benchmark/natural matched pairs, take
  `--traces <traces.json> --eval-json <eval.json> --cell <name> --cue <plain|elaborate>`
  (or loop over a small config table of the 8 sets). Join traces→eval rows by exact `problem`
  string to recover `example_id`, `sampler_correct`, gold answer.
- Judge model: **`gemini-3-flash-preview` via `google` provider** (the archived default
  `gpt-oss:120b`/ollama is unavailable locally — no local Ollama). Same model as the eval grader;
  cost is cents for ~600 trajectories.
- Per trajectory, output one row: `example_id, cell, cue, n_searches, commitment_step`
  (index of the search call after which the judge says the answer is committed; 0 = committed
  before any search), `post_commitment_searches, committed_answer, matches_gold, final_correct,
  stop_reason`.
- Output: `results/cue_commitment_locus/commitment_{cell}_{cue}.csv` (+ raw judge JSONL alongside,
  same pattern as the archived script).

**Summary analysis** (same script `--summarize` or a small companion): per cell, paired on
example_id:
1. PLAIN post-commitment search share (are most PLAIN searches verification ritual?).
2. Δ(commitment_step) and Δ(post_commitment_searches) PLAIN→ELABORATE: does truncation remove
   post-commitment searches only (Gemini/MedQA prediction) or shift commitment earlier
   (weak-model/facts_open prediction)?
3. Cross-tab flips (R→W / W→R) × whether commitment moved / whether pruned searches were
   pre-commitment. Wilcoxon for paired deltas, report medians (heavy right skew).

**Acceptance:** CSVs for all 8 sets (minus any cells lost in Phase 0), a printed summary table,
and 5 manually spot-checked judge verdicts per cell documented in the run report.

## Phase 2 — Streamlit flip-forensics inspector

New script `scripts/inspect_cue_flips.py`, follow the pattern of `scripts/inspect_frames_cues.py`
(argparse after `--`, run with `uv run streamlit run scripts/inspect_cue_flips.py -- ...`).
Purpose: manual forensics for (a) MedQA W→R flips — was PLAIN's wrong answer *induced by
retrieved content*? and (b) facts_open R→W flips (Gemma/Qwen) — was the fact ELABORATE got wrong
present in PLAIN's pruned-query results?

**Inputs:** the 6-cell table of eval JSONs above + `results/cue_traces/*_traces.json` +
(optional, if present) Phase 1 CSVs. All paths configurable via flags with these defaults.

**UI:**
- Sidebar: cell selector (dataset × model), flip-type filter (R→W, W→R, stable-correct,
  stable-wrong, all), question dropdown showing `example_id — first 80 chars` with flip badge.
- Header: full question (cue suffix visually separated), gold answer, per-cue correctness badges,
  search counts (PLAIN n vs ELABORATE n), and the Phase 1 commitment rows if available.
- Two columns, PLAIN | ELABORATE. Each column, top to bottom: final answer (from
  `sampler_response`, collapsible if long), then the trajectory as ordered expanders per step:
  thinking excerpt, search query (monospace), retrieved snippets (title + content, truncated
  ~800 chars each, expandable).
- **Highlighting (the core forensic aid):** case-insensitive highlight inside snippets/thinking/
  final answers of (1) the gold answer — green; (2) the PLAIN final short answer — blue;
  (3) the ELABORATE final short answer — orange. Extract short answers with a simple heuristic
  (last sentence / after "answer is"); make the highlight terms editable in a sidebar text box so
  the user can fix bad extractions on the fly.
- **Query-alignment panel** (above the columns): PLAIN queries vs ELABORATE queries as two lists
  with fuzzy matching (normalized token overlap ≥0.6 → "kept", else "pruned"/"new"); pruned PLAIN
  queries colored red. This shows tail-truncation vs plan-change at a glance.
- **Annotation widget** (per question, per cell): radio — `pruned evidence was load-bearing` /
  `retrieval-induced error` / `parametric recall compensated` / `grader issue` / `unclear` — plus
  a free-text note and the annotator-visible flip type. Persist to
  `results/cue_flip_annotations.csv` keyed on (cell, example_id); load existing annotations on
  start, upsert on save (write immediately, no batch-save button).

**Acceptance:** app launches with defaults and renders every cell that has traces; a question with
0 searches in one condition renders cleanly; annotations survive an app restart. Include an
`--export` mode (like `inspect_missed_leakage.py` had) that dumps a static markdown/CSV digest of
flipped questions per cell for cells to review offline.

---

## Non-goals / guardrails

- **No new agent eval runs, no Brave calls.** The only LLM calls in this plan are the Phase 1
  judge (gemini-3-flash-preview) — everything else is local file analysis.
- Don't modify existing eval/summarize scripts; new scripts only.
- Keep pairing/filtering conventions identical to the smoke analysis (example_id join,
  stop_reason filter) so numbers stay comparable with `results/cue_final_response_axes.csv` and
  project memory.
- If Logfire traces turn out unavailable for a cell, degrade gracefully (skip cell, log in
  manifest) and finish the rest — surface it in the final report rather than blocking.

## Suggested order & effort

1. Phase 0 (do first — retention risk): ~1h including matching validation.
2. Phase 2 skeleton next (it only needs traces + eval JSONs, and gives immediate value).
3. Phase 1 probe (judge prompts need care; spot-check before the full 600 calls).
4. Wire Phase 1 CSVs into the Phase 2 UI last.
