# Parametric-uncertainty / semantic-entropy handoff

Snapshot as of 2026-08-18. Written for a fresh Claude session to pick up this thread without
re-deriving context. Everything below is now on the **local machine** (this repo checkout) —
no need to SSH to srv3/Athena to read data, only to check on jobs still running there.

## What this thread is about

Building a semantic-entropy measure of each model's **parametric uncertainty** (how
consistent its answers are across repeated no-search rollouts of the same question), to test
whether it explains the search-triggering behavior studied in the sibling paper repo
(`/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex`, see
§"Paper tie-in" below and `accuracy_revision.md`, the current writing-ready synthesis of that
connection (there is no separate paper-notes file anymore). Key finding so far:
FRAMES search-call volume correlates strongly with semantic entropy (pooled ρ=0.33, p=3.7e-52,
monotone across every model tested); MedQA does not (ρ≈0.04, n.s.).

## Data inventory

### Raw no-search parametric probe (`--agent_type no_search`, plain-original phrasing)

`results/frames_parametric/<model_slug>/frames-cues_no_search_<tag>_run_<N>.json` (target 501 rows)
`results/medqa_parametric/<model_slug>/medqa-500_no_search_<tag>_run_<N>.json` (target 500 rows)

Each row: `{problem, correct_answer, sampler_response, sampler_correct, sampler_search_calls,
stop_reason, metrics, example_id}`. Grading is offline/regex (`scripts/regrade_regex.py`), not
a live LLM judge (`--no_grader` was used throughout) — `sampler_correct` in these particular
files is *not* meaningful (no_search runs skip grading); use `regrade_regex.heuristic_match`/
`relaxed_match` against `correct_answer`/`sampler_response` if you need offline correctness.

| model_slug | tag | FRAMES runs complete | MedQA runs complete |
|---|---|---|---|
| gemma4_31b | gemma4:31b | **5/5** (501 each) | **5/5** (500 each) |
| gpt-oss_120b | gpt-oss:120b | **5/5** (501 each) | **5/5** (500 each) |
| gpt-oss_20b | gpt-oss:20b | **5/5** (501 each) | **5/5** (500 each) |
| nemotron-3-nano_30b | nemotron-3-nano:30b | 5/5, but run_4=499/501, run_5=498/501 (a couple silently-skipped examples, minor) | **5/5** (500 each) |
| nemotron-cascade-2_30b | nemotron-cascade-2:30b | 3/3 (501 each); run_4 in progress on Athena (was 19/501 as of last check) | **3/3** (500 each) — run_3 just finished |
| qwen3.5_122b | qwen3.5:122b | **3/3** (501 each) | 2/3 (500 each); run_3 in progress on Athena (~68/500 as of last check); run_4/5 in progress on srv3 |

**No Gemini model has a no-search parametric probe at all** — never run for the cloud roster,
only the open-weight one above.

### LLM-judge semantic-entropy clustering

Clusterer = `gpt-oss:120b` via local Ollama, prompt asks it to group the N independent answers
by semantic equivalence (ignoring wording/reasoning differences); entropy = Shannon entropy
(bits) over the resulting cluster-size distribution. Validated by manual spot-check against full
response text on 24 examples (12 FRAMES + 12 MedQA) — 12/12 and 11/12 correct — **not** the
rigorous multi-annotator alt-test (see "Open items" below).

- **3-run version** (original, all 5 complete-3/3 models at the time):
  `results/{frames,medqa}_parametric/<model_slug>/<prefix>_<tag>_llm_clusters.json`
  — includes `nemotron-cascade-2_30b` (was clustered before its data-completeness issue was
  found; treat with caution — see caveats).
- **5-run version** (new, just finished — only the 4 models that already had 5/5 runs):
  `results/{frames,medqa}_parametric/<model_slug>/<prefix>_<tag>_llm_clusters_5run.json`
  — `gemma4_31b`, `gpt-oss_120b`, `gpt-oss_20b`, `nemotron-3-nano_30b` only. `qwen3.5_122b` and
  `nemotron-cascade-2_30b` do **not** have a 5-run cluster file yet (their raw 5-run data isn't
  done — see in-progress jobs below); once it lands, rerun clustering for just those two.

Both file schemas: `[{example_id, problem, correct_answer, semantic_entropy, cluster_ids}, ...]`.
Entropy over N samples only takes discrete values (3-run: 0/0.918/1.585 bits; 5-run: 0/0.722/
0.971/1.371/1.522/1.609/2.322 depending on the partition — richer resolution than 3-run).

Mean entropy by model (5-run, just computed):

| model | FRAMES mean entropy | MedQA mean entropy |
|---|---|---|
| gemma4_31b | 1.035 | 0.190 |
| gpt-oss_120b | 1.085 | 0.215 |
| gpt-oss_20b | 1.520 | 0.390 |
| nemotron-3-nano_30b | 1.166 | 0.322 |

(FRAMES entropy is consistently much higher than MedQA — open-domain multi-hop questions elicit
far less self-consistent answers than multiple-choice clinical vignettes, as expected.)

**Clustering script** (scratch, not committed — pull from srv3 if needed, or recreate; logic is
simple): `scripts/_scratch_cluster_parametric_llm_judge_5run.py` on `nlp-srv3` at
`/data/home/dvirla/parametric_search_tradeoff/scripts/`. Async, bounded-concurrency, resumable
via a `.gradecache.jsonl` cache file per model/dataset next to the output.

### Entropy-vs-search-calls analysis

`scripts/analyze_llm_entropy_vs_search_5run.py` (**final** — joins semantic entropy against
`sampler_search_calls` on the paired plain-original search run, per model, for the 4 non-cascade
models) → `results/param_vs_search_llm_5run/`: FRAMES pooled ρ=0.349 (p=1.8e-58), MedQA pooled
ρ=0.055 (p=0.015, barely significant, driven by 2/4 models only). An earlier 3-run version and an
even earlier k-of-3-correct binned measure both existed and were superseded by this one; both have
been deleted rather than kept as dead weight — do not recreate them. The causal necessity × cue
interaction test (§ "Causal necessity-vs-template test" below) was also rerun on this 5-run
entropy. Full writeup: `accuracy_revision.md` §1 ("The central reframe") — that file is the
writing-ready paper synthesis (thesis reframe, proposed main.tex edits per location, and a reading
list per new subsection for whoever drafts the actual paper text), not a chronological log.

## In-progress jobs (as of last check, will be further along by the time you read this)

| Job | Where | What | Status snapshot |
|---|---|---|---|
| qwen3.5:122b MedQA run_3 | Athena job 134535 (or its successor if it died/requeued — check `squeue -u dvirla` on `athena`) | finishing the 3rd MedQA run | ~68/500 |
| nemotron-cascade-2:30b run_4/run_5 | Athena job 134539 | FRAMES+MedQA run_4 and run_5 | early, run_4 FRAMES ~19/501 |
| qwen3.5:122b run_4/run_5 | `nlp-srv3`, background process (`logs_qwen_45.log` in the repo root there) | FRAMES+MedQA run_4 and run_5 | ~115-120/501 into FRAMES run_4 |
| ~~No-search LLM regrade~~ | `scripts/regrade_no_search_llm.py` | LLM-judge regrade of all 5 no-search rollouts x 4 models x 2 datasets | **COMPLETE** (20,015/20,015 rows, `results/no_search_llm_grades/`). Downstream scripts already rerun: `scripts/analyze_entropy_vs_correctness.py`, `scripts/analyze_no_search_accuracy_llm.py`, `scripts/make_no_search_oracle_figure.py`. Final numbers in `accuracy_revision.md` §1.1/§1.4 and `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md` Stage 0/Stage -1. |

Check via `ssh nlp-srv3` / `ssh athena` (both have recurring transient DNS resolution
hiccups — retry a few times before concluding something's actually down).
Once these land, pull the new files the same way this session did (`scp` from
`nlp-srv3:/data/home/dvirla/parametric_search_tradeoff/results/...` and
`athena:/home/dvirla/parametric_search_tradeoff/results/...`), then rerun the 5-run clusterer
for `qwen3.5_122b` and `nemotron-cascade-2_30b`.

## Known caveats / pitfalls (don't rediscover these the hard way)

1. **A `--resume` bug in the no-search eval pipeline caused real data loss once this session**
   (destroyed a complete 501-row FRAMES run on srv3, silently regenerating from scratch instead
   of skipping). Root cause was never fully confirmed (a theory about the shared
   `agent_name="no_search_agent"` in `scripts/run_qa_eval_experiment.py`'s `no_search` branch —
   see git history around this date for a since-reverted attempted fix — was rejected by the
   user as unconfirmed and not applied). **Practical rule adopted since**: for `no_search` runs,
   invoke `run_qa_eval_experiment.py` directly per `run_name` rather than looping through
   `scripts/run_parametric_probe_experiment.sh`'s `r=1..RUNS` (which always touches `run_1`
   first); when resuming a partially-complete file, watch the first few log lines for
   `Loaded N existing results` / a `pending` count that matches `target - N`, not the full
   target, before trusting it unattended.
2. **`nemotron-cascade-2_30b`'s original 3-run data had a real gap** (MedQA `run_3` was stuck at
   3/500, FRAMES `run_3` at 309/501) that got backfilled this session — now genuinely 3/3
   complete on both datasets (see table above). Its **3-run cluster file
   (`llm_clusters.json`, not `_5run`) may have been computed before this fix** — if you use it,
   double check `n` in the output matches 501/500, not the earlier partial coverage.
3. **Semantic-entropy resolution is coarse.** 3 samples → 3 entropy levels; 5 samples → up to 7.
   Don't over-interpret fine-grained entropy differences with only 3-5 rollouts backing them.
4. **The LLM-judge clusterer was validated by manual spot-check only** (24 examples, one
   reviewer), not a statistically rigorous procedure. `data/clustering_goldset.jsonl` has the
   start of a real gold-set (27 MuSiQue-hop examples) but only 1 annotator — short of the
   ≥3-annotator, 50-100-item bar needed for a proper alt-test (Calderon et al. 2025,
   arXiv:2501.10970) if you want to formally justify the judge later.
5. **Sentence-embedding clustering (cosine similarity on full responses) was tried and rejected**
   as a non-LLM-judge alternative — it fails on exactly the hard cases (same-topic/different-
   entity answers, refusal-vs-committed-answer) because full-response embeddings are dominated
   by shared boilerplate/reasoning text, not the committed answer.
6. **No Gemini no-search probe exists** — if you want Gemini in this analysis, it needs to be
   run from scratch (cloud API, not Ollama — costs real money, unlike the free local-Ollama
   models above).

## Causal necessity-vs-template test (separate thread, built on top of this entropy data)

A follow-up thread used this same entropy data as a pre-treatment "necessity" covariate to test a
causal question: when a cue (e.g. `confident_parametric`, "no need to search") changes search
volume, does the SIZE of that change depend on the model's own necessity, or is it a
necessity-blind uniform override? Design: paired within-subject (every example run under both
`plain` and the cue, same model/decoding config), `calls ~ entropy + is_cue + entropy:is_cue` OLS
with cluster-robust SEs by example, FDR-corrected across all (dataset, model, cue) cells. Run at
both 3-run and 5-run entropy resolution — same qualitative result both times (see below), which is
itself evidence the finding isn't a resolution artifact.

Headline: no single answer — `nemotron-3-nano_30b` is necessity-*aware* across 6 cues (effect
shrinks as necessity rises); `gemma4_31b` is necessity-*anti-calibrated* under its most
instruction-like cues (`confident_parametric`, `multiturn`, `searchmulti` — suppression gets worse
where necessity is higher) but calibrated under `query`; `gpt-oss_120b` shows no necessity-
dependence anywhere (cleanest "blind template override" case); MedQA has nothing that survives FDR
correction. Scripts: `scripts/analyze_necessity_vs_template_search{,_5run}.py`,
`scripts/make_necessity_vs_template_figure{,_5run}.py`. Full writeup with all caveats:
`accuracy_revision.md` §1 ("The central reframe") and §6 (script/data index).

**Separate, sharper follow-up (no cue involved at all)**: "are a model's search calls calibrated
with its own uncertainty," checked with split-half replication using the two independent no-cue
`plain` rollouts that already exist (`frames_cues_full`/`medqa_grid` vs. their literal-repeat
`_rerun` counterparts) instead of a single rollout. FRAMES replicates cleanly in all 4 models
(ρ_A≈ρ_B, p<1e-6 each). MedQA splits exactly along model lines: `gemma4_31b`/`gpt-oss_120b`
replicate as real (both runs significant); `gpt-oss_20b`/`nemotron-3-nano_30b` replicate as a
genuine null (both runs n.s.) — not a power issue, an actual model-level split. This is the
strongest-yet evidence for the correlational claim, though still not causal (nothing is
manipulated here — that's still only §7/§8's cue-interaction test). Scripts:
`scripts/analyze_baseline_calibration.py`, `scripts/make_baseline_calibration_figure.py`. Full
writeup: `accuracy_revision.md` §1 ("The central reframe").

**Third follow-up: mechanism decomposition (why/how do cues suppress search relative to the
calibrated plain line?)**. A first attempt pooled all cues into one `calls ~ entropy +
C(condition)` model and compared partial R² — correctly flagged by the user as the wrong lens,
since pooling every cue into one nuisance factor discards which specific cue does what (script and
output deleted, do not recreate this approach). The fix: decompose each cue's already-estimated
(§7/§8) effect into a LEVEL SHIFT (`b_is_cue`, suppression
unrelated to necessity) vs. a SLOPE CHANGE (`b_interaction`, change in necessity-sensitivity
itself). On FRAMES, most cues (50/62 cells) are pure level shifts — calibration survives
underneath them; only `gemma4_31b`/`confident_parametric`/`multiturn`/`searchmulti` and
`gpt-oss_20b`/`multiturn` show real calibration erosion, while `nemotron-3-nano_30b` under several
cues shows the opposite (calibration sharpens). MedQA shows zero significant slope changes at all
(0/60) — nothing to erode, consistent with its weak baseline calibration. Scripts:
`scripts/analyze_cue_suppression_mechanism.py`,
`scripts/make_cue_suppression_mechanism_figure.py`. Full writeup: `accuracy_revision.md` §1 ("The central reframe").

## Mediation: does a cue's accuracy cost route through its search-volume change?

Two follow-up attempts, one a dead end worth flagging so it isn't re-attempted, one a real result:

1. **Observational Baron-Kenny mediation is methodologically broken — do not redo this as a naive
   `calls ~ is_cue` / `correct ~ is_cue + calls` regression** (script and output deleted after
   this was established). Diagnostic: within every entropy
   stratum, realized search-call count and correctness are strongly NEGATIVELY correlated
   (r=-0.32 to -0.66 on one model/dataset alone; negative in 93% of 88 cells tested) — the
   signature of an endogenous mediator, since search-call count is generated during the same
   rollout as the answer (an agent searches more largely BECAUSE it's struggling, not the other
   way around). The cue's effect on calls (a-path) stays causally clean; the calls-to-accuracy
   path does not identify without an actual manipulation of the mediator.
2. **A real manipulated-mediator test: `gemma4-frames-robust-*` SFT checkpoints**
   (`results/frames_cue_eval_test/gemma4-frames-robust-q4km/`, 102-question FRAMES test set),
   trained on rollouts curated to have |Δsearch calls vs plain| <= 1 AND a correct answer.
   Comparing base gemma4:31b vs. this SFT'd version on the same 102 questions x same 8 cues,
   regex-graded both ways (`scripts/analyze_sft_intervention_mediation.py`,
   `results/sft_intervention_mediation/`): for `direct`, restoring ~54% of the search-volume swing
   also restores ~48% of the accuracy swing (roughly proportional -- search-volume-mediated). For
   `confident_parametric`, search volume barely recovers (still 77% of the original swing) yet
   accuracy recovers almost completely (down to 18% of original cost, no longer significant) --
   the accuracy fix did NOT come from restoring search volume, consistent with this specific cue
   eroding calibration itself (see mechanism-decomposition finding above) rather than just volume.
   Caveat: SFT jointly retrains toward small-delta-and-correct, not a single-variable mediator
   manipulation; n=102 is small. Full writeup: `accuracy_revision.md` §1 ("The central reframe").

## Unified framework (synthesizes everything above into one reusable protocol)

Everything from "Causal necessity-vs-template test" through "Mediation" above is one four-stage
protocol, not four separate analyses: Stage 0 (necessity instrument, i.e. the entropy data on this
page) → Stage 1 (baseline calibration, split-half replicated) → Stage 2 (fragility under a cue,
decomposed into a level-shift-vs-slope-change taxonomy: Null / Uniform volume shift / Erosion /
Inversion / Sharpening -- level shift alone is NOT calibration-neutral, it's a different KIND of
miscalibration than a slope change, not an absence of one; see the framework doc for why) →
Stage 3 (consequence attribution, requires a manipulated mediator like the SFT checkpoints, not
naive observational mediation). Written up as a general methodology in
`docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`, with a scorecard figure instantiating it across all 4
models x 2 datasets: `scripts/make_epistemic_alignment_scorecard.py` ->
`results/epistemic_alignment_scorecard/`. If a future session wants to apply this same protocol to
a new model (e.g. once `qwen3.5_122b` or `nemotron-cascade-2_30b`'s 5-run data completes, per the
in-progress jobs above), that doc has the exact requirements and stage-by-stage recipe.

## Does search add value at all? (prerequisite check, changes how the MedQA story reads)

Whether search actually helps in each domain, and — for MedQA specifically — whether that's
because the tool doesn't help, or because models mostly don't invoke it to begin with. Both are
now LLM-graded (final, post-regrade — an earlier regex-graded version, plus a "pool every rollout
ever collected" oracle ceiling, were both used first and then discarded: the oracle became
internally invalid once the corrected no-search floor exceeded its old regex-graded ceiling on
MedQA, and re-deriving a valid one would need LLM-regrading every cue condition, ~60k more
gradings, out of scope; both are gone from `results/`, do not recreate them). Current scripts:
`scripts/analyze_no_search_accuracy_llm.py`, `scripts/analyze_medqa_search_conditional.py`.
Figure: `results/search_oracle/no_search_oracle_comparison.png` (clean 2-bar no-search-floor vs.
`plain`, no oracle bar).

**FRAMES: search is genuinely load-bearing** (+10 to +30pp over no-search). **MedQA: search adds
essentially nothing in aggregate** — but the aggregate conflates two populations. Only 4-20% of
MedQA examples ever get a search call under `plain`; on the 80-96% that don't, `plain` ≈ no-search
trivially. On the small subset that *does* get searched, accuracy is consistently *worse* than the
model's own no-search accuracy on those same examples, in all 4 models (-3.8pp to -9.3pp). So the
correct claim is not "the tool doesn't help" — it's "search is rarely invoked, and when invoked,
correlates with a real accuracy cost." Full numbers: `accuracy_revision.md` §1.4 and the "Stage -1"
section of `docs/EPISTEMIC_ALIGNMENT_FRAMEWORK.md`.

**This means MedQA's weak/absent baseline calibration elsewhere in this doc is not simply "search
doesn't matter there so miscalibration is harmless"** — the searched subset shows a real,
directional cost. Written up in full in both docs above.

## The volume/accuracy decoupling headline finding

Independent of the entropy/calibration machinery above: among the 90 "level shift only" cells from
the mechanism decomposition (where the necessity-tracking slope is statistically intact), |Δ search
volume| and |Δ accuracy| are essentially uncorrelated (ρ=+0.016, p=0.88); 48% of cells with a large
volume swing (|Δcalls|>1.0) show negligible accuracy change (<3pp); MedQA's `searchmulti2/3` cues
inflate near-zero baseline volume 8-38× with at most ±2.8pp accuracy movement. This is the
project's most self-contained, easiest-to-defend claim — it needs no entropy instrument at all, just
two observed quantities. Script: `scripts/analyze_volume_accuracy_decoupling.py` →
`results/cue_suppression_mechanism/volume_vs_accuracy_delta.csv`. Full writeup:
`accuracy_revision.md` §1.0 (the document's lead section).

## Paper tie-in

The sibling paper (`/home/dvirla/projects/Info-Seeking-Agentic-Behavior-Analysis/main.tex`)
currently claims tool-use is "decoupled from parametric knowledge / epistemic uncertainty" —
too strong given the ρ=0.33 FRAMES baseline-calibration finding above, but the volume/accuracy
decoupling finding above supports a more precise version of the same instinct. A full accounting
of exactly which paper sentences need to change, why, and proposed replacement text is in
`accuracy_revision.md` — the single, current proposal document (a standalone earlier draft of this
existed and has been deleted; `accuracy_revision.md` supersedes it entirely). **No edits have been
applied to the paper** — that document is a proposal only, per explicit instruction.
