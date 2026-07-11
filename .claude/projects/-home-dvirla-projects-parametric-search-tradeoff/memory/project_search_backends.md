---
name: project_search_backends
description: "Pluggable search backends (brave/wiki/local) and the FRAMES + MedQA local indexes for money-free, low-noise retrieval experiments."
metadata: 
  node_type: memory
  type: project
  originSessionId: 10839a4a-2700-438c-b076-2010e126f420
---

To run search experiments without paying the Brave API and to reduce retrieval
noise, the agent search tool is now pluggable. All three implement the same
duck-typed contract `search(query, max_results) -> [{"title","snippet"}]`:

- `src/services/brave_search.py` — paid Brave web search (original).
- `src/services/wiki_search.py` — free live MediaWiki search (`WikipediaSearchService`). No key, cached in `.wiki_cache/`. Returns short keyword-highlighted snippets.
- `src/services/local_index_search.py` — `LocalIndexSearchService`, offline BM25/dense index over a local passage corpus. Returns full passages.

`scripts/run_qa_eval_experiment.py` and `scripts/run_frames_example.py` select via
`--search-backend {brave,wiki,local}` (+ `--index-dir`, `--local-backend {bm25,dense}`).

**FRAMES** is registered in `qa_eval.py._load_dataset` (`google/frames-benchmark`,
824-row test split, maps `Prompt`/`Answer` → `problem`/`gold answer`). No gold hop
decompositions, so it supports RQ1 search-volume + accuracy only, NOT the per-hop
E/CP/PR/M taxonomy. **Why:** FRAMES is the external-validity check for the
search-collapse claim (see [[project_missed_hop_paradox]]).

**Local FRAMES corpus** built by `scripts/build_frames_index.py` into
`data/frames_index/` (corpus.jsonl, titles.json, manifest.json, pages_cache/):
2476 gold pages + 1× random distractors → ~79k passages, ~99% gold coverage.
`titles.json` makes rebuilds reproducible/convergent.

**Demonstrated payoff:** on FRAMES example 0, swapping live-wiki → local BM25 cut
gemini-flash searches 41→15 and flipped a waffling answer to the correct one. Live
snippets are too obscured and cause redundant search loops.

**Pitfall (cost real time):** the MediaWiki `prop=extracts` endpoint gets throttled
under burst concurrency and returns EMPTY extracts. Never cache empty results and
keep fetch workers modest (~6) — otherwise empties get cached permanently and tank
gold coverage (~27% loss before the fix).

Reproducibility caveat: random distractors are only stable because `titles.json`
is persisted; deleting it (or `--refresh-distractors`) re-samples a new set.

**Local MedQA corpus** (built 2026-07-10, `scripts/build_medqa_index.py` →
`data/medqa_index/`): 125,065 passages from `MedRAG/textbooks`, the 18 medical textbooks
MedQA-USMLE was written from. **Far easier than FRAMES**: no gold-page identification (the corpus
is question-set-aligned by construction), **no distractors** (18 whole textbooks are ~95%
irrelevant to any one question), no fetching/caching, and no chunking — the HF corpus is already
capped at 999 chars. Deterministic and API-key-free, so *more* reproducible than the FRAMES index.
Validated by `scripts/validate_medqa_index.py`. Smoke: 5 Qs, gemini-3.1-pro-preview, baseline agent
→ acc 0.80, 3.8 search calls, zero Brave. Design + numbers: `docs/PLAN_medqa_local_index.md`.

**Pitfall — do NOT validate MedQA with answer-string recall.** It is *phrasing-bound*, not
content-bound: gold answers are compositional management phrases ("Reassure the mother",
"IV fluids and monitoring") present verbatim in only ~54% of the 98MB corpus, and the wrong options
come from the same phrasing distribution and appear at the same ~56% rate. So the
gold-vs-distractor contrast **cannot come out positive even for a perfect index** — the metric has
no discriminative power by construction, and its sign at floor-level counts is noise. Judge index
health by structural sanity + topical eyeball dump + an end-to-end run with nonzero
`sampler_search_calls`.

**Pitfall — `search()` latency scales with QUERY length, not corpus size.** ~0.6 s for a ~15-term
query vs **~5.8 s** for a raw 150-term MedQA vignette (rank_bm25 loops per query term). Agent
queries are short (0.5–1.3 s observed), so eval is cheap; but a 100-question raw-vignette
validation takes ~10 min. Don't mistake this for a corpus-size problem.

**Comparability pitfall:** switching brave → local invalidates cross-run comparability (search
volume lives on a different scale under a different retriever). Local runs need their OWN baselines;
never mix backends in one contrast — this blocks reusing the Brave 2×3 cue grid
([[project_cue_search_truncation_smoke]]).

**facts_open is NOT yet unlocked** and is much harder: `facts_open_filtered.csv` has only
`example_id, problem, gold answer` — no source URLs — and bridge entities are *described, not
named*, so gold pages must be mined from existing Brave traces + LLM annotation, then
coverage-validated.
