---
name: project_search_backends
description: "Pluggable search backends (brave/wiki/local) and the FRAMES local Wikipedia index for money-free, low-noise retrieval experiments."
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
