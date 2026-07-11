# PLAN — Local BM25 index for MedQA (free search tool calls)

**Goal.** Give MedQA the same money-free, low-noise offline retrieval that FRAMES has
(`data/frames_index/` + `LocalIndexSearchService`), so cue/behavioral experiments can be run
without Brave API spend or rate limits.

**Status:** planned 2026-07-10. facts_open is a *separate, harder* follow-up (no gold source
links in `facts_open_filtered.csv`) — not in scope here.

---

## Why MedQA is easy

FRAMES shipped gold `wiki_links` per question, so `build_frames_index.py` fetched exactly those
pages + random distractors. MedQA has no such field — but it doesn't need one. MedQA-USMLE was
constructed to be answerable from **18 English medical textbooks**, and the MedRAG project
publishes a pre-chunked version of exactly that corpus on HuggingFace. The corpus is
question-set-aligned *by construction*, so there is:

- no per-question gold-page identification,
- no MediaWiki scraping (and none of the `prop=extracts` empty-cache throttling pitfall that cost
  real time on FRAMES — see `project_search_backends` memory),
- **no distractor sampling** — 18 whole textbooks are ~95% irrelevant to any given question, so
  natural distractors come free. Do *not* add synthetic distractors.

## Verified facts (measured 2026-07-10, do not re-derive)

`load_dataset("MedRAG/textbooks", split="train")`:

| Property | Value |
|---|---|
| rows | 125,847 |
| features | `id` (str), `title` (str), `content` (str), `contents` (str) |
| `title` | the **book name only**, e.g. `Anatomy_Gray` — 18 distinct values, not a per-passage title |
| `contents` | `f"{title}. {content}"` — redundant, ignore it; index `content` |
| `content` length | mean 777 chars, median 865, p90 997, **max 999** |
| passages < 120 chars | 782 |
| download | 18 parquet files, a few seconds, no auth |

The 18 titles: `Anatomy_Gray, Biochemistry_Lippinco, Cell_Biology_Alberts, First_Aid_Step1,
First_Aid_Step2, Gynecology_Novak, Histology_Ross, Immunology_Janeway, InternalMed_Harrison,
Neurology_Adams, Obstentrics_Williams, Pathology_Robbins, Pathoma_Husain, Pediatrics_Nelson,
Pharmacology_Katzung, Physiology_Levy, Psichiatry_DSM-5, Surgery_Schwartz`
(sic — `Obstentrics`, `Psichiatry`, `Lippinco` are misspelled upstream; treat as opaque keys).

**Because `max(len(content)) == 999`, the corpus is already chunked to the same 1000-char budget
FRAMES uses.** Do not re-chunk. Emit passages as-is; only drop `len < --min-chars`.

Cost profile at 125,847 passages (measured, in-process):

| Step | Cost |
|---|---|
| tokenize | 4.7 s |
| `BM25Okapi(...)` build | 3.6 s |
| peak RSS | 2.1 GB |
| `get_scores()` per query | **~0.6 s** (long clinical-vignette query, ~15 terms) |

0.6 s/query is the one number to be aware of: startup is trivial but per-search latency is not
free. At Gemini's observed ~14 searches/question that's ~8 s/question of pure BM25. Acceptable —
in-memory `rank_bm25` stays. If it ever bites, swap `rank_bm25` → `bm25s` behind the existing
`LocalIndexSearchService._init_bm25` / `_bm25_search` seam; **do not do this pre-emptively.**

## What already works — do not touch

- `src/services/local_index_search.py` — `LocalIndexSearchService(index_dir, backend=...)` reads
  `corpus.jsonl` and uses **only** the `title` and `text` fields of each row. It is corpus-agnostic.
- `scripts/run_qa_eval_experiment.py` — already exposes `--search-backend local --index-dir <dir>
  --local-backend {bm25,dense}` (lines 27–30, 118–125). `--index-dir` defaults to
  `data/frames_index` but is a free parameter.

⇒ **No changes to either file.** The entire deliverable is a build script + a validation script.

## Output contract

`data/medqa_index/` mirroring `data/frames_index/`:

```
corpus.jsonl    one passage per line: {"doc_id","title","passage_id","text","source_id"}
manifest.json   build stats / config
```

- `doc_id` — integer index of the **book** (0–17), stable under the sorted title order.
- `passage_id` — integer index of the passage **within its book**, in dataset order.
- `source_id` — the upstream `id` (e.g. `Anatomy_Gray_0`), kept for traceability.
- `title` — the **prettified** book name (see below).
- `text` — the upstream `content`, verbatim.
- No `is_gold` field (meaningless here; the FRAMES writer emits it, we omit it — the search
  service never reads it).

### Title prettification (required)

`LocalIndexSearchService.search()` surfaces `title` straight into the agent's tool result as the
result's title. `Obstentrics_Williams` is noise in a model's context. Map to human names:

```python
BOOK_TITLES = {
    "Anatomy_Gray":          "Gray's Anatomy",
    "Biochemistry_Lippinco": "Lippincott's Illustrated Reviews: Biochemistry",
    "Cell_Biology_Alberts":  "Molecular Biology of the Cell (Alberts)",
    "First_Aid_Step1":       "First Aid for the USMLE Step 1",
    "First_Aid_Step2":       "First Aid for the USMLE Step 2 CK",
    "Gynecology_Novak":      "Berek & Novak's Gynecology",
    "Histology_Ross":        "Ross & Pawlina Histology",
    "Immunology_Janeway":    "Janeway's Immunobiology",
    "InternalMed_Harrison":  "Harrison's Principles of Internal Medicine",
    "Neurology_Adams":       "Adams and Victor's Principles of Neurology",
    "Obstentrics_Williams":  "Williams Obstetrics",
    "Pathology_Robbins":     "Robbins & Cotran Pathologic Basis of Disease",
    "Pathoma_Husain":        "Pathoma: Fundamentals of Pathology",
    "Pediatrics_Nelson":     "Nelson Textbook of Pediatrics",
    "Pharmacology_Katzung":  "Katzung's Basic & Clinical Pharmacology",
    "Physiology_Levy":       "Berne & Levy Physiology",
    "Psichiatry_DSM-5":      "DSM-5",
    "Surgery_Schwartz":      "Schwartz's Principles of Surgery",
}
```

Assert at build time that the dataset's title set == `BOOK_TITLES.keys()`; fail loudly on drift
rather than silently passing an unmapped key through.

---

## Task 1 — `scripts/build_medqa_index.py`

Model it on `scripts/build_frames_index.py` (same CLI shape, same manifest/dense conventions), but
much simpler: no fetching, no caching, no distractors, no chunking.

```
uv run python scripts/build_medqa_index.py                     # bm25, default
uv run python scripts/build_medqa_index.py --backend both      # also embeddings.npy
```

Args: `--index-dir data/medqa_index`, `--min-chars 120`, `--backend {bm25,dense,both}` (default
`bm25`), `--dense-model sentence-transformers/all-MiniLM-L6-v2`.

Steps:
1. `load_dataset("MedRAG/textbooks", split="train")`.
2. Validate the title set against `BOOK_TITLES` (hard fail on mismatch).
3. Assign `doc_id` from `sorted(BOOK_TITLES)`; `passage_id` counts within each book in dataset order.
4. Drop `len(content.strip()) < min_chars` (~782 rows) and any empty/whitespace-only content.
5. Write `corpus.jsonl` in the contract above.
6. `--backend dense|both`: **reuse `build_frames_index.py`'s dense block verbatim in spirit** —
   E5-family prefix auto-detection (`passage: ` / `query: `), `embeddings.npy` as float32, and
   record `dense_model`, `dense_dim`, `query_prefix` in the manifest. `LocalIndexSearchService`
   depends on those three manifest keys; omitting them silently degrades dense retrieval.
7. `manifest.json`: `n_books`, `n_passages`, `n_dropped_short`, `min_chars`, `source_dataset`,
   plus the dense keys when applicable.

Dense embedding of 125k passages on CPU is slow (tens of minutes). BM25 is the default and the
only backend needed for the immediate experiments — do not run `--backend both` as part of this task.

## Task 2 — `scripts/validate_medqa_index.py`

Independent of Task 1's internals; depends only on the output contract above. Purpose: prove the
corpus actually supports MedQA before any eval run, so a retrieval hole can never be mistaken for
a behavioral effect.

```
uv run python scripts/validate_medqa_index.py --index-dir data/medqa_index --n 100 --k 10
```

Load MedQA the same way `qa_eval.py` does (`GBaker/MedQA-USMLE-4-options`, split `test`, fields
renamed `question`→`problem`, `answer`→`gold answer`), take a seeded sample of `--n`, and report:

1. **Sanity** — `n_passages`, `n_books`, manifest agreement with `corpus.jsonl` line count.
2. **Answer-string recall@k** — fraction of sampled questions where the lowercased `gold answer`
   occurs as a substring in any of the top-`k` passages retrieved for the **raw question text**.
3. **Distractor contrast** — same recall computed for the three *wrong* options (from `options` /
   `answer_idx`). ~~This is the load-bearing metric.~~ **← This was wrong. See "Correction" below.**
4. **Latency** — mean seconds per `search()` call.
5. Dump 5 sampled (question, top-3 titles, top-1 snippet head) triples for eyeballing.

⚠️ **Interpretation caveat, state it in the script's output.** Answer-string recall is a *weak
lower bound* for MedQA: gold answers are often management steps or diagnoses phrased differently in
a textbook ("Reassurance and follow-up"), and the raw multi-sentence vignette is a poor BM25 query
— the agent issues short focused queries instead. A low absolute recall is **not** grounds to
reject the corpus. Do not add a `sys.exit(1)` gate on an arbitrary recall number.

### ❗ Correction (measured 2026-07-10, after the script was written)

The distractor contrast **cannot work for MedQA**, and the plan was wrong to name it the
load-bearing metric. Measured on the real index (n=100, k=10, seed=0, bm25):

| metric | gold | distractors |
|---|---|---|
| present **anywhere** in the 98 MB corpus | 0.54 | 0.56 |
| recall@10, raw-vignette query | 0.010 | 0.027 |
| recall@10, focused last-sentence query (n=50) | 0.06 | — |

The wrong options are drawn from the **same phrasing distribution** as the gold answer, so they
appear in the corpus at the **same rate**. Substring matching therefore measures answer *phrasing*,
not corpus *content*, and the contrast has **no discriminative power by construction** — it cannot
come out positive even for a perfect index. At floor-level counts (1 gold hit; 8 distractor hits
out of 300) its sign is noise. Focusing the query barely moves it (0.00 → 0.06). Gold answers are
compositional management phrases — `"Moxifloxacin and admission to the medical floor"`,
`"Reassure the mother"`, `"IV fluids and monitoring"` — that textbooks state in other words.

`validate_medqa_index.py` was updated accordingly: substring recall is demoted to a labelled
**regression tripwire**, a **corpus-coverage** section (present-anywhere, gold vs distractor) was
added to separate "corpus lacks content" from "query was bad", and the misleading `PASS SIGNAL` /
`WARNING` verdict was removed. Index health rests on structural sanity, the topical eyeball dump,
and the end-to-end smoke.

## Task 3 — DONE (2026-07-10)

Build ran clean: **125,065 passages** (125,847 − 782 short), manifest agrees with the actual
`corpus.jsonl` line count, `doc_id` is 1:1 with book, `passage_id` contiguous within book, text
lengths bounded to [120, 999], no raw upstream keys leaking into `title`.

End-to-end smoke (5 questions, `baseline` agent, `gemini-3.1-pro-preview`, grader
`gemini-3-flash-preview`, `--query_template plain`, **zero Brave calls**):

```
uv run python scripts/run_qa_eval_experiment.py \
  --dataset medqa --agent_type baseline \
  --search-backend local --index-dir data/medqa_index --local-backend bm25 \
  --model_name gemini-3.1-pro-preview --provider_name Google \
  --grader_model gemini-3-flash-preview --grader_provider Google \
  --query_template plain --num_examples 5 --num_workers 1 --seed 0 \
  --run_name smoke_local --output_dir results/medqa_local_smoke
```

→ `correct = 0.80`, `search_calls = 3.8` (mean). **The index is live and the agent uses it.**

**Latency, corrected.** `search()` cost scales with *query length*, not corpus size:
~0.6 s for a ~15-term query, but **~5.8 s** for a raw 150-term MedQA vignette. Real agent queries
are short (observed 0.5–1.3 s/call in the smoke run), so eval cost is fine — but a `--n 100`
validation run over raw vignettes takes ~10 min. Don't be alarmed by it.

**Retrieval quality is topical, not always precise.** Some raw-vignette queries retrieve
off-target passages (a paediatric jaundice vignette pulled DNA-polymerase text). This is a
property of stuffing a whole vignette into BM25 and does not reflect what the agent does.

---

## Experimental caveat (carry into any writeup)

Switching Brave → local BM25 **invalidates comparability with the completed 2×3 Brave cue grid**
(`project_cue_search_truncation_smoke`). On FRAMES, live-wiki → local BM25 moved one example's
search count 41 → 15. Search-volume lives on a different scale under a different retriever, so
local-index runs need their **own PLAIN baselines**; never mix the two in one contrast.
