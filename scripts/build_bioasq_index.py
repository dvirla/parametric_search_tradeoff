"""
Build a local search corpus for BioASQ from rag-datasets/rag-mini-bioasq's text-corpus config.

The official BioASQ task data (bigbio/bioasq_task_b) requires free registration at bioasq.org to
download -- HF even disables its dataset viewer for it. rag-datasets/rag-mini-bioasq sidesteps
this entirely: it's purpose-built for RAG eval, with a `text-corpus` config (40221 PubMed
passages: {id, passage}) kept SEPARATE from the `question-answer-passages` QA config (4719 rows:
{id, question, answer, relevant_passage_ids}), already linked by numeric passage id. No fetching,
no distractor synthesis, no API keys -- this script only filters and formats.

CAVEAT (discovered empirically, 2026-08-20): ~30% of text-corpus rows have passage == the
literal string "nan" (missing source text), which also means ~30% of relevant_passage_ids
references across the QA set point to text that isn't retrievable by any index -- this caps
achievable gold-passage recall well below 100% regardless of retrieval quality. --min-chars
(default 40, comfortably above len("nan")==3) drops these rows during the build; validate with
scripts/validate_bioasq_index.py, which reports recall against only the SURVIVING gold ids.

`doc_id` is kept as the corpus's OWN numeric passage id (NOT resequenced, unlike FRAMES/MedQA),
so it's directly comparable to `relevant_passage_ids` in the QA config for recall validation.

Writes:
    <index-dir>/corpus.jsonl      one passage per line {doc_id,title,passage_id,text}
    <index-dir>/manifest.json     build stats / config
    <index-dir>/embeddings.npy    (only with --backend dense|both) passage embeddings

The corpus is then served by src/services/local_index_search.LocalIndexSearchService.

Usage:
    uv run python scripts/build_bioasq_index.py
    uv run python scripts/build_bioasq_index.py --backend both      # also embed
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset


def setup_args():
    p = argparse.ArgumentParser(description="Build the local BioASQ search corpus.")
    p.add_argument("--index-dir", default="data/bioasq_index", help="Output directory.")
    p.add_argument("--min-chars", type=int, default=40,
                   help="Drop passages shorter than this (also drops the literal 'nan' rows -- see CAVEAT above).")
    p.add_argument("--backend", choices=["bm25", "dense", "both"], default="bm25",
                   help="bm25 needs no extra artifacts; dense/both also writes embeddings.npy.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def main():
    args = setup_args()
    os.makedirs(args.index_dir, exist_ok=True)

    print("Loading rag-datasets/rag-mini-bioasq (text-corpus config, passages split)...")
    ds = load_dataset("rag-datasets/rag-mini-bioasq", "text-corpus", split="passages")
    print(f"  {len(ds)} rows.")

    print("Filtering short/missing passages + writing corpus.jsonl...")
    corpus_path = os.path.join(args.index_dir, "corpus.jsonl")
    n_passages = 0
    n_dropped_short = 0
    with open(corpus_path, "w") as out:
        for row in ds:
            text = (row["passage"] or "").strip()
            if len(text) < args.min_chars:
                n_dropped_short += 1
                continue
            out.write(json.dumps({
                "doc_id": row["id"], "title": f"PubMed passage {row['id']}",
                "passage_id": 0, "text": text,
            }) + "\n")
            n_passages += 1

    manifest = {
        "n_source_rows": len(ds),
        "n_passages": n_passages,
        "n_dropped_short": n_dropped_short,
        "min_chars": args.min_chars,
        "source_dataset": "rag-datasets/rag-mini-bioasq (text-corpus)",
    }
    print(f"  wrote {n_passages} passages ({n_dropped_short} rows dropped for len < {args.min_chars}, "
          f"incl. ~30% literal-'nan' rows -- see docstring CAVEAT).")

    if args.backend in ("dense", "both"):
        print(f"Embedding passages with {args.dense_model} (this is slow on CPU)...")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        is_e5 = "e5" in args.dense_model.lower()
        passage_prefix = "passage: " if is_e5 else ""
        query_prefix = "query: " if is_e5 else ""
        texts = [passage_prefix + json.loads(l)["text"] for l in open(corpus_path)]
        model = SentenceTransformer(args.dense_model)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True).astype("float32")
        np.save(os.path.join(args.index_dir, "embeddings.npy"), emb)
        manifest["dense_model"] = args.dense_model
        manifest["dense_dim"] = int(emb.shape[1])
        manifest["query_prefix"] = query_prefix
        print(f"  saved embeddings.npy {emb.shape} (query_prefix={query_prefix!r})")

    with open(os.path.join(args.index_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Corpus at {args.index_dir}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
