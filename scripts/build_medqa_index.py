"""
Build a local search corpus for MedQA from the MedRAG/textbooks corpus.

Unlike FRAMES, MedQA has no per-question gold Wikipedia links to fetch. Instead
MedQA-USMLE was constructed to be answerable from 18 English medical textbooks,
and the MedRAG project publishes a pre-chunked version of exactly that corpus on
HuggingFace (`MedRAG/textbooks`). The corpus is already chunked to <=999 chars
per passage, so this script does NOT re-chunk, fetch, cache, or add distractors
(the 18 whole textbooks are ~95% irrelevant to any given question, so natural
distractors come free). It only validates, prettifies titles, drops overly-short
passages, and writes:

    <index-dir>/corpus.jsonl      one passage per line {doc_id,title,passage_id,text,source_id}
    <index-dir>/manifest.json     build stats / config
    <index-dir>/embeddings.npy    (only with --backend dense|both) passage embeddings

The corpus is then served by src/services/local_index_search.LocalIndexSearchService.

Usage:
    uv run python scripts/build_medqa_index.py
    uv run python scripts/build_medqa_index.py --backend both      # also embed
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset

# Upstream book keys (opaque, some misspelled: "Obstentrics", "Psichiatry",
# "Lippinco") mapped to human-readable titles surfaced to the agent via
# LocalIndexSearchService.search()'s "title" field.
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


def setup_args():
    p = argparse.ArgumentParser(description="Build the local MedQA search corpus.")
    p.add_argument("--index-dir", default="data/medqa_index", help="Output directory.")
    p.add_argument("--min-chars", type=int, default=120, help="Drop passages shorter than this.")
    p.add_argument("--backend", choices=["bm25", "dense", "both"], default="bm25",
                   help="bm25 needs no extra artifacts; dense/both also writes embeddings.npy.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def main():
    args = setup_args()
    os.makedirs(args.index_dir, exist_ok=True)

    print("Loading MedRAG/textbooks (train split)...")
    ds = load_dataset("MedRAG/textbooks", split="train")
    print(f"  {len(ds)} rows.")

    # Validate the title set against BOOK_TITLES before touching anything else —
    # fail loudly on drift rather than silently passing an unmapped key through.
    dataset_titles = set(ds.unique("title"))
    expected_titles = set(BOOK_TITLES.keys())
    if dataset_titles != expected_titles:
        missing = expected_titles - dataset_titles
        extra = dataset_titles - expected_titles
        raise AssertionError(
            "MedRAG/textbooks title set does not match BOOK_TITLES.\n"
            f"  missing from dataset: {sorted(missing)}\n"
            f"  unexpected in dataset: {sorted(extra)}\n"
            "Update BOOK_TITLES in scripts/build_medqa_index.py to match."
        )
    print(f"  title set matches BOOK_TITLES ({len(expected_titles)} books).")

    # doc_id is a stable integer index over sorted(BOOK_TITLES), independent of
    # dataset row order.
    sorted_book_keys = sorted(BOOK_TITLES.keys())
    doc_id_by_book = {book: i for i, book in enumerate(sorted_book_keys)}

    print("Filtering short passages + writing corpus.jsonl...")
    corpus_path = os.path.join(args.index_dir, "corpus.jsonl")
    passage_id_by_book = {book: 0 for book in sorted_book_keys}
    n_passages = 0
    n_dropped_short = 0
    with open(corpus_path, "w") as out:
        for row in ds:
            book = row["title"]
            text = (row["content"] or "").strip()
            if len(text) < args.min_chars:
                n_dropped_short += 1
                continue
            pid = passage_id_by_book[book]
            passage_id_by_book[book] = pid + 1
            out.write(json.dumps({
                "doc_id": doc_id_by_book[book],
                "title": BOOK_TITLES[book],
                "passage_id": pid,
                "text": text,
                "source_id": row["id"],
            }) + "\n")
            n_passages += 1

    manifest = {
        "n_books": len(sorted_book_keys),
        "n_passages": n_passages,
        "n_dropped_short": n_dropped_short,
        "min_chars": args.min_chars,
        "source_dataset": "MedRAG/textbooks",
    }
    print(f"  wrote {n_passages} passages ({n_dropped_short} rows dropped for len < {args.min_chars}).")

    if args.backend in ("dense", "both"):
        print(f"Embedding passages with {args.dense_model} (this is slow on CPU)...")
        import numpy as np
        from sentence_transformers import SentenceTransformer
        # E5-family models require asymmetric prefixes: "passage: " on documents
        # and "query: " on queries. Auto-detect by model name.
        is_e5 = "e5" in args.dense_model.lower()
        passage_prefix = "passage: " if is_e5 else ""
        query_prefix = "query: " if is_e5 else ""
        texts = [passage_prefix + json.loads(l)["text"] for l in open(corpus_path)]
        model = SentenceTransformer(args.dense_model)
        emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True).astype("float32")
        np.save(os.path.join(args.index_dir, "embeddings.npy"), emb)
        # Record the embedding model, dim, and query prefix so the search service
        # encodes queries with the *same* model + convention (otherwise FAISS
        # raises a dimension mismatch / retrieval silently degrades).
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
