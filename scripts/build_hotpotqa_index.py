"""
Build a local search corpus for HotpotQA from its own distractor-config contexts.

Unlike FRAMES, HotpotQA needs no MediaWiki fetching: the `distractor` config already ships,
per question, 10 Wikipedia paragraphs (2 gold + 8 distractor) as `context: {title, sentences}`.
This script pools `context` across the ENTIRE validation split (7405 questions) into one shared
corpus -- deduping by title, since the same paragraph recurs across many questions' distractor
sets -- which gives natural cross-question distractors (like FRAMES's pooling) for free. Each
paragraph is already Wikipedia-lead-length, so passages are NOT re-chunked by default; an
oversized paragraph (rare) is split on its own sentence boundaries via --max-chars, not FRAMES's
header-regex chunker (HotpotQA doesn't have section headers -- it's lead paragraphs only).

Writes:
    <index-dir>/corpus.jsonl      one passage per line {doc_id,title,passage_id,text}
    <index-dir>/manifest.json     build stats / config
    <index-dir>/embeddings.npy    (only with --backend dense|both) passage embeddings

The corpus is then served by src/services/local_index_search.LocalIndexSearchService.

Usage:
    uv run python scripts/build_hotpotqa_index.py
    uv run python scripts/build_hotpotqa_index.py --backend both      # also embed
"""

import os
import sys
import json
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datasets import load_dataset


def setup_args():
    p = argparse.ArgumentParser(description="Build the local HotpotQA search corpus.")
    p.add_argument("--index-dir", default="data/hotpotqa_index", help="Output directory.")
    p.add_argument("--split", default="validation", choices=["train", "validation"],
                   help="HotpotQA distractor split to pool contexts from (validation is the eval split; answers are hidden on test).")
    p.add_argument("--max-chars", type=int, default=1500,
                   help="Split an oversized paragraph on sentence boundaries above this length.")
    p.add_argument("--min-chars", type=int, default=40, help="Drop passages shorter than this.")
    p.add_argument("--backend", choices=["bm25", "dense", "both"], default="bm25",
                   help="bm25 needs no extra artifacts; dense/both also writes embeddings.npy.")
    p.add_argument("--dense-model", default="sentence-transformers/all-MiniLM-L6-v2")
    return p.parse_args()


def split_long(text: str, sentences: list[str], max_chars: int):
    """Split a paragraph into <=max_chars chunks on its own sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks, buf = [], ""
    for sent in sentences:
        if buf and len(buf) + len(sent) > max_chars:
            chunks.append(buf.strip())
            buf = sent
        else:
            buf += sent
    if buf.strip():
        chunks.append(buf.strip())
    return chunks


def main():
    args = setup_args()
    os.makedirs(args.index_dir, exist_ok=True)

    print(f"Loading hotpotqa/hotpot_qa (distractor config, {args.split} split)...")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.split)
    print(f"  {len(ds)} questions.")

    print("Pooling context paragraphs (deduping by title)...")
    sentences_by_title: dict[str, list[str]] = {}
    for row in ds:
        ctx = row["context"]
        for title, sentences in zip(ctx["title"], ctx["sentences"]):
            if title in sentences_by_title:
                continue
            if "".join(sentences).strip():
                sentences_by_title[title] = sentences
    print(f"  {len(sentences_by_title)} unique titles pooled from {len(ds)} questions.")

    print("Chunking + writing corpus.jsonl...")
    corpus_path = os.path.join(args.index_dir, "corpus.jsonl")
    n_passages = 0
    n_dropped_short = 0
    with open(corpus_path, "w") as out:
        for doc_id, (title, sentences) in enumerate(sorted(sentences_by_title.items())):
            text = "".join(sentences).strip()
            for pid, chunk_text in enumerate(split_long(text, sentences, args.max_chars)):
                if len(chunk_text) < args.min_chars:
                    n_dropped_short += 1
                    continue
                out.write(json.dumps({
                    "doc_id": doc_id, "title": title, "passage_id": pid, "text": chunk_text,
                }) + "\n")
                n_passages += 1

    manifest = {
        "n_questions_pooled": len(ds),
        "n_unique_titles": len(sentences_by_title),
        "n_passages": n_passages,
        "n_dropped_short": n_dropped_short,
        "max_chars": args.max_chars,
        "min_chars": args.min_chars,
        "split": args.split,
        "source_dataset": "hotpotqa/hotpot_qa (distractor)",
    }
    print(f"  wrote {n_passages} passages ({n_dropped_short} chunks dropped for len < {args.min_chars}).")

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
