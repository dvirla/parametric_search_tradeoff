"""Embed the unique question texts from the frames_cues_full results tree.

Writes an npz with parallel arrays 'texts' and 'embeddings' for predict_search_calls.py
--embeddings. All four models saw the same ~7k unique paraphrased questions, so one
embedding pass serves every per-model regression.

Usage:
    uv run python scripts/embed_questions.py \
        --model BAAI/bge-base-en-v1.5 \
        --output results/search_call_prediction/embeddings_bge-base.npz

    # Qwen3-Embedding is instruction-aware; --instruction prefixes each text with a task prompt
    uv run python scripts/embed_questions.py \
        --model Qwen/Qwen3-Embedding-0.6B \
        --instruction "Represent this question for predicting how much external search an LLM assistant would need to answer it" \
        --output results/search_call_prediction/embeddings_qwen3-0.6b.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.predict_search_calls import load_all, DEFAULT_RESULTS, DEFAULT_CUES_DIR  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--cues-dir", default=DEFAULT_CUES_DIR)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--instruction", default=None,
                    help="instruction prompt for instruction-aware models (Qwen3-Embedding)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    df = load_all(args.results_dir, args.cues_dir)
    texts = sorted(df.problem.unique())
    print(f"{len(texts)} unique texts to embed with {args.model}")

    model = SentenceTransformer(args.model, device="cpu")
    kwargs = {"batch_size": args.batch_size, "show_progress_bar": True,
              "normalize_embeddings": True}
    if args.instruction:
        kwargs["prompt"] = f"Instruct: {args.instruction}\nQuery: "
    emb = model.encode(texts, **kwargs)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, texts=np.array(texts, dtype=object), embeddings=emb)
    print(f"wrote {emb.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
