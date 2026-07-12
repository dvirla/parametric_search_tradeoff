"""Embed unique question texts into an npz for the search-call regressors.

Writes an npz with parallel arrays 'texts' and 'embeddings'. Two input modes:

  --table  results/search_call_prediction/frames_reconstructed_table.csv  --text-col prompt
        Embed the reconstructed *templated* prompts (the actual string the model saw). This is
        the mode the FFN pipeline uses. Use --text-col problem for the un-templated baseline.

  (default) assemble the frames_cues_full tree via load_all() and embed the `problem` column
        — the legacy behaviour kept for the sklearn predictor.

Qwen3-Embedding is instruction-aware: --instruction prefixes every text with a task prompt in the
model's `Instruct: {task}\nQuery: ` query format (only the query side needs the instruction). On a
GPU box, pass --device cuda --dtype bfloat16 to load the 8B model.

Usage (templated, Qwen3-8B, GPU):
    uv run python scripts/embed_questions.py \
        --table results/search_call_prediction/frames_reconstructed_table.csv --text-col prompt \
        --model Qwen/Qwen3-Embedding-8B --device cuda --dtype bfloat16 \
        --instruction "Given a question posed to an AI assistant, represent it to predict how many external web-search calls the assistant will make to answer it, paying attention to phrasing cues about answer length, format, directness, and confidence." \
        --output results/search_call_prediction/embeddings_qwen3-8b_templated.npz

    # un-templated baseline (same model + instruction, `problem` column)
    uv run python scripts/embed_questions.py \
        --table results/search_call_prediction/frames_reconstructed_table.csv --text-col problem \
        --model Qwen/Qwen3-Embedding-8B --device cuda --dtype bfloat16 \
        --instruction "<same task>" \
        --output results/search_call_prediction/embeddings_qwen3-8b_untemplated.npz
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.predict_search_calls import load_all, DEFAULT_RESULTS, DEFAULT_CUES_DIR  # noqa: E402


def collect_texts(args) -> list[str]:
    if args.table:
        df = pd.read_csv(args.table)
        if args.text_col not in df.columns:
            raise SystemExit(f"--text-col '{args.text_col}' not in {args.table} "
                             f"(columns: {list(df.columns)})")
        texts = df[args.text_col].dropna().astype(str).unique().tolist()
    else:
        df = load_all(args.results_dir, args.cues_dir)
        texts = df[args.text_col].dropna().astype(str).unique().tolist()
    texts = sorted(texts)
    if args.limit:
        texts = texts[: args.limit]
    return texts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default=None,
                    help="CSV with a text column to embed (e.g. frames_reconstructed_table.csv). "
                         "If omitted, assemble frames_cues_full via load_all().")
    ap.add_argument("--text-col", default="problem",
                    help="column to embed: 'prompt' (templated) or 'problem' (un-templated)")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--cues-dir", default=DEFAULT_CUES_DIR)
    ap.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    ap.add_argument("--instruction", default=None,
                    help="instruction prompt for instruction-aware models (Qwen3-Embedding)")
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--dtype", default=None, choices=["float16", "bfloat16", "float32"],
                    help="torch dtype for model weights (use bfloat16 for the 8B on GPU)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-seq-length", type=int, default=None,
                    help="override the model's max sequence length (Qwen defaults are large)")
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N texts (smoke test)")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    texts = collect_texts(args)
    print(f"{len(texts)} unique texts to embed with {args.model} "
          f"(col={args.text_col}, device={args.device}, dtype={args.dtype})")

    st_kwargs = {"device": args.device, "trust_remote_code": True}
    if args.dtype:
        st_kwargs["model_kwargs"] = {"torch_dtype": args.dtype}
    model = SentenceTransformer(args.model, **st_kwargs)
    if args.max_seq_length:
        model.max_seq_length = args.max_seq_length

    enc = {"batch_size": args.batch_size, "show_progress_bar": True,
           "normalize_embeddings": True}
    if args.instruction:
        # Qwen3-Embedding query format: "Instruct: {task}\nQuery: {text}"
        enc["prompt"] = f"Instruct: {args.instruction}\nQuery: "
    emb = model.encode(texts, **enc)
    emb = np.asarray(emb, dtype=np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez_compressed(args.output, texts=np.array(texts, dtype=object), embeddings=emb)
    print(f"wrote {emb.shape} embeddings to {args.output}")


if __name__ == "__main__":
    main()
