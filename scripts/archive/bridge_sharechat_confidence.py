"""
Merge per-model ShareChat atomic_fact_confidence CSVs into a single unified CSV
that unified_interplay_analysis.py's load_sharechat() can consume.

Usage:
    # Gemini only (today):
    uv run python scripts/bridge_sharechat_confidence.py \
      --confidence-csv gemini-3-pro-preview:results/curated_sharechat/atomic_fact_confidence_gemini.csv \
      --output-dir results/curated_sharechat/interplay_analysis

    # All three models (once qwen/nemotron CSVs are generated):
    uv run python scripts/bridge_sharechat_confidence.py \
      --confidence-csv gemini-3-pro-preview:results/curated_sharechat/atomic_fact_confidence_gemini.csv \
      --confidence-csv qwen3.5:122b:results/curated_sharechat/atomic_fact_confidence_qwen.csv \
      --confidence-csv nemotron-3-nano:30b:results/curated_sharechat/atomic_fact_confidence_nemotron.csv \
      --output-dir results/curated_sharechat/interplay_analysis
"""

import argparse
import csv
import os
import sys

import pandas as pd


def parse_model_path(spec: str) -> tuple[str, str]:
    """Split 'model_name:path' on the first colon only (model names may contain colons)."""
    idx = spec.index(":")
    return spec[:idx], spec[idx + 1 :]


def normalize_model(name: str) -> str:
    return name.replace(":", "_")


def load_one(model_name: str, path: str) -> pd.DataFrame:
    csv.field_size_limit(sys.maxsize)
    df = pd.read_csv(path)

    if "entropy" not in df.columns:
        raise ValueError(
            f"'{path}' is missing the 'entropy' column — "
            "only confidence CSVs (not attribution CSVs) are accepted."
        )

    norm = normalize_model(model_name)
    df["model"] = norm
    df["search_calls"] = df["search_calls"].fillna(0).astype(int)

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-model ShareChat confidence CSVs into a unified CSV for unified_interplay_analysis.py."
    )
    parser.add_argument(
        "--confidence-csv",
        metavar="MODEL:PATH",
        action="append",
        required=True,
        dest="confidence_csvs",
        help="model_name:path pair (repeat for each model). Colons in model names are fine.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write sharechat_confidence.csv into.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for spec in args.confidence_csvs:
        model_name, path = parse_model_path(spec)
        print(f"  Loading {model_name} from {path} ...")
        df = load_one(model_name, path)
        print(f"    {len(df):,} rows")
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True, sort=False)

    out_path = os.path.join(args.output_dir, "sharechat_confidence.csv")
    combined.to_csv(out_path, index=False)

    print(f"\nWrote {len(combined):,} rows to {out_path}")
    print("\nRows per model:")
    print(combined["model"].value_counts().to_string())


if __name__ == "__main__":
    main()
