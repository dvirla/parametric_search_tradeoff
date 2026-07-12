"""Reconstruct the full templated prompt each model actually saw, per FRAMES cue-grid row.

The persisted `problem` field in results/frames_cues_full is the *un-templated* question: the
6 output-directive cues (plain/natural/elaborate/polite/direct/query) are applied at runtime by
qa_eval.py and never saved, so there are only ~2004 unique `problem` texts per model (verbose_plain
and verbose_polite share an identical `problem`). Any text feature of `problem` is therefore blind
to the exact output-directive cue that drives search behavior.

This script re-applies the qa_eval template constants (imported, not re-typed, so they stay
byte-identical to what the model saw) to `problem`, keyed by the cue token parsed from each result
filename, producing a `prompt` column = the actual string the model received. That yields 14
distinct prompts per question (7014 unique texts/model).

Mapping (frames-cues is in PLAIN_QUERY_DATASETS, so *_plain and epi_* ran bare):
    *_plain      -> PLAIN_QUERY_TEMPLATE      (identity)
    *_natural    -> NATURAL_QUERY_TEMPLATE
    *_elaborate  -> ELABORATE_QUERY_TEMPLATE
    *_polite     -> POLITE_QUERY_TEMPLATE
    *_direct     -> DIRECT_QUERY_TEMPLATE
    *_query      -> QUERY_TEMPLATE
    epi_strong_* -> PLAIN_QUERY_TEMPLATE      (identity; cue baked into `problem`)

Usage:
    uv run python scripts/build_reconstructed_prompts.py \
        --results-dir results/frames_cues_full \
        --out results/search_call_prediction/frames_reconstructed_table.csv
"""
import argparse
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.predict_search_calls import load_all, DEFAULT_RESULTS, DEFAULT_CUES_DIR  # noqa: E402
from src.services.qa_eval import (  # noqa: E402
    PLAIN_QUERY_TEMPLATE,
    NATURAL_QUERY_TEMPLATE,
    ELABORATE_QUERY_TEMPLATE,
    POLITE_QUERY_TEMPLATE,
    DIRECT_QUERY_TEMPLATE,
    QUERY_TEMPLATE,
)

STYLE_TEMPLATE = {
    "plain": PLAIN_QUERY_TEMPLATE,
    "natural": NATURAL_QUERY_TEMPLATE,
    "elaborate": ELABORATE_QUERY_TEMPLATE,
    "polite": POLITE_QUERY_TEMPLATE,
    "direct": DIRECT_QUERY_TEMPLATE,
    "query": QUERY_TEMPLATE,
}


def reconstruct(condition: str, problem: str) -> str:
    """Apply the runtime query-template named by `condition` to the un-templated `problem`."""
    if condition.startswith("epi_"):
        tmpl = PLAIN_QUERY_TEMPLATE
    else:
        # condition is "<phrasing>_<style>", e.g. "verbose_polite" / "terse_natural"
        style = condition.split("_", 1)[1]
        if style not in STYLE_TEMPLATE:
            raise ValueError(f"unknown style '{style}' in condition '{condition}'")
        tmpl = STYLE_TEMPLATE[style]
    # str.format only parses the template's own braces; braces inside `problem` are inserted
    # verbatim as data (never re-parsed), so questions containing { } are safe.
    return tmpl.format(Question=problem)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--cues-dir", default=DEFAULT_CUES_DIR)
    ap.add_argument("--out", default="results/search_call_prediction/frames_reconstructed_table.csv")
    args = ap.parse_args()

    df = load_all(args.results_dir, args.cues_dir)
    n0 = len(df)
    df = df[~df.stopped].reset_index(drop=True)
    print(f"assembled {n0} rows; dropped {n0 - len(df)} budget-exhausted; {len(df)} remain")

    df["prompt"] = [reconstruct(c, p) for c, p in zip(df.condition, df.problem)]

    # --- correctness checks -------------------------------------------------
    # per (model, example_id): one distinct prompt per condition (no accidental collisions)
    per = df.groupby(["model", "example_id"]).agg(
        n_cond=("condition", "nunique"), n_prompt=("prompt", "nunique"))
    bad = per[per.n_cond != per.n_prompt]
    assert bad.empty, f"prompt collisions within example groups:\n{bad.head()}"

    # spot-check the suffix each template appends
    def sample(cond):
        s = df[df.condition == cond]
        return s.prompt.iloc[0] if len(s) else ""
    for cond, needle in [
        ("verbose_natural", "Please answer in 2-4 sentences."),
        ("verbose_elaborate", "at least 8-10 sentences"),
        ("verbose_polite", "truly appreciate your time"),
        ("verbose_direct", "final answer only."),
        ("verbose_query", "Exact Answer:"),
    ]:
        p = sample(cond)
        assert needle in p, f"{cond}: expected {needle!r} in reconstructed prompt, got:\n{p[:200]}"
    # plain / epi are identity
    for cond in ("verbose_plain", "terse_plain", "epi_strong_boost", "epi_strong_hedge"):
        s = df[df.condition == cond]
        if len(s):
            assert (s.prompt == s.problem).all(), f"{cond}: prompt != problem (should be identity)"
    print("reconstruction checks passed.")

    for model, sub in df.groupby("model"):
        print(f"  {model:<26} rows={len(sub):>5}  conds={sub.condition.nunique():>2}  "
              f"examples={sub.example_id.nunique():>4}  unique_prompts={sub.prompt.nunique():>5}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
