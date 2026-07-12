"""Assemble a multi-dataset search-call table (FRAMES + MuSiQue + ShareChat) with reconstructed
templated prompts, for the per-model FFN predictor.

Each dataset contributes rows in the same schema as the FRAMES cue grid:
    dataset, model, condition, example_id (namespaced group key), problem, prompt, search_calls

Template reconstruction (prompt = TEMPLATE.format(Question=problem), imported from qa_eval so the
strings stay byte-identical to what the model saw):
    FRAMES cue grid            -> per-cue template (see build_reconstructed_prompts)
    musique-formal             -> QUERY_TEMPLATE     (structured Exact-Answer format)
    musique-natural            -> NATURAL_QUERY_TEMPLATE  ("Please answer in 2-4 sentences.")
    curated-sharechat          -> NATURAL_QUERY_TEMPLATE
    curated-sharechat-benchmark-> QUERY_TEMPLATE

Group keys (GroupKFold anti-leakage; a question's paraphrases must never straddle train/test):
    frames::<example_id>     — the FRAMES item id (14 cue rows per question)
    musique::<example_id>    — the MuSiQue id; formal carries it, natural recovers it by joining
                               `problem` to data/musique_val_natural.jsonl `text` (paraphrase pair)
    sharechat::<i>           — curated[i] and benchmark[i] are the SAME question (index-aligned;
                               verified topically + identical order across models)

Model names are canonicalised via viz.canonical_model (merges gemini-3-pro/3.1, normalises slugs);
only the FRAMES roster is kept, so fine-tuned nemotron-musique checkpoints are dropped
automatically. Budget-exhausted rows (stop_reason set) are dropped.

Usage:
    uv run python scripts/build_multi_dataset_table.py \
        --out results/search_call_prediction/search_calls_multi_table.csv
"""
import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.predict_search_calls import load_all, DEFAULT_RESULTS, DEFAULT_CUES_DIR  # noqa: E402
from scripts.build_reconstructed_prompts import reconstruct as frames_reconstruct  # noqa: E402
from src.services.qa_eval import QUERY_TEMPLATE, NATURAL_QUERY_TEMPLATE  # noqa: E402
from src.viz import canonical_model  # noqa: E402

# The FRAMES roster (canonicalised). Anything canonicalising outside this set — e.g. the
# fine-tuned nemotron-3-nano-musique-* checkpoints — is dropped.
FRAMES_MODELS = {"gemini-3-pro-preview", "gemma4_31b", "nemotron-3-nano_30b", "qwen3.5_122b"}


def _search_calls(r):
    return int(r.get("sampler_search_calls", 0) or 0)


def _stopped(r):
    return bool(r.get("stop_reason"))


def _model_from_name(path, prefix):
    """Extract the model token between '<prefix>_baseline_' and '_run_1.json'."""
    m = re.match(rf"{re.escape(prefix)}_baseline_(.+)_run_1\.json$", os.path.basename(path))
    return m.group(1) if m else None


# ---------------------------------------------------------------- FRAMES

def load_frames(results_dir, cues_dir):
    df = load_all(results_dir, cues_dir)
    df = df[~df.stopped].reset_index(drop=True)
    df["prompt"] = [frames_reconstruct(c, p) for c, p in zip(df.condition, df.problem)]
    df["dataset"] = "frames"
    df["model"] = df.model.map(canonical_model)
    df["example_id"] = "frames::" + df.example_id.astype(str)
    return df[["dataset", "model", "condition", "example_id", "problem", "prompt", "search_calls"]]


# ---------------------------------------------------------------- MuSiQue

def _musique_natural_id_map(natural_jsonl):
    m = {}
    with open(natural_jsonl) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                m[str(r["text"]).strip()] = str(r["example_id"])
    return m


def load_musique(archive_dir, natural_jsonl):
    rows = []
    nat_map = _musique_natural_id_map(natural_jsonl)
    specs = [
        ("musique-formal", "musique_formal", QUERY_TEMPLATE, None),
        ("musique-natural", "musique_natural", NATURAL_QUERY_TEMPLATE, nat_map),
    ]
    unmatched = {}
    for prefix, condition, tmpl, id_map in specs:
        for path in sorted(glob.glob(os.path.join(archive_dir, prefix, f"{prefix}_baseline_*_run_1.json"))):
            tok = _model_from_name(path, prefix)
            if tok is None:
                continue
            model = canonical_model(tok)
            if model not in FRAMES_MODELS:
                continue
            for r in json.load(open(path)):
                if _stopped(r):
                    continue
                problem = str(r.get("problem", ""))
                if id_map is None:
                    ex = r.get("example_id")
                    if ex is None:
                        continue
                else:
                    ex = id_map.get(problem.strip())
                    if ex is None:
                        unmatched[condition] = unmatched.get(condition, 0) + 1
                        continue
                rows.append({"dataset": "musique", "model": model, "condition": condition,
                             "example_id": f"musique::{ex}", "problem": problem,
                             "prompt": tmpl.format(Question=problem), "search_calls": _search_calls(r)})
    if unmatched:
        print(f"  musique-natural rows dropped (no id match): {unmatched}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- ShareChat

def load_sharechat(sharechat_dir):
    rows = []
    specs = [
        ("curated-sharechat", "sharechat_curated", NATURAL_QUERY_TEMPLATE),
        ("curated-sharechat-benchmark", "sharechat_benchmark", QUERY_TEMPLATE),
    ]
    for prefix, condition, tmpl in specs:
        for path in sorted(glob.glob(os.path.join(sharechat_dir, f"{prefix}_baseline_*_run_1.json"))):
            tok = _model_from_name(path, prefix)
            if tok is None:
                continue
            model = canonical_model(tok)
            if model not in FRAMES_MODELS:
                continue
            for i, r in enumerate(json.load(open(path))):  # index i = same question in both files
                if _stopped(r):
                    continue
                problem = str(r.get("problem", ""))
                rows.append({"dataset": "sharechat", "model": model, "condition": condition,
                             "example_id": f"sharechat::{i}", "problem": problem,
                             "prompt": tmpl.format(Question=problem), "search_calls": _search_calls(r)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--cues-dir", default=DEFAULT_CUES_DIR)
    ap.add_argument("--musique-dir", default="results/archive")
    ap.add_argument("--sharechat-dir", default="results/curated_sharechat")
    ap.add_argument("--natural-jsonl", default="data/musique_val_natural.jsonl")
    ap.add_argument("--out", default="results/search_call_prediction/search_calls_multi_table.csv")
    args = ap.parse_args()

    frames = load_frames(args.results_dir, args.cues_dir)
    musique = load_musique(args.musique_dir, args.natural_jsonl)
    sharechat = load_sharechat(args.sharechat_dir)
    df = pd.concat([frames, musique, sharechat], ignore_index=True)

    # sanity: within a group, one prompt per condition (no accidental collisions)
    per = df.groupby(["model", "example_id"]).agg(nc=("condition", "nunique"), np=("prompt", "nunique"))
    bad = per[per.nc != per.np]
    assert bad.empty, f"prompt collisions within groups:\n{bad.head()}"

    print("\nrows per model x dataset:")
    print(pd.crosstab(df.model, df.dataset).to_string())
    print("\nconditions per model:")
    for m, sub in df.groupby("model"):
        print(f"  {m:<26} conds={sub.condition.nunique():>2}  groups={sub.example_id.nunique():>5}  rows={len(sub):>5}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
