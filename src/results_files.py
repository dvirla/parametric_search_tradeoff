"""File-based readers for paper-figure inputs (no database).

Replaces the query side of the retired ``results_db`` SQLite store: per-example
accuracy/search numbers are read straight from the baseline eval JSONs under
``results/musique-<phrasing>/``. Natural-phrasing records carry no ``example_id``,
so it is recovered from the paraphrase->id pairing files (the same mapping the old
ingester used).

Model-name canonicalization lives in :mod:`src.viz` (``viz.canonical_model``).
"""
from __future__ import annotations

import glob
import json
import os
import re

import pandas as pd

from src import viz

# phrasing -> directory holding its `<dataset>_baseline_<model>_run_<N>[_reevaluated].json`
PHRASING_DIRS = {
    "formal": "results/musique-formal",
    "natural": "results/musique-natural",
    "natural2": "results/musique-natural2",
}

# paraphrase text -> example_id (natural / natural2 baselines lack example_id).
_PAIRING_FILES = (
    "data/musique_val_natural.jsonl",
    "data/musique_val_natural2.jsonl",
)

_BASELINE_RE = re.compile(
    r"^(?P<dataset>.+?)_baseline_(?P<model>.+)_run_(?P<run>\d+)(?P<reeval>_reevaluated)?$"
)

_pairing_cache: dict[str, str] | None = None


def _to_bool_or_none(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("true", "1"):
        return True
    if s in ("false", "0"):
        return False
    return None


def _to_int_or_none(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip().lower() in ("none", "null")):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _load_pairing() -> dict[str, str]:
    """Combined {paraphrase_text: example_id} map for the musique naturals."""
    global _pairing_cache
    if _pairing_cache is not None:
        return _pairing_cache
    out: dict[str, str] = {}
    for path in _PAIRING_FILES:
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("text") and r.get("example_id"):
                    out[r["text"]] = r["example_id"]
    _pairing_cache = out
    return out


def parse_baseline_filename(path: str) -> dict | None:
    """Parse ``<dataset>_baseline_<model>_run_<N>[_reevaluated].json``.

    Returns None for trace files and anything that is not a baseline eval.
    """
    stem = os.path.basename(path)
    if not stem.endswith(".json"):
        return None
    stem = stem[: -len(".json")]
    if "_baseline_agent_" in stem or "_traces" in stem:
        return None
    m = _BASELINE_RE.match(stem)
    if not m:
        return None
    return {
        "raw_model": m.group("model"),
        "run_name": f"run_{m.group('run')}",
        "grading_version": "reevaluated" if m.group("reeval") else "original",
    }


def load_baseline_eval(path: str) -> pd.DataFrame:
    """Read one baseline eval JSON into a tidy per-example DataFrame.

    Columns: example_id, problem, correct_answer, sampler_response,
    sampler_correct (bool|None), sampler_search_calls (int|None). Missing
    example_ids are recovered from the paraphrase pairing via ``problem`` text.
    """
    with open(path) as f:
        data = json.load(f)
    pairing = _load_pairing()
    rows = []
    for rec in data:
        eid = rec.get("example_id") or pairing.get(rec.get("problem"))
        rows.append({
            "example_id": eid,
            "problem": rec.get("problem"),
            "correct_answer": rec.get("correct_answer"),
            "sampler_response": rec.get("sampler_response"),
            "sampler_correct": _to_bool_or_none(rec.get("sampler_correct")),
            "sampler_search_calls": _to_int_or_none(rec.get("sampler_search_calls")),
        })
    return pd.DataFrame(rows)


def _find_baseline_file(dirpath: str, model_canonical: str,
                        grading: str = "reevaluated") -> str | None:
    """Best baseline file in ``dirpath`` for a canonical model slug.

    Folds raw filename slugs (colon/dot/size variants) onto the canonical id, so
    e.g. natural2's ``gemini-3.1-pro-preview`` matches a request for
    ``gemini-3-pro-preview``. Prefers the requested grading, then reevaluated,
    then original.
    """
    by_grading: dict[str, str] = {}
    for p in sorted(glob.glob(os.path.join(dirpath, "*_baseline_*_run_*.json"))):
        meta = parse_baseline_filename(p)
        if meta is None:
            continue
        if viz.canonical_model(meta["raw_model"]) != model_canonical:
            continue
        by_grading.setdefault(meta["grading_version"], p)
    for g in (grading, "reevaluated", "original"):
        if g in by_grading:
            return by_grading[g]
    return None


def paired_eval_files(phrasings, model_canonical: str, grading: str = "reevaluated",
                      phrasing_dirs: dict[str, str] | None = None) -> pd.DataFrame:
    """Inner-join baseline evals across phrasings on example_id, for one model.

    File-based drop-in for the old ``results_db.paired_eval``. Returns one row per
    common example_id with ``<phrasing>_correct`` / ``<phrasing>_searches`` columns,
    or an empty DataFrame if any phrasing's file is missing.
    """
    phrasing_dirs = phrasing_dirs or PHRASING_DIRS
    merged: pd.DataFrame | None = None
    for ph in phrasings:
        dirpath = phrasing_dirs.get(ph)
        path = _find_baseline_file(dirpath, model_canonical, grading) if dirpath else None
        if path is None:
            return pd.DataFrame()
        df = (load_baseline_eval(path)
              .dropna(subset=["example_id"])
              .drop_duplicates("example_id", keep="last")
              [["example_id", "sampler_correct", "sampler_search_calls"]]
              .rename(columns={"sampler_correct": f"{ph}_correct",
                               "sampler_search_calls": f"{ph}_searches"}))
        merged = df if merged is None else merged.merge(df, on="example_id", how="inner")
    return merged if merged is not None else pd.DataFrame()
