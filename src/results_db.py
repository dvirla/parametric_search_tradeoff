"""SQLite-backed results store for the parametric-search-tradeoff project.

This module is the *analysis-layer* source of truth. The JSON files under
``results/`` remain the *capture-layer* source of truth: experiment runners keep
writing them, and :mod:`scripts.ingest_results` loads them into ``results.db``.
The database is fully rebuildable from the files at any time, so it can be
deleted and re-ingested without losing anything.

Design notes
------------
* One file, no server. Nested per-hop / per-run data is stored as JSON-text
  columns (``cluster_ids``, ``runs_json``) plus a lossless ``raw_json`` of every
  original record, so no field is ever dropped.
* Identity is normalized at ingest. ``canonical_model`` collapses the
  punctuation variants of the *same* model (``qwen3.5:122b`` -> ``qwen3.5_122b``)
  but never silently merges models that might be genuinely different (e.g.
  ``gemini-3.1-pro-preview`` vs ``gemini-3-pro-preview`` — those are surfaced by
  the verifier so a human can add an explicit alias if they are the same).
* Idempotent ingest: a run is keyed on its absolute ``source_path``; re-ingesting
  an unchanged file is a no-op, a changed file (newer mtime) replaces that run's
  rows inside a single transaction.

The query helpers return tidy pandas DataFrames and replace the hardcoded globs
that used to live in the analysis scripts.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from typing import Iterable

import pandas as pd

# Reuse the existing model-identity maps rather than duplicating them.
try:  # allow ``import src.results_db`` and ``import results_db`` both to work
    from src import viz
except ImportError:  # pragma: no cover - fallback when src/ is on sys.path
    import viz  # type: ignore

DEFAULT_DB_PATH = "results/results.db"

# ─── Identity normalization ─────────────────────────────────────────────────────

# Explicit alias -> canonical map. Seeded from viz plus the known colon variants.
# Only punctuation/ollama-vs-filename variants of the *same* model belong here.
# Ambiguous version pairs (gemini-3 vs gemini-3.1) are intentionally NOT merged.
_EXPLICIT_ALIASES = {
    "qwen3.5:122b": "qwen3.5_122b",
    "nemotron-3-nano:30b": "nemotron-3-nano_30b",
    "nemotron-3-nano": "nemotron-3-nano_30b",
    # gemini-3.1-pro-preview is the same model as gemini-3-pro-preview (confirmed
    # by the project owner); fold it so cross-phrasing joins line up on one slug.
    "gemini-3.1-pro-preview": "gemini-3-pro-preview",
}


def canonical_model(raw_slug: str) -> str:
    """Map a raw model slug from a filename to its canonical id.

    Explicit aliases win; otherwise we only normalize ``:`` -> ``_`` (the same
    convention ``viz.load_interplay_summary`` already applies). Models that are
    not known punctuation-variants are returned essentially as-is, so distinct
    models are never collapsed by accident.
    """
    raw_slug = raw_slug.strip()
    if raw_slug in _EXPLICIT_ALIASES:
        return _EXPLICIT_ALIASES[raw_slug]
    norm = raw_slug.replace(":", "_")
    return _EXPLICIT_ALIASES.get(norm, norm)


# dataset id (as it appears in filenames) -> (base_dataset, phrasing)
_PHRASING_MAP = {
    "musique-formal": ("musique", "formal"),
    "musique-natural": ("musique", "natural"),
    "musique-natural2": ("musique", "natural2"),
    "curated-sharechat": ("sharechat", "real"),
    "curated-sharechat-benchmark": ("sharechat", "benchmark"),
    "sharechat": ("sharechat", "real"),
    "sharechat-benchmark": ("sharechat", "benchmark"),
    "facts-search": ("facts", "search"),
    "facts-param": ("facts", "param"),
    "nq": ("nq", None),
}


def normalize_phrasing(dataset_id: str) -> tuple[str, str | None]:
    """Split a filename dataset id into (base_dataset, phrasing)."""
    if dataset_id in _PHRASING_MAP:
        return _PHRASING_MAP[dataset_id]
    return dataset_id, None


# ─── Coercion helpers (older files sometimes store "False"/"None" as strings) ────

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
    if s in ("none", "null", ""):
        return None
    return None


def _to_int_or_none(v):
    if v is None or v == "" or (isinstance(v, str) and v.strip().lower() in ("none", "null")):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None


def _bool_to_db(v):
    """bool|None -> 1/0/None for SQLite storage."""
    b = _to_bool_or_none(v)
    return None if b is None else int(b)


def _sha1_id(text: str) -> str:
    return "sha1_" + hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:16]


# ─── Connection + schema ────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    canonical_slug TEXT PRIMARY KEY,
    display_name   TEXT,
    family         TEXT,
    is_tuned       INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS model_aliases (
    alias          TEXT PRIMARY KEY,
    canonical_slug TEXT REFERENCES models(canonical_slug)
);
CREATE TABLE IF NOT EXISTS runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    base_dataset    TEXT,
    phrasing        TEXT,
    artifact_type   TEXT,
    model_slug      TEXT,
    raw_model_slug  TEXT,
    agent_type      TEXT,
    run_name        TEXT,
    mode            TEXT,
    grading_version TEXT,
    source_path     TEXT UNIQUE,
    source_mtime    REAL,
    ingested_at     TEXT,
    n_examples      INTEGER,
    raw_meta        TEXT
);
CREATE TABLE IF NOT EXISTS eval_records (
    run_id               INTEGER REFERENCES runs(run_id),
    row_index            INTEGER,
    example_id           TEXT,
    problem              TEXT,
    correct_answer       TEXT,
    sampler_response     TEXT,
    sampler_correct      INTEGER,
    sampler_search_calls INTEGER,
    stop_reason          TEXT,
    raw_json             TEXT,
    PRIMARY KEY (run_id, row_index)
);
CREATE TABLE IF NOT EXISTS parametric_hops (
    run_id           INTEGER REFERENCES runs(run_id),
    example_id       TEXT,
    hop_index        INTEGER,
    num_hops         INTEGER,
    question         TEXT,
    gold_answer      TEXT,
    num_correct      INTEGER,
    num_runs         INTEGER,
    semantic_entropy REAL,
    cluster_ids      TEXT,
    runs_json        TEXT,
    is_stale         INTEGER,
    PRIMARY KEY (run_id, example_id, hop_index)
);
CREATE TABLE IF NOT EXISTS parametric_aggregate (
    run_id       INTEGER REFERENCES runs(run_id),
    example_id   TEXT,
    question     TEXT,
    gold_answer  TEXT,
    reasoning    TEXT,
    final_answer TEXT,
    is_correct   INTEGER,
    search_calls INTEGER,
    PRIMARY KEY (run_id, example_id)
);
CREATE INDEX IF NOT EXISTS idx_runs_lookup
    ON runs(base_dataset, phrasing, model_slug, grading_version);
CREATE INDEX IF NOT EXISTS idx_eval_example   ON eval_records(example_id);
CREATE INDEX IF NOT EXISTS idx_phops_example  ON parametric_hops(example_id);
CREATE INDEX IF NOT EXISTS idx_pagg_example   ON parametric_aggregate(example_id);
"""


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(_SCHEMA)
    register_model_aliases(conn)
    conn.commit()


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_schema(conn)
    return conn


def register_model_aliases(conn: sqlite3.Connection) -> None:
    """Seed the models/model_aliases tables from viz + the explicit alias map."""
    for canon, label in viz.SLUG_TO_LABEL.items():
        canon_c = canonical_model(canon)
        is_tuned = 1 if ("sft" in label.lower() or "aug" in canon_c.lower()) else 0
        conn.execute(
            "INSERT OR IGNORE INTO models(canonical_slug, display_name, is_tuned) VALUES (?,?,?)",
            (canon_c, label, is_tuned),
        )
        conn.execute(
            "INSERT OR IGNORE INTO model_aliases(alias, canonical_slug) VALUES (?,?)",
            (canon, canon_c),
        )
    for alias, canon in _EXPLICIT_ALIASES.items():
        conn.execute(
            "INSERT OR IGNORE INTO models(canonical_slug, display_name) VALUES (?,?)",
            (canon, viz.SLUG_TO_LABEL.get(canon, canon)),
        )
        conn.execute(
            "INSERT OR IGNORE INTO model_aliases(alias, canonical_slug) VALUES (?,?)",
            (alias, canon),
        )
    conn.commit()


def _ensure_model(conn: sqlite3.Connection, raw_slug: str) -> str:
    """Register a (possibly new) model slug and return its canonical id."""
    canon = canonical_model(raw_slug)
    conn.execute(
        "INSERT OR IGNORE INTO models(canonical_slug, display_name) VALUES (?,?)",
        (canon, viz.SLUG_TO_LABEL.get(canon, canon)),
    )
    conn.execute(
        "INSERT OR IGNORE INTO model_aliases(alias, canonical_slug) VALUES (?,?)",
        (raw_slug, canon),
    )
    return canon


# ─── Filename parsing ───────────────────────────────────────────────────────────

_BASELINE_RE = re.compile(
    r"^(?P<dataset>.+?)_baseline_(?P<model>.+)_run_(?P<run>\d+)(?P<reeval>_reevaluated)?$"
)
_PARAMETRIC_RE = re.compile(
    r"^(?P<dataset>.+?)_parametric_uncertainty_(?P<model>.+)$"
)


def parse_baseline_filename(path: str) -> dict | None:
    """Parse ``<dataset>_baseline_<model>_run_<N>[_reevaluated].json``.

    Returns None for files that are not baseline evals (e.g. ``*_traces*.json``).
    """
    stem = os.path.basename(path)
    if not stem.endswith(".json"):
        return None
    stem = stem[: -len(".json")]
    if "_baseline_agent_" in stem or "_traces" in stem:
        return None  # trace file, not an eval
    m = _BASELINE_RE.match(stem)
    if not m:
        return None
    dataset = m.group("dataset")
    base_dataset, phrasing = normalize_phrasing(dataset)
    return {
        "dataset_id": dataset,
        "base_dataset": base_dataset,
        "phrasing": phrasing,
        "raw_model": m.group("model"),
        "run_name": f"run_{m.group('run')}",
        "grading_version": "reevaluated" if m.group("reeval") else "original",
    }


def parse_parametric_filename(path: str) -> dict | None:
    stem = os.path.basename(path)
    if not stem.endswith(".json"):
        return None
    stem = stem[: -len(".json")]
    m = _PARAMETRIC_RE.match(stem)
    if not m:
        return None
    base_dataset = m.group("dataset")  # 'musique' or 'sharechat'
    phrasing = "benchmark" if "benchmark" in path.lower() else None
    return {
        "base_dataset": base_dataset,
        "phrasing": phrasing,
        "raw_model": m.group("model"),
    }


# ─── Natural-phrasing pairing (text -> example_id) ──────────────────────────────

_PAIRING_FILES = (
    "data/musique_val_natural.jsonl",
    "data/musique_val_natural2.jsonl",
)
_pairing_cache: dict[str, str] | None = None


def _load_pairing(extra_paths: Iterable[str] = ()) -> dict[str, str]:
    """Build a combined {paraphrase_text: example_id} map for musique naturals."""
    global _pairing_cache
    if _pairing_cache is not None and not extra_paths:
        return _pairing_cache
    out: dict[str, str] = {}
    for path in list(_PAIRING_FILES) + list(extra_paths):
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
    if not extra_paths:
        _pairing_cache = out
    return out


# ─── Run bookkeeping (idempotency) ──────────────────────────────────────────────

def _existing_run(conn: sqlite3.Connection, source_path: str) -> sqlite3.Row | None:
    cur = conn.execute(
        "SELECT run_id, source_mtime FROM runs WHERE source_path=?", (source_path,)
    )
    return cur.fetchone()


def _delete_run_rows(conn: sqlite3.Connection, run_id: int) -> None:
    for tbl in ("eval_records", "parametric_hops", "parametric_aggregate"):
        conn.execute(f"DELETE FROM {tbl} WHERE run_id=?", (run_id,))
    conn.execute("DELETE FROM runs WHERE run_id=?", (run_id,))


def _should_skip(conn: sqlite3.Connection, source_path: str, mtime: float) -> bool:
    row = _existing_run(conn, source_path)
    return bool(row and row["source_mtime"] is not None and row["source_mtime"] >= mtime)


# ─── Ingest: baseline evals ─────────────────────────────────────────────────────

def ingest_baseline_eval(conn: sqlite3.Connection, path: str) -> dict:
    """Ingest one baseline eval JSON. Returns a small status dict."""
    meta = parse_baseline_filename(path)
    if meta is None:
        return {"status": "skipped", "reason": "not a baseline eval", "path": path}
    abspath = os.path.abspath(path)
    mtime = os.path.getmtime(path)
    if _should_skip(conn, abspath, mtime):
        return {"status": "unchanged", "path": path}

    with open(path) as f:
        data = json.load(f)

    pairing = _load_pairing()
    canon = _ensure_model(conn, meta["raw_model"])

    old = _existing_run(conn, abspath)
    if old:
        _delete_run_rows(conn, old["run_id"])

    cur = conn.execute(
        """INSERT INTO runs(base_dataset, phrasing, artifact_type, model_slug,
                            raw_model_slug, agent_type, run_name, mode,
                            grading_version, source_path, source_mtime,
                            ingested_at, n_examples, raw_meta)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            meta["base_dataset"], meta["phrasing"], "baseline_eval", canon,
            meta["raw_model"], "baseline", meta["run_name"], None,
            meta["grading_version"], abspath, mtime,
            time.strftime("%Y-%m-%dT%H:%M:%S"), len(data),
            json.dumps({"dataset_id": meta["dataset_id"]}),
        ),
    )
    run_id = cur.lastrowid

    n_synth = 0
    rows = []
    for i, rec in enumerate(data):
        eid = rec.get("example_id")
        if not eid:
            eid = pairing.get(rec.get("problem"))
        if not eid:
            eid = _sha1_id(rec.get("problem", ""))
            n_synth += 1
        rows.append((
            run_id, i, eid, rec.get("problem"), rec.get("correct_answer"),
            rec.get("sampler_response"), _bool_to_db(rec.get("sampler_correct")),
            _to_int_or_none(rec.get("sampler_search_calls")),
            None if rec.get("stop_reason") in (None, "None") else str(rec.get("stop_reason")),
            json.dumps(rec, default=str),
        ))
    conn.executemany(
        """INSERT OR REPLACE INTO eval_records
           (run_id, row_index, example_id, problem, correct_answer, sampler_response,
            sampler_correct, sampler_search_calls, stop_reason, raw_json)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return {"status": "ingested", "path": path, "run_id": run_id,
            "n_examples": len(data), "n_synthesized_ids": n_synth,
            "table": "eval_records"}


# ─── Ingest: parametric uncertainty (hops + formal aggregate) ───────────────────

def ingest_parametric(conn: sqlite3.Connection, path: str) -> dict:
    meta = parse_parametric_filename(path)
    if meta is None:
        return {"status": "skipped", "reason": "not a parametric file", "path": path}
    abspath = os.path.abspath(path)
    mtime = os.path.getmtime(path)
    if _should_skip(conn, abspath, mtime):
        return {"status": "unchanged", "path": path}

    with open(path) as f:
        data = json.load(f)

    canon = _ensure_model(conn, meta["raw_model"])
    old = _existing_run(conn, abspath)
    if old:
        _delete_run_rows(conn, old["run_id"])

    cur = conn.execute(
        """INSERT INTO runs(base_dataset, phrasing, artifact_type, model_slug,
                            raw_model_slug, agent_type, run_name, mode,
                            grading_version, source_path, source_mtime,
                            ingested_at, n_examples, raw_meta)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            meta["base_dataset"], meta["phrasing"], "parametric", canon,
            meta["raw_model"], "parametric", "run_1", None, "original",
            abspath, mtime, time.strftime("%Y-%m-%dT%H:%M:%S"), len(data), None,
        ),
    )
    run_id = cur.lastrowid

    hop_rows, agg_rows = [], []
    for ex in data:
        eid = ex.get("example_id") or _sha1_id(ex.get("aggregate_question", ""))
        sqs = ex.get("sub_questions_results") or []
        num_hops = len(sqs)
        for sq in sqs:
            runs = sq.get("runs") or []
            hop_rows.append((
                run_id, eid, _to_int_or_none(sq.get("hop_index")), num_hops,
                sq.get("question"), sq.get("gold_answer"),
                _to_int_or_none(sq.get("num_correct")),
                len(runs) if runs else _to_int_or_none(sq.get("num_runs")),
                sq.get("semantic_entropy"),
                json.dumps(sq.get("cluster_ids")),
                json.dumps(runs, default=str),
                _bool_to_db(ex.get("is_stale")),
            ))
        agg = ex.get("aggregate_result")
        if agg:
            agg_rows.append((
                run_id, eid, agg.get("question"), agg.get("gold_answer"),
                agg.get("reasoning"), agg.get("final_answer"),
                _bool_to_db(agg.get("is_correct")),
                _to_int_or_none(agg.get("search_calls")),
            ))
    conn.executemany(
        """INSERT OR REPLACE INTO parametric_hops
           (run_id, example_id, hop_index, num_hops, question, gold_answer,
            num_correct, num_runs, semantic_entropy, cluster_ids, runs_json, is_stale)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        hop_rows,
    )
    conn.executemany(
        """INSERT OR REPLACE INTO parametric_aggregate
           (run_id, example_id, question, gold_answer, reasoning, final_answer,
            is_correct, search_calls)
           VALUES (?,?,?,?,?,?,?,?)""",
        agg_rows,
    )
    conn.commit()
    return {"status": "ingested", "path": path, "run_id": run_id,
            "n_examples": len(data), "n_hops": len(hop_rows),
            "n_aggregate": len(agg_rows), "table": "parametric_hops"}


# ─── Query API ──────────────────────────────────────────────────────────────────

def _filter_runs(conn, base_dataset=None, phrasing=None, model=None,
                 artifact_type=None, grading=None, run_name=None) -> pd.DataFrame:
    q = "SELECT * FROM runs WHERE 1=1"
    params: list = []
    if base_dataset is not None:
        q += " AND base_dataset=?"; params.append(base_dataset)
    if phrasing is not None:
        q += " AND phrasing=?"; params.append(phrasing)
    if model is not None:
        q += " AND model_slug=?"; params.append(canonical_model(model))
    if artifact_type is not None:
        q += " AND artifact_type=?"; params.append(artifact_type)
    if grading is not None:
        q += " AND grading_version=?"; params.append(grading)
    if run_name is not None:
        q += " AND run_name=?"; params.append(run_name)
    return pd.read_sql_query(q, conn, params=params)


def _resolve_eval_runs(conn, base_dataset=None, phrasing=None, model=None,
                       grading="reevaluated", run_name=None) -> pd.DataFrame:
    """Pick one run per (dataset,phrasing,model,run_name), preferring ``grading``.

    Falls back to the 'original' grading when no row exists for the requested
    grading version of a given (dataset,phrasing,model,run_name) group.
    """
    runs = _filter_runs(conn, base_dataset, phrasing, model,
                        artifact_type="baseline_eval", run_name=run_name)
    if runs.empty:
        return runs
    group_cols = ["base_dataset", "phrasing", "model_slug", "run_name"]
    pref = {"reevaluated": 0, "original": 1}
    runs["_rank"] = runs["grading_version"].map(lambda g: pref.get(grading, 0)
                                                if g == grading else 1)
    chosen = (runs.sort_values(["_rank", "grading_version"])
                   .groupby(group_cols, dropna=False, as_index=False)
                   .first())
    return chosen.drop(columns=["_rank"])


def load_eval(conn, base_dataset=None, phrasing=None, model=None,
              grading="reevaluated", run_name=None) -> pd.DataFrame:
    """Per-example eval records, one chosen run per group (see _resolve_eval_runs)."""
    runs = _resolve_eval_runs(conn, base_dataset, phrasing, model, grading, run_name)
    if runs.empty:
        return pd.DataFrame(columns=[
            "model", "base_dataset", "phrasing", "grading_version", "example_id",
            "problem", "correct_answer", "sampler_response",
            "sampler_correct", "sampler_search_calls"])
    ids = runs["run_id"].tolist()
    recs = pd.read_sql_query(
        f"SELECT * FROM eval_records WHERE run_id IN ({','.join('?'*len(ids))}) "
        f"ORDER BY run_id, row_index",
        conn, params=ids)
    meta = runs[["run_id", "base_dataset", "phrasing", "model_slug",
                 "grading_version"]].rename(columns={"model_slug": "model"})
    df = recs.merge(meta, on="run_id", how="left")
    df["sampler_correct"] = df["sampler_correct"].map(
        {1: True, 0: False}).where(df["sampler_correct"].notna(), None)
    return df


def load_parametric_hops(conn, model=None, dataset="musique") -> pd.DataFrame:
    runs = _filter_runs(conn, base_dataset=dataset, model=model,
                        artifact_type="parametric")
    if runs.empty:
        return pd.DataFrame()
    ids = runs["run_id"].tolist()
    hops = pd.read_sql_query(
        f"SELECT * FROM parametric_hops WHERE run_id IN ({','.join('?'*len(ids))})",
        conn, params=ids)
    meta = runs[["run_id", "model_slug"]].rename(columns={"model_slug": "model"})
    return hops.merge(meta, on="run_id", how="left")


def load_canonical_entropy(conn) -> dict[tuple[str, str, int], float]:
    """{(model, example_id, hop_index): semantic_entropy} from parametric hops.

    DB-backed drop-in for :func:`viz.load_canonical_entropy`.
    """
    df = load_parametric_hops(conn)
    out: dict[tuple[str, str, int], float] = {}
    for r in df.itertuples(index=False):
        if r.semantic_entropy is not None and r.hop_index is not None:
            out[(r.model, r.example_id, int(r.hop_index))] = float(r.semantic_entropy)
    return out


def load_formal_aggregate(conn, model=None) -> pd.DataFrame:
    """The 'formal original question' eval, in baseline-eval column shape."""
    runs = _filter_runs(conn, model=model, artifact_type="parametric")
    if runs.empty:
        return pd.DataFrame()
    ids = runs["run_id"].tolist()
    agg = pd.read_sql_query(
        f"SELECT * FROM parametric_aggregate WHERE run_id IN ({','.join('?'*len(ids))})",
        conn, params=ids)
    meta = runs[["run_id", "model_slug", "base_dataset"]].rename(
        columns={"model_slug": "model"})
    df = agg.merge(meta, on="run_id", how="left")
    df["phrasing"] = "formal"
    return df.rename(columns={
        "question": "problem", "gold_answer": "correct_answer",
        "final_answer": "sampler_response", "is_correct": "sampler_correct",
        "search_calls": "sampler_search_calls"})


def paired_eval(conn, base_dataset, phrasings, model,
                grading="reevaluated", formal_from_aggregate=False) -> pd.DataFrame:
    """Inner-join evals across phrasings on example_id for one model.

    Returns one row per common example_id with ``<phrasing>_correct`` and
    ``<phrasing>_searches`` columns. This is the primitive the phrasing
    comparison script needs.

    If ``formal_from_aggregate`` is True, the ``formal`` phrasing is sourced from
    ``parametric_aggregate`` instead of materialized ``musique-formal`` files.
    """
    per: dict[str, pd.DataFrame] = {}
    for ph in phrasings:
        if ph == "formal" and formal_from_aggregate:
            d = load_formal_aggregate(conn, model=model)
        else:
            d = load_eval(conn, base_dataset=base_dataset, phrasing=ph,
                          model=model, grading=grading)
        if d.empty:
            per[ph] = pd.DataFrame(columns=["example_id", "sampler_correct",
                                            "sampler_search_calls"])
            continue
        d = d[["example_id", "sampler_correct", "sampler_search_calls"]].copy()
        d = d.rename(columns={
            "sampler_correct": f"{ph}_correct",
            "sampler_search_calls": f"{ph}_searches"})
        # When two rows share an example_id (a genuine duplicate in the source
        # paraphrases), keep the LAST in file order — matching the dict-overwrite
        # semantics of the original per-phrasing analysis scripts.
        per[ph] = d.dropna(subset=["example_id"]).drop_duplicates(
            "example_id", keep="last")

    merged: pd.DataFrame | None = None
    for ph in phrasings:
        merged = per[ph] if merged is None else merged.merge(
            per[ph], on="example_id", how="inner")
    return merged if merged is not None else pd.DataFrame()
