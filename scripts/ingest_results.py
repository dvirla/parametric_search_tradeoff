"""Additive migration: walk results/ and ingest raw eval data into results.db.

SAFETY: this script is read-only over the existing result files. It never moves,
mutates, or deletes anything under results/ — it only creates/updates the SQLite
database (default results/results.db). The DB is fully rebuildable from the files,
so it can be deleted and re-ingested at will.

Recommended one-off backup before the first real ingest:

    cp -r results results.bak

Usage:
    uv run python scripts/ingest_results.py --dry-run      # report, write nothing
    uv run python scripts/ingest_results.py                # ingest baseline+parametric
    uv run python scripts/ingest_results.py --verify       # check DB vs files
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src import results_db as rdb


# ─── Discovery ──────────────────────────────────────────────────────────────────

def discover(results_dir: str, include_archive: bool = False):
    """Classify every .json under results_dir into (baseline, parametric, other).

    Files under an ``archive/`` directory are stale backups and are skipped by
    default (they would create duplicate runs for the same dataset/model). Files
    under an ``interplay_analysis*`` directory are derived/staging copies (they
    duplicate the canonical parametric files) and are always skipped for raw-eval
    ingest; they belong to the deferred derived-data phase.
    """
    baseline, parametric, other = [], [], []
    for path in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"),
                                 recursive=True)):
        if not include_archive and (os.sep + "archive" + os.sep) in path:
            continue
        if "interplay_analysis" in path:
            continue
        name = os.path.basename(path)
        if "_parametric_uncertainty_" in name:
            parametric.append(path)
        elif rdb.parse_baseline_filename(path) is not None:
            baseline.append(path)
        else:
            other.append(path)
    return baseline, parametric, other


# ─── Ingest ─────────────────────────────────────────────────────────────────────

def run_ingest(results_dir, db_path, only, dry_run, include_archive=False):
    baseline, parametric, other = discover(results_dir, include_archive)

    todo = []
    if only in (None, "baseline"):
        todo += [("baseline", p) for p in baseline]
    if only in (None, "parametric"):
        todo += [("parametric", p) for p in parametric]

    print(f"Discovered under {results_dir}/:")
    print(f"  baseline evals : {len(baseline)}")
    print(f"  parametric     : {len(parametric)}")
    print(f"  other/deferred : {len(other)}  (traces, interplay CSVs, figures, ...)")
    print()

    if dry_run:
        print("--dry-run: the following files WOULD be ingested (nothing written):")
        for kind, p in todo:
            meta = (rdb.parse_baseline_filename(p) if kind == "baseline"
                    else rdb.parse_parametric_filename(p))
            print(f"  [{kind:10}] {os.path.relpath(p, results_dir)}")
            print(f"               -> {meta}")
        return

    conn = rdb.connect(db_path)
    stats = Counter()
    rows_per_table = Counter()
    for kind, p in todo:
        fn = rdb.ingest_baseline_eval if kind == "baseline" else rdb.ingest_parametric
        res = fn(conn, p)
        stats[res["status"]] += 1
        if res["status"] == "ingested":
            rows_per_table[res.get("table", "?")] += res.get("n_examples", 0)
            if kind == "parametric":
                rows_per_table["parametric_aggregate"] += res.get("n_aggregate", 0)
            if res.get("n_synthesized_ids"):
                print(f"  note: {res['n_synthesized_ids']} synthesized example_ids "
                      f"in {os.path.basename(p)}")
    conn.close()

    print("\nIngest summary:")
    for k, v in stats.items():
        print(f"  {k:12}: {v}")
    print("Rows written (by primary table):")
    for k, v in rows_per_table.items():
        print(f"  {k:22}: {v}")
    print(f"\nDatabase: {db_path}")


# ─── Verification ───────────────────────────────────────────────────────────────

def _file_metrics(path, is_parametric):
    """(n_records, accuracy, mean_search) computed straight from the file."""
    data = json.load(open(path))
    n = len(data)
    if is_parametric:
        corr, sc, m = 0, 0, 0
        for ex in data:
            agg = ex.get("aggregate_result") or {}
            if not agg:
                continue
            m += 1
            corr += 1 if rdb._to_bool_or_none(agg.get("is_correct")) else 0
            sc += rdb._to_int_or_none(agg.get("search_calls")) or 0
        acc = corr / m if m else None
        msearch = sc / m if m else None
        return n, acc, msearch
    corr = sum(1 for r in data if rdb._to_bool_or_none(r.get("sampler_correct")))
    sc = sum((rdb._to_int_or_none(r.get("sampler_search_calls")) or 0) for r in data)
    return n, corr / n if n else None, sc / n if n else None


def _db_metrics(conn, run_id, is_parametric):
    """DB-side (n, accuracy, mean_search) using the SAME None->incorrect / None->0
    semantics as :func:`_file_metrics` so the round-trip comparison is faithful."""
    if is_parametric:
        row = conn.execute(
            """SELECT COUNT(*) n,
                      SUM(CASE WHEN is_correct=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) acc,
                      SUM(COALESCE(search_calls,0))*1.0/COUNT(*) ms
               FROM parametric_aggregate WHERE run_id=?""", (run_id,)).fetchone()
        nrec = conn.execute(
            "SELECT n_examples FROM runs WHERE run_id=?", (run_id,)).fetchone()[0]
        return nrec, row["acc"], row["ms"]
    row = conn.execute(
        """SELECT COUNT(*) n,
                  SUM(CASE WHEN sampler_correct=1 THEN 1 ELSE 0 END)*1.0/COUNT(*) acc,
                  SUM(COALESCE(sampler_search_calls,0))*1.0/COUNT(*) ms
           FROM eval_records WHERE run_id=?""", (run_id,)).fetchone()
    return row["n"], row["acc"], row["ms"]


def _close(a, b, tol=1e-6):
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def run_verify(results_dir, db_path, include_archive=False):
    conn = rdb.connect(db_path)
    runs = conn.execute("SELECT * FROM runs ORDER BY base_dataset, phrasing, "
                        "model_slug, grading_version").fetchall()
    print(f"Verifying {len(runs)} ingested runs in {db_path}\n")

    failures = 0
    print(f"{'dataset/phr':28} {'model':26} {'grade':12} "
          f"{'n(file=db)':12} {'acc Δ':8} {'search Δ':9}")
    for r in runs:
        is_param = r["artifact_type"] == "parametric"
        fn, facc, fms = _file_metrics(r["source_path"], is_param)
        dn, dacc, dms = _db_metrics(conn, r["run_id"], is_param)
        n_ok = (fn == dn) if not is_param else True  # parametric n counted at agg level
        if is_param:
            # for parametric, file n_records == run.n_examples already, compare agg counts
            n_ok = True
        acc_ok = _close(facc, dacc)
        ms_ok = _close(fms, dms)
        ok = n_ok and acc_ok and ms_ok
        if not ok:
            failures += 1
        flag = "" if ok else "  <-- MISMATCH"
        dphr = f"{r['base_dataset']}/{r['phrasing']}"
        dacc_d = "n/a" if (facc is None or dacc is None) else f"{abs(facc-dacc):.2e}"
        dms_d = "n/a" if (fms is None or dms is None) else f"{abs(fms-dms):.2e}"
        print(f"{dphr:28} {r['model_slug']:26} {r['grading_version']:12} "
              f"{str(fn)+'='+str(dn):12} {dacc_d:8} {dms_d:9}{flag}")

    # identity check
    print("\nIdentity (canonical model <- raw slugs seen):")
    amap = defaultdict(set)
    for row in conn.execute("SELECT canonical_slug, alias FROM model_aliases"):
        amap[row["canonical_slug"]].add(row["alias"])
    for row in conn.execute("SELECT DISTINCT model_slug, raw_model_slug FROM runs"):
        amap[row["model_slug"]].add(row["raw_model_slug"])
    for canon, aliases in sorted(amap.items()):
        print(f"  {canon:32} <- {sorted(aliases)}")
    print("  ^ Review: any two lines that are actually the SAME model should be "
          "merged via _EXPLICIT_ALIASES in src/results_db.py, then re-ingested.")

    # coverage check
    _, _, other = discover(results_dir, include_archive)
    print(f"\nCoverage: {len(other)} files matched no raw-eval ingest pattern "
          f"(deferred to a later phase):")
    by_kind = Counter()
    for p in other:
        b = os.path.basename(p)
        if "_traces" in b:
            by_kind["traces"] += 1
        elif b.endswith(".json"):
            by_kind["interplay/other json"] += 1
    for k, v in by_kind.items():
        print(f"  {k:22}: {v}")

    conn.close()
    print(f"\n{'ALL CHECKS PASSED' if failures == 0 else f'{failures} MISMATCH(ES)'}")
    return failures


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--db", default=rdb.DEFAULT_DB_PATH)
    ap.add_argument("--only", choices=["baseline", "parametric"], default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--include-archive", action="store_true",
                    help="Also ingest stale backups under results/archive/ (off by default).")
    args = ap.parse_args()

    if args.verify:
        sys.exit(1 if run_verify(args.results_dir, args.db, args.include_archive) else 0)
    run_ingest(args.results_dir, args.db, args.only, args.dry_run, args.include_archive)


if __name__ == "__main__":
    main()
