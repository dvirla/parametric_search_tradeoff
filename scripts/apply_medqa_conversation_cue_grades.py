"""
Merges the JSONL cache produced by regrade_medqa_conversation_cues_llm.py
(results/medqa_conversation_cue_llm_grades/<model>_<cond>.jsonl) back into the
actual production grid files (results/medqa_grid/<model>/*orig_<cond>.json),
setting each row's `sampler_correct` field. Deliberately a separate, explicit
step from grading itself -- these are the files make_aggregate_cue_tradeoff_figure.py
and everything else in the pipeline read directly, so the merge is reviewable
(--dry-run prints exactly what would change) before it touches production data.

Backs up every file it touches to <path>.pre_conv_cue_grade_backup.json
alongside the original, once, before the first write (idempotent -- running
twice does not overwrite an existing backup with an already-merged version).

Usage:
    uv run python scripts/apply_medqa_conversation_cue_grades.py --dry-run
    uv run python scripts/apply_medqa_conversation_cue_grades.py
"""
import argparse
import glob
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(REPO, "results", "medqa_conversation_cue_llm_grades")
GRID_DIR = os.path.join(REPO, "results", "medqa_grid")

MODELS = ["gemma4_31b", "gpt-oss_20b", "gpt-oss_120b", "nemotron-3-nano_30b",
          "nemotron-cascade-2_30b", "qwen3.5_122b"]
CONDITIONS = ["multiturn", "searchmulti", "confident_parametric"]


def find_file(model, cond):
    matches = glob.glob(os.path.join(GRID_DIR, model, f"*orig_{cond}.json"))
    return matches[0] if len(matches) == 1 else None


def load_cache(model, cond):
    path = os.path.join(CACHE_DIR, f"{model}_{cond}.jsonl")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[row["example_id"]] = row["correct"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    total_updated = 0
    for model in MODELS:
        for cond in CONDITIONS:
            path = find_file(model, cond)
            if path is None:
                print(f"  ! {model}/{cond}: grid file not found, skipping")
                continue
            grades = load_cache(model, cond)
            if not grades:
                print(f"  {model}/{cond}: no cached grades, skipping")
                continue

            rows = json.load(open(path))
            n_updated = 0
            n_already = 0
            n_still_missing = 0
            for row in rows:
                eid = row["example_id"]
                if row.get("sampler_correct") is not None:
                    n_already += 1
                    continue
                if eid in grades:
                    if not args.dry_run:
                        row["sampler_correct"] = grades[eid]
                    n_updated += 1
                else:
                    n_still_missing += 1

            print(f"  {model}/{cond}: {n_updated} to update, {n_already} already graded, "
                  f"{n_still_missing} still missing from cache (n={len(rows)} total)")
            total_updated += n_updated

            if args.dry_run or n_updated == 0:
                continue

            backup_path = path + ".pre_conv_cue_grade_backup.json"
            if not os.path.exists(backup_path):
                with open(backup_path, "w") as f:
                    json.dump(json.load(open(path)), f, indent=2)
                print(f"    backed up original to {backup_path}")

            with open(path, "w") as f:
                json.dump(rows, f, indent=2)
            print(f"    wrote {path}")

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {total_updated} rows total across "
          f"{len(MODELS)} models x {len(CONDITIONS)} conditions.")


if __name__ == "__main__":
    main()
