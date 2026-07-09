"""Phase 0 of docs/PLAN_cue_mechanism_analysis.md: download PLAIN/ELABORATE cue traces from
Logfire for the 6 cells (facts_open x {gemini-3.1-pro-preview, qwen3.5:122b, gemma4:31b},
medqa x same 3 models) and write them to a deterministic per-cell/cue layout, plus a manifest CSV.

Reuses scripts/download_traces.py's Logfire query + trace-extraction machinery unmodified (imports
it as a module) — this script only adds the cue/dataset disambiguation and matching logic that
plain `--eval-json` filtering can't do here (see below).

Why not just `--eval-json` per cell as scripts/download_traces.py's docstring example shows: the
eval JSON's `problem` field holds the ORIGINAL dataset question (src/services/qa_eval.py stores
`row.get("problem")`, not the cue-templated `query` actually sent to the model), so it is IDENTICAL
between the plain and elaborate eval JSON for a given example_id. The cue suffix
("\n\nPlease answer with a detailed explanation - at least 8-10 sentences") only appears in the
actual first user message captured in the Logfire trace. `download_traces.py`'s own
`clean_problem()` does not strip that suffix, so exact-matching a trace's cleaned problem against
`--eval-json`'s (suffix-free) problem set would return zero hits for the elaborate condition.

Instead: pull ALL baseline_agent traces per model (one Logfire query per model, since PLAIN and
ELABORATE runs both used agent_name=f"baseline_agent_{run_name}" with run_name in {"plain",
"elaborate"} -- see scripts/run_qa_eval_experiment.py), then locally (a) split by agent_name
suffix into cue, (b) strip the elaborate suffix to recover the base question, (c) match the base
question against each dataset's eval-json problem set to assign dataset + example_id.
"""
import argparse
import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.download_traces import get_agent_traces_from_logfire  # noqa: E402

ELABORATE_SUFFIX = "\n\nPlease answer with a detailed explanation - at least 8-10 sentences"

# cell -> (dataset_slug, model_name (Logfire attr value), model_slug (filename-safe),
#          plain eval json, elaborate eval json)
CELLS: List[Tuple[str, str, str, str, str, str]] = [
    ("facts_open_gemini", "facts-open", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview",
     "results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_plain.json",
     "results/cue_smoke/facts-open_baseline_gemini-3.1-pro-preview_elaborate.json"),
    ("facts_open_qwen", "facts-open", "qwen3.5:122b", "qwen3.5_122b",
     "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_plain.json",
     "results/facts_qwen100/facts-open_baseline_qwen3.5:122b_elaborate.json"),
    ("facts_open_gemma", "facts-open", "gemma4:31b", "gemma4_31b",
     "results/facts_gemma100/facts-open_baseline_gemma4:31b_plain.json",
     "results/facts_gemma100/facts-open_baseline_gemma4:31b_elaborate.json"),
    ("medqa_gemini", "medqa", "gemini-3.1-pro-preview", "gemini-3.1-pro-preview",
     "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_plain.json",
     "results/cue_smoke_medqa/medqa_baseline_gemini-3.1-pro-preview_elaborate.json"),
    ("medqa_qwen", "medqa", "qwen3.5:122b", "qwen3.5_122b",
     "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_plain.json",
     "results/cue_smoke_medqa/medqa_baseline_qwen3.5:122b_elaborate.json"),
    ("medqa_gemma", "medqa", "gemma4:31b", "gemma4_31b",
     "results/cue_smoke_medqa/medqa_baseline_gemma4:31b_plain.json",
     "results/cue_smoke_medqa/medqa_baseline_gemma4:31b_elaborate.json"),
]


def load_eval_rows(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def base_question(problem: str, cue: str) -> Optional[str]:
    """Strip the elaborate cue suffix (if present) to recover the base dataset question.

    Returns None if cue == 'elaborate' but the suffix isn't present (unexpected trace, e.g. from
    an unrelated run_name that happens to also end in '_elaborate').
    """
    p = problem.strip()
    if cue == "elaborate":
        if p.endswith(ELABORATE_SUFFIX):
            return p[: -len(ELABORATE_SUFFIX)].strip()
        return None
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default="results/cue_traces")
    ap.add_argument("--lookback-days", type=int, default=7)
    ap.add_argument("--agent-name", default="baseline_agent")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Query Logfire once per unique model (agent_name broad-matches both _plain and _elaborate).
    models = sorted({c[2] for c in CELLS})
    per_model_traces: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for model_name in models:
        print(f"\n=== Querying Logfire for model_name LIKE '%{model_name}%' "
              f"(lookback {args.lookback_days}d) ===")
        traces_by_agent = get_agent_traces_from_logfire(
            agent_name=args.agent_name,
            model_name=model_name,
            limit=None,
            eval_problems=None,
            lookback_days=args.lookback_days,
        )
        per_model_traces[model_name] = traces_by_agent
        for agent, traces in traces_by_agent.items():
            print(f"  {agent}: {len(traces)} raw traces")

    manifest_rows = []
    for cell, dataset, model_name, model_slug, plain_path, elab_path in CELLS:
        traces_by_agent = per_model_traces.get(model_name, {})
        for cue, eval_path in (("plain", plain_path), ("elaborate", elab_path)):
            agent_key = f"{args.agent_name}_{cue}"
            raw_traces = traces_by_agent.get(agent_key, [])

            eval_rows = load_eval_rows(eval_path)
            eval_problem_set = {r["problem"].strip() for r in eval_rows if "problem" in r}
            n_eval = len(eval_rows)

            matched = []
            unmatched_no_suffix = 0
            unmatched_no_eval_hit = 0
            for tr in raw_traces:
                bq = base_question(tr["problem"], cue)
                if bq is None:
                    unmatched_no_suffix += 1
                    continue
                if bq not in eval_problem_set:
                    unmatched_no_eval_hit += 1
                    continue
                # Normalize the saved trace's top-level `problem` to the base question so that
                # downstream scripts can join trace -> eval row by exact `problem` string
                # regardless of cue (the elaborate suffix is still fully present inside
                # message_trace's actual first user-message part, nothing is lost).
                tr_out = dict(tr)
                tr_out["problem"] = bq
                matched.append(tr_out)

            n_traces_raw = len(raw_traces)
            n_matched = len(matched)
            match_rate = (n_matched / n_eval) if n_eval else 0.0

            out_path = os.path.join(args.output_dir, f"{dataset}_{model_slug}_{cue}_traces.json")
            with open(out_path, "w") as f:
                json.dump(matched, f, indent=2)

            print(f"[{cell}/{cue}] n_eval={n_eval} n_traces_raw={n_traces_raw} "
                  f"n_matched={n_matched} match_rate={match_rate:.1%} "
                  f"(unmatched: no_suffix={unmatched_no_suffix}, no_eval_hit={unmatched_no_eval_hit}) "
                  f"-> {out_path}")

            manifest_rows.append({
                "cell": cell, "dataset": dataset, "model_slug": model_slug, "cue": cue,
                "n_eval": n_eval, "n_traces_raw": n_traces_raw, "n_matched": n_matched,
                "match_rate": round(match_rate, 4),
                "unmatched_no_suffix": unmatched_no_suffix,
                "unmatched_no_eval_hit": unmatched_no_eval_hit,
                "output_path": out_path,
            })

    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nWrote manifest: {manifest_path}")

    low = [r for r in manifest_rows if r["n_eval"] > 0 and r["match_rate"] < 0.9]
    if low:
        print("\nCELLS BELOW 90% MATCH RATE (investigate):")
        for r in low:
            print(f"  {r['cell']}/{r['cue']}: {r['match_rate']:.1%}")


if __name__ == "__main__":
    main()
