"""
Prepare a staging directory so that analyze_parametric_search_interplay.py can run
on musique-natural traces with full LLM attribution.

The script does two things:
  1. Filters musique_parametric_uncertainty_<model>.json to the 103 natural questions
     and replaces aggregate_question with the natural-language paraphrase.
  2. Copies the traces file into the staging dir under the naming convention that
     analyze_parametric_search_interplay.py expects.

Usage:
  uv run python scripts/stage_musique_natural_interplay.py \\
    --model gemini-3-pro-preview \\
    --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_gemini-3-pro-preview.json \\
    --traces-json      results/musique-natural/musique_val_search_gemini-3-pro-baseline_agent_run_1_traces_20260420_164047.json \\
    --natural-jsonl    data/musique_natural.jsonl \\
    --output-dir       results/musique-natural/interplay_analysis_stage

Then run:
  uv run python scripts/analyze_parametric_search_interplay.py \\
    --eval-dir   results/musique-natural/interplay_analysis_stage \\
    --output-dir results/musique-natural/interplay_analysis \\
    --use-llm --llm-gold-check --check-multi-hop
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="gemini-3-pro-preview")
    p.add_argument("--uncertainty-json", required=True,
                   help="musique_parametric_uncertainty_<model>.json from the original musique run")
    p.add_argument("--traces-json", required=True,
                   help="Traces file from the musique-natural eval run")
    p.add_argument("--natural-jsonl", default="data/musique_natural.jsonl")
    p.add_argument("--output-dir", default="results/musique-natural/interplay_analysis_stage")
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Build example_id -> natural question map
    id_to_natural: dict[str, str] = {}
    with open(args.natural_jsonl) as f:
        for line in f:
            row = json.loads(line)
            id_to_natural[row["example_id"]] = row["text"]
    print(f"Loaded {len(id_to_natural)} natural questions from {args.natural_jsonl}")

    # Patch uncertainty file
    with open(args.uncertainty_json) as f:
        uncertainty = json.load(f)

    patched = []
    for entry in uncertainty:
        if entry["example_id"] in id_to_natural:
            e = dict(entry)
            e["aggregate_question"] = id_to_natural[e["example_id"]]
            patched.append(e)

    print(f"Patched {len(patched)}/{len(uncertainty)} entries with natural questions")

    uncertainty_out = os.path.join(args.output_dir, f"musique_parametric_uncertainty_{args.model}.json")
    with open(uncertainty_out, "w") as f:
        json.dump(patched, f, indent=2)
    print(f"Wrote {uncertainty_out}")

    # Extract timestamp from traces filename for the expected naming pattern
    ts_match = re.search(r"(\d{8}_\d{6})", os.path.basename(args.traces_json))
    timestamp = ts_match.group(1) if ts_match else "000000_000000"
    traces_out = os.path.join(args.output_dir, f"musique_val_search_{args.model}_traces_{timestamp}.json")
    shutil.copy2(args.traces_json, traces_out)
    print(f"Copied traces -> {traces_out}")

    # Verify matching
    def clean(prob: str) -> str:
        prob = prob.split("\n\nYour response should be in the following format:")[0]
        return prob.split("\n\nPlease answer")[0].strip()

    eval_map = {e["aggregate_question"] for e in patched}
    with open(traces_out) as f:
        traces = json.load(f)
    matched = sum(1 for t in traces if clean(t["problem"]) in eval_map)
    print(f"Trace-to-eval match: {matched}/{len(traces)}")
    if matched < len(traces):
        print("  WARNING: some traces did not match — check question text alignment")

    print(f"\nStaging complete. Now run:")
    print(f"  uv run python scripts/analyze_parametric_search_interplay.py \\")
    print(f"    --eval-dir   {args.output_dir} \\")
    print(f"    --output-dir {os.path.dirname(args.output_dir)}/interplay_analysis \\")
    print(f"    --use-llm --llm-gold-check --check-multi-hop")


if __name__ == "__main__":
    main()
