"""Join cue-grid eval results with their Logfire traces and report thinking-token stats.

Works for both the FRAMES and MedQA cue grids via `--dataset`. Joins each grid
result row (accuracy, search calls) to its downloaded trace (thinking tokens)
on (dataset-phrasing, condition, cleaned problem text), then reports mean
thinking/response/search stats per (model, dataset, condition) and a
"substitution test" per non-plain condition vs its plain baseline: does a
drop in search calls correlate with a rise in thinking tokens (substitution)
or a drop in both (suppression)?

Usage:
    uv run python scripts/analyze_thinking_tokens.py --dataset frames
    uv run python scripts/analyze_thinking_tokens.py --dataset medqa

Expects:
    results/frames_cues_full/<model>/*.json + results/frames_cues_traces/<model>/*.json
    results/medqa_grid/<model>/*.json       + results/medqa_grid_traces/<model>/*.json

Outputs into results/{frames,medqa}_token_analysis/: joined_tokens.csv, report.md
"""

import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr

ROOT = Path("results")


def clean_problem(problem: str) -> str:
    popqa_separator = "Portuguese\n\nNow answer the following:\n\nQuestion: "
    if popqa_separator in problem:
        problem = problem.split(popqa_separator)[1]

    separators = [
        "\n\nYour response should be in the following format:",
        "\n\nPlease answer in 2-4 sentences.",
        "\n\nPlease answer with a detailed explanation - at least 8-10 sentences",
        "\n\nIf it isn't too much trouble, I would be ever so grateful if you could very kindly help me with this. Please don't go to any bother on my account — I truly appreciate your time and effort, and thank you so very much in advance for your help.",
        "\n\nJust answer the question directly — final answer only."
    ]
    for sep in separators:
        if sep in problem:
            problem = problem.split(sep)[0]

    if "\nAnswer:" in problem:
        problem = problem.split("\nAnswer:")[0]

    return problem.strip()


def count_words(text):
    return len(text.split()) if text else 0


def extract_thinking(trace):
    full_thinking = ""
    for msg in trace.get("message_trace", []):
        for part in msg.get("parts", []):
            if part.get("type") == "thinking":
                full_thinking = part.get("content", "")
                break
        if full_thinking:
            break
            
    return {
        "thinking_chars": len(full_thinking),
        "thinking_words": count_words(full_thinking)
    }


# ---------------------------------------------------------------------------
# Dataset-specific filename / agent-name -> (phrasing, condition) parsing.
# ---------------------------------------------------------------------------

def _frames_result_filename(filename: str):
    if "frames-cues_baseline_" not in filename:
        return None
    cond = re.sub(r"^frames-cues_baseline_.*?(epi_strong_|terse_|verbose_)", r"\1", filename).replace(".json", "")
    if cond.startswith("epi_strong_"):
        phrasing = "epi_strong"
    elif cond.startswith("terse_"):
        phrasing = "terse"
    elif cond.startswith("verbose_"):
        phrasing = "verbose"
    else:
        phrasing = "unknown"
    return phrasing, cond


def _frames_trace_agent(agent_name: str):
    cond = agent_name.split("baseline_agent_")[-1]
    if cond.startswith("epi_strong_"):
        phrasing = "epi_strong"
    elif cond.startswith("terse_"):
        phrasing = "terse"
    elif cond.startswith("verbose_"):
        phrasing = "verbose"
    else:
        phrasing = "unknown"
    return phrasing, cond


def _frames_plain_key(phrasing: str):
    if phrasing == "epi_strong":
        return "verbose", "verbose_plain"
    return phrasing, f"{phrasing}_plain"


def _frames_trace_fallback(phrasing, cond, prob, traces_by_prob):
    return None


def _medqa_result_filename(filename: str):
    if "orig_" in filename:
        return "orig", filename.split("orig_")[-1].replace(".json", "")
    elif "terse_" in filename:
        return "terse", filename.split("terse_")[-1].replace(".json", "")
    return None


def _medqa_trace_agent(agent_name: str):
    phrasing = "orig" if "_orig_" in agent_name else "terse" if "_terse_" in agent_name else "unknown"
    cond_parts = agent_name.split(f"_{phrasing}_")
    cond = cond_parts[-1] if len(cond_parts) > 1 else agent_name.split("_")[-1]
    return phrasing, cond


def _medqa_plain_key(phrasing: str):
    return phrasing, "plain"


def _medqa_trace_fallback(phrasing, cond, prob, traces_by_prob):
    # Some early medqa traces don't carry an "_orig_"/"_terse_" marker in the
    # agent name (e.g. bare "baseline_agent_plain") and land under "unknown".
    if phrasing == "orig":
        return traces_by_prob.get(("unknown", cond, prob))
    return None


DATASET_CONFIGS = {
    "frames": dict(
        grid_dir="frames_cues_full",
        traces_dir="frames_cues_traces",
        output_dir="frames_token_analysis",
        report_title="Frames",
        parse_result_filename=_frames_result_filename,
        parse_trace_agent=_frames_trace_agent,
        plain_key=_frames_plain_key,
        trace_fallback=_frames_trace_fallback,
    ),
    "medqa": dict(
        grid_dir="medqa_grid",
        traces_dir="medqa_grid_traces",
        output_dir="medqa_token_analysis",
        report_title="MedQA",
        parse_result_filename=_medqa_result_filename,
        parse_trace_agent=_medqa_trace_agent,
        plain_key=_medqa_plain_key,
        trace_fallback=_medqa_trace_fallback,
    ),
}


def join_and_extract(cfg) -> pd.DataFrame:
    all_data = []
    grid_dir = ROOT / cfg["grid_dir"]
    traces_dir = ROOT / cfg["traces_dir"]

    if not grid_dir.exists() or not traces_dir.exists():
        print(f"Missing directories: {grid_dir} or {traces_dir}")
        return pd.DataFrame()

    for model_dir in grid_dir.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        results_by_prob = {}
        for res_file in model_dir.glob("*.json"):
            parsed = cfg["parse_result_filename"](res_file.name)
            if parsed is None:
                continue
            phrasing, cond = parsed
            try:
                with open(res_file) as f:
                    data = json.load(f)
                for item in data:
                    prob = clean_problem(item.get("problem", ""))
                    if prob:
                        results_by_prob[(phrasing, cond, prob)] = item
            except Exception as e:
                print(f"Error reading {res_file}: {e}")

        model_traces_dir = traces_dir / model
        if not model_traces_dir.exists():
            continue

        traces_by_prob = {}
        for trace_file in model_traces_dir.glob("*.json"):
            try:
                with open(trace_file) as f:
                    traces = json.load(f)
                for item in traces:
                    prob = clean_problem(item.get("problem", ""))
                    phrasing, cond = cfg["parse_trace_agent"](item.get("agent_name", ""))
                    if prob:
                        traces_by_prob[(phrasing, cond, prob)] = item
            except Exception as e:
                print(f"Error reading {trace_file}: {e}")

        for key, res_item in results_by_prob.items():
            phrasing, cond, prob = key
            trace_item = traces_by_prob.get(key) or cfg["trace_fallback"](phrasing, cond, prob, traces_by_prob)
            if not trace_item:
                continue

            t_stats = extract_thinking(trace_item)
            resp = res_item.get("sampler_response", "")

            all_data.append({
                "model": model,
                "dataset": phrasing,
                "condition": cond,
                "example_id": res_item.get("example_id"),
                "problem": prob,
                "thinking_words": t_stats["thinking_words"],
                "thinking_chars": t_stats["thinking_chars"],
                "response_words": count_words(resp),
                "response_chars": len(resp),
                "search_calls": res_item.get("sampler_search_calls", 0)
            })

    return pd.DataFrame(all_data)


def substitution_test(m_df: pd.DataFrame, cfg) -> list[dict]:
    sub_rows = []
    for phrasing in m_df["dataset"].unique():
        ds_df = m_df[m_df["dataset"] == phrasing]

        plain_phrasing, plain_cond = cfg["plain_key"](phrasing)
        plain_df = m_df[(m_df["dataset"] == plain_phrasing) & (m_df["condition"] == plain_cond)]
        plain_df = plain_df.set_index("example_id")
        if len(plain_df) == 0:
            continue

        for cond in ds_df["condition"].unique():
            if (phrasing, cond) == (plain_phrasing, plain_cond):
                continue
            cond_df = ds_df[ds_df["condition"] == cond].set_index("example_id")

            joined = cond_df.join(plain_df, lsuffix="_cue", rsuffix="_plain", how="inner")
            if len(joined) == 0:
                continue

            joined["delta_search"] = joined["search_calls_cue"] - joined["search_calls_plain"]
            joined["delta_thinking"] = joined["thinking_words_cue"] - joined["thinking_words_plain"]

            substitute_count = sum((joined["delta_search"] < 0) & (joined["delta_thinking"] > 0))
            suppress_count = sum((joined["delta_search"] < 0) & (joined["delta_thinking"] < 0))
            n_decreased_search = sum(joined["delta_search"] < 0)

            if len(joined["delta_search"].unique()) > 1 and len(joined["delta_thinking"].unique()) > 1:
                corr, pval = pearsonr(joined["delta_search"], joined["delta_thinking"])
            else:
                corr, pval = float("nan"), float("nan")

            sub_rows.append({
                "dataset": phrasing,
                "condition": cond,
                "n_pairs": len(joined),
                "n_search_decreased": n_decreased_search,
                "substitutes (search<0 & think>0)": substitute_count,
                "suppresses (search<0 & think<0)": suppress_count,
                "correlation(delta_S, delta_T)": corr,
                "pval": pval
            })
    return sub_rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=sorted(DATASET_CONFIGS), required=True)
    args = ap.parse_args()
    cfg = DATASET_CONFIGS[args.dataset]

    out_dir = ROOT / cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    df = join_and_extract(cfg)
    if len(df) == 0:
        print("No joined data found.")
        return

    df.to_csv(out_dir / "joined_tokens.csv", index=False)

    report = [f"# {cfg['report_title']} Token Analysis Report\n"]

    for model in df["model"].unique():
        m_df = df[df["model"] == model]
        report.append(f"## Model: {model}\n")

        agg = m_df.groupby(["dataset", "condition"])[["thinking_words", "response_words", "search_calls"]].mean().reset_index()
        report.append("### Mean Stats\n")
        report.append(agg.to_markdown(index=False) + "\n\n")

        report.append("### Substitution Test (vs plain)\n")
        sub_rows = substitution_test(m_df, cfg)
        if sub_rows:
            report.append(pd.DataFrame(sub_rows).to_markdown(index=False) + "\n\n")
        else:
            report.append("No paired data found for plain baseline.\n\n")

    report_path = out_dir / "report.md"
    with open(report_path, "w") as f:
        f.write("\n".join(report))

    print(f"Analysis complete. Saved to {report_path}")


if __name__ == "__main__":
    main()
