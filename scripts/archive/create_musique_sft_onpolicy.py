"""
Create MusiQue SFT dataset via on-policy rejection sampling — Procedure 1 only.

Simpler alternative to create_musique_sft_data.py: prompt the rollout model with a
system instruction to search whenever uncertain, then keep traces where R1 (correct
answer) and R2 (all uncertain hops searched) both hold. No formal rollouts, no
M-patching, no mediator agent.

Prerequisites (same as create_musique_sft_data.py):
    1. run_musique_parametric_uncertainty.py → uncertainty JSON
    2. paraphrase_musique_natural.py        → natural paraphrase JSONL

Usage:
    uv run python scripts/create_musique_sft_onpolicy.py \\
        --uncertainty-json results/musique_parametric/musique_parametric_uncertainty_<model>.json \\
        --natural-jsonl data/musique_train_natural.jsonl \\
        --model nemotron-mini:4b \\
        --provider ollama \\
        --k 5 \\
        --output-dir data/sft/musique_onpolicy \\
        --resume
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic_ai import Tool

from src.services.agent_sampler import AgentAsSampler
from src.services.base_agent import BaseAgent
from src.services.brave_search import BraveSearchService
from scripts.archive.create_musique_sft_data import (
    _check_ollama,
    compute_rollout_attribution,
    get_hop_uncertainty_labels,
    load_natural_jsonl,
    load_progress,
    load_uncertainty_json,
    messages_to_chatml,
    process_procedure1,
    run_k_rollouts,
    save_progress,
)

UNCERTAIN_SEARCH_SYSTEM_PROMPT = """\
You are a helpful AI assistant answering factual questions. You have access to a web search tool.

Use the search tool whenever you are not fully confident about a fact. Do not rely on your \
training data alone when uncertain — search for the answer instead. For facts you are highly \
confident about, you may answer directly without searching.

Provide a clear, natural-language answer of 2-4 sentences. Always state the specific answer \
explicitly.\
"""


def setup_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="On-policy SFT data collection via Procedure 1 (R1+R2) only."
    )
    p.add_argument("--uncertainty-json", required=True,
                   help="Output of run_musique_parametric_uncertainty.py")
    p.add_argument("--natural-jsonl", required=True,
                   help="Output of paraphrase_musique_natural.py")
    p.add_argument("--model", required=True,
                   help="Rollout model name (e.g. nemotron-mini:4b)")
    p.add_argument("--provider", default="ollama",
                   help="Rollout model provider (default: ollama)")
    p.add_argument("--k", type=int, default=5,
                   help="Rollouts per question (default: 5)")
    p.add_argument("--output-dir", default="data/sft/musique_onpolicy",
                   help="Output directory (default: data/sft/musique_onpolicy)")
    p.add_argument("--r3-strict", action="store_true",
                   help="Also reject rollouts that over-searched certain hops (off by default)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-processed example_ids")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Rollout sampling temperature (default: 1.0)")
    p.add_argument("--top-p", type=float, default=0.95,
                   help="Rollout nucleus sampling top-p (default: 0.95)")
    p.add_argument("--top-k", type=int, default=20,
                   help="Rollout top-k sampling; Ollama/vLLM only (default: 20)")
    p.add_argument("--attribution-model", default="gpt-oss:20b",
                   help="Model for LLM hop attribution (default: gpt-oss:20b)")
    p.add_argument("--attribution-provider", default="ollama",
                   help="Provider for attribution model (default: ollama)")
    p.add_argument("--attribution-ollama-url", default=None,
                   help="Ollama base URL for the attribution model. Defaults to OLLAMA_BASE_URL "
                        "or http://localhost:11434/v1/.")
    p.add_argument("--grader-ollama-url", default=None,
                   help="Ollama base URL for the grader. Defaults to OLLAMA_BASE_URL "
                        "or http://localhost:11434/v1/.")
    return p.parse_args()


def main() -> None:
    args = setup_args()
    _check_ollama()
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading uncertainty JSON...")
    uncertainty_data = load_uncertainty_json(args.uncertainty_json)
    print(f"  {len(uncertainty_data)} examples")

    print("Loading natural paraphrases...")
    natural_map = load_natural_jsonl(args.natural_jsonl)
    print(f"  {len(natural_map)} natural questions")

    examples = [ex for ex in uncertainty_data if ex["example_id"] in natural_map]
    print(f"  {len(examples)} examples with both inputs\n")

    progress_path = os.path.join(args.output_dir, "progress.json")
    done_ids = load_progress(progress_path) if args.resume else set()

    out_path = os.path.join(args.output_dir, "procedure1_onpolicy_sft.jsonl")
    if not args.resume:
        open(out_path, "w").close()

    print(f"Initializing rollout agent ({args.model} via {args.provider})...")
    search_service = BraveSearchService()
    rollout_agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        tools=[Tool(search_service.search)],
        output_type=str,
        system_prompt=UNCERTAIN_SEARCH_SYSTEM_PROMPT,
        agent_name="sft_onpolicy_rollout_agent",
    )

    print(f"Initializing attribution agent ({args.attribution_model} via {args.attribution_provider} @ {args.attribution_ollama_url or 'default'})...")
    from scripts.archive.create_musique_sft_data import QueryAttribution
    attribution_agent = BaseAgent(
        provider_name=args.attribution_provider,
        model_name=args.attribution_model,
        output_type=QueryAttribution,
        agent_name="sft_attribution_agent",
        ollama_base_url=args.attribution_ollama_url,
    )

    print(f"Initializing grader (gpt-oss:120b via ollama @ {args.grader_ollama_url or 'default'})...")
    grader = AgentAsSampler(BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        agent_name="sft_grader",
        ollama_base_url=args.grader_ollama_url,
    ))

    print(f"\n--- On-policy SFT data generation: K={args.k}, {len(examples)} examples ---\n")

    total = 0

    for i, ex in enumerate(examples):
        eid = ex["example_id"]
        if eid in done_ids:
            print(f"[{i + 1}/{len(examples)}] Skipping {eid} (done)")
            continue

        natural_rec = natural_map[eid]
        natural_q = natural_rec["text"]
        gold_answer = ex["aggregate_answer"]
        sub_questions = ex["sub_questions_results"]
        uncertainty_labels = get_hop_uncertainty_labels(sub_questions)

        print(f"[{i + 1}/{len(examples)}] {eid}")
        print(f"  natural: {natural_q[:80]}")
        print(f"  uncertainty: { {sq['hop_index']: uncertainty_labels[sq['hop_index']] for sq in sub_questions} }")

        print(f"  {args.k} rollouts...")
        rollouts = run_k_rollouts(rollout_agent, natural_q, gold_answer, grader, args.k)

        print(f"  attributing correct rollouts...")
        rollout_attribution_data = []
        for r_idx, r in enumerate(rollouts):
            if not r["is_correct"]:
                rollout_attribution_data.append(([], []))
                continue
            print(f"    attributing rollout {r_idx + 1}/{len(rollouts)}...")
            qctxs, attrs = compute_rollout_attribution(
                r["messages"], sub_questions, ex["aggregate_question"], attribution_agent
            )
            rollout_attribution_data.append((qctxs, attrs))

        proc1_ex = process_procedure1(
            rollouts, rollout_attribution_data,
            sub_questions, uncertainty_labels, r3_strict=args.r3_strict,
        )

        with open(out_path, "a") as f:
            for ex_out in proc1_ex:
                f.write(json.dumps(ex_out, ensure_ascii=False) + "\n")

        total += len(proc1_ex)
        done_ids.add(eid)
        save_progress(progress_path, done_ids)

        print(f"  -> +{len(proc1_ex)} examples (total: {total})")

    print("\n--- Done ---")
    print(f"Procedure 1 (on-policy SFT): {total:5d} examples  →  {out_path}")


if __name__ == "__main__":
    main()
