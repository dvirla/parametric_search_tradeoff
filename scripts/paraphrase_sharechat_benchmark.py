"""
Rewrite natural ShareChat questions into explicit multi-hop benchmark-style questions.

ShareChat questions are already natural and conversational:
  "Can you name any times in history where a military had the chance to concede peace,
   but chose war to reclaim lost territory and ended up losing the entire country?"

A benchmark question reveals the reasoning chain explicitly:
  "What historical instances involved a military that rejected a peace offer to reclaim
   lost territory and subsequently lost the entire country?"

Uses few-shot examples drawn from data/musique_natural.jsonl (which contains both the
original MusiQue benchmark question and the natural rewrite), presented in reverse to
show the model what the target register looks like.

Outputs data/sharechat_benchmark.jsonl preserving all original fields plus
`benchmark_question`.

Usage:
  uv run python scripts/paraphrase_sharechat_benchmark.py [--model gpt-oss:20b] [--limit N]
"""

import argparse
import asyncio
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.base_agent import BaseAgent

SOURCE_FILE = "data/sharechat_info_seeking_v3.jsonl"
MUSIQUE_NATURAL_FILE = "data/musique_natural.jsonl"
OUTPUT_FILE = "data/sharechat_benchmark.jsonl"

# How many musique_natural pairs to embed as few-shot examples.
# Chosen to cover 2-hop, 3-hop, and different surface forms (who/what/why).
FEW_SHOT_IDS = [
    "2hop__40485_40502",    # Huguenots / Dutch Republic — "tell me about" → "What was the population of…"
    "2hop__81379_84616",    # Moon River / Audrey Hepburn — "I'm curious about" → "For what did… win a Tony"
    "3hop1__159728_91191_156667",  # Jordan religion 3-hop — "I've been wondering about" → "What was the name of…"
    "2hop__17192_77606",    # Israel geography — "Could you explain why" → "In ancient times, why was…"
]

SYSTEM_PROMPT = """\
You are a prompt rewriter. You will be given a natural, conversational user question \
and a description of the underlying reasoning hops required to answer it. Your task is \
to rewrite the question into a concise, explicit multi-hop benchmark question — the kind \
that appears in academic QA datasets such as MusiQue, HotpotQA, or 2WikiMultiHopQA.

Rules:
1. The rewritten question must explicitly encode the hop chain. Use nested relative clauses, \
"the X of Y" constructions, or "where/who/which" subordinate clauses that force the reader \
to resolve each intermediate step.
2. Replace any named entity that is the *answer to an earlier hop* with a descriptive phrase \
that requires the reader to first resolve that hop. The final entity should NOT be named \
directly if it is itself an intermediate answer.
3. The question must be answerable with a short, precise response (a name, date, place, \
number, or brief phrase). Avoid open-ended framings like "explain" or "tell me about".
4. Preserve the same ultimate information need and expected answer as the original question.
5. Return only the rewritten benchmark question — no explanation, no preamble, no quotes.

Below are examples of natural questions and their corresponding benchmark rewrites:

{few_shots}
Now rewrite the following question.\
"""

FEW_SHOT_TEMPLATE = """\
---
Natural question: {natural}
Reasoning hops: {hops}
Benchmark question: {benchmark}
"""

REWRITE_TEMPLATE = """\
Natural question: {question}
Reasoning hops: {reasoning}

Benchmark question:\
"""


def load_few_shot_examples(musique_path: str, ids: list[str]) -> str:
    by_id = {}
    with open(musique_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("example_id") in ids:
                    by_id[rec["example_id"]] = rec
            except json.JSONDecodeError:
                pass

    blocks = []
    for eid in ids:
        rec = by_id.get(eid)
        if rec is None:
            continue
        hops_str = " → ".join(
            f"[{sq['question']}]" for sq in rec.get("sub_questions", [])
        )
        blocks.append(
            FEW_SHOT_TEMPLATE.format(
                natural=rec["text"],
                hops=hops_str,
                benchmark=rec["original_question"],
            )
        )
    return "\n".join(blocks)


def load_sharechat_filtered() -> list[dict]:
    df = pd.read_json(SOURCE_FILE, lines=True)
    df = (
        df[
            df["is_info_seeking"].astype(bool)
            & (df["reasoning_hops"] > 1)
            & ~df["is_time_dependent"].astype(bool)
        ]
        .reset_index(drop=True)
    )
    return df.to_dict("records")


def load_existing_texts(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    seen = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    seen.add(json.loads(line)["text"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return seen


async def rewrite_one(agent: BaseAgent, item: dict) -> str:
    prompt = REWRITE_TEMPLATE.format(
        question=item["text"],
        reasoning=item.get("reasoning", ""),
    )
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def main(model_name: str, limit: int | None) -> None:
    few_shots = load_few_shot_examples(MUSIQUE_NATURAL_FILE, FEW_SHOT_IDS)
    system_prompt = SYSTEM_PROMPT.format(few_shots=few_shots)

    data = load_sharechat_filtered()
    if limit is not None:
        data = data[:limit]

    existing_texts = load_existing_texts(OUTPUT_FILE)
    pending = [d for d in data if d["text"] not in existing_texts]

    print(f"Total filtered ShareChat questions: {len(data)}")
    print(f"Already processed: {len(existing_texts)}")
    print(f"To process: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    agent = BaseAgent(
        provider_name="ollama",
        model_name=model_name,
        system_prompt=system_prompt,
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    with open(OUTPUT_FILE, "a") as out_f:
        for i, item in enumerate(pending):
            try:
                benchmark_q = await rewrite_one(agent, item)
            except Exception as e:
                print(f"  [!] Failed on item {i}: {e}")
                continue

            record = {
                "text": item["text"],
                "benchmark_question": benchmark_q,
                "subset": item.get("subset", ""),
                "is_info_seeking": item.get("is_info_seeking", True),
                "category": item.get("category", ""),
                "answerable_in_paragraph": item.get("answerable_in_paragraph", False),
                "is_time_dependent": item.get("is_time_dependent", False),
                "reasoning_hops": item.get("reasoning_hops"),
                "reasoning": item.get("reasoning", ""),
                "source": "sharechat_benchmark",
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"[{i + 1}/{len(pending)}] {item['text'][:70]}")
            print(f"        → {benchmark_q[:80]}")

    print(f"\nDone. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rewrite ShareChat questions into benchmark multi-hop format"
    )
    parser.add_argument(
        "--model",
        default="gpt-oss:20b",
        help="Ollama model name (default: gpt-oss:20b)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N questions",
    )
    args = parser.parse_args()

    asyncio.run(main(model_name=args.model, limit=args.limit))
