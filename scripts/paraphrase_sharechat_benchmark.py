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
import csv
import json
import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.services.base_agent import BaseAgent

SOURCE_FILE = "data/curated_sharechat_wildchat.csv"
MUSIQUE_NATURAL_FILE = "data/musique_natural.jsonl"
OUTPUT_FILE = "data/sharechat_benchmark.csv"

# How many musique_natural pairs to embed as few-shot examples.
# Chosen to cover 2-hop, 3-hop, and different surface forms (who/what/why).
FEW_SHOT_IDS = [
    "2hop__40485_40502",    # Huguenots / Dutch Republic — "tell me about" → "What was the population of…"
    "2hop__81379_84616",    # Moon River / Audrey Hepburn — "I'm curious about" → "For what did… win a Tony"
    "3hop1__159728_91191_156667",  # Jordan religion 3-hop — "I've been wondering about" → "What was the name of…"
    "2hop__17192_77606",    # Israel geography — "Could you explain why" → "In ancient times, why was…"
]

SYSTEM_PROMPT = """\
You are a prompt rewriter. You will be given a natural, conversational user question. \
Your task is to rewrite it into a concise, explicit multi-hop benchmark question — the kind \
that appears in academic QA datasets such as MusiQue, HotpotQA, or 2WikiMultiHopQA.

Rules:
1. First, silently identify the underlying reasoning chain (the sequence of facts a reader \
must resolve to answer the question). Then construct the benchmark question so that chain \
is encoded explicitly.
2. The rewritten question must make the hop chain visible. Use nested relative clauses, \
"the X of Y" constructions, or "where/who/which" subordinate clauses that force the reader \
to resolve each intermediate step.
3. Replace any named entity that is the *answer to an earlier hop* with a descriptive phrase \
that requires the reader to first resolve that hop. The final entity should NOT be named \
directly if it is itself an intermediate answer.
4. The question must be answerable with a short, precise response (a name, date, place, \
number, or brief phrase). Avoid open-ended framings like "explain" or "tell me about".
5. Preserve the same ultimate information need and expected answer as the original question.
6. Return only the rewritten benchmark question — no explanation, no preamble, no quotes.

The examples below include explicit reasoning hops to illustrate the structure a good \
benchmark question encodes; you will not be given hops for the question you must rewrite.

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


def load_sharechat_data() -> list[dict]:
    df = pd.read_csv(SOURCE_FILE)
    return df[["text"]].to_dict("records")


def load_existing_texts(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    try:
        return set(pd.read_csv(output_path)["text"].tolist())
    except Exception:
        return set()


async def rewrite_one(agent: BaseAgent, item: dict) -> str:
    prompt = REWRITE_TEMPLATE.format(question=item["text"])
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def main(model_name: str, limit: int | None) -> None:
    few_shots = load_few_shot_examples(MUSIQUE_NATURAL_FILE, FEW_SHOT_IDS)
    system_prompt = SYSTEM_PROMPT.format(few_shots=few_shots)

    data = load_sharechat_data()
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
    write_header = not os.path.exists(OUTPUT_FILE)

    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=["text", "benchmark_question", "source"])
        if write_header:
            writer.writeheader()

        for i, item in enumerate(pending):
            try:
                benchmark_q = await rewrite_one(agent, item)
            except Exception as e:
                print(f"  [!] Failed on item {i}: {e}")
                continue

            writer.writerow({
                "text": item["text"],
                "benchmark_question": benchmark_q,
                "source": "sharechat_benchmark",
            })
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
