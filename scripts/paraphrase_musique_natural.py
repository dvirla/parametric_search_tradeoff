"""
Paraphrase MusiQue benchmark questions into natural, goal-oriented user questions.

The benchmark format exposes the hop chain explicitly:
  "What administrative territorial entity includes the place where Bill Cockcroft was educated?"

A natural user would ask the same information need without revealing the retrieval strategy:
  "I'm curious about Bill Cockcroft's background — what region of the country did he study in?"

Outputs data/musique_natural.jsonl in the same schema as sharechat_info_seeking_v3.jsonl,
with an added `answer` field so the evaluation pipeline can grade responses.

Usage:
  uv run python scripts/paraphrase_musique_natural.py [--hops 2] [--model gpt-oss:20b] [--limit N]
"""

import argparse
import asyncio
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent

SOURCE_FILE = "results/musique_parametric/musique_parametric_uncertainty_gemini-3-pro-preview.json"
OUTPUT_FILE = "data/musique_natural.jsonl"

SYSTEM_PROMPT = """\
You are a prompt rewriter. You will be given a multi-hop benchmark question and the \
chain of sub-questions that reveal its structure. Your task is to rewrite the question \
so it sounds like a genuine, curious user asking — not like a database traversal.

Rules:
1. Do NOT mention any intermediate entity that appears only as a bridge in the hop chain. \
The user asking the question would not know that entity exists.
2. Do NOT use the "X of Y where Z did W" pattern. Avoid nested relative clauses that signal \
the hop structure.
3. Frame the question so the answer naturally warrants 2–4 explanatory sentences, not a \
single word or number. Prefer "why", "how", "tell me about", "explain", "what can you tell \
me about" phrasing where it fits naturally. If the answer is inherently factual (a location, \
a date), still frame it as "I'm curious about…" or "Could you tell me a bit about…" to \
invite context.
4. The rewritten question must preserve the same ultimate information need and have the same \
gold answer.
5. Return only the rewritten question text — no explanation, no preamble, no quotes.
"""

PARAPHRASE_TEMPLATE = """\
Original benchmark question: {question}

Hop chain (do NOT expose these steps in the rewrite):
{hops}

Gold answer: {answer}

Rewrite this as a natural user question following the rules:"""


def format_hops(sub_questions: list[dict]) -> str:
    lines = []
    for sq in sub_questions:
        lines.append(f"  Step {sq['hop_index'] + 1}: {sq['question']} → {sq['gold_answer']}")
    return "\n".join(lines)


def load_existing_ids(output_path: str) -> set[str]:
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ids.add(json.loads(line)["example_id"])
                except (KeyError, json.JSONDecodeError):
                    pass
    return ids


async def paraphrase_one(agent: BaseAgent, item: dict) -> str:
    sub_questions = item["sub_questions_results"]
    prompt = PARAPHRASE_TEMPLATE.format(
        question=item["aggregate_question"],
        hops=format_hops(sub_questions),
        answer=item["aggregate_answer"],
    )
    response = await agent.arun(prompt)
    return response.output.strip().strip('"').strip("'")


async def main(hops_filter: int | None, model_name: str, limit: int | None):
    with open(SOURCE_FILE) as f:
        data = json.load(f)

    if hops_filter is not None:
        data = [d for d in data if len(d.get("sub_questions_results", [])) == hops_filter]

    if limit is not None:
        data = data[:limit]

    existing_ids = load_existing_ids(OUTPUT_FILE)
    pending = [d for d in data if d["example_id"] not in existing_ids]

    print(f"Total questions (after hop filter): {len(data)}")
    print(f"Already processed: {len(existing_ids)}")
    print(f"To process: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    agent = BaseAgent(
        provider_name="ollama",
        model_name=model_name,
        system_prompt=SYSTEM_PROMPT,
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)

    with open(OUTPUT_FILE, "a") as out_f:
        for i, item in enumerate(pending):
            try:
                paraphrased = await paraphrase_one(agent, item)
            except Exception as e:
                print(f"  [!] Failed on {item['example_id']}: {e}")
                continue

            record = {
                "text": paraphrased,
                "answer": item["aggregate_answer"],
                "example_id": item["example_id"],
                "original_question": item["aggregate_question"],
                "reasoning_hops": len(item["sub_questions_results"]),
                "sub_questions": [
                    {"question": sq["question"], "answer": sq["gold_answer"]}
                    for sq in item["sub_questions_results"]
                ],
                "is_info_seeking": True,
                "is_time_dependent": item.get("is_stale", False),
                "category": "factual_lookup",
                "source": "musique_natural",
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"[{i + 1}/{len(pending)}] {item['aggregate_question'][:60]}")
            print(f"        → {paraphrased[:80]}")

    print(f"\nDone. Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paraphrase MusiQue questions into natural user questions")
    parser.add_argument("--hops", type=int, default=2, help="Filter to N-hop questions (default: 2)")
    parser.add_argument("--all-hops", action="store_true", help="Include all hop counts (overrides --hops)")
    parser.add_argument("--model", default="gpt-oss:20b", help="Ollama model name (default: gpt-oss:20b)")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N questions")
    args = parser.parse_args()

    hops_filter = None if args.all_hops else args.hops
    asyncio.run(main(hops_filter=hops_filter, model_name=args.model, limit=args.limit))
