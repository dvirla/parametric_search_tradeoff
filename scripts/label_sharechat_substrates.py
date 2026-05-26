"""
Auto-label ShareChat questions with factual substrate probe questions.

For each open-ended question, uses an LLM to identify as many specific,
objectively-verifiable factual probe questions as the question genuinely
requires. The count varies per question — simple factual questions may need
2–3, complex multi-faceted ones may need 5–6. These serve as "hops" for the
parametric uncertainty evaluation pipeline (run_sharechat_parametric_uncertainty.py).

Usage:
  uv run python scripts/label_sharechat_substrates.py \\
      --dataset data/curated_sharechat_wildchat.csv \\
      --output data/sharechat_substrates.jsonl \\
      --model gemma4:31b --provider ollama
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel, Field

from src.services.base_agent import BaseAgent


SYSTEM_PROMPT = """\
You identify the factual substrate of open-ended questions.

Given a question, produce as many short, specific probe questions as the question genuinely \
requires — typically 2–5, but use your judgment. Each probe must be about an objectively-verifiable \
fact that the world must make true for any well-grounded answer to the original question.

Rules:
1. Each probe question must be answerable with a concrete fact (a yes/no, a number, a named \
entity, a mechanism) — not an opinion or a prediction.
2. The probes must represent the *necessary factual claims* behind the original question, \
not sub-tasks for answering it. They should be answerable independently.
3. Cover distinct aspects — do not repeat equivalent claims in different words.
4. A simple factual question may need only 2–3 probes; a complex multi-faceted question may \
need 5–6. Do not pad with redundant probes to hit a target count.
5. Keep probes short and self-contained. No "Does this mean..." or "Given that..." phrasing.
6. Return ONLY a valid JSON array of strings. No explanation, no preamble.\
"""

SUBSTRATE_TEMPLATE = "Question: {question}\n\nProbe questions (JSON array only):"


class SubstrateQuestions(BaseModel):
    questions: list[str] = Field(
        description="List of factual substrate probe questions. Length varies by question complexity.",
        min_length=1,
    )


def make_example_id(question: str) -> str:
    """Stable, deterministic ID derived from question text."""
    return "sha_" + hashlib.md5(question.encode()).hexdigest()[:8]


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


def load_questions(dataset_path: str) -> list[str]:
    """Load unique question texts from the ShareChat CSV (single 'text' column)."""
    seen: set[str] = set()
    questions: list[str] = []
    with open(dataset_path, newline="") as f:
        for row in csv.DictReader(f):
            q = row["text"].strip()
            if q and q not in seen:
                seen.add(q)
                questions.append(q)
    return questions


async def label_one(agent: BaseAgent, question: str) -> list[str]:
    prompt = SUBSTRATE_TEMPLATE.format(question=question)
    response = await agent.arun(prompt)
    result: SubstrateQuestions = response.output
    return result.questions


async def run(args: argparse.Namespace) -> None:
    questions = load_questions(args.dataset)
    existing_ids = load_existing_ids(args.output)

    pending = [q for q in questions if make_example_id(q) not in existing_ids]

    print(f"Total questions:     {len(questions)}")
    print(f"Already processed:   {len(existing_ids)}")
    print(f"To process:          {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    agent = BaseAgent(
        provider_name=args.provider,
        model_name=args.model,
        output_type=SubstrateQuestions,
        system_prompt=SYSTEM_PROMPT,
        agent_name="sharechat_substrate_labeler",
    )

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "a") as out_f:
        for i, question in enumerate(pending):
            example_id = make_example_id(question)
            try:
                substrate_questions = await label_one(agent, question)
            except Exception as e:
                print(f"  [!] Failed on {example_id}: {e}")
                continue

            record = {
                "example_id": example_id,
                "question": question,
                "substrate_questions": [
                    {"hop_index": idx, "question": q}
                    for idx, q in enumerate(substrate_questions)
                ],
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"[{i+1}/{len(pending)}] {question[:70]}")
            for sq in record["substrate_questions"]:
                print(f"  [{sq['hop_index']}] {sq['question']}")

    print(f"\nDone. Output: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label ShareChat questions with factual substrate probe questions."
    )
    parser.add_argument("--dataset", default="data/curated_sharechat_wildchat.csv",
                        help="Path to ShareChat CSV with a 'text' column "
                             "(default: data/curated_sharechat_wildchat.csv).")
    parser.add_argument("--output", required=True,
                        help="Output JSONL path (e.g. data/sharechat_substrates.jsonl).")
    parser.add_argument("--model", default="gemma4:31b",
                        help="LLM model name (default: gemma4:31b).")
    parser.add_argument("--provider", default="ollama",
                        help="LLM provider (default: ollama).")
    parser.add_argument("--ollama-url", default=None,
                        help="Ollama base URL (defaults to OLLAMA_BASE_URL env var).")
    args = parser.parse_args()

    asyncio.run(run(args))
