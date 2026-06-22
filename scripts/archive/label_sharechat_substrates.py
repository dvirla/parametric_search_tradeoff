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
import re
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pydantic import BaseModel, Field

from src.services.base_agent import BaseAgent


SYSTEM_PROMPT = """\
You identify the factual substrate of open-ended questions — the general knowledge any \
well-grounded answer must draw from.

Given a question, produce as many short probes as the question genuinely requires — typically \
2–5, but use your judgment. Each probe must be about an objectively-verifiable fact, but its \
**scope should match the scope of the original question**. Open-ended questions (those that \
invite multiple valid answers, ask for examples, lists, or comparisons) demand probes that test \
the broader knowledge space, not probes that pre-commit to one specific instance.

Rules:
1. **Match the question's generality.** If the original asks "what are examples of X?", probe \
the model's general knowledge of X (e.g. "Which historical events fit the pattern X?"), not \
whether a single named instance satisfies X. If the original asks "how does Y work?", probe the \
mechanism in general, not one specific case. Pre-committing to one answer (one dynasty, one \
compound, one person) when the question invites many is the most common failure mode — avoid it.
2. Each probe must be answerable with a concrete fact (a yes/no, a number, a named entity, a \
mechanism, a list) — not an opinion or a prediction. "Yes/no about specific instance X" is \
acceptable only when the original question is itself about X.
3. Probes should test the **types of knowledge** the answer requires (categories, mechanisms, \
quantities, criteria, exemplars), not pick the answer for the responder. A good probe lets the \
model demonstrate breadth where breadth was asked for.
4. **Strongly prefer independent probes.** Each probe should stand on its own and be answerable \
without knowing the answer to any other probe. Re-state the relevant entity or context \
explicitly in each probe rather than referring back to a previous one.
5. **Dependencies are a last resort.** Only if rephrasing into an independent probe would \
distort the original meaning may a probe reference a previous probe's answer using the \
placeholder `{hop_K}`, where `K` is the 0-indexed position of the prior probe. When you use \
`{hop_K}`, list `K` in this probe's `depends_on`. References must be strictly backward \
(K < this probe's index) — no forward or self-references. Typical legitimate use: the first \
probe asks the model to enumerate a set ("Which inorganic compounds are gases at STP?"), and a \
later probe asks for a property "of {hop_0}".
6. Cover distinct aspects — do not repeat equivalent claims in different words.
7. A simple factual question may need only 2–3 probes; a complex multi-faceted question may \
need 5–6. Do not pad with redundant probes to hit a target count.
8. Keep probes short. No "Does this mean..." or "Given that..." phrasing — express any \
conditioning via a `{hop_K}` placeholder instead.
9. Return ONLY the structured output. No explanation, no preamble.

Examples:
- Original: "Can you name any times in history where a military rejected peace and lost the war?"
  Bad (over-specific):  "Did the Sui dynasty reject Goguryeo's peace offer in 614?"  ← pre-picks one answer
  Good: "Which historical wars involved one side rejecting an offered peace settlement?" \
"For those cases, which side ultimately lost sovereignty over the disputed territory?"

- Original: "What inorganic gases exist at STP?"
  Bad (over-specific):  "Is neon a gas at STP?"  ← tests one instance, not the set
  Good: "Which inorganic compounds with no carbon are gases at 0°C and 1 atm?" \
"What are the boiling points of {hop_0}?"\
"""

SUBSTRATE_TEMPLATE = "Question: {question}\n\nProbe questions:"


PLACEHOLDER_RE = re.compile(r"\{hop_(\d+)\}")


class Substrate(BaseModel):
    question: str = Field(
        description=(
            "Short, specific probe question. May contain `{hop_K}` placeholders (0-indexed) "
            "that will be replaced with a previous probe's answer at probing time. Use "
            "placeholders only when an independent rephrasing would distort the meaning."
        )
    )
    depends_on: list[int] = Field(
        default_factory=list,
        description=(
            "0-indexed positions of previous probes this probe depends on. Must match the set "
            "of {hop_K} placeholders in the question text and contain only strictly backward "
            "references (each index < this probe's own index)."
        ),
    )


class SubstrateQuestions(BaseModel):
    questions: list[Substrate] = Field(
        description="List of factual substrate probes. Length varies by question complexity.",
        min_length=1,
    )


def extract_placeholder_refs(text: str) -> set[int]:
    """Return the set of hop indices referenced by `{hop_K}` placeholders in `text`."""
    return {int(m.group(1)) for m in PLACEHOLDER_RE.finditer(text)}


def validate_substrates(substrates: list[dict]) -> list[str]:
    """Validate a list of substrate dicts (each with hop_index, question, depends_on).

    Returns a list of human-readable error strings; empty list means valid.
    Catches: forward/self references, out-of-range indices, placeholder/depends_on mismatch.
    """
    errors: list[str] = []
    n = len(substrates)
    for sq in substrates:
        idx = sq.get("hop_index")
        text = sq.get("question", "")
        declared = set(sq.get("depends_on", []))
        referenced = extract_placeholder_refs(text)

        for k in referenced:
            if k >= idx:
                errors.append(
                    f"hop {idx}: placeholder {{hop_{k}}} is not strictly backward "
                    f"(must be < {idx})."
                )
            elif k < 0 or k >= n:
                errors.append(
                    f"hop {idx}: placeholder {{hop_{k}}} is out of range [0, {n})."
                )

        for k in declared:
            if k >= idx:
                errors.append(
                    f"hop {idx}: depends_on contains {k}, which is not strictly backward "
                    f"(must be < {idx})."
                )
            elif k < 0 or k >= n:
                errors.append(
                    f"hop {idx}: depends_on contains {k}, which is out of range [0, {n})."
                )

        if referenced != declared:
            missing = referenced - declared
            extra = declared - referenced
            parts = []
            if missing:
                parts.append(f"placeholders not declared: {sorted(missing)}")
            if extra:
                parts.append(f"declared but no placeholder: {sorted(extra)}")
            errors.append(f"hop {idx}: depends_on/placeholder mismatch — {'; '.join(parts)}.")
    return errors


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


async def label_one(agent: BaseAgent, question: str) -> list[Substrate]:
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

            substrate_records = [
                {
                    "hop_index": idx,
                    "question": s.question,
                    "depends_on": sorted(set(s.depends_on)),
                }
                for idx, s in enumerate(substrate_questions)
            ]
            errors = validate_substrates(substrate_records)
            if errors:
                print(f"  [!] Validation errors on {example_id}, skipping:")
                for err in errors:
                    print(f"      {err}")
                continue

            record = {
                "example_id": example_id,
                "question": question,
                "substrate_questions": substrate_records,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            print(f"[{i+1}/{len(pending)}] {question[:70]}")
            for sq in record["substrate_questions"]:
                deps = f"  (depends on {sq['depends_on']})" if sq["depends_on"] else ""
                print(f"  [{sq['hop_index']}] {sq['question']}{deps}")

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
