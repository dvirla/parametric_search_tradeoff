"""EntityQuestions dataset support: loader, stratified sampler, exact-match grader."""

import glob
import json
import os
import random

from pydantic import BaseModel, Field


class ConciseAnswer(BaseModel):
    explanation: str = Field(description="Brief reasoning or explanation for the answer.")
    answer: str = Field(description="The concise, final answer.")


ENTITY_QUESTIONS_QUERY_TEMPLATE = """
Answer the following question concisely. Your answer should be plain text — no LaTeX, no markup, no extra formatting. If there are multiple answers, separate them with a comma.

Here are some examples:

Question: What kind of work does John Ruskin do?
Answer: art critic, poet, architect

Question: Which country is Mount Stromlo located in?
Answer: Australia

Question: What is the official language of Brazil?
Answer: Portuguese

Now answer the following:

Question: {Question}
Answer:""".strip()


def load_entity_questions(data_dir: str = "data/EntityQuestions/dev") -> list[dict]:
    """Load all EntityQuestions P*.dev.json files and normalize to evaluation format."""
    pattern = os.path.join(data_dir, "P*.dev.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No EntityQuestions files found matching {pattern}")

    examples = []
    for filepath in files:
        # Extract property ID from filename, e.g. "P17" from "P17.dev.json"
        source_file = os.path.basename(filepath).replace(".dev.json", "")
        with open(filepath, "r") as f:
            records = json.load(f)
        for record in records:
            examples.append({
                "problem": record["question"],
                "gold answer": record["answers"],  # list[str]
                "source_file": source_file,
            })

    print(f"Loaded {len(examples)} EntityQuestions examples from {len(files)} files.")
    return examples


def stratified_sample(examples: list[dict], num_examples: int, seed: int = 0) -> list[dict]:
    """Proportionally sample from each source_file group."""
    groups: dict[str, list[dict]] = {}
    for ex in examples:
        key = ex.get("source_file", "unknown")
        groups.setdefault(key, []).append(ex)

    total = len(examples)
    rng = random.Random(seed)
    sampled = []
    allocations = {}

    # Proportional allocation
    for key, group in groups.items():
        allocations[key] = round(num_examples * len(group) / total)

    # Adjust for rounding: add/remove from the largest group
    diff = num_examples - sum(allocations.values())
    if diff != 0:
        largest_key = max(allocations, key=lambda k: len(groups[k]))
        allocations[largest_key] += diff

    for key, group in groups.items():
        n = min(allocations.get(key, 0), len(group))
        if n > 0:
            sampled.extend(rng.sample(group, n))

    return sampled


def exact_match_grade(model_answer: str, gold_answers: list[str]) -> bool:
    """Check that ALL gold answers appear in the model answer (case-insensitive).

    The model answer may list multiple items (e.g. "poet, playwright").
    Each gold answer must appear as a substring of the normalized model answer.
    """
    normalized = model_answer.strip().lower()
    return all(g.strip().lower() in normalized for g in gold_answers)


def extract_answer_text(response_output) -> str:
    """Extract answer string from ConciseAnswer or raw output."""
    if isinstance(response_output, ConciseAnswer):
        return response_output.answer
    return str(response_output)


def extract_explanation(response_output) -> str:
    """Extract explanation from ConciseAnswer if available."""
    if isinstance(response_output, ConciseAnswer):
        return response_output.explanation
    return ""
