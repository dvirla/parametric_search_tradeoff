"""EntityQuestions dataset support: loader, stratified sampler, exact-match grader."""

import glob
import json
import os
import random

from pydantic import BaseModel, Field

ENTITY_STYLE_DATASETS: set[str] = {"entity-questions", "popqa"}


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


def load_entity_questions(
    data_dir: str | None = None,
    split: str = "dev",
    relations: list[str] | None = None,
) -> list[dict]:
    """Load all EntityQuestions P*.{split}.json files and normalize to evaluation format.

    Args:
        data_dir: Directory containing the split files. Defaults to
            ``data/EntityQuestions/{split}``.
        split: One of "dev", "test", or "train".
        relations: Optional list of property IDs (e.g. ["P26", "P50"]) to
            restrict the loaded examples. All relations are loaded when None.
    """
    if data_dir is None:
        data_dir = f"data/EntityQuestions/{split}"
    pattern = os.path.join(data_dir, f"P*.{split}.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No EntityQuestions files found matching {pattern}")

    examples = []
    for filepath in files:
        if 'P136' in filepath:
            print(f"Skipping {filepath} due to known issues with these properties.")
            continue
        # Extract property ID from filename, e.g. "P17" from "P17.train.json"
        source_file = os.path.basename(filepath).split(".")[0]
        if relations and source_file not in relations:
            continue
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


def load_popqa() -> list[dict]:
    """Load PopQA dataset from HuggingFace and normalize to evaluation format."""
    from datasets import load_dataset
    ds = load_dataset("akariasai/PopQA", split="test")
    examples = []
    for row in ds:
        aliases_raw = row.get("o_aliases")
        if isinstance(aliases_raw, list):
            aliases = [str(a).strip() for a in aliases_raw if a]
        elif isinstance(aliases_raw, str) and aliases_raw:
            aliases = [a.strip() for a in aliases_raw.split(",") if a.strip()]
        else:
            aliases = []
        examples.append({
            "problem": row["question"],
            "gold answer": [row["obj"]],
            "source_file": str(row["prop"]),
            "answer_aliases": aliases,
        })
    print(f"Loaded {len(examples)} PopQA examples.")
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


def _normalize(text: str) -> str:
    """Lowercase, strip, remove trailing periods."""
    return text.strip().lower().rstrip(".")


def _word_set(text: str) -> set[str]:
    """Split normalized text into a set of words."""
    return set(_normalize(text).split())


def _relaxed_match(candidate: str, gold: str) -> bool:
    """Match two strings, tolerating missing middle names/initials.

    "Bruce Murray" matches "Bruce C. Murray" because the words of the
    shorter form are a subset of the words of the longer form.
    """
    if _normalize(candidate) == _normalize(gold):
        return True
    c_words = _word_set(candidate)
    g_words = _word_set(gold)
    shorter, longer = (c_words, g_words) if len(c_words) <= len(g_words) else (g_words, c_words)
    return len(shorter) >= 1 and shorter.issubset(longer)


def exact_match_grade(model_answer: str, gold_answers: list[str]) -> bool:
    """Check that ALL gold answers are matched by the model answer.

    Handles comma-separated model answers and relaxed name matching
    (tolerating missing middle names/initials).
    """
    candidates = [c.strip() for c in model_answer.split(",") if c.strip()]

    for gold in gold_answers:
        # Try the full model answer first (in case commas aren't delimiters)
        if _relaxed_match(model_answer, gold):
            continue
        # Then try each comma-separated part
        if any(_relaxed_match(c, gold) for c in candidates):
            continue
        return False
    return True


def any_match_grade(model_answer: str, acceptable_answers: list[str]) -> bool:
    """Check if model answer matches ANY of the acceptable answers.

    Unlike exact_match_grade (which requires ALL gold answers matched),
    this treats each answer as an alternative surface form.
    """
    candidates = [c.strip() for c in model_answer.split(",") if c.strip()]
    for acceptable in acceptable_answers:
        if _relaxed_match(model_answer, acceptable):
            return True
        if any(_relaxed_match(c, acceptable) for c in candidates):
            return True
    return False


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
