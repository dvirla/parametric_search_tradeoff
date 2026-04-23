"""
Classify each atomic fact in atomic_fact_confidence.csv as essential/necessary
to answer the corresponding question or not, using gemini-flash as classifier.

Outputs atomic_fact_necessity.csv with added columns:
  - is_essential  (True/False)
  - necessity_reasoning  (one-sentence explanation)
"""

import asyncio
import csv
import os
import sys

from pydantic import BaseModel
from tqdm.asyncio import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

INPUT_CSV  = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_confidence.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_necessity.csv')

CLASSIFIER_MODEL = "gemini-3-flash-preview"
CONCURRENCY = 10   # Google API rows in-flight at once

class NecessityClassification(BaseModel):
    is_essential: bool
    necessity_reasoning: str


SYSTEM_PROMPT = (
    "You are a precise factual analyst. "
    "Given a question and one atomic fact extracted from a model's answer, "
    "decide whether that fact is ESSENTIAL to correctly answering the question.\n\n"
    "ESSENTIAL means: the fact directly constitutes or is a required component of the answer. "
    "Without it the answer would be incomplete or wrong.\n\n"
    "NOT ESSENTIAL means: the fact is background context, elaboration, or supporting detail "
    "that is interesting but not required to answer the question.\n\n"
    "Set is_essential to true or false, and provide a one-sentence necessity_reasoning."
)

USER_TEMPLATE = (
    "Question: {question}\n\n"
    "Atomic fact: {atomic_fact}\n\n"
    "Is this atomic fact essential/necessary to correctly answer the question?"
)

# Only rows whose model field matches one of these are classified
VALID_MODELS = {'gemini-3-pro-preview', 'nemotron-3-nano', 'qwen3.5:122b'}


async def classify_row(
    row: dict,
    classifier: BaseAgent,
    sem: asyncio.Semaphore,
    writer_lock: asyncio.Lock,
    out_f,
    writer: csv.DictWriter,
) -> None:
    atomic_fact = row.get('atomic_fact', '').strip()
    problem     = row.get('problem', '').strip()
    if not atomic_fact or not problem:
        return

    prompt = USER_TEMPLATE.format(question=problem, atomic_fact=atomic_fact)

    async with sem:
        try:
            result = await classifier.arun(prompt)
            classification: NecessityClassification = result.output
        except Exception as e:
            print(f"\nClassifier error for fact '{atomic_fact[:60]}...': {e}")
            classification = NecessityClassification(is_essential=False, necessity_reasoning=f"error: {e}")

    out_row = dict(row)
    out_row['is_essential']        = classification.is_essential
    out_row['necessity_reasoning'] = classification.necessity_reasoning

    async with writer_lock:
        writer.writerow(out_row)
        out_f.flush()


async def main():
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    output_fieldnames = input_fieldnames + ['is_essential', 'necessity_reasoning']

    # --- Resumability ---
    processed_keys: set[tuple[str, str, str]] = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                processed_keys.add((row['model'], row['problem'], row['atomic_fact']))
        print(f"Resuming: {len(processed_keys)} rows already classified")

    pending = [
        r for r in rows
        if r.get('model', '') in VALID_MODELS
        and r.get('atomic_fact', '').strip()
        and (r['model'], r['problem'], r['atomic_fact'].strip()) not in processed_keys
    ]
    print(f"Rows to classify: {len(pending)}")

    if not pending:
        print("Nothing to do.")
        return

    # --- Classifier agent (gemini-flash, same pattern as sampler in evaluate_atomic_fact_confidence.py) ---
    classifier = BaseAgent(
        provider_name="Google",
        model_name=CLASSIFIER_MODEL,
        output_type=NecessityClassification,
        agent_name="necessity_classifier",
        system_prompt=SYSTEM_PROMPT,
    )

    sem         = asyncio.Semaphore(CONCURRENCY)
    writer_lock = asyncio.Lock()

    mode = 'a' if os.path.exists(OUTPUT_CSV) else 'w'
    with open(OUTPUT_CSV, mode, newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        if mode == 'w':
            writer.writeheader()

        tasks = [
            classify_row(row, classifier, sem, writer_lock, out_f, writer)
            for row in pending
        ]
        await tqdm.gather(*tasks, desc="Classifying atomic facts")

    print(f"\nDone. Output written to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
