import asyncio
import csv
import json
import os
import math
import re
import sys
from typing import List
from tqdm.asyncio import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_attribution.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_confidence.csv')

NUM_SAMPLES = 5

# Maps model_label (as it appears in the CSV) → (provider_name, model_name)
SAMPLER_MODEL_CONFIG = {
    "gemini-3-pro-preview": ("Google", "gemini-3-pro-preview"),
    "nemotron-3-nano":      ("ollama", "nemotron-3-nano:30b"),
    "qwen3.5:122b":                ("ollama", "qwen3.5:122b"),
}

# Concurrency limits per backend.
# gpt-oss:120b handles paraphrase + clustering; keep it from being overwhelmed.
# Gemini is API-bound so we can be more generous.
# Nemotron is on local GPUs; set conservatively.
CONCURRENCY = {
    "gpt_oss": 4,    # paraphrase_agent + clustering_agent
    "gemini":  12,   # Google API rows (network I/O)
    "nemotron": 4,   # Local ollama GPU rows
    "qwen3.5:122b": 2,   # Local ollama GPU rows
}

# Max rows being processed end-to-end at once (caps total in-flight work)
ROW_CONCURRENCY = 16


# --- Prompts ---
PARAPHRASE_SYSTEM_PROMPT = """\
You convert atomic facts into concise, open-ended factual questions.
Given a fact, produce a single clear WH-question (who / what / when / where / how / which) whose correct answer would confirm that fact.

CRITICAL RULE: Never produce a yes/no question. A yes/no question is any question whose natural answer is simply "yes" or "no", typically starting with: did, do, does, is, are, was, were, has, have, can, could, would, will, should.

Output only the question text — no preamble, no labels, no punctuation other than the question mark.

Examples:

Fact: Thomas Jefferson authored the Virginia Statute for Religious Freedom.
BAD (yes/no): Did Thomas Jefferson author the Virginia Statute for Religious Freedom?
GOOD: Who authored the Virginia Statute for Religious Freedom?

Fact: The blood-brain barrier prevents most large-molecule drugs from entering the brain.
BAD (yes/no): Does the blood-brain barrier prevent drugs from entering the brain?
GOOD: What is the role of the blood-brain barrier in limiting drug access to the brain?

Fact: Python uses indentation rather than braces to define code blocks.
BAD (yes/no): Does Python use indentation to define code blocks?
GOOD: How does Python delimit code blocks?

Fact: The Great Wall of China stretches approximately 21,196 kilometres in total length.
BAD (yes/no): Is the Great Wall of China approximately 21,196 kilometres long?
GOOD: How long is the Great Wall of China?\
"""

SEMANTIC_CLUSTERING_PROMPT = """You are judging whether multiple model responses to the same question give semantically equivalent answers.

Question: "{question}"

I have {num_answers} full responses to this question from different runs of the same model. Each response may contain reasoning, explanation, and a final answer (often marked as "Exact Answer:" or in a \\boxed{{}}).

Focus on the FINAL ANSWER / EXACT ANSWER in each response — ignore differences in reasoning, explanation length, or confidence levels. Only the bottom-line factual answer matters for clustering.

{answers_text}

Your Task:
Group these responses into clusters based on whether their final answers are semantically equivalent.
- Responses whose final answers convey the same fact belong in the SAME cluster, even if the wording or explanation differs.
- Equivalent formats should be clustered together (e.g., "3" and "three", "Jan 1 2020" and "January 1, 2020", "USA" and "United States of America", "FDA" and "US Food and Drug Administration").
- Responses whose final answers contradict each other or give different factual content must be in DIFFERENT clusters.
- "I don't know" / refusal / "no answer" responses should be clustered together, separate from factual answers.
- If a response has no clear final answer, treat the overall conclusion as the answer.

Return ONLY a JSON array of arrays of 1-based indices, with no explanation or surrounding text.
Example for 5 responses where 1,3,5 give the same answer and 2,4 give a different answer: [[1, 3, 5], [2, 4]]
Example for 5 responses that all give the same answer: [[1, 2, 3, 4, 5]]
"""


def _parse_paraphrase(text: str) -> str:
    """Extract question from plain-text paraphrase response."""
    text = text.strip()
    # If the model added a label like "Question: ..." strip it
    text = re.sub(r'^(?:question|q)\s*:\s*', '', text, flags=re.IGNORECASE)
    # Take the first sentence/line that ends with '?'
    for line in text.splitlines():
        line = line.strip()
        if line.endswith('?'):
            return line
    # Fallback: return stripped text (model obeyed instructions)
    return text or ""


def _parse_clusters(text: str, num_samples: int) -> List[List[int]]:
    """Extract [[...], [...]] cluster structure from plain-text response."""
    # Find the first JSON array in the response
    match = re.search(r'\[\s*\[.*?\]\s*\]', text, re.DOTALL)
    if match:
        try:
            clusters = json.loads(match.group())
            if isinstance(clusters, list) and all(isinstance(c, list) for c in clusters):
                return clusters
        except json.JSONDecodeError:
            pass
    # Fallback: each answer in its own cluster
    return [[i] for i in range(1, num_samples + 1)]


def compute_shannon_entropy(cluster_sizes: List[int]) -> float:
    """Compute Shannon entropy H = -sum(p * log2(p)) over cluster distribution."""
    total = sum(cluster_sizes)
    if total == 0:
        return 0.0
    entropy = 0.0
    for size in cluster_sizes:
        if size > 0:
            p = size / total
            entropy -= p * math.log2(p)
    return entropy


async def process_row(
    row: dict,
    paraphrase_agent: BaseAgent,
    sampler_agents: dict,
    clustering_agent: BaseAgent,
    gpt_oss_sem: asyncio.Semaphore,
    sampler_sems: dict,
    writer_lock: asyncio.Lock,
    out_f,
    writer: csv.DictWriter,
) -> None:
    atomic_fact = row.get('atomic_fact', '').strip()
    if not atomic_fact:
        return

    model_label = row.get('model', '')
    sampler_agent = sampler_agents.get(model_label)
    if sampler_agent is None:
        print(f"\nNo sampler configured for model '{model_label}', skipping row")
        return

    sampler_sem = sampler_sems[model_label]

    # Step 1 — Paraphrase (serialised through gpt_oss_sem)
    async with gpt_oss_sem:
        try:
            para_result = await paraphrase_agent.arun(atomic_fact)
            paraphrased_question = _parse_paraphrase(para_result.output) or f"What is known about: {atomic_fact.rstrip('.')}?"
        except Exception as e:
            print(f"\nParaphrase error for fact '{atomic_fact[:60]}...': {e}")
            paraphrased_question = f"What is known about: {atomic_fact.rstrip('.')}?"

    # Step 2 — Sample ×5 in parallel (each through its own backend semaphore)
    async def _sample(i: int) -> str:
        async with sampler_sem:
            try:
                result = await sampler_agent.arun(paraphrased_question)
                return result.output
            except Exception as e:
                print(f"\nSampler error (sample {i+1}) for '{paraphrased_question[:60]}...': {e}")
                return ""

    samples = await asyncio.gather(*[_sample(i) for i in range(NUM_SAMPLES)])

    # Step 3 — Cluster + Entropy (serialised through gpt_oss_sem)
    answers_text = "\n\n".join(
        f"--- Response {i+1} ---\n{s}" for i, s in enumerate(samples)
    )
    cluster_prompt = SEMANTIC_CLUSTERING_PROMPT.format(
        question=paraphrased_question,
        num_answers=NUM_SAMPLES,
        answers_text=answers_text,
    )
    async with gpt_oss_sem:
        try:
            cluster_result = await clustering_agent.arun(cluster_prompt)
            clusters = _parse_clusters(cluster_result.output, NUM_SAMPLES)

            all_indices = set()
            for cluster in clusters:
                all_indices.update(cluster)
            expected = set(range(1, NUM_SAMPLES + 1))
            if all_indices != expected:
                print(f"\nWARNING: cluster indices {all_indices} != {expected}, falling back")
                clusters = [[i] for i in range(1, NUM_SAMPLES + 1)]
        except Exception as e:
            print(f"\nClustering error for '{paraphrased_question[:60]}...': {e}")
            clusters = [[i] for i in range(1, NUM_SAMPLES + 1)]

    cluster_sizes = [len(c) for c in clusters]
    entropy = compute_shannon_entropy(cluster_sizes)

    out_row = dict(row)
    out_row['paraphrased_question'] = paraphrased_question
    for i, s in enumerate(samples):
        out_row[f'sample_{i+1}'] = s
    out_row['num_clusters'] = len(clusters)
    out_row['cluster_sizes'] = str(cluster_sizes)
    out_row['entropy'] = round(entropy, 4)

    async with writer_lock:
        writer.writerow(out_row)
        out_f.flush()


async def main():
    # --- Load input ---
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    # --- Resumability ---
    output_fieldnames = (input_fieldnames or []) + [
        'paraphrased_question',
        'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5',
        'num_clusters', 'cluster_sizes', 'entropy',
    ]

    processed_keys = set()
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_keys.add((row['model'], row['problem'], row['atomic_fact']))
        print(f"Resuming: {len(processed_keys)} rows already processed")

    pending_rows = [
        r for r in rows
        if r.get('atomic_fact', '').strip()
        and (r['model'], r['problem'], r['atomic_fact'].strip()) not in processed_keys
    ]
    print(f"Rows to process: {len(pending_rows)}")

    # --- Initialise agents ---
    print("Initialising agents...")
    paraphrase_agent = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=str,
        agent_name="paraphrase_agent",
        use_thinking=False,
        temperature=0,
        system_prompt=PARAPHRASE_SYSTEM_PROMPT,
    )
    sampler_agents = {
        label: BaseAgent(
            provider_name=provider,
            model_name=model_name,
            output_type=str,
            agent_name=f"sampler_agent_{label}",
            use_thinking=True,
            system_prompt="Answer the question with very short and concise answer, no need for explanations and reasoning.",
        )
        for label, (provider, model_name) in SAMPLER_MODEL_CONFIG.items()
    }
    clustering_agent = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=str,
        agent_name="clustering_agent",
        use_thinking=False,
        temperature=0,
    )

    # --- Semaphores ---
    gpt_oss_sem = asyncio.Semaphore(CONCURRENCY["gpt_oss"])
    sampler_sems = {
        "gemini-3-pro-preview": asyncio.Semaphore(CONCURRENCY["gemini"]),
        "nemotron-3-nano":      asyncio.Semaphore(CONCURRENCY["nemotron"]),
        "qwen3.5:122b":      asyncio.Semaphore(CONCURRENCY["qwen3.5:122b"]),
    }
    row_sem = asyncio.Semaphore(ROW_CONCURRENCY)
    writer_lock = asyncio.Lock()

    # --- Process rows ---
    mode = 'a' if os.path.exists(OUTPUT_CSV) else 'w'
    with open(OUTPUT_CSV, mode, newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        if mode == 'w':
            writer.writeheader()

        async def _bounded_process(row):
            async with row_sem:
                await process_row(
                    row, paraphrase_agent, sampler_agents, clustering_agent,
                    gpt_oss_sem, sampler_sems, writer_lock, out_f, writer,
                )

        tasks = [_bounded_process(row) for row in pending_rows]
        await tqdm.gather(*tasks, desc="Processing atomic facts")

    print(f"\nDone. Output written to {OUTPUT_CSV}")


if __name__ == "__main__":
    asyncio.run(main())
