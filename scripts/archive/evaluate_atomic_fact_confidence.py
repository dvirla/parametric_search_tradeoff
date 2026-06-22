import argparse
import asyncio
import csv
import json
import os
import math
import re
import sys
from typing import List
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

csv.field_size_limit(sys.maxsize)

INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'curated_sharechat', 'atomic_fact_attribution.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'curated_sharechat')

MODEL_ARG_MAP = {
    "gemini":   "gemini-3-pro-preview",
    "nemotron": "nemotron-3-nano:30b",
    "qwen":     "qwen3.5:122b",
}

NUM_SAMPLES = 5

# Maps model_label (as it appears in the CSV) → (provider_name, model_name)
SAMPLER_MODEL_CONFIG = {
    "gemini-3-pro-preview": ("Google", "gemini-3-pro-preview"),
    "nemotron-3-nano:30b":  ("ollama", "nemotron-3-nano:30b"),
    "qwen3.5:122b":         ("ollama", "qwen3.5:122b"),
}


def _row_key(row: dict) -> str:
    return f"{row.get('model','')}|{row.get('problem','')}|{row.get('atomic_fact','').strip()}"


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
Group these responses into clusters based on whether their final answers are semantically similar or equivalent.
- Responses whose final answers convey the same fact or compatible facts belong in the SAME cluster, even if the wording, phrasing, or level of detail differs.
- Equivalent formats should be clustered together (e.g., "3" and "three", "Jan 1 2020" and "January 1, 2020", "USA" and "United States of America", "FDA" and "US Food and Drug Administration").
- Partially overlapping answers that point to the same entity or fact should be clustered together (e.g., "Paris" and "Paris, France", "Einstein" and "Albert Einstein").
- Only place responses in DIFFERENT clusters if their final answers are clearly contradictory or mutually exclusive factual claims.
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


async def main():
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(description="Evaluate atomic fact confidence for a single model.")
    parser.add_argument('--model', required=True, choices=list(MODEL_ARG_MAP.keys()),
                        help="Which model's rows to process: gemini, nemotron, or qwen")
    parser.add_argument(
        '--ollama-base-urls', nargs='+',
        default=['http://localhost:11434'],
        metavar='URL',
        help=(
            "Ollama base URLs, one per parallel worker/GPU instance. "
            "Example: --ollama-base-urls http://localhost:11434 http://localhost:11435 "
            "http://localhost:11436 http://localhost:11437"
        ),
    )
    args = parser.parse_args()

    selected_label = MODEL_ARG_MAP[args.model]
    output_csv = os.path.join(RESULTS_DIR, f'atomic_fact_confidence_{args.model}.csv')
    cache_file = output_csv.replace('.csv', '_phase_cache.json')

    # --- Load input ---
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        input_fieldnames = reader.fieldnames
        rows = list(reader)

    print(f"Loaded {len(rows)} rows from {INPUT_CSV}")
    print(f"Model: {selected_label}  →  output: {output_csv}")

    output_fieldnames = (input_fieldnames or []) + [
        'paraphrased_question',
        'sample_1', 'sample_2', 'sample_3', 'sample_4', 'sample_5',
        'num_clusters', 'cluster_sizes', 'entropy',
    ]

    # --- Load already-written output rows ---
    processed_keys = set()
    if os.path.exists(output_csv):
        with open(output_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_keys.add((row['model'], row['problem'], row['atomic_fact']))
        print(f"Resuming: {len(processed_keys)} rows already written to output")

    # --- Load phase cache (intermediate results between phases) ---
    phase_cache: dict = {}
    if os.path.exists(cache_file):
        with open(cache_file, 'r', encoding='utf-8') as f:
            phase_cache = json.load(f)
        print(f"Phase cache: {len(phase_cache)} entries loaded")

    def _save_cache():
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(phase_cache, f)

    pending_rows = [
        r for r in rows
        if r.get('atomic_fact', '').strip()
        and r.get('model', '') == selected_label
        and (r['model'], r['problem'], r['atomic_fact'].strip()) not in processed_keys
    ]
    print(f"Rows pending final output: {len(pending_rows)}")

    if not pending_rows:
        print("Nothing to do.")
        return

    # --- Initialise agents ---
    # Normalise Ollama URLs to always end with /v1/
    def _normalise_ollama_url(url: str) -> str:
        url = url.rstrip('/')
        return url + '/v1/' if not url.endswith('/v1') else url + '/'

    ollama_urls = [_normalise_ollama_url(u) for u in args.ollama_base_urls]
    num_workers = len(ollama_urls)
    print(f"Initialising agents ({num_workers} sampler worker(s))...")

    paraphrase_agent = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=str,
        agent_name="paraphrase_agent",
        use_thinking=False,
        temperature=0,
        system_prompt=PARAPHRASE_SYSTEM_PROMPT,
        timeout=300,
    )
    provider, model_name = SAMPLER_MODEL_CONFIG[selected_label]
    sampler_agents = [
        BaseAgent(
            provider_name=provider,
            model_name=model_name,
            output_type=str,
            agent_name=f"sampler_agent_{selected_label}_w{i}",
            use_thinking=True,
            system_prompt="Answer the question with very short and concise answer, no need for explanations and reasoning.",
            ollama_base_url=url if provider == "ollama" else None,
        )
        for i, url in enumerate(ollama_urls)
    ]
    clustering_agent = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=str,
        agent_name="clustering_agent",
        use_thinking=False,
        temperature=0,
        timeout=300,
    )

    cache_lock = asyncio.Lock()

    async def _save_cache_async():
        async with cache_lock:
            _save_cache()

    # ── Phase 1: Paraphrase all rows (gpt-oss:120b stays loaded) ────────────
    phase1_todo = [r for r in pending_rows if _row_key(r) not in phase_cache]
    print(f"\nPhase 1 — Paraphrase: {len(phase1_todo)} rows")
    for row in tqdm(phase1_todo, desc="Phase 1: Paraphrasing (gpt-oss)"):
        key = _row_key(row)
        atomic_fact = row.get('atomic_fact', '').strip()
        try:
            result = await paraphrase_agent.arun(atomic_fact)
            question = _parse_paraphrase(result.output)
            if question:
                phase_cache[key] = {'paraphrased_question': question}
                _save_cache()
            else:
                print(f"\nEmpty paraphrase for '{atomic_fact[:60]}...', skipping")
        except Exception as e:
            print(f"\nParaphrase error for '{atomic_fact[:60]}...': {e}")

    # ── Phase 2: Sample all rows in parallel across workers ─────────────────
    phase2_todo = [
        r for r in pending_rows
        if _row_key(r) in phase_cache and 'samples' not in phase_cache[_row_key(r)]
    ]
    print(f"\nPhase 2 — Sampling: {len(phase2_todo)} rows × {NUM_SAMPLES} samples"
          f" across {num_workers} worker(s)")

    row_queue: asyncio.Queue = asyncio.Queue()
    for row in phase2_todo:
        row_queue.put_nowait(row)

    phase2_pbar = tqdm(total=len(phase2_todo), desc=f"Phase 2: Sampling ({selected_label})")

    async def _sampling_worker(agent: BaseAgent):
        while True:
            try:
                row = row_queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            key = _row_key(row)
            question = phase_cache[key]['paraphrased_question']
            samples = []
            for i in range(NUM_SAMPLES):
                try:
                    result = await agent.arun(question)
                    samples.append(result.output)
                except Exception as e:
                    print(f"\nSampler error (sample {i+1}) for '{question[:60]}...': {e}")
                    samples.append("")
            phase_cache[key]['samples'] = samples
            await _save_cache_async()
            phase2_pbar.update(1)

    await asyncio.gather(*[_sampling_worker(a) for a in sampler_agents])
    phase2_pbar.close()

    # ── Phase 3: Cluster + write output (gpt-oss:120b stays loaded) ─────────
    phase3_todo = [
        r for r in pending_rows
        if _row_key(r) in phase_cache and 'samples' in phase_cache[_row_key(r)]
    ]
    print(f"\nPhase 3 — Clustering: {len(phase3_todo)} rows")
    mode = 'a' if os.path.exists(output_csv) else 'w'
    with open(output_csv, mode, newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        if mode == 'w':
            writer.writeheader()

        for row in tqdm(phase3_todo, desc="Phase 3: Clustering (gpt-oss)"):
            key = _row_key(row)
            cached = phase_cache[key]
            question = cached['paraphrased_question']
            samples = cached['samples']

            answers_text = "\n\n".join(
                f"--- Response {i+1} ---\n{s}" for i, s in enumerate(samples)
            )
            cluster_prompt = SEMANTIC_CLUSTERING_PROMPT.format(
                question=question,
                num_answers=NUM_SAMPLES,
                answers_text=answers_text,
            )
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
                print(f"\nClustering error for '{question[:60]}...': {e}")
                clusters = [[i] for i in range(1, NUM_SAMPLES + 1)]

            cluster_sizes = [len(c) for c in clusters]
            entropy = compute_shannon_entropy(cluster_sizes)

            out_row = dict(row)
            out_row['paraphrased_question'] = question
            for i, s in enumerate(samples):
                out_row[f'sample_{i+1}'] = s
            out_row['num_clusters'] = len(clusters)
            out_row['cluster_sizes'] = str(cluster_sizes)
            out_row['entropy'] = round(entropy, 4)

            writer.writerow(out_row)
            out_f.flush()

    print(f"\nDone. Output written to {output_csv}")


if __name__ == "__main__":
    asyncio.run(main())
