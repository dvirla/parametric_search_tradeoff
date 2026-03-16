import csv
import os
import math
import sys
from typing import List
from pydantic import BaseModel, Field
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

INPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_attribution.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), '..', 'results', 'sharechat', 'atomic_fact_confidence.csv')

NUM_SAMPLES = 5

# --- Pydantic Models ---
class ParaphrasedQuestion(BaseModel):
    question: str

class ClusteringResult(BaseModel):
    clusters: List[List[int]] = Field(
        description="List of clusters (1-based indices). Each cluster is a list of answer indices that are semantically equivalent."
    )

# --- Prompts ---
PARAPHRASE_SYSTEM_PROMPT = (
    "You convert atomic facts into concise, answerable factual questions. "
    "Given a fact, produce a single clear question whose correct answer would verify that fact. "
    "Output only the question, nothing else."
)

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

Return the clusters as a list of lists of 1-based indices.
Example for 5 responses where 1,3,5 give the same answer and 2,4 give a different answer: [[1, 3, 5], [2, 4]]
Example for 5 responses that all give the same answer: [[1, 2, 3, 4, 5]]
"""


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


def main():
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

    # --- Initialise agents ---
    print("Initialising agents...")
    paraphrase_agent = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:20b",
        output_type=ParaphrasedQuestion,
        agent_name="paraphrase_agent",
        use_thinking=False,
        temperature=0,
        system_prompt=PARAPHRASE_SYSTEM_PROMPT,
    )
    sampler_agent = BaseAgent(
        provider_name="Google",
        model_name="gemini-3-pro-preview",
        output_type=str,
        agent_name="sampler_agent",
        use_thinking=True,
        system_prompt="Answer the question with very short and concise answer, no need for explanations and reasoning."
    )
    clustering_agent = BaseAgent(
        provider_name="ollama",
        model_name="gpt-oss:120b",
        output_type=ClusteringResult,
        agent_name="clustering_agent",
        use_thinking=False,
        temperature=0,
    )

    # --- Process rows ---
    mode = 'a' if os.path.exists(OUTPUT_CSV) else 'w'
    with open(OUTPUT_CSV, mode, newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=output_fieldnames)
        if mode == 'w':
            writer.writeheader()

        for row in tqdm(rows, desc="Processing atomic facts"):
            atomic_fact = row.get('atomic_fact', '').strip()
            if not atomic_fact:
                continue

            key = (row['model'], row['problem'], atomic_fact)
            if key in processed_keys:
                continue

            # Step 1 — Paraphrase
            try:
                para_result = paraphrase_agent.run(atomic_fact)
                paraphrased_question = para_result.output.question
            except Exception as e:
                print(f"\nParaphrase error for fact '{atomic_fact[:60]}...': {e}")
                paraphrased_question = f"Is it true that: {atomic_fact}?"

            # Step 2 — Sample ×5
            samples = []
            for i in range(NUM_SAMPLES):
                try:
                    sample_result = sampler_agent.run(paraphrased_question)
                    samples.append(sample_result.output)
                except Exception as e:
                    print(f"\nSampler error (sample {i+1}) for '{paraphrased_question[:60]}...': {e}")
                    samples.append("")

            # Step 3 — Cluster + Entropy
            answers_text = "\n\n".join(
                f"--- Response {i+1} ---\n{s}" for i, s in enumerate(samples)
            )
            cluster_prompt = SEMANTIC_CLUSTERING_PROMPT.format(
                question=paraphrased_question,
                num_answers=NUM_SAMPLES,
                answers_text=answers_text,
            )
            try:
                cluster_result = clustering_agent.run(cluster_prompt)
                clusters = cluster_result.output.clusters

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

            writer.writerow(out_row)
            out_f.flush()

    print(f"\nDone. Output written to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
