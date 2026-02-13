import json
import csv
import os
import math
import argparse
import glob
from typing import List, Dict, Optional
from collections import defaultdict
from pydantic import BaseModel, Field
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.base_agent import BaseAgent
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
DEFAULT_JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "Google")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-3-flash-preview")

LOGS_BASE = os.path.join(os.path.dirname(__file__), '..', 'logs')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'semantic_entropy')

# The 7 target model/dataset combinations
TARGET_CONFIGS = [
    {
        "dataset": "facts_one_hop",
        "model": "claude_4_5_haiku",
        "glob_pattern": "facts-search_no_search_claude-haiku-4-5-20251001_run_*.json",
    },
    {
        "dataset": "facts_one_hop",
        "model": "gemini_3_pro",
        "glob_pattern": "facts-search_no_search_gemini-3-pro-preview_run_*.json",
    },
    {
        "dataset": "facts_one_hop",
        "model": "nemotron-3-nano",
        "glob_pattern": "facts-search_no_search_nemotron-3-nano:30b_run_*.json",
    },
    {
        "dataset": "facts_one_hop",
        "model": "glm_4_7",
        "glob_pattern": "facts-search_no_search_glm-4.7-flash:bf16_run_*.json",
    },
    {
        "dataset": "facts_parametric",
        "model": "claude_4_5_haiku",
        "glob_pattern": "facts-param_no_search_claude-haiku-4-5-20251001_run_*.json",
    },    
    {
        "dataset": "facts_parametric",
        "model": "gemini_3_pro",
        "glob_pattern": "facts_parametric_eval_results_gemini_3_pro_no_search_run_*.json",
    },
    {
        "dataset": "natural_questions",
        "model": "glm-4.7-flash",
        "glob_pattern": "natural_questions_eval_results_glm_4_7_flash_no_search_run_*.json",
    },
    {
        "dataset": "natural_questions",
        "model": "nemotron-3-nano",
        "glob_pattern": "nq_no_search_nemotron-3-nano:30b_run_*.json",
    },
    {
        "dataset": "popqa",
        "model": "gemini_3_pro",
        "glob_pattern": "popqa_no_search_gemini-3-pro-preview_run_*.json",
    },
    {
        "dataset": "popqa",
        "model": "nemotron_3_nano",
        "glob_pattern": "popqa_no_search_nemotron-3-nano:30b_run_*.json",
    }       
]

# --- Pydantic Model for LLM Judge ---
class ClusteringResult(BaseModel):
    clusters: List[List[int]] = Field(description="List of clusters (1-based indices). Each cluster is a list of answer indices that are semantically equivalent.")

# --- Clustering Prompt ---
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


def load_runs(config: Dict) -> Dict[str, List[Dict]]:
    """Load all run files for a config and index entries by problem."""
    dataset = config["dataset"]
    model = config["model"]
    pattern = config["glob_pattern"]

    # Find the model directory
    dataset_dir = os.path.join(LOGS_BASE, dataset)
    # Search for the glob pattern in all subdirectories of the dataset dir
    search_pattern = os.path.join(dataset_dir, '**', pattern)
    files = sorted(glob.glob(search_pattern, recursive=True))

    if not files:
        print(f"  WARNING: No files found for {dataset}/{model} with pattern {pattern}")
        return {}

    print(f"  Found {len(files)} run files for {dataset}/{model}")

    # Index by problem: problem -> list of entries (one per run)
    problem_entries: Dict[str, List[Dict]] = defaultdict(list)
    for filepath in files:
        with open(filepath, 'r') as f:
            data = json.load(f)
        for entry in data:
            problem_entries[entry['problem']].append(entry)

    return dict(problem_entries)


def clean_problem(problem: str) -> str:
    """Removes the instruction suffix from the problem string."""
    separator = "\n\nYour response should be in the following format:"
    popqa_seperator = "Portuguese\n\nNow answer the following:\n\nQuestion: "
    if separator in problem:
        return problem.split(separator)[0].strip()
    elif popqa_seperator in problem:
        problem = problem.split(popqa_seperator)[1].strip().split('\nAnswer:')[0]
    return problem.strip()


def get_correct_answer_str(entry: Dict) -> str:
    """Get correct answer as a string, handling both str and list formats."""
    ca = entry.get('correct_answer', '')
    if isinstance(ca, list):
        return ', '.join(str(a) for a in ca)
    return str(ca)


def process_config(config: Dict, judge: BaseAgent, output_dir: str) -> Optional[Dict]:
    """Process a single model/dataset config and write per-question CSV."""
    dataset = config["dataset"]
    model = config["model"]
    print(f"\nProcessing {dataset}/{model}...")

    problem_entries = load_runs(config)
    if not problem_entries:
        return None

    output_filename = f"semantic_entropy_{dataset}__{model}.csv"
    output_path = os.path.join(output_dir, output_filename)

    fieldnames = [
        "problem", "num_runs", "num_clusters", "entropy",
        "answers_summary", "cluster_sizes", "correct_answer", "majority_answer_correct"
    ]

    # Resume support: load already-processed problems
    processed_problems = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            if reader.fieldnames == fieldnames:
                for row in reader:
                    processed_problems.add(row['problem'])
        print(f"  Resuming: {len(processed_problems)} problems already processed")

    # Filter to problems with all runs and not yet processed
    problems_to_process = []
    for problem, entries in problem_entries.items():
        problem_clean = clean_problem(problem)
        if problem_clean not in processed_problems:
            problems_to_process.append((problem, entries))

    if not problems_to_process:
        print(f"  All problems already processed, skipping.")
        # Still read the CSV for summary stats
        return read_summary_from_csv(output_path, dataset, model)

    # Open file for writing/appending
    mode = 'a' if os.path.exists(output_path) else 'w'
    results_for_summary = []

    with open(output_path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if mode == 'w':
            writer.writeheader()

        for problem, entries in tqdm(problems_to_process, desc=f"  {dataset}/{model}"):
            problem_clean = clean_problem(problem)
            raw_responses = [e['sampler_response'] for e in entries]
            correct_answer = get_correct_answer_str(entries[0])
            num_runs = len(raw_responses)

            # Build prompt for LLM judge — pass full raw responses
            answers_text = "\n\n".join(
                f"--- Response {i+1} ---\n{r}" for i, r in enumerate(raw_responses)
            )
            prompt = SEMANTIC_CLUSTERING_PROMPT.format(
                question=problem_clean[:1000],
                num_answers=num_runs,
                answers_text=answers_text
            )

            # Call judge
            try:
                result = judge.run(prompt)
                clustering = result.output
                clusters = clustering.clusters

                # Validate clusters: all indices should be 1-based and cover all answers
                all_indices = set()
                for cluster in clusters:
                    all_indices.update(cluster)
                expected = set(range(1, num_runs + 1))
                if all_indices != expected:
                    print(f"  WARNING: Cluster indices {all_indices} != expected {expected} for problem: {problem_clean[:50]}...")
                    # Fall back: treat each answer as its own cluster
                    clusters = [[i] for i in range(1, num_runs + 1)]

            except Exception as e:
                print(f"  ERROR clustering for problem '{problem_clean[:50]}...': {e}")
                # Fallback: each answer is its own cluster (max entropy)
                clusters = [[i] for i in range(1, num_runs + 1)]

            cluster_sizes = [len(c) for c in clusters]
            entropy = compute_shannon_entropy(cluster_sizes)

            # Check if majority answer is correct
            largest_cluster = max(clusters, key=len)
            majority_idx = largest_cluster[0] - 1  # Convert to 0-based
            majority_correct = entries[majority_idx].get('sampler_correct', False)

            answers_summary = " | ".join(f"[{i+1}] {r[:80]}" for i, r in enumerate(raw_responses))

            row = {
                "problem": problem_clean,
                "num_runs": num_runs,
                "num_clusters": len(clusters),
                "entropy": round(entropy, 4),
                "answers_summary": answers_summary,
                "cluster_sizes": str(cluster_sizes),
                "correct_answer": correct_answer,
                "majority_answer_correct": majority_correct,
            }
            writer.writerow(row)
            f.flush()
            results_for_summary.append(row)

    # Build summary stats from full CSV
    return read_summary_from_csv(output_path, dataset, model)


def read_summary_from_csv(csv_path: str, dataset: str, model: str) -> Optional[Dict]:
    """Read a per-question CSV and compute aggregate summary stats."""
    if not os.path.exists(csv_path):
        return None

    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    entropies = [float(r['entropy']) for r in rows]
    num_clusters_list = [int(r['num_clusters']) for r in rows]
    majority_correct_list = [r['majority_answer_correct'].lower() == 'true' for r in rows]

    entropies_sorted = sorted(entropies)
    n = len(entropies_sorted)
    median = entropies_sorted[n // 2] if n % 2 == 1 else (entropies_sorted[n // 2 - 1] + entropies_sorted[n // 2]) / 2

    return {
        "dataset": dataset,
        "model": model,
        "num_questions": len(rows),
        "mean_entropy": round(sum(entropies) / len(entropies), 4),
        "median_entropy": round(median, 4),
        "max_entropy": round(max(entropies), 4),
        "min_entropy": round(min(entropies), 4),
        "mean_num_clusters": round(sum(num_clusters_list) / len(num_clusters_list), 4),
        "pct_zero_entropy": round(sum(1 for e in entropies if e == 0) / len(entropies) * 100, 2),
        "pct_majority_correct": round(sum(majority_correct_list) / len(majority_correct_list) * 100, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate semantic entropy across no-search runs using LLM-as-a-judge clustering.")
    parser.add_argument('--judge_provider', type=str, default=DEFAULT_JUDGE_PROVIDER, help=f"LLM provider for the judge (default: {DEFAULT_JUDGE_PROVIDER})")
    parser.add_argument('--judge_model', type=str, default=DEFAULT_JUDGE_MODEL, help=f"LLM model for the judge (default: {DEFAULT_JUDGE_MODEL})")
    parser.add_argument('--filter_dataset', type=str, default=None, help="Only process this dataset (e.g., facts_one_hop)")
    parser.add_argument('--filter_model', type=str, default=None, help="Only process this model (e.g., glm_4_7)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Initialize judge
    print(f"Initializing judge: {args.judge_provider}/{args.judge_model}")
    judge = BaseAgent(
        model_name=args.judge_model,
        provider_name=args.judge_provider,
        output_type=ClusteringResult,
        agent_name="semantic_entropy_judge",
        use_thinking=False,
        temperature=0,
    )

    # Filter configs
    configs = TARGET_CONFIGS
    if args.filter_dataset:
        configs = [c for c in configs if c["dataset"] == args.filter_dataset]
    if args.filter_model:
        configs = [c for c in configs if c["model"] == args.filter_model]

    if not configs:
        print("No matching configurations found. Check --filter_dataset and --filter_model values.")
        print("Available datasets:", sorted(set(c["dataset"] for c in TARGET_CONFIGS)))
        print("Available models:", sorted(set(c["model"] for c in TARGET_CONFIGS)))
        return

    print(f"Processing {len(configs)} configuration(s)...")

    # Process each config
    summaries = []
    for config in configs:
        summary = process_config(config, judge, RESULTS_DIR)
        if summary:
            summaries.append(summary)

    # Write aggregate summary CSV
    if summaries:
        summary_path = os.path.join(RESULTS_DIR, "semantic_entropy_summary.csv")
        summary_fieldnames = [
            "dataset", "model", "num_questions", "mean_entropy", "median_entropy",
            "max_entropy", "min_entropy", "mean_num_clusters", "pct_zero_entropy", "pct_majority_correct"
        ]
        with open(summary_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
            writer.writeheader()
            for s in summaries:
                writer.writerow(s)
        print(f"\nSummary written to {summary_path}")

        # Print summary table
        print("\n" + "=" * 90)
        print(f"{'Dataset':<22} {'Model':<20} {'N':>4} {'Mean H':>7} {'Med H':>7} {'0-Ent%':>7} {'Maj%':>6}")
        print("-" * 90)
        for s in summaries:
            print(f"{s['dataset']:<22} {s['model']:<20} {s['num_questions']:>4} "
                  f"{s['mean_entropy']:>7.4f} {s['median_entropy']:>7.4f} "
                  f"{s['pct_zero_entropy']:>6.1f}% {s['pct_majority_correct']:>5.1f}%")
        print("=" * 90)


if __name__ == "__main__":
    main()
