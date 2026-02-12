"""
Analyze semantic entropy results and visualize relationships to:
1. Accuracy (majority_answer_correct) per model & dataset
2. Pre-search confidence levels from analysis_by_gemini_3_flash.csv
3. Search amounts per problem from baseline JSON files

Usage:
    python scripts/analyze_semantic_entropy.py                        # vanilla baseline
    python scripts/analyze_semantic_entropy.py --variant with_sys_instruct  # sys_instruct baseline
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
import numpy as np
from collections import defaultdict
from scipy import stats

# Set style
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
SEMANTIC_ENTROPY_DIR = os.path.join(BASE_DIR, 'semantic_entropy')

EPISTEMIC_STATE_ORDER = ['IGNORANCE', 'AMBIGUITY', 'HIGH_CERTAINTY', 'DECISIVE']

# Maps semantic entropy file naming → (dataset_dir, model_dir) in logs/
MODEL_DIR_MAP = {
    ('facts_one_hop', 'claude_4_5_haiku'): ('facts_one_hop', 'claude_4_5_haiku'),
    ('facts_parametric', 'claude_4_5_haiku'): ('facts_parametric', 'claude_4_5_haiku'),
    ('facts_one_hop', 'gemini_3_pro'): ('facts_one_hop', 'gemini_3_pro'),
    ('facts_parametric', 'gemini_3_pro'): ('facts_parametric', 'gemini_3_pro'),
    ('facts_one_hop', 'nemotron-3-nano'): ('facts_one_hop', 'nemotron-3-nano'),
    ('facts_one_hop', 'glm_4_7'): ('facts_one_hop', 'glm_4_7'),
    ('natural_questions', 'glm-4.7-flash'): ('natural_questions', 'glm-4.7-flash'),
    ('natural_questions', 'nemotron-3-nano'): ('natural_questions', 'nemotron-3-nano'),
}

# Maps (dataset, model) → glob pattern for no-search run JSONs (from calculate_semantic_entropy.py TARGET_CONFIGS)
NO_SEARCH_GLOB_MAP = {
    ('facts_one_hop', 'claude_4_5_haiku'): 'facts-search_no_search_claude-haiku-4-5-20251001_run_*.json',
    ('facts_one_hop', 'gemini_3_pro'): 'facts-search_no_search_gemini-3-pro-preview_run_*.json',
    ('facts_one_hop', 'nemotron-3-nano'): 'facts-search_no_search_nemotron-3-nano:30b_run_*.json',
    ('facts_one_hop', 'glm_4_7'): 'facts-search_no_search_glm-4.7-flash:bf16_run_*.json',
    ('natural_questions', 'glm-4.7-flash'): 'natural_questions_eval_results_glm_4_7_flash_no_search_run_*.json',
    ('natural_questions', 'nemotron-3-nano'): 'nq_no_search_nemotron-3-nano:30b_run_*.json',
    ('facts_parametric', 'gemini_3_pro'): 'facts_parametric_eval_results_gemini_3_pro_no_search_run_*.json',
    ('facts_parametric', 'claude_4_5_haiku'): 'facts-param_no_search_claude-haiku-4-5-20251001_run_*.json',
}

# Normalize model names for cross-dataset aggregation
MODEL_DISPLAY_NAMES = {
    'nemotron-3-nano': 'Nemotron 3 Nano',
    'glm_4_7': 'GLM 4.7 Flash',
    'glm-4.7-flash': 'GLM 4.7 Flash',
    'gemini_3_pro': 'Gemini 3 Pro',
    'claude_4_5_haiku': 'Claude 4.5 Haiku',
}


def clean_problem(problem: str) -> str:
    separator = "\n\nYour response should be in the following format:"
    if separator in problem:
        return problem.split(separator)[0].strip()
    return problem.strip()


def get_problem_id(problem: str) -> str:
    return clean_problem(problem)[:50] + "..."


def load_semantic_entropy_files() -> pd.DataFrame:
    """Load all semantic entropy CSVs and return a single DataFrame."""
    pattern = os.path.join(SEMANTIC_ENTROPY_DIR, 'semantic_entropy_*__*.csv')
    files = glob.glob(pattern)

    frames = []
    for f in files:
        basename = os.path.basename(f)
        # Parse dataset and model from filename: semantic_entropy_{dataset}__{model}.csv
        parts = basename.replace('semantic_entropy_', '').replace('.csv', '').split('__')
        if len(parts) != 2:
            continue
        dataset, model = parts

        df = pd.read_csv(f)
        df['dataset'] = dataset
        df['model'] = model
        df['problem_id'] = df['problem'].apply(get_problem_id)
        frames.append(df)

    if not frames:
        print("No semantic entropy files found!")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(combined)} semantic entropy records across {len(frames)} model/dataset combos")
    return combined


def load_analysis_csv(dataset_dir: str, model_dir: str, variant: str = 'vanilla') -> pd.DataFrame | None:
    """Load analysis_by_gemini_3_flash.csv for a given model/dataset."""
    if variant == 'with_sys_instruct':
        filename = 'baseline_with_sys_instruct_analysis_by_gemini_3_flash.csv'
    else:
        filename = 'analysis_by_gemini_3_flash.csv'

    csv_path = os.path.join(BASE_DIR, dataset_dir, model_dir, filename)
    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    # Rename columns to match conventions
    column_mapping = {
        'judge1_confidence': 'epistemic_state',  # legacy CSV format
        'judge2_hypothesis_correct': 'hypothesis_correct',
        'judge2_query_biased': 'is_search_query_biased',
        'judge2_snippet_has_answer': 'snippet_has_answer',
        'judge2_answer_flipped': 'answer_flipped',
    }
    # Only rename judge1_confidence if epistemic_state doesn't already exist (new CSV format)
    if 'epistemic_state' in df.columns:
        del column_mapping['judge1_confidence']
    df.rename(columns=column_mapping, inplace=True)

    # Map old epistemic state values to new ones if present
    old_to_new = {
        'TABULA_RASA': 'IGNORANCE',
        'WEAK_GUESS': 'AMBIGUITY',
        'STRONG_HYPOTHESIS': 'HIGH_CERTAINTY',
        'NO_SEARCH': 'DECISIVE',
    }
    if 'epistemic_state' in df.columns:
        df['epistemic_state'] = df['epistemic_state'].replace(old_to_new)
    return df


def load_baseline_search_calls(dataset_dir: str, model_dir: str, variant: str = 'vanilla') -> dict:
    """Load baseline JSON and extract search call counts per problem_id."""
    model_path = os.path.join(BASE_DIR, dataset_dir, model_dir)

    if variant == 'with_sys_instruct':
        # Match sys_instruct baseline JSONs (not traces)
        patterns = [
            os.path.join(model_path, '*baseline*with_sys_instruct*.json'),
        ]
        files = []
        for p in patterns:
            files = [f for f in glob.glob(p) if 'traces' not in f]
            if files:
                break
    else:
        # Match vanilla baseline JSONs (not traces, not sys_instruct)
        patterns = [
            os.path.join(model_path, '*baseline*run_1.json'),
            os.path.join(model_path, '*baseline*run*.json'),
        ]
        files = []
        for p in patterns:
            files = [f for f in glob.glob(p) if 'traces' not in f and 'sys_instruct' not in f]
            if files:
                break

    if not files:
        return {}

    filepath = files[0]
    search_map = {}
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        for entry in data:
            pid = get_problem_id(entry.get('problem', ''))
            search_map[pid] = entry.get('sampler_search_calls', 0)
    except Exception as e:
        print(f"  Error loading {filepath}: {e}")

    return search_map


def load_no_search_mean_accuracy(dataset: str, model: str, dataset_dir: str, model_dir: str) -> dict:
    """Load all no-search run JSONs for a (dataset, model) pair and compute mean(sampler_correct) per problem_id."""
    glob_pattern = NO_SEARCH_GLOB_MAP.get((dataset, model))
    if not glob_pattern:
        return {}

    model_path = os.path.join(BASE_DIR, dataset_dir, model_dir)
    files = sorted(glob.glob(os.path.join(model_path, glob_pattern)))

    if not files:
        return {}

    # Collect correctness per problem across all runs
    problem_correct: dict[str, list[float]] = defaultdict(list)
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            for entry in data:
                pid = get_problem_id(entry.get('problem', ''))
                correct = 1.0 if entry.get('sampler_correct', False) else 0.0
                problem_correct[pid].append(correct)
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    # Compute mean accuracy per problem
    accuracy_map = {pid: sum(vals) / len(vals) for pid, vals in problem_correct.items() if vals}
    print(f"  Loaded mean accuracy from {len(files)} no-search runs ({len(accuracy_map)} problems)")
    return accuracy_map


def build_merged_data(variant: str = 'vanilla') -> pd.DataFrame:
    """Merge semantic entropy with analysis CSV and search call data."""
    se_df = load_semantic_entropy_files()
    if se_df.empty:
        return se_df

    merged_frames = []

    for (dataset, model), (dataset_dir, model_dir) in MODEL_DIR_MAP.items():
        subset = se_df[(se_df['dataset'] == dataset) & (se_df['model'] == model)].copy()
        if subset.empty:
            print(f"  No semantic entropy data for {dataset}/{model}")
            continue

        label = f"{dataset}/{model}"
        print(f"\nProcessing {label} ({len(subset)} problems)...")

        # Load analysis CSV
        analysis_df = load_analysis_csv(dataset_dir, model_dir, variant)
        if analysis_df is not None:
            subset = subset.merge(analysis_df, on='problem_id', how='left')
            matched = subset['epistemic_state'].notna().sum()
            print(f"  Matched {matched}/{len(subset)} with analysis CSV")
        else:
            print(f"  No analysis CSV found for {label} (variant={variant})")

        # Load search calls
        search_map = load_baseline_search_calls(dataset_dir, model_dir, variant)
        if search_map:
            subset['num_searches'] = subset['problem_id'].map(search_map)
            matched = subset['num_searches'].notna().sum()
            print(f"  Matched {matched}/{len(subset)} with search call data")
        else:
            print(f"  No baseline search data found for {label} (variant={variant})")

        # Load mean accuracy from no-search runs
        accuracy_map = load_no_search_mean_accuracy(dataset, model, dataset_dir, model_dir)
        if accuracy_map:
            subset['mean_accuracy'] = subset['problem_id'].map(accuracy_map)
            matched = subset['mean_accuracy'].notna().sum()
            print(f"  Matched {matched}/{len(subset)} with mean accuracy data")
        else:
            print(f"  No no-search run data found for {label}")

        subset['label'] = label
        merged_frames.append(subset)

    if not merged_frames:
        return pd.DataFrame()

    return pd.concat(merged_frames, ignore_index=True)


# ─── Plotting functions ────────────────────────────────────────────────────────


def plot_entropy_vs_accuracy(df: pd.DataFrame, output_dir: str):
    """Plot 1: Semantic entropy distribution by correctness for each model/dataset."""
    labels = df['label'].unique()
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, label in enumerate(sorted(labels)):
        ax = axes[i]
        sub = df[df['label'] == label].copy()
        sub['Correct'] = sub['majority_answer_correct'].map(
            {True: 'Correct', False: 'Incorrect', 'True': 'Correct', 'False': 'Incorrect'}
        )

        sns.boxplot(data=sub, x='Correct', y='entropy', palette='Set2', ax=ax,
                    order=['Correct', 'Incorrect'])
        sns.stripplot(data=sub, x='Correct', y='entropy', color='black', alpha=0.3,
                      jitter=True, size=3, ax=ax, order=['Correct', 'Incorrect'])

        # Counts
        correct_n = (sub['Correct'] == 'Correct').sum()
        incorrect_n = (sub['Correct'] == 'Incorrect').sum()
        ax.set_title(f"{label}\n(correct={correct_n}, incorrect={incorrect_n})", fontsize=10)
        ax.set_ylabel('Semantic Entropy' if i == 0 else '')
        ax.set_xlabel('')

    fig.suptitle('Semantic Entropy vs. Accuracy', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_vs_accuracy.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_vs_accuracy.png")

    # Also plot mean entropy bar chart
    summary = df.copy()
    summary['Correct'] = summary['majority_answer_correct'].map(
        {True: 'Correct', False: 'Incorrect', 'True': 'Correct', 'False': 'Incorrect'}
    )
    means = summary.groupby(['label', 'Correct'])['entropy'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(max(8, 3 * n), 5))
    sns.barplot(data=means, x='label', y='entropy', hue='Correct', palette='Set2', ax=ax)
    ax.set_title('Mean Semantic Entropy by Accuracy')
    ax.set_ylabel('Mean Entropy')
    ax.set_xlabel('')
    plt.xticks(rotation=25, ha='right')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_entropy_by_accuracy.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated mean_entropy_by_accuracy.png")


def plot_entropy_vs_confidence(df: pd.DataFrame, output_dir: str):
    """Plot 2: Semantic entropy distribution by pre-search confidence level."""
    plot_df = df.dropna(subset=['epistemic_state']).copy()
    valid_conf = [c for c in EPISTEMIC_STATE_ORDER if c in plot_df['epistemic_state'].values]
    plot_df = plot_df[plot_df['epistemic_state'].isin(valid_conf)]

    if plot_df.empty:
        print("Skipping entropy_vs_confidence: no valid confidence data.")
        return

    labels = sorted(plot_df['label'].unique())
    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, label in enumerate(labels):
        ax = axes[i]
        sub = plot_df[plot_df['label'] == label]

        sns.boxplot(data=sub, x='epistemic_state', y='entropy', palette='muted',
                    ax=ax, order=valid_conf)
        sns.stripplot(data=sub, x='epistemic_state', y='entropy', color='black',
                      alpha=0.3, jitter=True, size=3, ax=ax, order=valid_conf)

        ax.set_title(f"{label} (n={len(sub)})", fontsize=10)
        ax.set_ylabel('Semantic Entropy' if i == 0 else '')
        ax.set_xlabel('Epistemic State')
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Semantic Entropy vs. Epistemic State', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_vs_epistemic_state.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_vs_epistemic_state.png")

    # Aggregated view: mean entropy per confidence level across all models
    agg = plot_df.groupby(['label', 'epistemic_state'])['entropy'].agg(['mean', 'count']).reset_index()
    agg.columns = ['label', 'epistemic_state', 'mean_entropy', 'count']

    fig, ax = plt.subplots(figsize=(max(8, 3 * n), 5))
    sns.barplot(data=agg, x='epistemic_state', y='mean_entropy', hue='label',
                palette='viridis', ax=ax, order=valid_conf)
    ax.set_title('Mean Semantic Entropy by Epistemic State')
    ax.set_ylabel('Mean Entropy')
    ax.set_xlabel('Epistemic State')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=7)
    plt.legend(title='Model/Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_entropy_by_epistemic_state.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated mean_entropy_by_epistemic_state.png")


def plot_entropy_vs_searches(df: pd.DataFrame, output_dir: str):
    """Plot 3: Semantic entropy vs number of searches per problem."""
    plot_df = df.dropna(subset=['num_searches']).copy()
    plot_df['num_searches'] = plot_df['num_searches'].astype(int)

    if plot_df.empty:
        print("Skipping entropy_vs_searches: no search data.")
        return

    labels = sorted(plot_df['label'].unique())
    n = len(labels)

    # Scatter plot per model
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, label in enumerate(labels):
        ax = axes[i]
        sub = plot_df[plot_df['label'] == label]

        ax.scatter(sub['entropy'], sub['num_searches'], alpha=0.5, s=30, edgecolors='w', linewidth=0.5)

        # Regression line
        if len(sub) > 2:
            z = np.polyfit(sub['entropy'], sub['num_searches'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(sub['entropy'].min(), sub['entropy'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=1.5)

            r, pval = stats.pearsonr(sub['entropy'], sub['num_searches'])
            ax.annotate(f"r={r:.3f}, p={pval:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

        ax.set_title(f"{label} (n={len(sub)})", fontsize=10)
        ax.set_xlabel('Semantic Entropy')
        ax.set_ylabel('Number of Searches' if i == 0 else '')

    fig.suptitle('Semantic Entropy vs. Search Count', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_vs_searches_scatter.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_vs_searches_scatter.png")

    # Box plot: bin entropy into ranges
    plot_df['entropy_bin'] = pd.cut(
        plot_df['entropy'],
        bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
        labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]']
    )

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, label in enumerate(labels):
        ax = axes[i]
        sub = plot_df[plot_df['label'] == label]

        sns.boxplot(data=sub, x='entropy_bin', y='num_searches', palette='coolwarm', ax=ax)
        ax.set_title(f"{label} (n={len(sub)})", fontsize=10)
        ax.set_xlabel('Entropy Range')
        ax.set_ylabel('Number of Searches' if i == 0 else '')
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Search Count by Entropy Range', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'searches_by_entropy_bin.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated searches_by_entropy_bin.png")


def plot_combined_heatmap(df: pd.DataFrame, output_dir: str):
    """Heatmap: entropy bins vs confidence level, colored by mean search count."""
    plot_df = df.dropna(subset=['epistemic_state', 'num_searches']).copy()
    valid_conf = [c for c in EPISTEMIC_STATE_ORDER if c in plot_df['epistemic_state'].values]
    plot_df = plot_df[plot_df['epistemic_state'].isin(valid_conf)]

    if plot_df.empty:
        print("Skipping combined heatmap: insufficient data.")
        return

    plot_df['entropy_bin'] = pd.cut(
        plot_df['entropy'],
        bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
        labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]']
    )

    pivot = plot_df.pivot_table(
        values='num_searches', index='entropy_bin', columns='epistemic_state',
        aggfunc='mean'
    )
    pivot = pivot.reindex(columns=valid_conf)

    # Count pivot for annotations
    count_pivot = plot_df.pivot_table(
        values='num_searches', index='entropy_bin', columns='epistemic_state',
        aggfunc='count'
    ).reindex(columns=valid_conf).fillna(0).astype(int)

    # Combined annotations: "mean (n=count)"
    annot = pd.DataFrame('', index=pivot.index, columns=pivot.columns)
    for r in pivot.index:
        for c in pivot.columns:
            val = pivot.loc[r, c] if pd.notna(pivot.loc[r, c]) else 0
            cnt = count_pivot.loc[r, c] if r in count_pivot.index and c in count_pivot.columns else 0
            annot.loc[r, c] = f"{val:.1f}\n(n={int(cnt)})" if cnt > 0 else ""

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=annot, fmt='', cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Mean Searches'})
    ax.set_title('Mean Search Count by Entropy & Epistemic State (All Models)')
    ax.set_xlabel('Epistemic State')
    ax.set_ylabel('Semantic Entropy Range')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_epistemic_state_search_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_epistemic_state_search_heatmap.png")


def plot_entropy_vs_mean_accuracy(df: pd.DataFrame, output_dir: str):
    """Scatter plot and bar chart of entropy vs continuous mean accuracy from no-search runs."""
    plot_df = df.dropna(subset=['mean_accuracy']).copy()
    if plot_df.empty:
        print("Skipping entropy_vs_mean_accuracy: no mean_accuracy data.")
        return

    labels = sorted(plot_df['label'].unique())
    n = len(labels)

    # --- Scatter plot with regression line per model ---
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True, squeeze=False)
    axes = axes[0]

    for i, label in enumerate(labels):
        ax = axes[i]
        sub = plot_df[plot_df['label'] == label]

        ax.scatter(sub['entropy'], sub['mean_accuracy'], alpha=0.5, s=30, edgecolors='w', linewidth=0.5)

        if len(sub) > 2:
            z = np.polyfit(sub['entropy'], sub['mean_accuracy'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(sub['entropy'].min(), sub['entropy'].max(), 100)
            ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=1.5)

            r, pval = stats.pearsonr(sub['entropy'], sub['mean_accuracy'])
            ax.annotate(f"r={r:.3f}, p={pval:.3f}", xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=8, va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

        ax.set_title(f"{label} (n={len(sub)})", fontsize=10)
        ax.set_xlabel('Semantic Entropy')
        ax.set_ylabel('Mean Accuracy (no-search runs)' if i == 0 else '')
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle('Semantic Entropy vs. Mean Expected Accuracy', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_vs_mean_accuracy_scatter.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_vs_mean_accuracy_scatter.png")

    # --- Bar chart: mean expected accuracy by entropy bin ---
    plot_df['entropy_bin'] = pd.cut(
        plot_df['entropy'],
        bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
        labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]']
    )

    agg = plot_df.groupby(['label', 'entropy_bin'], observed=False)['mean_accuracy'].agg(['mean', 'count']).reset_index()
    agg.columns = ['label', 'entropy_bin', 'mean_accuracy', 'count']
    agg = agg[agg['count'] > 0]

    fig, ax = plt.subplots(figsize=(max(10, 3 * n), 5))
    sns.barplot(data=agg, x='entropy_bin', y='mean_accuracy', hue='label', palette='viridis', ax=ax)
    ax.set_title('Mean Expected Accuracy by Entropy Bin')
    ax.set_ylabel('Mean Accuracy')
    ax.set_xlabel('Entropy Range')
    ax.set_ylim(0, 1.05)
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=7)
    plt.legend(title='Model/Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_accuracy_by_entropy_bin.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated mean_accuracy_by_entropy_bin.png")


def compute_and_plot_ece(df: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Compute Expected Calibration Error and plot reliability diagram.

    Maps entropy → confidence via confidence = 1 - entropy / log2(5).
    Returns a DataFrame with per-model ECE results.
    """
    plot_df = df.dropna(subset=['mean_accuracy']).copy()
    if plot_df.empty:
        print("Skipping ECE: no mean_accuracy data.")
        return pd.DataFrame()

    max_entropy = np.log2(5)
    plot_df['confidence'] = (1 - plot_df['entropy'] / max_entropy).clip(0, 1)

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)

    labels = sorted(plot_df['label'].unique())
    ece_records = []

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')

    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    for idx, label in enumerate(labels):
        sub = plot_df[plot_df['label'] == label]
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for j in range(n_bins):
            lo, hi = bin_edges[j], bin_edges[j + 1]
            in_bin = sub[(sub['confidence'] >= lo) & (sub['confidence'] < hi if j < n_bins - 1 else sub['confidence'] <= hi)]
            if len(in_bin) > 0:
                bin_accs.append(in_bin['mean_accuracy'].mean())
                bin_confs.append(in_bin['confidence'].mean())
                bin_counts.append(len(in_bin))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)

        # Compute ECE
        total = sum(bin_counts)
        ece = sum(
            (count / total) * abs(acc - conf)
            for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
            if count > 0
        ) if total > 0 else np.nan

        ece_records.append({
            'Model/Dataset': label,
            'ECE': round(ece, 4) if not np.isnan(ece) else np.nan,
            'N': total,
            'Mean_Confidence': round(sub['confidence'].mean(), 4),
            'Mean_Accuracy': round(sub['mean_accuracy'].mean(), 4),
        })

        # Plot reliability diagram
        valid = [i for i in range(n_bins) if bin_counts[i] > 0]
        if valid:
            ax.plot(
                [bin_confs[i] for i in valid],
                [bin_accs[i] for i in valid],
                'o-', color=colors[idx], label=f"{label} (ECE={ece:.3f})", markersize=5
            )

    ax.set_xlabel('Confidence (1 - entropy/log2(5))', fontsize=12)
    ax.set_ylabel('Mean Accuracy', fontsize=12)
    ax.set_title('Reliability Diagram (ECE)', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability_diagram_ece.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated reliability_diagram_ece.png")

    # --- Aggregated reliability diagram: pool by model across datasets ---
    plot_df['display_name'] = plot_df['model'].map(MODEL_DISPLAY_NAMES).fillna(plot_df['model'])
    agg_names = sorted(plot_df['display_name'].unique())

    fig_agg, ax_agg = plt.subplots(figsize=(8, 7))
    ax_agg.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    agg_colors = plt.cm.tab10(np.linspace(0, 1, len(agg_names)))

    for idx, display_name in enumerate(agg_names):
        sub = plot_df[plot_df['display_name'] == display_name]
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for j in range(n_bins):
            lo, hi = bin_edges[j], bin_edges[j + 1]
            in_bin = sub[(sub['confidence'] >= lo) & (sub['confidence'] < hi if j < n_bins - 1 else sub['confidence'] <= hi)]
            if len(in_bin) > 0:
                bin_accs.append(in_bin['mean_accuracy'].mean())
                bin_confs.append(in_bin['confidence'].mean())
                bin_counts.append(len(in_bin))
            else:
                bin_accs.append(np.nan)
                bin_confs.append(np.nan)
                bin_counts.append(0)

        total = sum(bin_counts)
        ece = sum(
            (count / total) * abs(acc - conf)
            for acc, conf, count in zip(bin_accs, bin_confs, bin_counts)
            if count > 0
        ) if total > 0 else np.nan

        agg_label = f"{display_name} (aggregated)"
        ece_records.append({
            'Model/Dataset': agg_label,
            'ECE': round(ece, 4) if not np.isnan(ece) else np.nan,
            'N': total,
            'Mean_Confidence': round(sub['confidence'].mean(), 4),
            'Mean_Accuracy': round(sub['mean_accuracy'].mean(), 4),
        })

        valid = [i for i in range(n_bins) if bin_counts[i] > 0]
        if valid:
            datasets = sorted(sub['dataset'].unique())
            ds_suffix = f" [{', '.join(datasets)}]" if len(datasets) > 1 else ''
            ax_agg.plot(
                [bin_confs[i] for i in valid],
                [bin_accs[i] for i in valid],
                'o-', color=agg_colors[idx],
                label=f"{display_name}{ds_suffix} (ECE={ece:.3f}, n={total})",
                markersize=5
            )

    ax_agg.set_xlabel('Confidence (1 - entropy/log2(5))', fontsize=12)
    ax_agg.set_ylabel('Mean Accuracy', fontsize=12)
    ax_agg.set_title('Reliability Diagram — Aggregated Across Datasets', fontsize=14)
    ax_agg.set_xlim(0, 1)
    ax_agg.set_ylim(0, 1)
    ax_agg.legend(fontsize=8, loc='lower right')
    ax_agg.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'reliability_diagram_ece_aggregated.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated reliability_diagram_ece_aggregated.png")

    ece_df = pd.DataFrame(ece_records)
    ece_df.to_csv(os.path.join(output_dir, 'ece_results.csv'), index=False)
    print("Saved ece_results.csv")

    print("\nECE Results:")
    print(ece_df.to_string(index=False))

    return ece_df


def plot_risk_coverage(df: pd.DataFrame, output_dir: str):
    """Plot risk-coverage curve: cumulative accuracy vs coverage fraction, sorted by entropy."""
    plot_df = df.dropna(subset=['mean_accuracy']).copy()
    if plot_df.empty:
        print("Skipping risk_coverage: no mean_accuracy data.")
        return

    labels = sorted(plot_df['label'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    fig, ax = plt.subplots(figsize=(9, 6))

    for idx, label in enumerate(labels):
        sub = plot_df[plot_df['label'] == label].sort_values('entropy').reset_index(drop=True)
        n = len(sub)
        if n == 0:
            continue

        coverage = np.arange(1, n + 1) / n
        cumulative_acc = sub['mean_accuracy'].expanding().mean().values

        ax.plot(coverage, cumulative_acc, color=colors[idx], label=label, linewidth=1.5)

        # Baseline reference line (overall mean accuracy)
        baseline_acc = sub['mean_accuracy'].mean()
        ax.axhline(y=baseline_acc, color=colors[idx], linestyle=':', alpha=0.4, linewidth=1)

    ax.set_xlabel('Coverage Fraction (sorted by entropy, most confident first)', fontsize=11)
    ax.set_ylabel('Cumulative Mean Accuracy', fontsize=11)
    ax.set_title('Risk-Coverage Curve', fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_coverage_curve.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated risk_coverage_curve.png")


def plot_aggregated_cross_dataset(df: pd.DataFrame, ece_df: pd.DataFrame, output_dir: str):
    """Aggregate metrics across datasets for models that appear in multiple datasets."""
    plot_df = df.dropna(subset=['mean_accuracy']).copy()
    if plot_df.empty:
        print("Skipping cross-dataset aggregation: no mean_accuracy data.")
        return

    plot_df['display_name'] = plot_df['model'].map(MODEL_DISPLAY_NAMES).fillna(plot_df['model'])

    # Compute per-model aggregated metrics
    agg_records = []
    for display_name, group in plot_df.groupby('display_name'):
        majority_correct = group['majority_answer_correct'].map(
            {True: 1, False: 0, 'True': 1, 'False': 0}
        )

        record = {
            'Model': display_name,
            'Datasets': ', '.join(sorted(group['dataset'].unique())),
            'N': len(group),
            'Mean_Entropy': round(group['entropy'].mean(), 4),
            'Mean_Accuracy': round(group['mean_accuracy'].mean(), 4),
            'Majority_Vote_Accuracy': round(majority_correct.mean(), 4),
        }

        # Add ECE if available
        model_labels = group['label'].unique()
        if not ece_df.empty:
            model_eces = ece_df[ece_df['Model/Dataset'].isin(model_labels)]
            if not model_eces.empty:
                record['Mean_ECE'] = round(model_eces['ECE'].mean(), 4)

        agg_records.append(record)

    if not agg_records:
        print("No cross-dataset records to aggregate.")
        return

    agg_df = pd.DataFrame(agg_records).sort_values('Model')
    agg_df.to_csv(os.path.join(output_dir, 'cross_dataset_aggregated_metrics.csv'), index=False)
    print("Saved cross_dataset_aggregated_metrics.csv")

    print("\nCross-Dataset Aggregated Metrics:")
    print(agg_df.to_string(index=False))

    # Bar chart: grouped bars for key metrics
    metric_cols = ['Mean_Entropy', 'Mean_Accuracy', 'Majority_Vote_Accuracy']
    if 'Mean_ECE' in agg_df.columns:
        metric_cols.append('Mean_ECE')

    melted = agg_df.melt(id_vars='Model', value_vars=metric_cols, var_name='Metric', value_name='Value')

    fig, ax = plt.subplots(figsize=(max(10, 2.5 * len(agg_df)), 6))
    sns.barplot(data=melted, x='Model', y='Value', hue='Metric', palette='Set2', ax=ax)
    ax.set_title('Cross-Dataset Aggregated Metrics by Model', fontsize=14)
    ax.set_ylabel('Value')
    ax.set_xlabel('')
    plt.xticks(rotation=25, ha='right')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.3f', label_type='edge', fontsize=7)
    plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_dataset_summary_bars.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated cross_dataset_summary_bars.png")


def compute_correlations(df: pd.DataFrame, output_dir: str):
    """Calculate and save correlation statistics."""
    stats_list = []

    epistemic_state_mapping = {
        'IGNORANCE': 0,
        'AMBIGUITY': 1,
        'HIGH_CERTAINTY': 2,
        'DECISIVE': 3,
    }

    for label in sorted(df['label'].unique()):
        sub = df[df['label'] == label]

        # 1. Entropy vs Accuracy (point-biserial)
        acc_col = sub['majority_answer_correct'].map(
            {True: 1, False: 0, 'True': 1, 'False': 0}
        ).dropna()
        entropy_for_acc = sub.loc[acc_col.index, 'entropy']
        if len(acc_col) > 2:
            r, p = stats.pointbiserialr(acc_col, entropy_for_acc)
            stats_list.append({
                'Model/Dataset': label, 'Relationship': 'Entropy vs. Accuracy',
                'Method': 'Point-Biserial', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(acc_col),
            })

        # 2. Entropy vs Mean Accuracy (Pearson — continuous)
        if 'mean_accuracy' in sub.columns:
            acc_sub = sub.dropna(subset=['mean_accuracy'])
            if len(acc_sub) > 2:
                r, p = stats.pearsonr(acc_sub['entropy'], acc_sub['mean_accuracy'])
                stats_list.append({
                    'Model/Dataset': label, 'Relationship': 'Entropy vs. Mean Accuracy',
                    'Method': 'Pearson', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(acc_sub),
                })

        # 3. Entropy vs Confidence (Spearman)
        conf_sub = sub.dropna(subset=['epistemic_state']).copy()
        conf_sub['conf_ord'] = conf_sub['epistemic_state'].map(epistemic_state_mapping)
        conf_sub = conf_sub.dropna(subset=['conf_ord'])
        if len(conf_sub) > 2:
            r, p = stats.spearmanr(conf_sub['entropy'], conf_sub['conf_ord'])
            stats_list.append({
                'Model/Dataset': label, 'Relationship': 'Entropy vs. Epistemic State',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(conf_sub),
            })

        # 4. Entropy vs Searches (Pearson & Spearman)
        search_sub = sub.dropna(subset=['num_searches'])
        if len(search_sub) > 2:
            r, p = stats.pearsonr(search_sub['entropy'], search_sub['num_searches'])
            stats_list.append({
                'Model/Dataset': label, 'Relationship': 'Entropy vs. Search Count',
                'Method': 'Pearson', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(search_sub),
            })
            r, p = stats.spearmanr(search_sub['entropy'], search_sub['num_searches'])
            stats_list.append({
                'Model/Dataset': label, 'Relationship': 'Entropy vs. Search Count',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(search_sub),
            })

    # --- Aggregated correlations: pool data by model across datasets ---
    df_agg = df.copy()
    df_agg['display_name'] = df_agg['model'].map(MODEL_DISPLAY_NAMES).fillna(df_agg['model'])

    for display_name in sorted(df_agg['display_name'].unique()):
        sub = df_agg[df_agg['display_name'] == display_name]
        agg_label = f"{display_name} (aggregated)"

        # 1. Entropy vs Accuracy (point-biserial)
        acc_col = sub['majority_answer_correct'].map(
            {True: 1, False: 0, 'True': 1, 'False': 0}
        ).dropna()
        entropy_for_acc = sub.loc[acc_col.index, 'entropy']
        if len(acc_col) > 2:
            r, p = stats.pointbiserialr(acc_col, entropy_for_acc)
            stats_list.append({
                'Model/Dataset': agg_label, 'Relationship': 'Entropy vs. Accuracy',
                'Method': 'Point-Biserial', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(acc_col),
            })

        # 2. Entropy vs Mean Accuracy (Pearson — continuous)
        if 'mean_accuracy' in sub.columns:
            acc_sub = sub.dropna(subset=['mean_accuracy'])
            if len(acc_sub) > 2:
                r, p = stats.pearsonr(acc_sub['entropy'], acc_sub['mean_accuracy'])
                stats_list.append({
                    'Model/Dataset': agg_label, 'Relationship': 'Entropy vs. Mean Accuracy',
                    'Method': 'Pearson', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(acc_sub),
                })

        # 3. Entropy vs Confidence (Spearman)
        conf_sub = sub.dropna(subset=['epistemic_state']).copy()
        conf_sub['conf_ord'] = conf_sub['epistemic_state'].map(epistemic_state_mapping)
        conf_sub = conf_sub.dropna(subset=['conf_ord'])
        if len(conf_sub) > 2:
            r, p = stats.spearmanr(conf_sub['entropy'], conf_sub['conf_ord'])
            stats_list.append({
                'Model/Dataset': agg_label, 'Relationship': 'Entropy vs. Epistemic State',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(conf_sub),
            })

        # 4. Entropy vs Searches (Pearson & Spearman)
        search_sub = sub.dropna(subset=['num_searches'])
        if len(search_sub) > 2:
            r, p = stats.pearsonr(search_sub['entropy'], search_sub['num_searches'])
            stats_list.append({
                'Model/Dataset': agg_label, 'Relationship': 'Entropy vs. Search Count',
                'Method': 'Pearson', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(search_sub),
            })
            r, p = stats.spearmanr(search_sub['entropy'], search_sub['num_searches'])
            stats_list.append({
                'Model/Dataset': agg_label, 'Relationship': 'Entropy vs. Search Count',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(search_sub),
            })

    if not stats_list:
        print("No correlations calculated.")
        return

    stats_df = pd.DataFrame(stats_list)

    print("\n" + "=" * 100)
    print("CORRELATION SUMMARY")
    print("=" * 100)
    print(stats_df.to_string(index=False))
    print("=" * 100 + "\n")

    stats_df.to_csv(os.path.join(output_dir, 'semantic_entropy_correlations.csv'), index=False)
    print("Saved semantic_entropy_correlations.csv")

    # Plot correlation summary as heatmap
    for relationship in stats_df['Relationship'].unique():
        rel_df = stats_df[stats_df['Relationship'] == relationship]
        # Pick the best method per relationship
        method = rel_df['Method'].iloc[0]
        rel_df = rel_df[rel_df['Method'] == method]

        if len(rel_df) < 2:
            continue

        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#2ecc71' if s else '#e74c3c' for s in rel_df['Significant']]
        bars = ax.barh(rel_df['Model/Dataset'], rel_df['Correlation'], color=colors)
        ax.set_xlabel(f'{method} Correlation')
        ax.set_title(f'{relationship}')
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

        for bar, pval in zip(bars, rel_df['P-Value']):
            width = bar.get_width()
            ax.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                    f'p={pval:.4f}', va='center', fontsize=8)

        plt.tight_layout()
        safe_name = relationship.lower().replace(' ', '_').replace('.', '')
        plt.savefig(os.path.join(output_dir, f'correlation_{safe_name}.png'), bbox_inches='tight', dpi=150)
        plt.close()

    print("Generated correlation bar charts.")

    # --- Aggregated bar charts: only the "(aggregated)" rows ---
    agg_stats_df = stats_df[stats_df['Model/Dataset'].str.endswith('(aggregated)')]
    if not agg_stats_df.empty:
        for relationship in agg_stats_df['Relationship'].unique():
            rel_df = agg_stats_df[agg_stats_df['Relationship'] == relationship]
            method = rel_df['Method'].iloc[0]
            rel_df = rel_df[rel_df['Method'] == method]

            if len(rel_df) < 2:
                continue

            fig, ax = plt.subplots(figsize=(8, 4))
            colors = ['#2ecc71' if s else '#e74c3c' for s in rel_df['Significant']]
            bars = ax.barh(rel_df['Model/Dataset'], rel_df['Correlation'], color=colors)
            ax.set_xlabel(f'{method} Correlation')
            ax.set_title(f'{relationship} (Aggregated Across Datasets)')
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)

            for bar, pval in zip(bars, rel_df['P-Value']):
                width = bar.get_width()
                ax.text(width + 0.01 * np.sign(width), bar.get_y() + bar.get_height() / 2,
                        f'p={pval:.4f}', va='center', fontsize=8)

            plt.tight_layout()
            safe_name = relationship.lower().replace(' ', '_').replace('.', '')
            plt.savefig(os.path.join(output_dir, f'correlation_{safe_name}_aggregated.png'), bbox_inches='tight', dpi=150)
            plt.close()

        print("Generated aggregated correlation bar charts.")


def main():
    parser = argparse.ArgumentParser(description="Analyze semantic entropy relationships.")
    parser.add_argument(
        '--variant', type=str, default='vanilla',
        choices=['vanilla', 'with_sys_instruct'],
        help="Which baseline analysis to use: 'vanilla' (default) or 'with_sys_instruct'."
    )
    args = parser.parse_args()

    variant = args.variant
    if variant == 'with_sys_instruct':
        output_dir = os.path.join(SEMANTIC_ENTROPY_DIR, 'analysis_with_sys_instruct')
    else:
        output_dir = os.path.join(SEMANTIC_ENTROPY_DIR, 'analysis')

    os.makedirs(output_dir, exist_ok=True)

    print(f"Variant: {variant}")
    print(f"Output directory: {output_dir}")
    print("Building merged dataset...")
    df = build_merged_data(variant)

    if df.empty:
        print("No data to analyze.")
        return

    # Drop rows that had no matching analysis CSV or search data for this variant
    has_analysis = df['epistemic_state'].notna() if 'epistemic_state' in df.columns else pd.Series(False, index=df.index)
    has_searches = df['num_searches'].notna() if 'num_searches' in df.columns else pd.Series(False, index=df.index)
    has_either = has_analysis | has_searches

    # For with_sys_instruct, only keep rows that actually matched (filters out NQ models etc.)
    if variant == 'with_sys_instruct':
        before = len(df)
        df = df[has_either].copy()
        if len(df) < before:
            print(f"Filtered to {len(df)} rows with {variant} data (from {before})")

    print(f"\nFinal dataset: {len(df)} rows, columns: {df.columns.tolist()}")
    print(f"Models/datasets: {df['label'].unique().tolist()}")

    plot_entropy_vs_accuracy(df, output_dir)
    plot_entropy_vs_confidence(df, output_dir)
    plot_entropy_vs_searches(df, output_dir)
    plot_combined_heatmap(df, output_dir)
    plot_entropy_vs_mean_accuracy(df, output_dir)
    ece_df = compute_and_plot_ece(df, output_dir)
    plot_risk_coverage(df, output_dir)
    plot_aggregated_cross_dataset(df, ece_df, output_dir)
    compute_correlations(df, output_dir)

    # Save merged data for further analysis
    export_cols = ['problem_id', 'dataset', 'model', 'label', 'entropy', 'num_clusters',
                   'majority_answer_correct']
    if 'epistemic_state' in df.columns:
        export_cols.append('epistemic_state')
    if 'num_searches' in df.columns:
        export_cols.append('num_searches')
    if 'mean_accuracy' in df.columns:
        export_cols.append('mean_accuracy')
    if 'agent_correct' in df.columns:
        export_cols.append('agent_correct')

    existing = [c for c in export_cols if c in df.columns]
    df[existing].to_csv(os.path.join(output_dir, 'merged_semantic_entropy_data.csv'), index=False)
    print(f"\nSaved merged data to {output_dir}/merged_semantic_entropy_data.csv")
    print(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
