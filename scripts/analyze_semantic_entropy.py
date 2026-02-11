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
from scipy import stats

# Set style
sns.set_theme(style="whitegrid")

BASE_DIR = os.path.join(os.path.dirname(__file__), '..', 'logs')
SEMANTIC_ENTROPY_DIR = os.path.join(BASE_DIR, 'semantic_entropy')

CONFIDENCE_ORDER = ['TABULA_RASA', 'WEAK_GUESS', 'STRONG_HYPOTHESIS', 'NO_SEARCH']

# Maps semantic entropy file naming → (dataset_dir, model_dir) in logs/
MODEL_DIR_MAP = {
    ('facts_one_hop', 'claude_4_5_haiku'): ('facts_one_hop', 'claude_4_5_haiku'),
    ('facts_one_hop', 'gemini_3_pro'): ('facts_one_hop', 'gemini_3_pro'),
    ('facts_one_hop', 'nemotron-3-nano'): ('facts_one_hop', 'nemotron-3-nano'),
    ('facts_one_hop', 'glm_4_7'): ('facts_one_hop', 'glm_4_7'),
    ('natural_questions', 'glm-4.7-flash'): ('natural_questions', 'glm-4.7-flash'),
    ('natural_questions', 'nemotron-3-nano'): ('natural_questions', 'nemotron-3-nano'),
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
        'judge1_confidence': 'pre_search_confidence',
        'judge2_hypothesis_correct': 'hypothesis_correct',
        'judge2_query_biased': 'is_search_query_biased',
        'judge2_snippet_has_answer': 'snippet_has_answer',
        'judge2_answer_flipped': 'answer_flipped',
    }
    df.rename(columns=column_mapping, inplace=True)
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
            matched = subset['pre_search_confidence'].notna().sum()
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
    plot_df = df.dropna(subset=['pre_search_confidence']).copy()
    valid_conf = [c for c in CONFIDENCE_ORDER if c in plot_df['pre_search_confidence'].values]
    plot_df = plot_df[plot_df['pre_search_confidence'].isin(valid_conf)]

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

        sns.boxplot(data=sub, x='pre_search_confidence', y='entropy', palette='muted',
                    ax=ax, order=valid_conf)
        sns.stripplot(data=sub, x='pre_search_confidence', y='entropy', color='black',
                      alpha=0.3, jitter=True, size=3, ax=ax, order=valid_conf)

        ax.set_title(f"{label} (n={len(sub)})", fontsize=10)
        ax.set_ylabel('Semantic Entropy' if i == 0 else '')
        ax.set_xlabel('Pre-Search Confidence')
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('Semantic Entropy vs. Pre-Search Confidence', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_vs_confidence.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_vs_confidence.png")

    # Aggregated view: mean entropy per confidence level across all models
    agg = plot_df.groupby(['label', 'pre_search_confidence'])['entropy'].agg(['mean', 'count']).reset_index()
    agg.columns = ['label', 'pre_search_confidence', 'mean_entropy', 'count']

    fig, ax = plt.subplots(figsize=(max(8, 3 * n), 5))
    sns.barplot(data=agg, x='pre_search_confidence', y='mean_entropy', hue='label',
                palette='viridis', ax=ax, order=valid_conf)
    ax.set_title('Mean Semantic Entropy by Confidence Level')
    ax.set_ylabel('Mean Entropy')
    ax.set_xlabel('Pre-Search Confidence')
    for c in ax.containers:
        ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=7)
    plt.legend(title='Model/Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_entropy_by_confidence.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated mean_entropy_by_confidence.png")


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
    plot_df = df.dropna(subset=['pre_search_confidence', 'num_searches']).copy()
    valid_conf = [c for c in CONFIDENCE_ORDER if c in plot_df['pre_search_confidence'].values]
    plot_df = plot_df[plot_df['pre_search_confidence'].isin(valid_conf)]

    if plot_df.empty:
        print("Skipping combined heatmap: insufficient data.")
        return

    plot_df['entropy_bin'] = pd.cut(
        plot_df['entropy'],
        bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
        labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]']
    )

    pivot = plot_df.pivot_table(
        values='num_searches', index='entropy_bin', columns='pre_search_confidence',
        aggfunc='mean'
    )
    pivot = pivot.reindex(columns=valid_conf)

    # Count pivot for annotations
    count_pivot = plot_df.pivot_table(
        values='num_searches', index='entropy_bin', columns='pre_search_confidence',
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
    ax.set_title('Mean Search Count by Entropy & Confidence (All Models)')
    ax.set_xlabel('Pre-Search Confidence')
    ax.set_ylabel('Semantic Entropy Range')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'entropy_confidence_search_heatmap.png'), bbox_inches='tight', dpi=150)
    plt.close()
    print("Generated entropy_confidence_search_heatmap.png")


def compute_correlations(df: pd.DataFrame, output_dir: str):
    """Calculate and save correlation statistics."""
    stats_list = []

    confidence_mapping = {
        'TABULA_RASA': 0,
        'WEAK_GUESS': 1,
        'STRONG_HYPOTHESIS': 2,
        'NO_SEARCH': 3,
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

        # 2. Entropy vs Confidence (Spearman)
        conf_sub = sub.dropna(subset=['pre_search_confidence']).copy()
        conf_sub['conf_ord'] = conf_sub['pre_search_confidence'].map(confidence_mapping)
        conf_sub = conf_sub.dropna(subset=['conf_ord'])
        if len(conf_sub) > 2:
            r, p = stats.spearmanr(conf_sub['entropy'], conf_sub['conf_ord'])
            stats_list.append({
                'Model/Dataset': label, 'Relationship': 'Entropy vs. Pre-Search Confidence',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(conf_sub),
            })

        # 3. Entropy vs Searches (Pearson & Spearman)
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
    has_analysis = df['pre_search_confidence'].notna() if 'pre_search_confidence' in df.columns else pd.Series(False, index=df.index)
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
    compute_correlations(df, output_dir)

    # Save merged data for further analysis
    export_cols = ['problem_id', 'dataset', 'model', 'label', 'entropy', 'num_clusters',
                   'majority_answer_correct']
    if 'pre_search_confidence' in df.columns:
        export_cols.append('pre_search_confidence')
    if 'num_searches' in df.columns:
        export_cols.append('num_searches')
    if 'agent_correct' in df.columns:
        export_cols.append('agent_correct')

    existing = [c for c in export_cols if c in df.columns]
    df[existing].to_csv(os.path.join(output_dir, 'merged_semantic_entropy_data.csv'), index=False)
    print(f"\nSaved merged data to {output_dir}/merged_semantic_entropy_data.csv")
    print(f"All outputs saved to {output_dir}/")


if __name__ == "__main__":
    main()
