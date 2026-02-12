import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
import glob
import json
from scipy import stats

# Set style
sns.set_theme(style="whitegrid")

# Global epistemic state order
EPISTEMIC_STATE_ORDER = ['IGNORANCE', 'AMBIGUITY', 'HIGH_CERTAINTY', 'DECISIVE']

def clean_problem(problem: str) -> str:
    """Removes the instruction suffix from the problem string."""
    separator = "\n\nYour response should be in the following format:"
    if separator in problem:
        return problem.split(separator)[0].strip()
    return problem.strip()

def get_problem_id(problem: str) -> str:
    return clean_problem(problem)[:50] + "..."

def load_no_search_baseline(json_dir):
    """
    Loads no-search run JSONs and calculates the confidence (agreement on correct answer) per problem.
    Returns a dict: {problem_id: confidence_score}
    """
    pattern = os.path.join(json_dir, '*no_search*.json')
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No no-search JSON files found in {json_dir}")
        return None
        
    print(f"Found {len(files)} no-search run files.")
    
    # Initialize counts
    problem_correct_counts = {}
    run_count = 0
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                for item in data:
                    problem_id = get_problem_id(item.get('problem', ''))
                    if problem_id not in problem_correct_counts:
                        problem_correct_counts[problem_id] = 0
                    if item.get('sampler_correct', False):
                        problem_correct_counts[problem_id] += 1
                run_count += 1
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    if run_count == 0:
        return None
        
    confidence_map = {pid: count / run_count for pid, count in problem_correct_counts.items()}
    print(f"Calculated no-search confidence (agreement) for {len(confidence_map)} problems across {run_count} runs.")
    return confidence_map

def load_traces_info(traces_path):
    """
    Loads traces and extracts search usage info.
    Returns a dict: {problem_id: num_searches}
    """
    if not os.path.exists(traces_path):
        print(f"Warning: Traces file {traces_path} not found.")
        return {}
        
    print(f"Loading traces from {traces_path}...")
    try:
        with open(traces_path, 'r') as f:
            traces = json.load(f)
    except Exception as e:
        print(f"Error reading traces JSON: {e}")
        return {}
        
    info_map = {}
    for trace in traces:
        problem_id = get_problem_id(trace.get('problem', ''))
        
        # Count tool calls
        num_searches = 0
        for msg in trace.get('message_trace', []):
            if msg.get('role') == 'assistant':
                for part in msg.get('parts', []):
                    if part.get('type') == 'tool_call':
                        num_searches += 1
                        
        info_map[problem_id] = num_searches
        
    print(f"Extracted info for {len(info_map)} traces.")
    return info_map

def load_and_preprocess(csv_path, json_dir=None, traces_path=None):
    """
    Loads the analysis CSV and calculates derived misalignment flags if they don't exist.
    """
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded raw data: {len(df)} rows.")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    # Map raw judge columns to expected names if needed
    column_mapping = {
        'judge1_confidence': 'epistemic_state',  # legacy CSV format
        'judge2_hypothesis_correct': 'hypothesis_correct',
        'judge2_query_biased': 'is_search_query_biased',
        'judge2_snippet_has_answer': 'snippet_has_answer',
        'judge2_answer_flipped': 'answer_flipped'
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

    # Convert boolean-like columns to actual booleans
    bool_cols = ['baseline_correct', 'agent_correct', 'snippet_has_answer', 'answer_flipped', 'is_search_query_biased']
    for col in bool_cols:
        if col in df.columns:
            # Handle string 'True'/'False' or 1/0
            df[col] = df[col].astype(str).map({'True': True, 'False': False, '1': True, '0': False, '1.0': True, '0.0': False})
            df[col] = df[col].fillna(False)

    # Update no-search confidence if json_dir is provided
    if json_dir:
        confidence_map = load_no_search_baseline(json_dir)
        if confidence_map is not None:
             # Map using problem_id
             df['no_search_confidence'] = df['problem_id'].map(confidence_map)
             
             # Fill missing if any (though they should match)
             num_missing = df['no_search_confidence'].isna().sum()
             if num_missing > 0:
                 print(f"Warning: {num_missing} problems in CSV not found in no-search JSONs.")
             
             # For backward compatibility / existing logic:
             # Define baseline_correct as high confidence (e.g., 1.0)
             df['baseline_correct'] = (df['no_search_confidence'] == 1.0)
             print(f"Updated 'no_search_confidence' and 'baseline_correct' (1.0 agreement) for {df['no_search_confidence'].notna().sum()} rows.")

    # Update search usage from traces if provided
    if traces_path:
        info_map = load_traces_info(traces_path)
        if info_map:
            # map using problem_id
            df['num_searches'] = df['problem_id'].map(info_map)
            print(f"Matched 'num_searches' for {df['num_searches'].notna().sum()} rows.")
        else:
             print("No trace info loaded.")

    # Apply 4th epistemic state: DECISIVE (agent didn't use search)
    if 'num_searches' in df.columns:
        if 'epistemic_state' not in df.columns:
             df['epistemic_state'] = np.nan

        # Identify rows where num_searches is 0
        no_search_mask = (df['num_searches'] == 0)
        df.loc[no_search_mask, 'epistemic_state'] = 'DECISIVE'
        print(f"Tagged {no_search_mask.sum()} records as 'DECISIVE'.")

    # Calculate Derived Flags
    
    # 1. Context Poisoning: Baseline correct (knew it) but Agent wrong (poisoned by search?)
    if 'is_context_poisoning' not in df.columns:
        if 'baseline_correct' in df.columns and 'agent_correct' in df.columns:
            df['is_context_poisoning'] = df['baseline_correct'] & (~df['agent_correct'])

    # 2. Performative Ignorance: Baseline correct (knew it) but acted unsure
    if 'is_performative_ignorance' not in df.columns and 'epistemic_state' in df.columns:
        if 'baseline_correct' in df.columns:
            df['is_performative_ignorance'] = df['baseline_correct'] & df['epistemic_state'].isin(['IGNORANCE', 'AMBIGUITY'])

    # 3. Confirmation Bias: Had a hypothesis and queried in a biased way
    if 'is_confirmation_bias' not in df.columns and 'epistemic_state' in df.columns:
        if 'is_search_query_biased' in df.columns:
            df['is_confirmation_bias'] = df['epistemic_state'].isin(['HIGH_CERTAINTY', 'AMBIGUITY']) & df['is_search_query_biased']

    # 4. Utilization Failure: Snippet had answer but Agent got it wrong
    if 'is_utilization_failure' not in df.columns:
        if 'snippet_has_answer' in df.columns and 'agent_correct' in df.columns:
            df['is_utilization_failure'] = df['snippet_has_answer'] & (~df['agent_correct'])

    return df

def plot_known_confidence_distribution(df, output_dir):
    """
    For questions the agent ALREADY KNEW (baseline_correct=True),
    what was their epistemic state?
    Includes DECISIVE as the 4th level.
    """
    if 'baseline_correct' not in df.columns or 'epistemic_state' not in df.columns:
        print("Skipping known_confidence_distribution: Missing columns.")
        return

    # Filter for Known Facts (Baseline Correct)
    known_df = df[df['baseline_correct'] == True].copy()

    if known_df.empty:
        print("No 'Known Facts' (baseline correct) found.")
        return

    # Normalize hypothesis_correct
    def normalize_correct(val):
        s = str(val).upper().strip()
        if s == 'YES': return 'Correct'
        if s == 'NO': return 'Incorrect'
        return 'N/A'

    if 'hypothesis_correct' in known_df.columns:
        known_df['hyp_accuracy'] = known_df['hypothesis_correct'].apply(normalize_correct)
    else:
        known_df['hyp_accuracy'] = 'N/A'

    # STRICTLY Filter for Valid Epistemic States
    valid_states = EPISTEMIC_STATE_ORDER
    known_df = known_df[known_df['epistemic_state'].isin(valid_states)]

    if known_df.empty:
        print("No valid epistemic state data for known facts.")
        return

    # Calculate percentages grouped by Epistemic State AND Hypothesis Accuracy
    counts = known_df.groupby(['epistemic_state', 'hyp_accuracy'], observed=False).size().reset_index(name='count')

    # Calculate global percentage relative to total filtered known facts
    total_known = len(known_df)
    counts['Percentage'] = (counts['count'] / total_known) * 100

    # Define order (for plotting consistency)
    order = EPISTEMIC_STATE_ORDER

    # Ensure categorical ordering
    counts['epistemic_state'] = pd.Categorical(counts['epistemic_state'], categories=order, ordered=True)
    counts = counts.sort_values(['epistemic_state', 'hyp_accuracy'])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=counts,
        x='epistemic_state',
        y='Percentage',
        hue='hyp_accuracy',
        palette='Set2',
        order=order
    )

    plt.title(f"Epistemic State Distribution on Known Facts (n={total_known})")
    plt.ylabel('Percentage of All Known Facts (%)')
    plt.xlabel('Epistemic State')
    plt.ylim(0, 115)
    plt.legend(title='Hypothesis Accuracy')

    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='edge')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'known_fact_epistemic_state_distribution.png'), bbox_inches='tight')
    plt.close()
    print("Generated known_fact_epistemic_state_distribution.png")

def plot_confidence_by_agreement(df, output_dir):
    """
    Creates a table and heatmap showing epistemic state distribution per agreement level.
    Includes DECISIVE.
    """
    if 'no_search_confidence' not in df.columns or 'epistemic_state' not in df.columns:
        print("Skipping epistemic_state_by_agreement: Missing columns.")
        return

    valid_states = EPISTEMIC_STATE_ORDER
    plot_df = df[df['epistemic_state'].isin(valid_states)].copy()

    if plot_df.empty:
        print("No valid epistemic state data for agreement analysis.")
        return

    # Determine number of runs
    unique_scores = plot_df['no_search_confidence'].unique()
    diffs = np.diff(sorted(unique_scores))
    if len(diffs) > 0 and np.min(diffs) > 0:
        inferred_runs = int(round(1.0 / np.min(diffs)))
    else:
        inferred_runs = 5

    print(f"Inferred {inferred_runs} runs for agreement labeling.")

    def format_agreement(val):
        runs_correct = round(val * inferred_runs)
        return f"{int(runs_correct)}/{inferred_runs}"

    plot_df['agreement_label'] = plot_df['no_search_confidence'].apply(format_agreement)

    # Pivot table for counts
    table = pd.crosstab(
        plot_df['agreement_label'],
        plot_df['epistemic_state'],
        normalize='index'
    ) * 100

    # Reorder columns and rows
    table = table.reindex(columns=valid_states).fillna(0)

    agreement_order = [f"{i}/{inferred_runs}" for i in range(inferred_runs + 1)]
    table = table.reindex(agreement_order).fillna(0)

    # Add n=X to index
    counts_per_level = plot_df['agreement_label'].value_counts()
    table.index = [f"{idx} (n={counts_per_level.get(idx, 0)})" for idx in table.index]

    print("\n--- Epistemic State Distribution per Agreement Level (%) ---")
    print(table.round(1).to_string())
    print("------------------------------------------------------------\n")

    # Save as CSV
    table.to_csv(os.path.join(output_dir, 'epistemic_state_by_agreement.csv'))

    # Plot as heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Epistemic State Distribution per Agreement Level (n={len(plot_df)})')
    plt.xlabel('Epistemic State')
    plt.ylabel(f'Agreement Level')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'epistemic_state_by_agreement_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Generated epistemic_state_by_agreement.csv and epistemic_state_by_agreement_heatmap.png")

def plot_hypothesis_analysis(df, output_dir):
    """
    Analyzes traces where the agent had HIGH_CERTAINTY or AMBIGUITY.
    """
    if 'epistemic_state' not in df.columns or 'hypothesis_correct' not in df.columns:
        print("Skipping hypothesis_analysis: Missing columns.")
        return

    # Filter for HIGH_CERTAINTY or AMBIGUITY
    hyp_df = df[df['epistemic_state'].isin(['HIGH_CERTAINTY', 'AMBIGUITY'])].copy()

    if hyp_df.empty:
        print("No HIGH_CERTAINTY or AMBIGUITY traces found.")
        return

    # Normalize hypothesis_correct
    def normalize_correct(val):
        s = str(val).upper().strip()
        if s == 'YES': return 'Correct'
        if s == 'NO': return 'Incorrect'
        return 'Unknown'

    hyp_df['hyp_accuracy'] = hyp_df['hypothesis_correct'].apply(normalize_correct)
    hyp_df = hyp_df[hyp_df['hyp_accuracy'] != 'Unknown']
    
    if hyp_df.empty:
        return

    # Accuracy of Initial Hypothesis by Epistemic State
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(
        x='epistemic_state',
        hue='hyp_accuracy',
        data=hyp_df,
        palette='Set2',
        order=['AMBIGUITY', 'HIGH_CERTAINTY'],
        hue_order=['Correct', 'Incorrect']
    )
    plt.title('Accuracy of Initial Hypothesis by Epistemic State')
    plt.xlabel('Epistemic State')
    plt.ylabel('Count')
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_accuracy_split.png'), bbox_inches='tight')
    plt.close()

    # Behaviors
    behavior_flags = {
        'is_confirmation_bias': 'Confirmation Bias',
        'answer_flipped': 'Answer Flipped',
        'is_utilization_failure': 'Utilization Failure'
    }
    existing_flags = [f for f in behavior_flags.keys() if f in hyp_df.columns]
    
    if not existing_flags:
        return

    melted = hyp_df.melt(
        id_vars=['hyp_accuracy', 'epistemic_state'],
        value_vars=existing_flags,
        var_name='Behavior',
        value_name='IsPresent'
    )
    summary = melted.groupby(['epistemic_state', 'hyp_accuracy', 'Behavior'], observed=True)['IsPresent'].mean().reset_index()
    summary['Percentage'] = summary['IsPresent'] * 100
    summary['Behavior Label'] = summary['Behavior'].map(behavior_flags)

    g = sns.catplot(
        data=summary,
        x='Behavior Label',
        y='Percentage',
        hue='hyp_accuracy',
        col='epistemic_state',
        kind='bar',
        palette='Set2',
        height=5,
        aspect=1.2,
        col_order=['AMBIGUITY', 'HIGH_CERTAINTY']
    )
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Search Behaviors given a Hypothesis')
    plt.savefig(os.path.join(output_dir, 'hypothesis_behaviors_split.png'), bbox_inches='tight')
    plt.close()

def plot_search_vs_nosearch_confidence(df, output_dir):
    """
    Correlation between no-search confidence (agreement) and number of searches.
    """
    if 'no_search_confidence' not in df.columns or 'num_searches' not in df.columns:
        return

    plot_df = df.dropna(subset=['no_search_confidence', 'num_searches']).copy()
    
    plt.figure(figsize=(10, 6))
    sns.stripplot(data=plot_df, x='no_search_confidence', y='num_searches', jitter=True, alpha=0.5, palette='viridis', hue='no_search_confidence', legend=False)
    sns.boxplot(data=plot_df, x='no_search_confidence', y='num_searches', showfliers=False, color='lightgray', boxprops={'alpha': 0.3})
    plt.title('Search Usage vs. No-Search Confidence (Agreement)')
    plt.xlabel('No-Search Confidence (Agreement %)')
    plt.ylabel('Number of Searches')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'search_vs_nosearch_confidence.png'), bbox_inches='tight')
    plt.close()

def plot_search_vs_presearch_confidence(df, output_dir):
    """
    Correlation between epistemic state and number of searches.
    EXCLUDES DECISIVE as it always has 0 searches (trivial).
    """
    if 'epistemic_state' not in df.columns or 'num_searches' not in df.columns:
        return

    # EXCLUDE DECISIVE for this specific plot as requested by user
    valid_states = [c for c in EPISTEMIC_STATE_ORDER if c != 'DECISIVE']
    plot_df = df[df['epistemic_state'].isin(valid_states)].copy()

    if plot_df.empty:
        return

    plot_df['epistemic_state'] = pd.Categorical(plot_df['epistemic_state'], categories=valid_states, ordered=True)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=plot_df, x='epistemic_state', y='num_searches', palette='Set3')
    sns.stripplot(data=plot_df, x='epistemic_state', y='num_searches', color='black', alpha=0.3, jitter=True, size=3)
    plt.title('Search Usage vs. Epistemic State (Excluding DECISIVE cases)')
    plt.xlabel('Epistemic State')
    plt.ylabel('Number of Searches')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'search_vs_epistemic_state.png'), bbox_inches='tight')
    plt.close()

def calculate_and_save_correlations(df, output_dir):
    """
    Calculates and saves correlation coefficients and p-values to a table.
    Ensures both No-Search Agreement and Pre-Search Confidence relationships are included.
    """
    stats_list = []
    
    # 1. No-Search Confidence (Agreement) vs Num Searches
    if 'no_search_confidence' in df.columns and 'num_searches' in df.columns:
        subset = df.dropna(subset=['no_search_confidence', 'num_searches'])
        if not subset.empty and len(subset) > 1:
            # Pearson
            p_corr, p_pval = stats.pearsonr(subset['no_search_confidence'], subset['num_searches'])
            stats_list.append({
                'Relationship': 'No-Search Confidence vs. Searches',
                'Method': 'Pearson (Linear)',
                'Correlation': round(p_corr, 4),
                'P-Value': round(p_pval, 6),
                'Significant': p_pval < 0.05,
                'N': len(subset)
            })
            # Spearman
            s_corr, s_pval = stats.spearmanr(subset['no_search_confidence'], subset['num_searches'])
            stats_list.append({
                'Relationship': 'No-Search Confidence vs. Searches',
                'Method': 'Spearman (Rank)',
                'Correlation': round(s_corr, 4),
                'P-Value': round(s_pval, 6),
                'Significant': s_pval < 0.05,
                'N': len(subset)
            })
        else:
            print(f"Warning: Insufficient data for No-Search Confidence correlation (N={len(subset)})")
    else:
        missing = []
        if 'no_search_confidence' not in df.columns: missing.append('no_search_confidence')
        if 'num_searches' not in df.columns: missing.append('num_searches')
        print(f"Warning: Missing columns for No-Search Confidence correlation: {missing}")

    # 2. Epistemic State (Ordinal) vs Num Searches
    mapping = {
        'IGNORANCE': 0,
        'AMBIGUITY': 1,
        'HIGH_CERTAINTY': 2,
        'DECISIVE': 3
    }

    if 'epistemic_state' in df.columns and 'num_searches' in df.columns:
        subset = df.dropna(subset=['epistemic_state', 'num_searches']).copy()
        subset['state_ordinal'] = subset['epistemic_state'].map(mapping)
        subset = subset.dropna(subset=['state_ordinal'])

        if not subset.empty and len(subset) > 1:
            s_corr, s_pval = stats.spearmanr(subset['state_ordinal'], subset['num_searches'])
            stats_list.append({
                'Relationship': 'Epistemic State vs. Searches',
                'Method': 'Spearman (Rank)',
                'Correlation': round(s_corr, 4),
                'P-Value': round(s_pval, 6),
                'Significant': s_pval < 0.05,
                'N': len(subset)
            })
        else:
             print(f"Warning: Insufficient data for Epistemic State vs. Searches correlation (N={len(subset)})")

    # 3. No-Search Confidence (Agreement) vs Epistemic State (Ordinal)
    if 'no_search_confidence' in df.columns and 'epistemic_state' in df.columns:
        subset = df.dropna(subset=['no_search_confidence', 'epistemic_state']).copy()
        subset['state_ordinal'] = subset['epistemic_state'].map(mapping)
        subset = subset.dropna(subset=['state_ordinal'])

        if not subset.empty and len(subset) > 1:
            s_corr, s_pval = stats.spearmanr(subset['no_search_confidence'], subset['state_ordinal'])
            stats_list.append({
                'Relationship': 'No-Search Confidence vs. Epistemic State',
                'Method': 'Spearman (Rank)',
                'Correlation': round(s_corr, 4),
                'P-Value': round(s_pval, 6),
                'Significant': s_pval < 0.05,
                'N': len(subset)
            })
        else:
             print(f"Warning: Insufficient data for No-Search Confidence vs. Epistemic State correlation (N={len(subset)})")

    if not stats_list:
        print("No correlations calculated.")
        return

    stats_df = pd.DataFrame(stats_list)
    
    print("\n" + "="*80)
    print("CORRELATION SUMMARY")
    print("="*80)
    print(stats_df.to_string(index=False))
    print("="*80 + "\n")
    
    output_path = os.path.join(output_dir, 'correlation_summary.csv')
    stats_df.to_csv(output_path, index=False)
    print(f"Saved correlation summary to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize analysis results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file from analysis.")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots.")
    parser.add_argument("--json-dir", help="Directory containing no-search JSON files (run 1-5).")
    parser.add_argument("--traces", help="Path to the agent's traces JSON file (to extract search usage).")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print("Loading and enhancing data...")
    df = load_and_preprocess(args.input, args.json_dir, args.traces)
    
    if df is not None:
        print(f"Data columns: {df.columns.tolist()}")
        
        plot_known_confidence_distribution(df, args.output_dir)
        plot_confidence_by_agreement(df, args.output_dir)
        plot_hypothesis_analysis(df, args.output_dir)
        plot_search_vs_nosearch_confidence(df, args.output_dir)
        plot_search_vs_presearch_confidence(df, args.output_dir)
        
        calculate_and_save_correlations(df, args.output_dir)
        
        print(f"All plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
