import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np
import glob
import json

# Set style
sns.set_theme(style="whitegrid")

def clean_problem(problem: str) -> str:
    """Removes the instruction suffix from the problem string."""
    separator = "\n\nYour response should be in the following format:"
    if separator in problem:
        return problem.split(separator)[0].strip()
    return problem.strip()

def get_problem_id(problem: str) -> str:
    return clean_problem(problem)[:50] + "..."

def load_no_search_baseline(json_dir, num_questions):
    """
    Loads no-search run JSONs and calculates the confidence (agreement on correct answer).
    Returns an array of floats (0.0 to 1.0).
    """
    pattern = os.path.join(json_dir, '*no_search*.json')
    files = glob.glob(pattern)
    
    if not files:
        print(f"Warning: No no-search JSON files found in {json_dir}")
        return None
        
    print(f"Found {len(files)} no-search run files.")
    
    # Initialize counts
    correct_counts = np.zeros(num_questions, dtype=int)
    run_count = 0
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                # Verify length matches or handle accordingly
                # Assuming standard list of dicts and order is preserved
                for i, item in enumerate(data):
                    if i < num_questions:
                        if item.get('sampler_correct', False):
                            correct_counts[i] += 1
                run_count += 1
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            
    if run_count == 0:
        return None
        
    confidence_scores = correct_counts / run_count
    print(f"Calculated no-search confidence (agreement) across {run_count} runs.")
    return confidence_scores

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
        'judge1_confidence': 'pre_search_confidence',
        'judge2_hypothesis_correct': 'hypothesis_correct',
        'judge2_query_biased': 'is_search_query_biased',
        'judge2_snippet_has_answer': 'snippet_has_answer',
        'judge2_answer_flipped': 'answer_flipped'
    }
    df.rename(columns=column_mapping, inplace=True)

    # Convert boolean-like columns to actual booleans
    bool_cols = ['baseline_correct', 'agent_correct', 'snippet_has_answer', 'answer_flipped', 'is_search_query_biased']
    for col in bool_cols:
        if col in df.columns:
            # Handle string 'True'/'False' or 1/0
            df[col] = df[col].astype(str).map({'True': True, 'False': False, '1': True, '0': False, '1.0': True, '0.0': False})
            df[col] = df[col].fillna(False)

    # Update no-search confidence if json_dir is provided
    if json_dir:
        confidence_scores = load_no_search_baseline(json_dir, len(df))
        if confidence_scores is not None:
             if len(confidence_scores) == len(df):
                 df['no_search_confidence'] = confidence_scores
                 # For backward compatibility / existing logic:
                 # Define baseline_correct as high confidence (e.g., 1.0 or >= 0.8)
                 # But let's keep the user's logic: 5/5 was previously used.
                 df['baseline_correct'] = (df['no_search_confidence'] == 1.0)
                 print("Updated 'no_search_confidence' and 'baseline_correct' (1.0 agreement).")
             else:
                 print(f"Warning: Length mismatch. DF: {len(df)}, No-Search: {len(confidence_scores)}")

    # Update search usage from traces if provided
    if traces_path:
        info_map = load_traces_info(traces_path)
        if info_map:
            # map using problem_id
            df['num_searches'] = df['problem_id'].map(info_map)
            print(f"Matched 'num_searches' for {df['num_searches'].notna().sum()} rows.")
        else:
             print("No trace info loaded.")

    # Calculate Derived Flags
    
    # 1. Context Poisoning: Baseline correct (knew it) but Agent wrong (poisoned by search?)
    if 'is_context_poisoning' not in df.columns:
        if 'baseline_correct' in df.columns and 'agent_correct' in df.columns:
            df['is_context_poisoning'] = df['baseline_correct'] & (~df['agent_correct'])

    # 2. Performative Ignorance: Baseline correct (knew it) but acted unsure
    if 'is_performative_ignorance' not in df.columns and 'pre_search_confidence' in df.columns:
        if 'baseline_correct' in df.columns:
            df['is_performative_ignorance'] = df['baseline_correct'] & df['pre_search_confidence'].isin(['TABULA_RASA', 'WEAK_GUESS'])

    # 3. Confirmation Bias: Had a hypothesis and queried in a biased way
    if 'is_confirmation_bias' not in df.columns and 'pre_search_confidence' in df.columns:
        if 'is_search_query_biased' in df.columns:
            df['is_confirmation_bias'] = df['pre_search_confidence'].isin(['STRONG_HYPOTHESIS', 'WEAK_GUESS']) & df['is_search_query_biased']

    # 4. Utilization Failure: Snippet had answer but Agent got it wrong
    if 'is_utilization_failure' not in df.columns:
        if 'snippet_has_answer' in df.columns and 'agent_correct' in df.columns:
            df['is_utilization_failure'] = df['snippet_has_answer'] & (~df['agent_correct'])

    return df

def plot_known_confidence_distribution(df, output_dir):
    """
    For questions the agent ALREADY KNEW (baseline_correct=True), 
    what was their stated confidence?
    Also breaks down by whether their internal hypothesis was correct (if applicable).
    """
    if 'baseline_correct' not in df.columns or 'pre_search_confidence' not in df.columns:
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

    # STRICTLY Filter for Valid Confidence Levels
    valid_confidence = ['TABULA_RASA', 'WEAK_GUESS', 'STRONG_HYPOTHESIS']
    known_df = known_df[known_df['pre_search_confidence'].isin(valid_confidence)]

    if known_df.empty:
        print("No valid confidence levels found in known facts.")
        return

    # Calculate percentages grouped by Confidence AND Hypothesis Accuracy
    # Let's count (Confidence, HypAccuracy) pairs
    counts = known_df.groupby(['pre_search_confidence', 'hyp_accuracy']).size().reset_index(name='count')
    
    # Calculate global percentage relative to total filtered known facts
    total_known = len(known_df)
    counts['Percentage'] = (counts['count'] / total_known) * 100
    
    # Filter out 0 counts to clean up plot
    counts = counts[counts['count'] > 0]

    # Define order (for plotting consistency)
    order = valid_confidence
    
    # Ensure categorical ordering
    counts['pre_search_confidence'] = pd.Categorical(counts['pre_search_confidence'], categories=order, ordered=True)
    counts = counts.sort_values(['pre_search_confidence', 'hyp_accuracy'])

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=counts,
        x='pre_search_confidence',
        y='Percentage',
        hue='hyp_accuracy',
        palette='Set2',
        order=order
    )
    
    plt.title(f"Confidence Distribution on Known Facts (n={total_known})\n(Performative Ignorance = Tabula Rasa / Weak Guess)")
    plt.ylabel('Percentage of All Known Facts (%)')
    plt.xlabel('Pre-Search Confidence')
    plt.ylim(0, 115)
    plt.legend(title='Hypothesis Accuracy')
    
    for c in ax.containers:
        ax.bar_label(c, fmt='%.1f%%', label_type='edge')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'known_fact_confidence_distribution.png'), bbox_inches='tight')
    plt.close()
    print("Generated known_fact_confidence_distribution.png")

def plot_confidence_by_agreement(df, output_dir):
    """
    Creates a table and heatmap showing confidence distribution per agreement level.
    """
    if 'no_search_confidence' not in df.columns or 'pre_search_confidence' not in df.columns:
        print("Skipping confidence_by_agreement: Missing columns.")
        return

    valid_confidence = ['TABULA_RASA', 'WEAK_GUESS', 'STRONG_HYPOTHESIS']
    plot_df = df[df['pre_search_confidence'].isin(valid_confidence)].copy()
    
    if plot_df.empty:
        print("No valid confidence data for agreement analysis.")
        return

    # Determine number of runs from the denominator of confidence scores if possible
    # Otherwise default to 5 as requested by user
    unique_scores = plot_df['no_search_confidence'].unique()
    # Find the smallest non-zero difference between sorted unique scores as an estimate of 1/N
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

    # Calculate counts to add (n=X) to labels
    counts_per_level = plot_df['agreement_label'].value_counts()
    
    def format_agreement_with_n(label):
        n = counts_per_level.get(label, 0)
        return f"{label} (n={n})"

    # Pivot table for counts
    table = pd.crosstab(
        plot_df['agreement_label'], 
        plot_df['pre_search_confidence'],
        normalize='index' # Percentages per agreement level (row)
    ) * 100
    
    # Reorder columns
    conf_order = [c for c in valid_confidence if c in table.columns]
    table = table[conf_order]
    
    # Reorder rows and add n=X to index
    agreement_order = [f"{i}/{inferred_runs}" for i in range(inferred_runs + 1)]
    table = table.reindex(agreement_order).fillna(0)
    table.index = [format_agreement_with_n(idx) for idx in table.index]

    print("\n--- Confidence Distribution per Agreement Level (%) ---")
    print(table.round(1).to_string())
    print("------------------------------------------------------\n")

    # Save as CSV
    table.to_csv(os.path.join(output_dir, 'confidence_by_agreement.csv'))
    
    # Plot as heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(table, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage (%)'})
    plt.title(f'Confidence Distribution per Agreement Level (n={len(plot_df)})')
    plt.xlabel('Pre-Search Confidence')
    plt.ylabel(f'Agreement Level (0/{inferred_runs} to {inferred_runs}/{inferred_runs})')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confidence_by_agreement_heatmap.png'), bbox_inches='tight')
    plt.close()
    print("Generated confidence_by_agreement.csv and confidence_by_agreement_heatmap.png")

def plot_hypothesis_analysis(df, output_dir):
    """
    Analyzes traces where the agent had a STRONG_HYPOTHESIS or WEAK_GUESS.
    Broken down by confidence level.
    """
    if 'pre_search_confidence' not in df.columns or 'hypothesis_correct' not in df.columns:
        print("Skipping hypothesis_analysis: Missing columns.")
        return

    # Filter for Strong Hypothesis or Weak Guess
    hyp_df = df[df['pre_search_confidence'].isin(['STRONG_HYPOTHESIS', 'WEAK_GUESS'])].copy()
    
    if hyp_df.empty:
        print("No STRONG_HYPOTHESIS or WEAK_GUESS traces found.")
        return

    # Debug: Check Confirmation Bias prevalence
    if 'is_confirmation_bias' in hyp_df.columns:
        print("\n--- DEBUG: Confirmation Bias Prevalence by Confidence ---")
        stats = hyp_df.groupby('pre_search_confidence')['is_confirmation_bias'].mean() * 100
        print(stats)
        print("----------------------------------------------------------\n")

    # Normalize hypothesis_correct
    def normalize_correct(val):
        s = str(val).upper().strip()
        if s == 'YES': return 'Correct'
        if s == 'NO': return 'Incorrect'
        return 'Unknown'

    hyp_df['hyp_accuracy'] = hyp_df['hypothesis_correct'].apply(normalize_correct)
    
    # Filter out Unknowns for cleaner plotting
    hyp_df = hyp_df[hyp_df['hyp_accuracy'] != 'Unknown']
    
    if hyp_df.empty:
        print("No graded hypothesis traces found.")
        return

    # --- Plot 1: Accuracy of Hypotheses by Confidence ---
    plt.figure(figsize=(8, 6))
    
    # We want to see how often Weak vs Strong guesses are correct
    ax = sns.countplot(
        x='pre_search_confidence', 
        hue='hyp_accuracy', 
        data=hyp_df, 
        palette='Set2', 
        order=['WEAK_GUESS', 'STRONG_HYPOTHESIS'],
        hue_order=['Correct', 'Incorrect']
    )
    plt.title('Accuracy of Initial Hypothesis by Confidence')
    plt.xlabel('Pre-Search Confidence')
    plt.ylabel('Count')
    
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_accuracy_split.png'), bbox_inches='tight')
    plt.close()
    print("Generated hypothesis_accuracy_split.png")

    # --- Plot 2: Behaviors by Hypothesis Accuracy and Confidence ---
    
    behavior_flags = {
        'is_confirmation_bias': 'Confirmation Bias',
        'answer_flipped': 'Answer Flipped',
        'is_utilization_failure': 'Utilization Failure'
    }
    
    existing_flags = [f for f in behavior_flags.keys() if f in hyp_df.columns]
    
    if not existing_flags:
        return

    melted = hyp_df.melt(
        id_vars=['hyp_accuracy', 'pre_search_confidence'],
        value_vars=existing_flags,
        var_name='Behavior',
        value_name='IsPresent'
    )
    
    # Calculate percentages
    summary = melted.groupby(['pre_search_confidence', 'hyp_accuracy', 'Behavior'])['IsPresent'].mean().reset_index()
    summary['Percentage'] = summary['IsPresent'] * 100
    summary['Behavior Label'] = summary['Behavior'].map(behavior_flags)

    # Use a FacetGrid to separate by Confidence
    g = sns.catplot(
        data=summary,
        x='Behavior Label',
        y='Percentage',
        hue='hyp_accuracy',
        col='pre_search_confidence',
        kind='bar',
        palette='Set2',
        height=5,
        aspect=1.2,
        col_order=['WEAK_GUESS', 'STRONG_HYPOTHESIS']
    )
    
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle('Search Behaviors given a Hypothesis (Weak vs Strong)')
    g.set_axis_labels("Behavior", "Prevalence (%)")
    g.set_titles("{col_name}")
    
    # Clean up axes limits
    for ax in g.axes.flat:
        ax.set_ylim(0, 100)

    plt.savefig(os.path.join(output_dir, 'hypothesis_behaviors_split.png'), bbox_inches='tight')
    plt.close()
    print("Generated hypothesis_behaviors_split.png")

def plot_search_vs_nosearch_confidence(df, output_dir):
    """
    Correlation between no-search confidence (agreement) and number of searches.
    """
    if 'no_search_confidence' not in df.columns or 'num_searches' not in df.columns:
        print("Skipping search_vs_nosearch_confidence: Missing columns.")
        return

    # Filter out missing values
    plot_df = df.dropna(subset=['no_search_confidence', 'num_searches']).copy()
    
    plt.figure(figsize=(10, 6))
    
    # Jitter the x-values slightly if they are discrete (like 0, 0.2, 0.4...) to see density better
    # Or just use a stripplot/boxplot
    
    # Round confidence to 1 decimal place for grouping if needed, but let's treat as continuous-ish
    # If we have 5 runs, values are 0, 0.2, 0.4, 0.6, 0.8, 1.0.
    
    sns.stripplot(
        data=plot_df,
        x='no_search_confidence',
        y='num_searches',
        jitter=True,
        alpha=0.5,
        palette='viridis',
        hue='no_search_confidence',
        legend=False
    )
    
    # Add mean line or boxplot
    sns.boxplot(
        data=plot_df,
        x='no_search_confidence',
        y='num_searches',
        showfliers=False, # Don't show outliers again
        color='lightgray',
        boxprops={'alpha': 0.3}
    )

    plt.title('Search Usage vs. No-Search Confidence (Agreement)')
    plt.xlabel('No-Search Confidence (Agreement %)')
    plt.ylabel('Number of Searches')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'search_vs_nosearch_confidence.png'), bbox_inches='tight')
    plt.close()
    print("Generated search_vs_nosearch_confidence.png")

def plot_search_vs_presearch_confidence(df, output_dir):
    """
    Correlation between pre-search confidence (Tabula Rasa, Weak Guess, Strong Hypothesis) and number of searches.
    """
    if 'pre_search_confidence' not in df.columns or 'num_searches' not in df.columns:
        print("Skipping search_vs_presearch_confidence: Missing columns.")
        return

    # Filter out missing values and ensure valid categories
    valid_conf = ['TABULA_RASA', 'WEAK_GUESS', 'STRONG_HYPOTHESIS']
    plot_df = df[df['pre_search_confidence'].isin(valid_conf)].copy()
    
    # Ensure order
    plot_df['pre_search_confidence'] = pd.Categorical(plot_df['pre_search_confidence'], categories=valid_conf, ordered=True)
    
    plt.figure(figsize=(10, 6))
    
    sns.boxplot(
        data=plot_df,
        x='pre_search_confidence',
        y='num_searches',
        palette='Set3'
    )
    
    # Add strip plot on top for visibility of distribution
    sns.stripplot(
        data=plot_df,
        x='pre_search_confidence',
        y='num_searches',
        color='black',
        alpha=0.3,
        jitter=True,
        size=3
    )

    plt.title('Search Usage vs. Pre-Search Confidence')
    plt.xlabel('Pre-Search Confidence')
    plt.ylabel('Number of Searches')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'search_vs_presearch_confidence.png'), bbox_inches='tight')
    plt.close()
    print("Generated search_vs_presearch_confidence.png")

def calculate_and_save_correlations(df, output_dir):
    """
    Calculates and saves correlation coefficients to a table.
    """
    stats = []

    # 1. No-Search Confidence (Agreement) vs Num Searches
    # Expect negative correlation (Higher agreement -> Fewer searches)
    if 'no_search_confidence' in df.columns and 'num_searches' in df.columns:
        subset = df.dropna(subset=['no_search_confidence', 'num_searches'])
        if not subset.empty and len(subset) > 1:
            pearson = subset['no_search_confidence'].corr(subset['num_searches'], method='pearson')
            spearman = subset['no_search_confidence'].corr(subset['num_searches'], method='spearman')
            stats.append({
                'Relationship': 'No-Search Confidence vs. Searches',
                'Method': 'Pearson (Linear)',
                'Correlation': round(pearson, 4),
                'N': len(subset)
            })
            stats.append({
                'Relationship': 'No-Search Confidence vs. Searches',
                'Method': 'Spearman (Rank)',
                'Correlation': round(spearman, 4),
                'N': len(subset)
            })

    # 2. Pre-Search Confidence (Ordinal) vs Num Searches
    # Expect negative correlation (Stronger hypothesis -> Fewer searches)
    if 'pre_search_confidence' in df.columns and 'num_searches' in df.columns:
        # Map to ordinal: Tabula Rasa (0) < Weak Guess (1) < Strong Hypothesis (2)
        mapping = {'TABULA_RASA': 0, 'WEAK_GUESS': 1, 'STRONG_HYPOTHESIS': 2}
        
        subset = df.dropna(subset=['pre_search_confidence', 'num_searches']).copy()
        subset['conf_ordinal'] = subset['pre_search_confidence'].map(mapping)
        
        # Drop rows where mapping failed
        subset = subset.dropna(subset=['conf_ordinal'])
        
        if not subset.empty and len(subset) > 1:
            spearman = subset['conf_ordinal'].corr(subset['num_searches'], method='spearman')
            stats.append({
                'Relationship': 'Pre-Search Confidence vs. Searches',
                'Method': 'Spearman (Rank)',
                'Correlation': round(spearman, 4),
                'N': len(subset)
            })

    if not stats:
        print("No correlations calculated (missing data or insufficient rows).")
        return

    stats_df = pd.DataFrame(stats)
    
    print("\n" + "="*60)
    print("CORRELATION SUMMARY")
    print("="*60)
    print(stats_df.to_string(index=False))
    print("="*60 + "\n")
    
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
    # Pass json_dir and traces if provided
    df = load_and_preprocess(args.input, args.json_dir, args.traces)
    
    if df is not None:
        print(f"Data ready with columns: {df.columns.tolist()}")
        
        plot_known_confidence_distribution(df, args.output_dir)
        plot_confidence_by_agreement(df, args.output_dir)
        plot_hypothesis_analysis(df, args.output_dir)
        
        # New plots
        plot_search_vs_nosearch_confidence(df, args.output_dir)
        plot_search_vs_presearch_confidence(df, args.output_dir)
        
        # Calculate stats
        calculate_and_save_correlations(df, args.output_dir)
        
        print(f"All plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
