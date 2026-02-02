import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import numpy as np

# Set style
sns.set_theme(style="whitegrid")

def load_and_preprocess(csv_path):
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

    # Calculate Derived Flags
    
    # 1. Context Poisoning: Baseline correct (knew it) but Agent wrong (poisoned by search?)
    if 'is_context_poisoning' not in df.columns:
        df['is_context_poisoning'] = df['baseline_correct'] & (~df['agent_correct'])

    # 2. Performative Ignorance: Baseline correct (knew it) but acted unsure
    if 'is_performative_ignorance' not in df.columns and 'pre_search_confidence' in df.columns:
        df['is_performative_ignorance'] = df['baseline_correct'] & df['pre_search_confidence'].isin(['TABULA_RASA', 'WEAK_GUESS'])

    # 3. Confirmation Bias: Had a hypothesis and queried in a biased way
    if 'is_confirmation_bias' not in df.columns and 'pre_search_confidence' in df.columns:
        df['is_confirmation_bias'] = df['pre_search_confidence'].isin(['STRONG_HYPOTHESIS', 'WEAK_GUESS']) & df['is_search_query_biased']

    # 4. Utilization Failure: Snippet had answer but Agent got it wrong
    if 'is_utilization_failure' not in df.columns:
        df['is_utilization_failure'] = df['snippet_has_answer'] & (~df['agent_correct'])

    return df

def plot_performative_ignorance_breakdown(df, output_dir):
    """
    Analyzes low-confidence traces (Weak Guess / Tabula Rasa) and breaks them down
    by whether the extracted hypothesis was actually correct.
    """
    if 'hypothesis_correct' not in df.columns:
        print("Skipping performative_ignorance_breakdown: 'hypothesis_correct' column missing.")
        return

    # Filter for Low Confidence (Weak Guess or Tabula Rasa)
    low_conf_df = df[df['pre_search_confidence'].isin(['WEAK_GUESS', 'TABULA_RASA'])].copy()

    if low_conf_df.empty:
        print("No low confidence traces found for Performative Ignorance breakdown.")
        return

    # Normalize hypothesis_correct values
    def categorize_hypothesis(row):
        conf = row['pre_search_confidence']
        # Handle nan/float/string safely
        hyp_corr = str(row['hypothesis_correct']).upper().strip() if pd.notna(row['hypothesis_correct']) else "NA"
        
        if conf == 'TABULA_RASA':
            return 'No Hypothesis (Tabula Rasa)'
        
        if hyp_corr == 'YES':
            return 'Correct Hypothesis (Under-confident)'
        elif hyp_corr == 'NO':
            return 'Incorrect Hypothesis (Justified Uncertainty)'
        else:
            return 'Weak Guess (Unverified)' 

    low_conf_df['hypothesis_status'] = low_conf_df.apply(categorize_hypothesis, axis=1)

    plt.figure(figsize=(10, 6))
    
    hue_order = [
        'Correct Hypothesis (Under-confident)', 
        'Incorrect Hypothesis (Justified Uncertainty)', 
        'No Hypothesis (Tabula Rasa)',
        'Weak Guess (Unverified)'
    ]
    existing_hues = [h for h in hue_order if h in low_conf_df['hypothesis_status'].unique()]
    
    # We want to count traces where baseline was correct vs incorrect, but strictly speaking 
    # "Performative Ignorance" usually implies baseline WAS correct. 
    # Let's show distribution over baseline correctness to see if "ignorance" was justified or performative. 
    
    try:
        ax = sns.countplot(
            data=low_conf_df,
            x='baseline_correct',
            hue='hypothesis_status',
            hue_order=existing_hues,
            palette='viridis'
        )
        
        plt.title('Breakdown of Low Confidence States by Baseline Correctness')
        plt.xlabel('Baseline (No Search) Correctness (True = Performative Ignorance)')
        plt.ylabel('Count of Traces')
        plt.legend(title='Hypothesis Status')
        
        for c in ax.containers:
            labels = [f"{int(v)}" if v > 0 else "" for v in c.datavalues]
            ax.bar_label(c, labels=labels, label_type='center', color='white', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performative_ignorance_breakdown.png'))
        plt.close()
        print("Generated performative_ignorance_breakdown.png")
    except Exception as e:
        print(f"Error plotting performative_ignorance_breakdown: {e}")

def plot_misalignment_prevalence(df, output_dir):
    """
    Bar chart showing the percentage of traces for each misalignment flag.
    """
    flags = [
        'is_context_poisoning',
        'is_performative_ignorance',
        'is_confirmation_bias',
        'is_utilization_failure'
    ]
    
    stats = []
    total_rows = len(df)
    for flag in flags:
        if flag in df.columns:
            count = df[flag].sum()
            percent = (count / total_rows) * 100
            stats.append({
                'Flag': flag.replace('is_', '').replace('_', ' ').title(), 
                'Percentage': percent,
                'Count': count
            })
    
    stats_df = pd.DataFrame(stats)
    
    if stats_df.empty:
        print("No misalignment flags found to plot.")
        return

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x='Flag', y='Percentage', data=stats_df, hue='Flag', palette='viridis', legend=False)
    plt.title(f'Prevalence of Misalignment Behaviors (N={total_rows})')
    plt.ylabel('Percentage of Traces (%)')
    plt.xlabel('Behavior')
    plt.ylim(0, 100)
    
    for i, row in stats_df.iterrows():
        label = f"{row['Percentage']:.1f}%\n(n={int(row['Count'])})"
        ax.text(i, row['Percentage'] + 1, label, ha='center', va='bottom')
        
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'misalignment_prevalence.png'))
    plt.close()
    print("Generated misalignment_prevalence.png")

def plot_search_utility_by_confidence(df, output_dir):
    """
    Stacked bar chart showing the Outcome of Search (Win/Loss/Redundant/Fail)
    grouped by the Agent's Pre-Search Confidence.
    """
    required_cols = ['pre_search_confidence', 'baseline_correct', 'agent_correct']
    if not all(col in df.columns for col in required_cols):
        print("Skipping search_utility_by_confidence: Missing columns.")
        return

    # Define Categories
    def classify_outcome(row):
        base = row['baseline_correct']
        agent = row['agent_correct']
        
        if not base and agent:
            return 'Win (Fixed)'
        elif base and not agent:
            return 'Loss (Poisoned)'
        elif base and agent:
            return 'Redundant (Verified)'
        else:
            return 'Fail (Still Wrong)'

    df['outcome'] = df.apply(classify_outcome, axis=1)
    
    # Order for Confidence
    conf_order = ['TABULA_RASA', 'WEAK_GUESS', 'STRONG_HYPOTHESIS']
    # Filter to only existing categories
    existing_confs = [c for c in conf_order if c in df['pre_search_confidence'].unique()]
    
    if not existing_confs:
        print("No confidence data found.")
        return

    df['pre_search_confidence'] = pd.Categorical(
        df['pre_search_confidence'], 
        categories=conf_order, 
        ordered=True
    )
    
    # Aggregate
    stats = df.groupby(['pre_search_confidence', 'outcome'], observed=False).size().reset_index(name='count')
    
    # Calculate totals per confidence level to normalize
    totals = stats.groupby('pre_search_confidence', observed=False)['count'].transform('sum')
    stats['percentage'] = (stats['count'] / totals) * 100
    
    # Pivot for plotting
    pivot_df = stats.pivot(index='pre_search_confidence', columns='outcome', values='percentage').fillna(0)
    
    # Reorder columns for logical flow: Win -> Redundant -> Fail -> Loss
    col_order = ['Win (Fixed)', 'Redundant (Verified)', 'Fail (Still Wrong)', 'Loss (Poisoned)']
    # Filter only existing columns
    col_order = [c for c in col_order if c in pivot_df.columns]
    pivot_df = pivot_df[col_order]
    
    if pivot_df.empty:
        print("No data for utility plot.")
        return

    # Colors
    color_map = {
        'Win (Fixed)': '#2ca02c',       # Green
        'Redundant (Verified)': '#f1c40f', # Yellow/Gold
        'Fail (Still Wrong)': '#7f7f7f', # Gray
        'Loss (Poisoned)': '#d62728'    # Red
    }
    colors = [color_map[c] for c in pivot_df.columns]
    
    # Plot
    ax = pivot_df.plot(kind='bar', stacked=True, color=colors, figsize=(10, 7), width=0.7)
    
    plt.title('Impact of Search by Internal Confidence Level')
    plt.ylabel('Percentage of Queries (%)')
    plt.xlabel('Pre-Search Confidence')
    plt.xticks(rotation=0)
    plt.legend(title='Search Outcome', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Add value labels
    for c in ax.containers:
        ax.bar_label(c, fmt='%.0f%%', label_type='center', color='white', fontweight='bold', padding=0)

    plt.savefig(os.path.join(output_dir, 'search_utility_by_confidence.png'))
    plt.close()
    print("Generated search_utility_by_confidence.png")

def plot_accuracy_comparison(df, output_dir):
    """
    Compare No Search vs Baseline Agent Accuracy.
    """
    if 'baseline_correct' in df.columns and 'agent_correct' in df.columns:
        baseline_acc = df['baseline_correct'].mean() * 100
        agent_acc = df['agent_correct'].mean() * 100
        
        data = pd.DataFrame({
            'Model': ['No Search (Baseline)', 'Search Agent'],
            'Accuracy': [baseline_acc, agent_acc]
        })
        
        plt.figure(figsize=(8, 6))
        ax = sns.barplot(x='Model', y='Accuracy', data=data, hue='Model', palette='magma', legend=False)
        plt.title('Accuracy Comparison: No Search vs Search Agent')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        
        for i, v in enumerate(data['Accuracy']):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center')
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_comparison.png'))
        plt.close()
        print("Generated accuracy_comparison.png")

def plot_hypothesis_analysis(df, output_dir):
    """
    Analyzes traces where the agent had a STRONG_HYPOTHESIS or WEAK_GUESS.
    1. How often is the Hypothesis actually correct?
    2. What are the search behaviors (Flipped, Fail) for Correct vs Incorrect Hypotheses?
    """
    if 'pre_search_confidence' not in df.columns or 'hypothesis_correct' not in df.columns:
        print("Skipping hypothesis_analysis: Missing columns.")
        return

    # Filter for Strong Hypothesis or Weak Guess
    hyp_df = df[df['pre_search_confidence'].isin(['STRONG_HYPOTHESIS', 'WEAK_GUESS'])].copy()
    
    if hyp_df.empty:
        print("No STRONG_HYPOTHESIS or WEAK_GUESS traces found.")
        return

    # Normalize hypothesis_correct
    def normalize_correct(val):
        s = str(val).upper().strip()
        if s == 'YES': return 'Correct'
        if s == 'NO': return 'Incorrect'
        return 'Unknown'

    hyp_df['hyp_accuracy'] = hyp_df['hypothesis_correct'].apply(normalize_correct)
    
    # Filter out Unknowns for cleaner plotting if they are few
    hyp_df = hyp_df[hyp_df['hyp_accuracy'] != 'Unknown']
    
    if hyp_df.empty:
        print("No graded hypothesis traces found (all Unknown).")
        return

    # --- Plot 1: Accuracy of Hypotheses ---
    plt.figure(figsize=(6, 6))
    ax = sns.countplot(x='hyp_accuracy', data=hyp_df, palette='Set2', order=['Correct', 'Incorrect'])
    plt.title('Accuracy of Hypotheses (n={})'.format(len(hyp_df)))
    plt.xlabel('Hypothesis Correctness')
    plt.ylabel('Count')
    for c in ax.containers:
        ax.bar_label(c, label_type='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_accuracy.png'))
    plt.close()
    print("Generated hypothesis_accuracy.png")

    # --- Plot 2: Behaviors by Hypothesis Accuracy ---
    
    # Melt for behavior flags
    behavior_flags = {
        'is_confirmation_bias': 'Confirmation Bias',
        'answer_flipped': 'Answer Flipped',
        'is_utilization_failure': 'Utilization Failure'
    }
    
    # Ensure cols exist
    existing_flags = [f for f in behavior_flags.keys() if f in hyp_df.columns]
    
    if not existing_flags:
        return

    melted = hyp_df.melt(
        id_vars=['hyp_accuracy'],
        value_vars=existing_flags,
        var_name='Behavior',
        value_name='IsPresent'
    )
    
    # Calculate percentages
    summary = melted.groupby(['hyp_accuracy', 'Behavior'])['IsPresent'].mean().reset_index()
    summary['Percentage'] = summary['IsPresent'] * 100
    summary['Behavior Label'] = summary['Behavior'].map(behavior_flags)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=summary,
        x='Behavior Label',
        y='Percentage',
        hue='hyp_accuracy',
        palette='Set2'
    )
    plt.title('Search Behaviors given a Hypothesis (Strong or Weak)')
    plt.ylabel('Prevalence (%)')
    plt.ylim(0, 100)
    plt.legend(title='Hypothesis Was Actually:')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hypothesis_behaviors.png'))
    plt.close()
    print("Generated hypothesis_behaviors.png")

def main():
    parser = argparse.ArgumentParser(description="Visualize analysis results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file from analysis.")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots.")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print("Loading and enhancing data...")
    df = load_and_preprocess(args.input)
    
    if df is not None:
        print(f"Data ready with columns: {df.columns.tolist()}")
        
        plot_misalignment_prevalence(df, args.output_dir)
        plot_search_utility_by_confidence(df, args.output_dir)
        plot_accuracy_comparison(df, args.output_dir)
        plot_performative_ignorance_breakdown(df, args.output_dir)
        plot_hypothesis_analysis(df, args.output_dir)
        
        print(f"All plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
