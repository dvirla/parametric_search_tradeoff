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

def load_no_search_baseline(json_dir, num_questions):
    """
    Loads no-search run JSONs and identifies questions where the agent got 5/5 correct.
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
        
    # We want 5/5 correct.
    is_stable_correct = (correct_counts == 5)
    print(f"Identified {is_stable_correct.sum()} questions with stable correct answers (5/5).")
    return is_stable_correct

def load_and_preprocess(csv_path, json_dir=None):
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

    # Update baseline_correct if json_dir is provided
    if json_dir:
        stable_correct = load_no_search_baseline(json_dir, len(df))
        if stable_correct is not None:
             if len(stable_correct) == len(df):
                 df['baseline_correct'] = stable_correct
                 print("Updated 'baseline_correct' based on no-search JSON evaluations (5/5 correct).")
             else:
                 print(f"Warning: Length mismatch. DF: {len(df)}, No-Search: {len(stable_correct)}")

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

def main():
    parser = argparse.ArgumentParser(description="Visualize analysis results.")
    parser.add_argument("--input", required=True, help="Path to input CSV file from analysis.")
    parser.add_argument("--output-dir", required=True, help="Directory to save plots.")
    parser.add_argument("--json-dir", help="Directory containing no-search JSON files (run 1-5).")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    print("Loading and enhancing data...")
    # Pass json_dir if provided
    df = load_and_preprocess(args.input, args.json_dir)
    
    if df is not None:
        print(f"Data ready with columns: {df.columns.tolist()}")
        
        plot_known_confidence_distribution(df, args.output_dir)
        plot_hypothesis_analysis(df, args.output_dir)
        
        print(f"All plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()
