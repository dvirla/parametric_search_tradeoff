import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from scipy import stats
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
import sys
import os
import glob
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Agent Comparison Analysis Tool")
    
    # Input/Output paths
    parser.add_argument('--logs-dir', type=str, default='logs/facts_100_queries/gemini_3_pro',
                        help='Directory containing agent result JSON files')
    parser.add_argument('--file-pattern', type=str, default='facts_eval_results_*.json',
                        help='Glob pattern for matching result files')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                        help='Directory to save plots and reports')
    
    # Analysis options
    parser.add_argument('--baseline-agent', type=str, default='baseline',
                        help='Substring to identify the baseline agent for statistical tests')
    parser.add_argument('--grouping-config', type=str, default=None,
                        help='JSON string or path to JSON file mapping Alias -> Regex Pattern for grouping agents.')
    
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable generating and saving plots')
    parser.add_argument('--plot-title', type=str, default='Gemini-3-Pro Performance Comparison',
                        help='Main title for the generated plots')

    return parser.parse_args()

def get_default_grouping_config() -> Dict[str, str]:
    """
    Returns the default mapping of Display Name -> Regex Pattern.
    """
    return {
        "baseline_with_sys_instruct": r"baseline.*with_sys_instruct",
        "baseline_run_1": r"baseline.*run[-_]?1",
        "no_search_avg": r"no[-_]?search",
        "v4_generalized_iterative_search": r"v4.*generalized",
        "v4_iterative_search": r"v4.*(?!.*generalized).*iterative",
        "v5_iterative_search": r"v5.*iterative",
        "v5_iterative_search": r"iterative"

    }

def load_datasets(logs_dir: str, file_pattern: str) -> Dict[str, pd.DataFrame]:
    """Load and process agent result files."""
    search_path = os.path.join(logs_dir, file_pattern)
    available_files = glob.glob(search_path)
    
    if not available_files:
        logger.error(f"ERROR: No files found matching pattern: {search_path}")
        sys.exit(1)
        
    logger.info(f"Found {len(available_files)} files.")
    
    raw_datasets = {}
    for filepath in available_files:
        filename = Path(filepath).stem
        # Basic cleanup for the key, though grouping will handle the rest
        clean_name = filename.replace('facts_eval_results_', '').replace('facts-search_', '').replace('.json', '')
        
        try:
            with open(filepath, 'r') as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"Skipping empty file: {filepath}")
                    continue
                data = json.loads(content)
                
            if not data:
                continue
                
            df = pd.DataFrame(data)
            
            # Normalize columns
            if 'sampler_search_calls' not in df.columns:
                if 'search_calls' in df.columns:
                    df['sampler_search_calls'] = df['search_calls']
                else:
                    df['sampler_search_calls'] = 0  # Default for no-search
            
            if 'sampler_correct' not in df.columns:
                 if 'correct' in df.columns:
                    df['sampler_correct'] = df['correct']
                 else:
                    logger.warning(f"Skipping {filename}: No correctness column found.")
                    continue

            # Ensure boolean/numeric correctness
            if df['sampler_correct'].dtype == object:
                df['sampler_correct'] = df['sampler_correct'].apply(lambda x: 1 if str(x).lower() == 'true' else 0)
            
            df['original_agent_name'] = clean_name
            df['question_id'] = range(len(df))
            
            raw_datasets[clean_name] = df
            
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

    return raw_datasets

def process_and_group_datasets(raw_datasets: Dict[str, pd.DataFrame], 
                             grouping_config: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    """
    Groups raw datasets based on regex patterns. 
    Aggregates multiple matches into a single average dataframe.
    """
    processed_datasets = {}
    used_keys = set()
    
    logger.info("\nGrouping and Aliasing Datasets:")
    
    for alias, pattern in grouping_config.items():
        regex = re.compile(pattern, re.IGNORECASE)
        # Only match keys that haven't been used yet
        matched_keys = [k for k in raw_datasets.keys() if k not in used_keys and regex.search(k)]
        
        if not matched_keys:
            logger.warning(f"  No matches found for alias '{alias}' (pattern: '{pattern}')")
            continue
            
        logger.info(f"  Alias '{alias}' matches: {matched_keys}")
        
        if len(matched_keys) == 1:
            # Single match: just rename
            df = raw_datasets[matched_keys[0]].copy()
            df['agent'] = alias
            processed_datasets[alias] = df
        else:
            # Multiple matches: Aggregate/Average
            logger.info(f"    -> Aggregating {len(matched_keys)} runs for '{alias}'")
            dfs = [raw_datasets[k] for k in matched_keys]
            
            # Truncate to minimum length to ensure alignment
            min_len = min(len(d) for d in dfs)
            dfs = [d.iloc[:min_len].copy() for d in dfs]
            
            # Create average dataframe
            avg_df = dfs[0].copy()
            
            # Average correctness (numerical 0.0-1.0)
            all_correct = np.stack([d['sampler_correct'].astype(float).values for d in dfs])
            avg_df['sampler_correct'] = np.mean(all_correct, axis=0)
            
            # Average search calls
            all_calls = np.stack([d['sampler_search_calls'].astype(float).values for d in dfs])
            avg_df['sampler_search_calls'] = np.mean(all_calls, axis=0)
            
            avg_df['agent'] = alias
            processed_datasets[alias] = avg_df
            
        used_keys.update(matched_keys)
        
    return processed_datasets

def print_separator(title: str):
    logger.info("\n" + "="*80)
    logger.info(title)
    logger.info("="*80)

def analyze_search_stats(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Calculate and print search statistics."""
    stats_list = []
    for agent_name, df in datasets.items():
        stats_list.append({
            'Agent': agent_name,
            'Mean Searches': df['sampler_search_calls'].mean(),
            'Median Searches': df['sampler_search_calls'].median(),
            'Std Searches': df['sampler_search_calls'].std(),
            'Min Searches': df['sampler_search_calls'].min(),
            'Max Searches': df['sampler_search_calls'].max()
        })
    
    stats_df = pd.DataFrame(stats_list)
    logger.info("\nSearch Call Statistics by Agent:")
    logger.info(stats_df.to_string(index=False))
    return stats_df

def analyze_zero_search(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Analyze zero-search vs with-search performance."""
    zero_search_stats = []
    
    for agent_name, df in datasets.items():
        # Precision handling for averaged runs (floats)
        zero_mask = df['sampler_search_calls'] < 0.1 
        with_mask = df['sampler_search_calls'] >= 0.1
        
        num_zero = zero_mask.sum()
        num_with = with_mask.sum()
        total = len(df)
        
        # Weighted accuracy for averaged runs
        zero_acc = (df[zero_mask]['sampler_correct'].sum() / num_zero * 100) if num_zero > 0 else 0
        with_acc = (df[with_mask]['sampler_correct'].sum() / num_with * 100) if num_with > 0 else 0
        
        zero_search_stats.append({
            'Agent': agent_name,
            'Zero Search Count': num_zero,
            'Zero Search %': (num_zero / total) * 100,
            'Zero Search Accuracy': zero_acc,
            'With Search Count': num_with,
            'With Search Accuracy': with_acc
        })

    stats_df = pd.DataFrame(zero_search_stats)
    logger.info("\nZero Search Statistics:")
    logger.info(stats_df.to_string(index=False))
    return stats_df

def run_statistical_tests(datasets: Dict[str, pd.DataFrame], baseline_name: str):
    """Run ANOVA and t-tests against baseline."""
    # Identify baseline
    baseline_keys = [k for k in datasets.keys() if baseline_name in k.lower()]
    
    if not baseline_keys:
        logger.warning(f"No baseline agent found matching '{baseline_name}'. Skipping baseline comparisons.")
    else:
        # Use first matching baseline for pairwise comparison
        base_key = baseline_keys[0]
        base_searches = datasets[base_key]['sampler_search_calls'].values
        
        for agent_name, df in datasets.items():
            if agent_name == base_key: continue
            
            agent_searches = df['sampler_search_calls'].values
            min_len = min(len(agent_searches), len(base_searches))
            
            # Skip if constant (variance=0)
            if np.std(agent_searches) < 1e-9 and np.std(base_searches) < 1e-9:
                continue

            try:
                # Use ttest_ind for potentially averaged data
                t_stat, p_val = stats.ttest_rel(agent_searches[:min_len], base_searches[:min_len])
                mean_diff = np.mean(agent_searches) - np.mean(base_searches)
                
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                logger.info(f"{agent_name} vs {base_key}: Diff={mean_diff:+.2f}, p={p_val:.4f} ({sig})")
            except Exception as e:
                pass

    # ANOVA
    valid_dfs = [df for df in datasets.values() if df['sampler_search_calls'].std() > 1e-9]
    if len(valid_dfs) > 1:
        min_len = min(len(df) for df in valid_dfs)
        all_searches = [df['sampler_search_calls'].values[:min_len] for df in valid_dfs]
        
        try:
            f_stat, anova_p = stats.f_oneway(*all_searches)
            sig = '***' if anova_p < 0.001 else '**' if anova_p < 0.01 else '*' if anova_p < 0.05 else 'ns'
            logger.info(f"\nOverall ANOVA (Search Calls): F={f_stat:.3f}, p={anova_p:.4f} ({sig})")
        except:
            pass

def analyze_aggregation(datasets: Dict[str, pd.DataFrame], baseline_name: str):
    """Analyze oracle coverage of different agent combinations."""
    agents = [k for k in datasets.keys() if 'no_search' not in k]
    
    if not agents: return

    combinations = {
        'All Search Agents': agents,
    }

    results = []
    min_len = min(len(df) for df in datasets.values())
    
    for name, agent_list in combinations.items():
        if not agent_list: continue
        
        matrix = np.array([datasets[a]['sampler_correct'].values[:min_len] for a in agent_list]).T
        
        # Oracle: Correct if ANY agent is correct (>0.5)
        oracle_correct = np.any(matrix > 0.5, axis=1).sum()
        oracle_acc = (oracle_correct / min_len) * 100
        
        # Best individual
        best_ind = max([datasets[a]['sampler_correct'].mean() for a in agent_list]) * 100
        
        results.append({
            'Combination': name,
            'Num Agents': len(agent_list),
            'Oracle Accuracy': oracle_acc,
            'Best Individual': best_ind,
            'Improvement': oracle_acc - best_ind
        })
        
    df = pd.DataFrame(results)
    if not df.empty:
        logger.info("\nAggregation Comparison:")
        logger.info(df.to_string(index=False))

def create_plots(datasets: Dict[str, pd.DataFrame], search_stats_df: pd.DataFrame, 
                 output_dir: str, title: str):
    """Generate and save visualization plots."""
    sns.set_style("whitegrid")
    
    # Sort agents by name or config order
    agents = list(datasets.keys())
    palette = sns.color_palette("husl", len(agents))
    color_map = dict(zip(agents, palette))
    
    # Figure: Comparison (1 row, 2 columns)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))  # Slightly taller for title
    fig.suptitle(title, fontsize=18, weight='bold')
    
    # --- Plot 1: Avg Search Calls (EXCLUDING no_search_avg) ---
    ax1 = axes[0]
    # Filter out no-search-avg using regex to be robust
    search_plot_agents = [a for a in agents if not re.search(r'no[-_]?search', a, re.IGNORECASE)]
    
    if search_plot_agents:
        # Re-slice search_stats_df
        plot_df = search_stats_df[search_stats_df['Agent'].isin(search_plot_agents)]
        
        means = plot_df['Mean Searches']
        # Calculate CIs only if we have variance/multiple runs, otherwise 0
        cis = []
        for agent in plot_df['Agent']:
             data = datasets[agent]['sampler_search_calls']
             if len(data) > 1 and data.std() > 0:
                 sem = stats.sem(data)
                 cis.append(sem * stats.t.ppf(0.975, len(data)-1))
             else:
                 cis.append(0)
        
        bar_colors = [color_map[agent] for agent in plot_df['Agent']]
        
        bars = ax1.bar(plot_df['Agent'], means, yerr=cis, capsize=5, alpha=0.8, color=bar_colors)
        ax1.set_xticklabels(plot_df['Agent'], rotation=45, ha='right')
        ax1.set_title('Average Search Calls')
        ax1.set_ylabel('Search Calls')
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ci = cis[i] if i < len(cis) else 0
            ax1.text(bar.get_x() + bar.get_width()/2., height + ci, 
                    f'{height:.2f}', ha='center', va='bottom')
    else:
        ax1.text(0.5, 0.5, 'No Search Agents to Display', ha='center', va='center')
        ax1.set_title('Average Search Calls')

    # --- Plot 2: Accuracy (INCLUDING no_search_avg) ---
    ax2 = axes[1]
    accs = [datasets[a]['sampler_correct'].mean() * 100 for a in agents]
    
    bars2 = ax2.bar(agents, accs, alpha=0.8, color=[color_map[a] for a in agents])
    ax2.set_xticklabels(agents, rotation=45, ha='right')
    ax2.set_title('Accuracy (%)')
    ax2.set_ylim(0, 100)
    ax2.set_ylabel('Accuracy (%)')
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)  # Adjust layout to make room for suptitle
    plt.savefig(os.path.join(output_dir, 'agent_comparison.png'), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get grouping config
    if args.grouping_config:
        try:
            if os.path.exists(args.grouping_config):
                with open(args.grouping_config, 'r') as f:
                    grouping_config = json.load(f)
            else:
                grouping_config = json.loads(args.grouping_config)
        except Exception as e:
            logger.error(f"Failed to parse grouping config: {e}")
            sys.exit(1)
    else:
        grouping_config = get_default_grouping_config()
    
    print_separator("LOADING AND GROUPING DATASETS")
    raw_datasets = load_datasets(args.logs_dir, args.file_pattern)
    datasets = process_and_group_datasets(raw_datasets, grouping_config)
    
    if not datasets:
        logger.error("No datasets to analyze after grouping.")
        return

    print_separator("SEARCH STATISTICS")
    search_stats_df = analyze_search_stats(datasets)
    
    print_separator("ZERO SEARCH ANALYSIS")
    zero_search_df = analyze_zero_search(datasets)
    
    print_separator("STATISTICAL TESTS")
    run_statistical_tests(datasets, args.baseline_agent)
    
    print_separator("AGGREGATION ANALYSIS")
    analyze_aggregation(datasets, args.baseline_agent)
    
    if not args.no_plots:
        print_separator("GENERATING PLOTS")
        create_plots(datasets, search_stats_df, args.output_dir, args.plot_title)

if __name__ == "__main__":
    main()
