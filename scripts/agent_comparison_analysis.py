import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from scipy import stats
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import sys
import os
import glob
import argparse
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from src.services.base_agent import BaseAgent
except ImportError:
    BaseAgent = None
    logger.warning("Warning: Could not import BaseAgent. LLM Agreement Analysis will be unavailable.")

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
                        help='Substring to identify the baseline agent')
    parser.add_argument('--no-plots', action='store_true',
                        help='Disable generating and saving plots')
    parser.add_argument('--run-llm-agreement', action='store_true',
                        help='Enable LLM-based agreement analysis (requires BaseAgent and Ollama)')
    parser.add_argument('--llm-model', type=str, default='gpt-oss:20b',
                        help='Ollama model name for agreement analysis')

    return parser.parse_args()

def load_datasets(logs_dir: str, file_pattern: str) -> Dict[str, pd.DataFrame]:
    """Load and process agent result files."""
    search_path = os.path.join(logs_dir, file_pattern)
    available_files = glob.glob(search_path)
    
    if not available_files:
        logger.error(f"ERROR: No files found matching pattern: {search_path}")
        sys.exit(1)
        
    logger.info(f"Found {len(available_files)} files.")
    
    files = {}
    for filepath in available_files:
        filename = Path(filepath).stem
        # Clean up agent name
        agent_name = filename.replace('facts_eval_results_', '').replace('_reevaluated', '').strip('_').replace('serper_', '')
        files[agent_name] = filepath
        
    datasets = {}
    for name, filepath in files.items():
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                df = pd.DataFrame(data)
                # Standardize column names
                if 'search_calls' in df.columns and 'sampler_search_calls' not in df.columns:
                    df['sampler_search_calls'] = df['search_calls']
                df['agent'] = name
                df['question_id'] = range(len(df))
                datasets[name] = df
        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")

    # Sort by name
    return dict(sorted(datasets.items()))

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
        zero_mask = df['sampler_search_calls'] == 0
        with_mask = df['sampler_search_calls'] > 0
        
        num_zero = zero_mask.sum()
        num_with = with_mask.sum()
        total = len(df)
        
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
    
    # Statistical Comparison (Chi-square)
    logger.info("\n--- Statistical Comparison: Zero-Search vs With-Search Accuracy ---")
    for agent_name, df in datasets.items():
        zero_mask = df['sampler_search_calls'] == 0
        with_mask = df['sampler_search_calls'] > 0
        
        zero_correct = df[zero_mask]['sampler_correct'].sum()
        with_correct = df[with_mask]['sampler_correct'].sum()
        
        num_zero = zero_mask.sum()
        num_with = with_mask.sum()
        
        if num_zero > 0 and num_with > 0:
            contingency = np.array([
                [zero_correct, num_zero - zero_correct],
                [with_correct, num_with - with_correct]
            ])
            try:
                chi2, p_val, _, _ = stats.chi2_contingency(contingency)
                sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
                logger.info(f"{agent_name}: χ²={chi2:.3f}, p={p_val:.4f} ({sig})")
            except ValueError as e:
                logger.info(f"{agent_name}: Chi-square failed: {e}")
                
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
            if agent_name in baseline_keys: continue
            
            agent_searches = df['sampler_search_calls'].values
            min_len = min(len(agent_searches), len(base_searches))
            
            t_stat, p_val = stats.ttest_rel(agent_searches[:min_len], base_searches[:min_len])
            mean_diff = np.mean(agent_searches) - np.mean(base_searches)
            
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            logger.info(f"{agent_name} vs {base_key}: Diff={mean_diff:+.2f}, p={p_val:.4f} ({sig})")

    # ANOVA
    min_len = min(len(df) for df in datasets.values())
    all_searches = [df['sampler_search_calls'].values[:min_len] for df in datasets.values()]
    
    if len(all_searches) > 1:
        f_stat, anova_p = stats.f_oneway(*all_searches)
        sig = '***' if anova_p < 0.001 else '**' if anova_p < 0.01 else '*' if anova_p < 0.05 else 'ns'
        logger.info(f"\nOverall ANOVA (Search Calls): F={f_stat:.3f}, p={anova_p:.4f} ({sig})")
        return anova_p
    return None

def analyze_lambda_correlation(datasets: Dict[str, pd.DataFrame]):
    """Analyze correlation between lambda values and search calls."""
    lambda_vals = []
    search_calls = []
    
    for name, df in datasets.items():
        val = 0.0
        if 'baseline' in name.lower():
            val = 0.0
        elif 'lambda_' in name:
            try:
                match = re.search(r'lambda_([0-9.]+)', name)
                if match: val = float(match.group(1))
                else: continue
            except:
                continue
        else:
            continue
            
        for calls in df['sampler_search_calls']:
            lambda_vals.append(val)
            search_calls.append(calls)
            
    if len(lambda_vals) > 1:
        corr = np.corrcoef(lambda_vals, search_calls)[0, 1]
        logger.info(f"Pearson correlation (Lambda vs Search Calls): {corr:.3f}")

def run_llm_agreement(datasets: Dict[str, pd.DataFrame], model_name: str, output_dir: str):
    """Run LLM-based agreement analysis."""
    if BaseAgent is None:
        logger.error("BaseAgent not imported. Skipping LLM agreement analysis.")
        return

    # Define Pydantic models for structured output
    class AnswerGroup(BaseModel):
        group_id: int
        representative_answer: str
        answer_variants: List[str]
        agent_names: List[str]
        reasoning: str

    class AnswerGroupingResult(BaseModel):
        question_id: int
        total_agents: int
        answer_groups: List[AnswerGroup]
        num_unique_groups: int
        largest_group_size: int
        agreement_score: float

    min_len = min(len(df) for df in datasets.values())
    
    try:
        logger.info(f"Initializing LLM agent ({model_name})...")
        grouping_agent = BaseAgent(
            provider_name="ollama",
            model_name=model_name,
            output_type=AnswerGroupingResult,
            agent_name="answer_grouping_agent"
        )
        
        results = []
        for q_idx in range(min_len):
            agent_responses = {}
            for name, df in datasets.items():
                resp = df.iloc[q_idx]['sampler_response']
                match = re.search(r'Exact Answer:\s*(.+?)(?:\n|$)', resp, re.IGNORECASE)
                agent_responses[name] = match.group(1).strip() if match else "NO_ANSWER"
            
            prompt = f"Question ID: {q_idx}\nAnalyze these answers for semantic equivalence:\n"
            for name, ans in agent_responses.items():
                prompt += f"- {name}: {ans}\n"
            
            try:
                res = grouping_agent.run(prompt).output
                results.append(res.model_dump())
                logger.info(f"Q{q_idx}: {res.num_unique_groups} groups, score={res.agreement_score:.2f}")
            except Exception as e:
                logger.error(f"Error on Q{q_idx}: {e}")

        # Save results
        out_file = os.path.join(output_dir, 'llm_agreement_analysis.json')
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved LLM agreement analysis to {out_file}")

    except Exception as e:
        logger.error(f"Failed to run LLM analysis: {e}")

def create_plots(datasets: Dict[str, pd.DataFrame], search_stats_df: pd.DataFrame, 
                zero_search_df: pd.DataFrame, output_dir: str):
    """Generate and save visualization plots."""
    sns.set_style("whitegrid")
    
    # Figure 1: Comparison
    fig1 = plt.figure(figsize=(20, 16))
    gs1 = fig1.add_gridspec(2, 2, hspace=0.5, wspace=0.3)
    
    # 1.1 Avg Search Calls
    ax1 = fig1.add_subplot(gs1[0, 0])
    means = search_stats_df['Mean Searches']
    sems = [stats.sem(datasets[a]['sampler_search_calls']) for a in search_stats_df['Agent']]
    cis = [sem * stats.t.ppf(0.975, len(datasets[a])-1) if len(datasets[a]) > 1 else 0 
           for a, sem in zip(search_stats_df['Agent'], sems)]
    
    ax1.bar(search_stats_df['Agent'], means, yerr=cis, capsize=5, alpha=0.7)
    ax1.set_xticklabels(search_stats_df['Agent'], rotation=45, ha='right')
    ax1.set_title('Average Search Calls (95% CI)')
    
    # 1.2 Accuracy
    ax2 = fig1.add_subplot(gs1[0, 1])
    accs = [df['sampler_correct'].mean() * 100 for df in datasets.values()]
    ax2.bar(datasets.keys(), accs, alpha=0.7, color='green')
    ax2.set_xticklabels(datasets.keys(), rotation=45, ha='right')
    ax2.set_title('Accuracy (%)')
    ax2.set_ylim(0, 100)
    
    # 1.3 Search Calls per Question
    ax3 = fig1.add_subplot(gs1[1, 0])
    for name, df in datasets.items():
        if 'baseline' not in name.lower():
            ax3.plot(df['sampler_search_calls'].values, alpha=0.5, label=name)
    ax3.set_title('Search Calls per Question')
    
    # 1.4 Zero vs With Search
    ax4 = fig1.add_subplot(gs1[1, 1])
    x = np.arange(len(zero_search_df))
    w = 0.35
    ax4.bar(x - w/2, zero_search_df['Zero Search Accuracy'], w, label='Zero Search')
    ax4.bar(x + w/2, zero_search_df['With Search Accuracy'], w, label='With Search')
    ax4.set_xticks(x)
    ax4.set_xticklabels(zero_search_df['Agent'], rotation=45, ha='right')
    ax4.set_title('Zero vs With Search Accuracy')
    ax4.legend()
    
    plt.savefig(os.path.join(output_dir, 'agent_comparison.png'), bbox_inches='tight')
    plt.close()
    
    # Figure 2: Heatmap
    min_len = min(len(df) for df in datasets.values())
    matrix = np.array([df['sampler_correct'].astype(int).values[:min_len] for df in datasets.values()])
    
    plt.figure(figsize=(max(16, min_len * 0.5), 6))
    sns.heatmap(matrix, cmap='RdYlGn', cbar_kws={'label': 'Correct (1) / Incorrect (0)'},
                yticklabels=list(datasets.keys()))
    plt.title('Per-Question Accuracy Heatmap')
    plt.xlabel('Question Index')
    plt.savefig(os.path.join(output_dir, 'accuracy_heatmap.png'), bbox_inches='tight')
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")

def analyze_aggregation(datasets: Dict[str, pd.DataFrame], baseline_name: str):
    """Analyze oracle coverage of different agent combinations."""
    baseline_agents = sorted([k for k in datasets.keys() if baseline_name in k.lower()])
    mmr_agents = sorted([k for k in datasets.keys() if 'mmr' in k.lower()])
    
    combinations = {
        'Baseline Only': baseline_agents,
        'MMR Only': mmr_agents,
    }
    
    if baseline_agents and mmr_agents:
        combinations['Baseline + MMR'] = baseline_agents + mmr_agents
        combinations['Single Baseline + MMR'] = [baseline_agents[0]] + mmr_agents

    results = []
    min_len = min(len(df) for df in datasets.values())
    
    for name, agents in combinations.items():
        if not agents: continue
        
        # Matrix: questions x agents (1 if correct, 0 if not)
        matrix = np.array([datasets[a]['sampler_correct'].values[:min_len] for a in agents]).T
        
        # Oracle: Correct if ANY agent is correct
        oracle_correct = np.any(matrix, axis=1).sum()
        oracle_acc = (oracle_correct / min_len) * 100
        
        # Best individual
        best_ind = max([datasets[a]['sampler_correct'].mean() for a in agents]) * 100
        
        results.append({
            'Combination': name,
            'Num Agents': len(agents),
            'Oracle Accuracy': oracle_acc,
            'Best Individual': best_ind,
            'Improvement': oracle_acc - best_ind
        })
        
    df = pd.DataFrame(results)
    logger.info("\nAggregation Comparison:")
    logger.info(df.to_string(index=False))

def main():
    args = parse_arguments()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    print_separator("LOADING DATASETS")
    datasets = load_datasets(args.logs_dir, args.file_pattern)
    
    print_separator("SEARCH STATISTICS")
    search_stats_df = analyze_search_stats(datasets)
    analyze_lambda_correlation(datasets)
    
    print_separator("ZERO SEARCH ANALYSIS")
    zero_search_df = analyze_zero_search(datasets)
    
    print_separator("STATISTICAL TESTS")
    run_statistical_tests(datasets, args.baseline_agent)
    
    print_separator("AGGREGATION ANALYSIS")
    analyze_aggregation(datasets, args.baseline_agent)
    
    if not args.no_plots:
        print_separator("GENERATING PLOTS")
        create_plots(datasets, search_stats_df, zero_search_df, args.output_dir)
        
    if args.run_llm_agreement:
        print_separator("LLM AGREEMENT ANALYSIS")
        run_llm_agreement(datasets, args.llm_model, args.output_dir)

if __name__ == "__main__":
    main()