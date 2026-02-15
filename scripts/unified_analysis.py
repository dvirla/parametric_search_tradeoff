"""
Unified analysis script for parametric search tradeoff experiments.

Replaces both analyze_semantic_entropy.py and visualize_results.py with a single
per-model analysis tool that supports:
- Multiple datasets with explicit paths (no hardcoded maps)
- 5 no-search + 5 baseline runs with stability analysis
- Cross-dataset aggregation with --aggregate
- Markdown report generation with embedded figure references

Usage:
    uv run python scripts/unified_analysis.py \
      --model-name "Gemini 3 Pro" \
      --datasets \
        "facts_one_hop:path/analysis.csv:path/json_dir:path/traces.json:path/entropy.csv" \
        "popqa:path/to/baseline_run_*_analysis.csv:path/json_dir::path/entropy.csv" \
      --output-dir results/gemini_3_pro \
      --aggregate

The analysis_csv_glob field supports glob patterns. If it matches multiple
files (e.g. per-run analysis CSVs), each gets a run_id column and the
resulting DataFrame has multiple rows per problem_id.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sns.set_theme(style="whitegrid")

EPISTEMIC_STATE_ORDER = ['IGNORANCE', 'AMBIGUITY', 'HIGH_CERTAINTY', 'DECISIVE']


# ─── Utility functions ─────────────────────────────────────────────────────────


def clean_problem(problem: str) -> str:
    separator = "\n\nYour response should be in the following format:"
    if separator in problem:
        return problem.split(separator)[0].strip()
    return problem.strip()


def get_problem_id(problem: str) -> str:
    return clean_problem(problem)[:50] + "..."


# ─── Report Builder ─────────────────────────────────────────────────────────────


class ReportBuilder:
    """Accumulates sections for a Markdown report."""

    def __init__(self, model_name: str, datasets: list[str], n_questions: int):
        self.sections: list[str] = []
        self.model_name = model_name
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        header = (
            f"# Analysis Report: {model_name}\n\n"
            f"Generated: {now}  \n"
            f"Datasets: {', '.join(datasets)}  \n"
            f"Total questions: {n_questions}\n"
        )
        self.sections.append(header)
        self._section_num = 0

    def add_section(self, title: str, content: str):
        self._section_num += 1
        self.sections.append(f"## {self._section_num}. {title}\n\n{content}\n")

    def build(self) -> str:
        return "\n".join(self.sections)


# ─── Data Loading ────────────────────────────────────────────────────────────────


def _normalize_analysis_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply column renaming, epistemic state normalization, and boolean conversion."""
    column_mapping = {
        'judge1_confidence': 'epistemic_state',
        'judge2_hypothesis_correct': 'hypothesis_correct',
        'judge2_query_biased': 'is_search_query_biased',
        'judge2_snippet_has_answer': 'snippet_has_answer',
        'judge2_answer_flipped': 'answer_flipped',
    }
    if 'epistemic_state' in df.columns:
        del column_mapping['judge1_confidence']
    df.rename(columns=column_mapping, inplace=True)

    old_to_new = {
        'TABULA_RASA': 'IGNORANCE',
        'WEAK_GUESS': 'AMBIGUITY',
        'STRONG_HYPOTHESIS': 'HIGH_CERTAINTY',
        'NO_SEARCH': 'DECISIVE',
    }
    if 'epistemic_state' in df.columns:
        df['epistemic_state'] = df['epistemic_state'].replace(old_to_new)

    # Convert boolean-like columns
    bool_cols = ['baseline_correct', 'agent_correct', 'snippet_has_answer',
                 'answer_flipped', 'is_search_query_biased']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).map(
                {'True': True, 'False': False, '1': True, '0': False,
                 '1.0': True, '0.0': False}
            )
            df[col] = df[col].fillna(False)

    # Ensure problem_id
    if 'problem_id' not in df.columns and 'problem' in df.columns:
        df['problem_id'] = df['problem'].apply(get_problem_id)

    return df


def _extract_run_id(filepath: str) -> int:
    """Try to extract a numeric run_id from a filename like 'baseline_run_3_analysis...csv'.

    Falls back to -1 if no pattern found (caller assigns sequential IDs).
    """
    basename = os.path.basename(filepath)
    match = re.search(r'run[_\-]?(\d+)', basename)
    return int(match.group(1)) if match else -1


def load_analysis_csvs(glob_or_path: str) -> pd.DataFrame | None:
    """Load one or more analysis CSVs matching a glob pattern.

    If the pattern matches multiple files, each gets a `run_id` column
    (extracted from filename or assigned sequentially). If it matches one
    file, `run_id=0`. Returns None if no files match.
    """
    files = sorted(glob.glob(glob_or_path))
    if not files:
        print(f"  No analysis CSVs matched: {glob_or_path}")
        return None

    frames = []
    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            df = _normalize_analysis_df(df)
            df['_source_file'] = filepath
            frames.append(df)
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    if not frames:
        return None

    if len(frames) == 1:
        frames[0]['run_id'] = 0
        print(f"  Loaded 1 analysis CSV ({len(frames[0])} rows): {files[0]}")
        return frames[0].drop(columns=['_source_file'])

    # Multiple files: assign run_ids
    for i, frame in enumerate(frames):
        extracted = _extract_run_id(frame['_source_file'].iloc[0])
        frame['run_id'] = extracted if extracted >= 0 else i

    combined = pd.concat(frames, ignore_index=True).drop(columns=['_source_file'])
    n_runs = combined['run_id'].nunique()
    print(f"  Loaded {len(files)} analysis CSVs ({len(combined)} rows, {n_runs} runs)")
    return combined


def load_traces_info(traces_path: str) -> dict:
    """Load traces and extract search usage per problem_id."""
    if not traces_path or not os.path.exists(traces_path):
        return {}

    try:
        with open(traces_path, 'r') as f:
            traces = json.load(f)
    except Exception as e:
        print(f"  Error reading traces: {e}")
        return {}

    info_map = {}
    for trace in traces:
        pid = get_problem_id(trace.get('problem', ''))
        num_searches = 0
        for msg in trace.get('message_trace', []):
            if msg.get('role') == 'assistant':
                for part in msg.get('parts', []):
                    if part.get('type') == 'tool_call':
                        num_searches += 1
        info_map[pid] = num_searches

    print(f"  Loaded trace info for {len(info_map)} problems")
    return info_map


def load_semantic_entropy_csv(csv_path: str) -> pd.DataFrame | None:
    """Load a single semantic entropy CSV."""
    if not csv_path or not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    if 'problem' in df.columns:
        df['problem_id'] = df['problem'].apply(get_problem_id)
    return df


def load_runs_from_json_dir(json_dir: str, glob_pattern: str) -> list[pd.DataFrame]:
    """Load run JSONs from json_dir matching glob_pattern.

    Returns list of DataFrames (one per file) with columns:
    problem_id, sampler_correct, sampler_search_calls, sampler_response, run_id
    """
    if not json_dir or not os.path.exists(json_dir):
        return []

    pattern = os.path.join(json_dir, glob_pattern)
    files = sorted(glob.glob(pattern))
    if not files:
        return []

    frames = []
    for i, filepath in enumerate(files):
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            rows = []
            for entry in data:
                rows.append({
                    'problem_id': get_problem_id(entry.get('problem', '')),
                    'sampler_correct': bool(entry.get('sampler_correct', False)),
                    'sampler_search_calls': entry.get('sampler_search_calls', 0),
                    'sampler_response': entry.get('sampler_response', ''),
                    'run_id': i,
                })
            frames.append(pd.DataFrame(rows))
        except Exception as e:
            print(f"  Error loading {filepath}: {e}")

    return frames


def compute_multi_run_stats(runs: list[pd.DataFrame], prefix: str) -> pd.DataFrame:
    """Compute per-problem statistics across multiple runs.

    Returns DataFrame with columns:
    problem_id, {prefix}_mean_correct, {prefix}_correctness_agreement,
    {prefix}_search_mean, {prefix}_search_var, {prefix}_answer_flip_rate,
    {prefix}_num_runs
    """
    if not runs:
        return pd.DataFrame()

    combined = pd.concat(runs, ignore_index=True)
    grouped = combined.groupby('problem_id')

    records = []
    for pid, group in grouped:
        group = group.sort_values('run_id')
        n = len(group)
        correct_vals = group['sampler_correct'].astype(float).values
        search_vals = group['sampler_search_calls'].astype(float).values

        mean_correct = correct_vals.mean()
        # Majority label
        majority = 1.0 if mean_correct >= 0.5 else 0.0
        agreement = (correct_vals == majority).mean()

        # Answer flip rate: fraction of adjacent pairs with different correctness
        if n > 1:
            flips = sum(1 for a, b in zip(correct_vals[:-1], correct_vals[1:]) if a != b)
            flip_rate = flips / (n - 1)
        else:
            flip_rate = 0.0

        records.append({
            'problem_id': pid,
            f'{prefix}_mean_correct': mean_correct,
            f'{prefix}_correctness_agreement': agreement,
            f'{prefix}_search_mean': search_vals.mean(),
            f'{prefix}_search_var': search_vals.var() if n > 1 else 0.0,
            f'{prefix}_answer_flip_rate': flip_rate,
            f'{prefix}_num_runs': n,
        })

    return pd.DataFrame(records)


# ─── Merge Pipeline ──────────────────────────────────────────────────────────────


def parse_dataset_spec(spec: str) -> dict:
    """Parse a colon-delimited dataset spec string.

    Format: LABEL:analysis_csv_glob:json_dir[:traces_json][:entropy_csv]
    The analysis_csv_glob field supports glob patterns to load multiple
    per-run analysis CSVs (e.g., 'logs/popqa/baseline_run_*_analysis.csv').
    Empty fields are treated as missing.
    """
    parts = spec.split(':')
    if len(parts) < 3:
        raise ValueError(
            f"Dataset spec must have at least 3 colon-separated fields "
            f"(LABEL:analysis_csv_glob:json_dir), got: {spec}"
        )

    result = {
        'label': parts[0],
        'analysis_glob': parts[1],
        'json_dir': parts[2],
        'traces_json': parts[3] if len(parts) > 3 and parts[3] else None,
        'entropy_csv': parts[4] if len(parts) > 4 and parts[4] else None,
    }
    return result


def build_dataset_frame(
    spec: dict,
    no_search_glob: str,
    baseline_glob: str,
    variant: str,
) -> pd.DataFrame | None:
    """Build merged DataFrame for a single dataset."""
    label = spec['label']
    print(f"\n--- Processing dataset: {label} ---")

    # Load analysis CSV(s) — supports glob for multiple per-run CSVs
    df = load_analysis_csvs(spec['analysis_glob'])
    if df is None:
        print(f"  Cannot proceed without analysis CSV for {label}")
        return None

    has_multi_run = 'run_id' in df.columns and df['run_id'].nunique() > 1
    if has_multi_run:
        print(f"  Multi-run analysis: {df['run_id'].nunique()} runs, {df['problem_id'].nunique()} unique problems")

    # Load no-search runs
    no_search_runs = load_runs_from_json_dir(spec['json_dir'], no_search_glob)
    if no_search_runs:
        print(f"  Found {len(no_search_runs)} no-search run files")
        ns_stats = compute_multi_run_stats(no_search_runs, 'no_search')
        df = df.merge(ns_stats, on='problem_id', how='left')
        matched = df['no_search_mean_correct'].notna().sum()
        print(f"  Matched {matched}/{len(df)} with no-search stats")

        # Backward compat: derive no_search_confidence and baseline_correct
        df['no_search_confidence'] = df['no_search_mean_correct']
        df['baseline_correct'] = (df['no_search_mean_correct'] == 1.0)
    else:
        print(f"  No no-search run files found matching {no_search_glob}")

    # Load baseline runs
    baseline_runs = load_runs_from_json_dir(spec['json_dir'], baseline_glob)
    if baseline_runs:
        print(f"  Found {len(baseline_runs)} baseline run files")
        bl_stats = compute_multi_run_stats(baseline_runs, 'baseline')
        df = df.merge(bl_stats, on='problem_id', how='left')
        matched = df['baseline_mean_correct'].notna().sum()
        print(f"  Matched {matched}/{len(df)} with baseline stats")

        # Also extract search counts from baseline for backward compat
        if 'num_searches' not in df.columns:
            df['num_searches'] = df['baseline_search_mean']
    else:
        print(f"  No baseline run files found matching {baseline_glob}")

    # Load traces (optional)
    if spec['traces_json']:
        trace_map = load_traces_info(spec['traces_json'])
        if trace_map:
            df['num_searches_from_traces'] = df['problem_id'].map(trace_map)
            matched = df['num_searches_from_traces'].notna().sum()
            print(f"  Matched {matched}/{len(df)} with trace search counts")
            # Use trace-based search counts as primary if no baseline search counts
            if 'num_searches' not in df.columns:
                df['num_searches'] = df['num_searches_from_traces']

    # Load semantic entropy (optional)
    entropy_df = load_semantic_entropy_csv(spec['entropy_csv'])
    if entropy_df is not None:
        entropy_cols = ['problem_id']
        for col in ['entropy', 'num_clusters', 'majority_answer_correct']:
            if col in entropy_df.columns:
                entropy_cols.append(col)
        df = df.merge(entropy_df[entropy_cols], on='problem_id', how='left')
        matched = df['entropy'].notna().sum() if 'entropy' in df.columns else 0
        print(f"  Matched {matched}/{len(df)} with entropy data")
    else:
        print(f"  No entropy data for {label}")

    # Apply DECISIVE tagging
    if 'num_searches' in df.columns:
        if 'epistemic_state' not in df.columns:
            df['epistemic_state'] = np.nan
        no_search_mask = (df['num_searches'] == 0)
        df.loc[no_search_mask, 'epistemic_state'] = 'DECISIVE'
        print(f"  Tagged {no_search_mask.sum()} records as DECISIVE")

    # Derived flags
    if 'baseline_correct' in df.columns and 'agent_correct' in df.columns:
        if 'is_context_poisoning' not in df.columns:
            df['is_context_poisoning'] = df['baseline_correct'] & (~df['agent_correct'])

    if 'epistemic_state' in df.columns and 'baseline_correct' in df.columns:
        if 'is_performative_ignorance' not in df.columns:
            df['is_performative_ignorance'] = (
                df['baseline_correct'] & df['epistemic_state'].isin(['IGNORANCE', 'AMBIGUITY'])
            )

    if 'epistemic_state' in df.columns and 'is_search_query_biased' in df.columns:
        if 'is_confirmation_bias' not in df.columns:
            df['is_confirmation_bias'] = (
                df['epistemic_state'].isin(['HIGH_CERTAINTY', 'AMBIGUITY']) & df['is_search_query_biased']
            )

    if 'snippet_has_answer' in df.columns and 'agent_correct' in df.columns:
        if 'is_utilization_failure' not in df.columns:
            df['is_utilization_failure'] = df['snippet_has_answer'] & (~df['agent_correct'])

    df['dataset'] = label
    return df


# ─── Section A: Epistemic State Analysis ─────────────────────────────────────────


def section_epistemic_state(df: pd.DataFrame, output_dir: str, report: ReportBuilder):
    """Generate epistemic state analysis plots and report section."""
    content_parts = []

    # A1: Known fact epistemic distribution
    if 'baseline_correct' in df.columns and 'epistemic_state' in df.columns:
        known_df = df[df['baseline_correct'] == True].copy()
        valid_states = EPISTEMIC_STATE_ORDER
        known_df = known_df[known_df['epistemic_state'].isin(valid_states)]

        if not known_df.empty:
            def normalize_correct(val):
                s = str(val).upper().strip()
                if s == 'YES':
                    return 'Correct'
                if s == 'NO':
                    return 'Incorrect'
                return 'N/A'

            if 'hypothesis_correct' in known_df.columns:
                known_df['hyp_accuracy'] = known_df['hypothesis_correct'].apply(normalize_correct)
            else:
                known_df['hyp_accuracy'] = 'N/A'

            total_known = len(known_df)
            counts = known_df.groupby(
                ['epistemic_state', 'hyp_accuracy'], observed=False
            ).size().reset_index(name='count')
            counts['Percentage'] = (counts['count'] / total_known) * 100
            counts['epistemic_state'] = pd.Categorical(
                counts['epistemic_state'], categories=valid_states, ordered=True
            )
            counts = counts.sort_values(['epistemic_state', 'hyp_accuracy'])

            plt.figure(figsize=(10, 6))
            ax = sns.barplot(
                data=counts, x='epistemic_state', y='Percentage',
                hue='hyp_accuracy', palette='Set2', order=valid_states
            )
            plt.title(f"Epistemic State Distribution on Known Facts (n={total_known})")
            plt.ylabel('Percentage of All Known Facts (%)')
            plt.xlabel('Epistemic State')
            plt.ylim(0, 115)
            plt.legend(title='Hypothesis Accuracy')
            for c in ax.containers:
                ax.bar_label(c, fmt='%.1f%%', label_type='edge')
            plt.tight_layout()
            fname = 'known_fact_epistemic_state_distribution.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(
                f"### Known Fact Epistemic Distribution\n\n"
                f"Among {total_known} known facts (baseline correct), the epistemic state distribution:\n\n"
                f"![]({fname})\n"
            )
            print(f"Generated {fname}")

    # A2: Agreement level vs epistemic heatmap
    if 'no_search_confidence' in df.columns and 'epistemic_state' in df.columns:
        valid_states = EPISTEMIC_STATE_ORDER
        plot_df = df[df['epistemic_state'].isin(valid_states)].copy()

        if not plot_df.empty and 'no_search_confidence' in plot_df.columns:
            unique_scores = plot_df['no_search_confidence'].dropna().unique()
            diffs = np.diff(sorted(unique_scores))
            if len(diffs) > 0 and np.min(diffs) > 0:
                inferred_runs = int(round(1.0 / np.min(diffs)))
            else:
                inferred_runs = 5

            def format_agreement(val):
                runs_correct = round(val * inferred_runs)
                return f"{int(runs_correct)}/{inferred_runs}"

            plot_df['agreement_label'] = plot_df['no_search_confidence'].apply(format_agreement)

            table = pd.crosstab(
                plot_df['agreement_label'], plot_df['epistemic_state'], normalize='index'
            ) * 100
            table = table.reindex(columns=valid_states).fillna(0)
            agreement_order = [f"{i}/{inferred_runs}" for i in range(inferred_runs + 1)]
            table = table.reindex(agreement_order).fillna(0)

            counts_per_level = plot_df['agreement_label'].value_counts()
            table.index = [f"{idx} (n={counts_per_level.get(idx, 0)})" for idx in table.index]

            table.to_csv(os.path.join(output_dir, 'epistemic_state_by_agreement.csv'))

            plt.figure(figsize=(10, 6))
            sns.heatmap(table, annot=True, fmt=".1f", cmap="YlGnBu",
                        cbar_kws={'label': 'Percentage (%)'})
            plt.title(f'Epistemic State Distribution per Agreement Level (n={len(plot_df)})')
            plt.xlabel('Epistemic State')
            plt.ylabel('Agreement Level')
            plt.tight_layout()
            fname = 'epistemic_state_by_agreement_heatmap.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(
                f"### Agreement Level vs Epistemic State\n\n![]({fname})\n\n"
                f"See `epistemic_state_by_agreement.csv` for full data.\n"
            )
            print(f"Generated {fname}")

    # A3: Hypothesis correctness & behaviors
    if 'epistemic_state' in df.columns and 'hypothesis_correct' in df.columns:
        hyp_df = df[df['epistemic_state'].isin(['HIGH_CERTAINTY', 'AMBIGUITY'])].copy()

        if not hyp_df.empty:
            def normalize_correct(val):
                s = str(val).upper().strip()
                if s == 'YES':
                    return 'Correct'
                if s == 'NO':
                    return 'Incorrect'
                return 'Unknown'

            hyp_df['hyp_accuracy'] = hyp_df['hypothesis_correct'].apply(normalize_correct)
            hyp_df = hyp_df[hyp_df['hyp_accuracy'] != 'Unknown']

            if not hyp_df.empty:
                plt.figure(figsize=(8, 6))
                ax = sns.countplot(
                    x='epistemic_state', hue='hyp_accuracy', data=hyp_df,
                    palette='Set2', order=['AMBIGUITY', 'HIGH_CERTAINTY'],
                    hue_order=['Correct', 'Incorrect']
                )
                plt.title('Accuracy of Initial Hypothesis by Epistemic State')
                plt.xlabel('Epistemic State')
                plt.ylabel('Count')
                for c in ax.containers:
                    ax.bar_label(c, label_type='center')
                plt.tight_layout()
                fname = 'hypothesis_accuracy_split.png'
                plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
                plt.close()
                content_parts.append(f"### Hypothesis Accuracy\n\n![]({fname})\n")
                print(f"Generated {fname}")

                # Behaviors
                behavior_flags = {
                    'is_confirmation_bias': 'Confirmation Bias',
                    'answer_flipped': 'Answer Flipped',
                    'is_utilization_failure': 'Utilization Failure',
                }
                existing_flags = [f for f in behavior_flags if f in hyp_df.columns]
                if existing_flags:
                    melted = hyp_df.melt(
                        id_vars=['hyp_accuracy', 'epistemic_state'],
                        value_vars=existing_flags,
                        var_name='Behavior', value_name='IsPresent',
                    )
                    summary = melted.groupby(
                        ['epistemic_state', 'hyp_accuracy', 'Behavior'], observed=True
                    )['IsPresent'].mean().reset_index()
                    summary['Percentage'] = summary['IsPresent'] * 100
                    summary['Behavior Label'] = summary['Behavior'].map(behavior_flags)

                    g = sns.catplot(
                        data=summary, x='Behavior Label', y='Percentage',
                        hue='hyp_accuracy', col='epistemic_state', kind='bar',
                        palette='Set2', height=5, aspect=1.2,
                        col_order=['AMBIGUITY', 'HIGH_CERTAINTY'],
                    )
                    g.fig.subplots_adjust(top=0.85)
                    g.fig.suptitle('Search Behaviors given a Hypothesis')
                    fname = 'hypothesis_behaviors_split.png'
                    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
                    plt.close()
                    content_parts.append(f"### Hypothesis Behaviors\n\n![]({fname})\n")
                    print(f"Generated {fname}")

    # A4: Search vs no-search confidence
    if 'no_search_confidence' in df.columns and 'num_searches' in df.columns:
        plot_df = df.dropna(subset=['no_search_confidence', 'num_searches']).copy()
        if not plot_df.empty:
            plt.figure(figsize=(10, 6))
            sns.stripplot(data=plot_df, x='no_search_confidence', y='num_searches',
                          jitter=True, alpha=0.5, palette='viridis',
                          hue='no_search_confidence', legend=False)
            sns.boxplot(data=plot_df, x='no_search_confidence', y='num_searches',
                        showfliers=False, color='lightgray', boxprops={'alpha': 0.3})
            plt.title('Search Usage vs. No-Search Confidence (Agreement)')
            plt.xlabel('No-Search Confidence (Agreement %)')
            plt.ylabel('Number of Searches')
            plt.tight_layout()
            fname = 'search_vs_nosearch_confidence.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Search vs No-Search Confidence\n\n![]({fname})\n")
            print(f"Generated {fname}")

    # A5: Search vs epistemic state (exclude DECISIVE)
    if 'epistemic_state' in df.columns and 'num_searches' in df.columns:
        valid_states = [c for c in EPISTEMIC_STATE_ORDER if c != 'DECISIVE']
        plot_df = df[df['epistemic_state'].isin(valid_states)].copy()
        if not plot_df.empty:
            plot_df['epistemic_state'] = pd.Categorical(
                plot_df['epistemic_state'], categories=valid_states, ordered=True
            )
            plt.figure(figsize=(10, 6))
            sns.boxplot(data=plot_df, x='epistemic_state', y='num_searches', palette='Set3')
            sns.stripplot(data=plot_df, x='epistemic_state', y='num_searches',
                          color='black', alpha=0.3, jitter=True, size=3)
            plt.title('Search Usage vs. Epistemic State (Excluding DECISIVE)')
            plt.xlabel('Epistemic State')
            plt.ylabel('Number of Searches')
            plt.tight_layout()
            fname = 'search_vs_epistemic_state.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Search vs Epistemic State\n\n![]({fname})\n")
            print(f"Generated {fname}")

    # A6: Misalignment flags summary
    flag_cols = ['is_context_poisoning', 'is_performative_ignorance',
                 'is_confirmation_bias', 'is_utilization_failure']
    existing_flags = [f for f in flag_cols if f in df.columns]
    if existing_flags:
        flag_summary = {}
        for col in existing_flags:
            flag_summary[col] = {
                'count': int(df[col].sum()),
                'rate': round(float(df[col].mean()) * 100, 2),
            }
        flag_df = pd.DataFrame(flag_summary).T
        flag_df.columns = ['Count', 'Rate (%)']
        flag_df.to_csv(os.path.join(output_dir, 'misalignment_flags_summary.csv'))
        content_parts.append(
            f"### Misalignment Flags\n\n{flag_df.to_markdown()}\n\n"
            f"See `misalignment_flags_summary.csv`.\n"
        )
        print("Generated misalignment_flags_summary.csv")

    if content_parts:
        report.add_section("Epistemic State Analysis", "\n".join(content_parts))
    else:
        report.add_section("Epistemic State Analysis", "Skipped: insufficient data.\n")


# ─── Section B: Semantic Entropy Plots ────────────────────────────────────────────


def section_semantic_entropy(df: pd.DataFrame, output_dir: str, report: ReportBuilder,
                             aggregate: bool):
    """Generate semantic entropy plots. Skipped if no entropy data."""
    if 'entropy' not in df.columns or df['entropy'].isna().all():
        report.add_section("Semantic Entropy", "Skipped: no entropy data.\n")
        return

    content_parts = []
    has_dataset_hue = aggregate and df['dataset'].nunique() > 1

    # B1: Entropy vs accuracy (boxplot)
    if 'majority_answer_correct' in df.columns:
        plot_df = df.dropna(subset=['entropy', 'majority_answer_correct']).copy()
        plot_df['Correct'] = plot_df['majority_answer_correct'].map(
            {True: 'Correct', False: 'Incorrect', 'True': 'Correct', 'False': 'Incorrect'}
        )

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.boxplot(data=plot_df, x='Correct', y='entropy', palette='Set2', ax=ax,
                        order=['Correct', 'Incorrect'])
            sns.stripplot(data=plot_df, x='Correct', y='entropy', color='black',
                          alpha=0.3, jitter=True, size=3, ax=ax,
                          order=['Correct', 'Incorrect'])
            correct_n = (plot_df['Correct'] == 'Correct').sum()
            incorrect_n = (plot_df['Correct'] == 'Incorrect').sum()
            ax.set_title(f"Semantic Entropy vs Accuracy (correct={correct_n}, incorrect={incorrect_n})")
            ax.set_ylabel('Semantic Entropy')
            ax.set_xlabel('')
            plt.tight_layout()
            fname = 'entropy_vs_accuracy.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Entropy vs Accuracy\n\n![]({fname})\n")
            print(f"Generated {fname}")

            # Mean entropy bar chart
            hue_col = 'dataset' if has_dataset_hue else None
            means = plot_df.groupby(['Correct'] + (['dataset'] if has_dataset_hue else []))['entropy'].mean().reset_index()
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=means, x='Correct', y='entropy', hue=hue_col, palette='Set2', ax=ax)
            ax.set_title('Mean Semantic Entropy by Accuracy')
            ax.set_ylabel('Mean Entropy')
            for c in ax.containers:
                ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=8)
            plt.tight_layout()
            fname = 'mean_entropy_by_accuracy.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"![]({fname})\n")
            print(f"Generated {fname}")

    # B2: Entropy vs epistemic state
    if 'epistemic_state' in df.columns:
        plot_df = df.dropna(subset=['epistemic_state', 'entropy']).copy()
        valid_conf = [c for c in EPISTEMIC_STATE_ORDER if c in plot_df['epistemic_state'].values]
        plot_df = plot_df[plot_df['epistemic_state'].isin(valid_conf)]

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=plot_df, x='epistemic_state', y='entropy', palette='muted',
                        ax=ax, order=valid_conf)
            sns.stripplot(data=plot_df, x='epistemic_state', y='entropy', color='black',
                          alpha=0.3, jitter=True, size=3, ax=ax, order=valid_conf)
            ax.set_title(f"Semantic Entropy vs Epistemic State (n={len(plot_df)})")
            ax.set_ylabel('Semantic Entropy')
            ax.set_xlabel('Epistemic State')
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            fname = 'entropy_vs_epistemic_state.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Entropy vs Epistemic State\n\n![]({fname})\n")
            print(f"Generated {fname}")

            # Mean entropy by state
            hue_col = 'dataset' if has_dataset_hue else None
            group_cols = ['epistemic_state'] + (['dataset'] if has_dataset_hue else [])
            agg = plot_df.groupby(group_cols)['entropy'].agg(['mean', 'count']).reset_index()
            agg.columns = group_cols + ['mean_entropy', 'count']

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=agg, x='epistemic_state', y='mean_entropy', hue=hue_col,
                        palette='viridis', ax=ax, order=valid_conf)
            ax.set_title('Mean Semantic Entropy by Epistemic State')
            ax.set_ylabel('Mean Entropy')
            ax.set_xlabel('Epistemic State')
            for c in ax.containers:
                ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=7)
            plt.tight_layout()
            fname = 'mean_entropy_by_epistemic_state.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"![]({fname})\n")
            print(f"Generated {fname}")

    # B3: Entropy vs search count
    if 'num_searches' in df.columns:
        plot_df = df.dropna(subset=['entropy', 'num_searches']).copy()
        plot_df['num_searches'] = plot_df['num_searches'].astype(int)

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(plot_df['entropy'], plot_df['num_searches'], alpha=0.5, s=30,
                       edgecolors='w', linewidth=0.5)
            if len(plot_df) > 2:
                z = np.polyfit(plot_df['entropy'], plot_df['num_searches'], 1)
                p = np.poly1d(z)
                x_range = np.linspace(plot_df['entropy'].min(), plot_df['entropy'].max(), 100)
                ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=1.5)
                r, pval = stats.pearsonr(plot_df['entropy'], plot_df['num_searches'])
                ax.annotate(f"r={r:.3f}, p={pval:.3f}", xy=(0.05, 0.95),
                            xycoords='axes fraction', fontsize=8, va='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
            ax.set_title(f"Semantic Entropy vs Search Count (n={len(plot_df)})")
            ax.set_xlabel('Semantic Entropy')
            ax.set_ylabel('Number of Searches')
            plt.tight_layout()
            fname = 'entropy_vs_searches_scatter.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Entropy vs Search Count\n\n![]({fname})\n")
            print(f"Generated {fname}")

            # Binned boxplot
            plot_df['entropy_bin'] = pd.cut(
                plot_df['entropy'],
                bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
                labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]'],
            )
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=plot_df, x='entropy_bin', y='num_searches',
                        palette='coolwarm', ax=ax)
            ax.set_title(f"Search Count by Entropy Range (n={len(plot_df)})")
            ax.set_xlabel('Entropy Range')
            ax.set_ylabel('Number of Searches')
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            fname = 'searches_by_entropy_bin.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"![]({fname})\n")
            print(f"Generated {fname}")

    # B4: Entropy vs mean accuracy
    mean_acc_col = 'no_search_mean_correct' if 'no_search_mean_correct' in df.columns else None
    if mean_acc_col:
        plot_df = df.dropna(subset=['entropy', mean_acc_col]).copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(plot_df['entropy'], plot_df[mean_acc_col], alpha=0.5, s=30,
                       edgecolors='w', linewidth=0.5)
            if len(plot_df) > 2:
                z = np.polyfit(plot_df['entropy'], plot_df[mean_acc_col], 1)
                p = np.poly1d(z)
                x_range = np.linspace(plot_df['entropy'].min(), plot_df['entropy'].max(), 100)
                ax.plot(x_range, p(x_range), 'r--', alpha=0.7, linewidth=1.5)
                r, pval = stats.pearsonr(plot_df['entropy'], plot_df[mean_acc_col])
                ax.annotate(f"r={r:.3f}, p={pval:.3f}", xy=(0.05, 0.95),
                            xycoords='axes fraction', fontsize=8, va='top',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))
            ax.set_title(f"Entropy vs Mean Accuracy (no-search, n={len(plot_df)})")
            ax.set_xlabel('Semantic Entropy')
            ax.set_ylabel('Mean Accuracy (no-search runs)')
            ax.set_ylim(-0.05, 1.05)
            plt.tight_layout()
            fname = 'entropy_vs_mean_accuracy_scatter.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Entropy vs Mean Accuracy\n\n![]({fname})\n")
            print(f"Generated {fname}")

            # Binned bar chart
            plot_df['entropy_bin'] = pd.cut(
                plot_df['entropy'],
                bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
                labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]'],
            )
            hue_col = 'dataset' if has_dataset_hue else None
            group_cols = ['entropy_bin'] + (['dataset'] if has_dataset_hue else [])
            agg = plot_df.groupby(group_cols, observed=False)[mean_acc_col].agg(
                ['mean', 'count']
            ).reset_index()
            agg.columns = group_cols + ['mean_accuracy', 'count']
            agg = agg[agg['count'] > 0]

            fig, ax = plt.subplots(figsize=(8, 5))
            sns.barplot(data=agg, x='entropy_bin', y='mean_accuracy', hue=hue_col,
                        palette='viridis', ax=ax)
            ax.set_title('Mean Accuracy by Entropy Bin')
            ax.set_ylabel('Mean Accuracy')
            ax.set_xlabel('Entropy Range')
            ax.set_ylim(0, 1.05)
            for c in ax.containers:
                ax.bar_label(c, fmt='%.2f', label_type='edge', fontsize=7)
            plt.tight_layout()
            fname = 'mean_accuracy_by_entropy_bin.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"![]({fname})\n")
            print(f"Generated {fname}")

    # B5: Combined heatmap (entropy bin x epistemic state)
    if 'epistemic_state' in df.columns and 'num_searches' in df.columns and 'entropy' in df.columns:
        plot_df = df.dropna(subset=['epistemic_state', 'num_searches', 'entropy']).copy()
        valid_conf = [c for c in EPISTEMIC_STATE_ORDER if c in plot_df['epistemic_state'].values]
        plot_df = plot_df[plot_df['epistemic_state'].isin(valid_conf)]

        if not plot_df.empty:
            plot_df['entropy_bin'] = pd.cut(
                plot_df['entropy'],
                bins=[-0.01, 0.0, 0.5, 1.0, 1.5, 2.5],
                labels=['0', '(0, 0.5]', '(0.5, 1]', '(1, 1.5]', '(1.5, 2.3]'],
            )
            pivot = plot_df.pivot_table(
                values='num_searches', index='entropy_bin', columns='epistemic_state',
                aggfunc='mean'
            ).reindex(columns=valid_conf)
            count_pivot = plot_df.pivot_table(
                values='num_searches', index='entropy_bin', columns='epistemic_state',
                aggfunc='count'
            ).reindex(columns=valid_conf).fillna(0).astype(int)

            annot = pd.DataFrame('', index=pivot.index, columns=pivot.columns)
            for r in pivot.index:
                for c_col in pivot.columns:
                    val = pivot.loc[r, c_col] if pd.notna(pivot.loc[r, c_col]) else 0
                    cnt = count_pivot.loc[r, c_col] if r in count_pivot.index and c_col in count_pivot.columns else 0
                    annot.loc[r, c_col] = f"{val:.1f}\n(n={int(cnt)})" if cnt > 0 else ""

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(pivot, annot=annot, fmt='', cmap='YlOrRd', ax=ax,
                        cbar_kws={'label': 'Mean Searches'})
            ax.set_title('Mean Search Count by Entropy & Epistemic State')
            ax.set_xlabel('Epistemic State')
            ax.set_ylabel('Semantic Entropy Range')
            plt.tight_layout()
            fname = 'entropy_epistemic_state_search_heatmap.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Entropy x Epistemic State Heatmap\n\n![]({fname})\n")
            print(f"Generated {fname}")

    if content_parts:
        report.add_section("Semantic Entropy", "\n".join(content_parts))
    else:
        report.add_section("Semantic Entropy", "Skipped: no entropy data.\n")


# ─── Section C: Statistical Tests ─────────────────────────────────────────────────


def section_statistical_tests(df: pd.DataFrame, output_dir: str, report: ReportBuilder):
    """Correlations, AUROC, logistic regression, ECE, risk-coverage."""
    content_parts = []
    has_entropy = 'entropy' in df.columns and not df['entropy'].isna().all()

    epistemic_state_mapping = {
        'IGNORANCE': 0, 'AMBIGUITY': 1, 'HIGH_CERTAINTY': 2, 'DECISIVE': 3,
    }

    # Determine mean accuracy column
    mean_acc_col = 'no_search_mean_correct' if 'no_search_mean_correct' in df.columns else None

    # ── Correlations ──
    stats_list = []

    # Entropy correlations
    if has_entropy:
        # Entropy vs accuracy (point-biserial)
        if 'majority_answer_correct' in df.columns:
            acc_col = df['majority_answer_correct'].map(
                {True: 1, False: 0, 'True': 1, 'False': 0}
            ).dropna()
            entropy_for_acc = df.loc[acc_col.index, 'entropy']
            if len(acc_col) > 2:
                r, p = stats.pointbiserialr(acc_col, entropy_for_acc)
                stats_list.append({
                    'Relationship': 'Entropy vs. Accuracy',
                    'Method': 'Point-Biserial', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(acc_col),
                })

        # Entropy vs mean accuracy (Pearson)
        if mean_acc_col:
            acc_sub = df.dropna(subset=[mean_acc_col, 'entropy'])
            if len(acc_sub) > 2:
                r, p = stats.pearsonr(acc_sub['entropy'], acc_sub[mean_acc_col])
                stats_list.append({
                    'Relationship': 'Entropy vs. Mean Accuracy',
                    'Method': 'Pearson', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(acc_sub),
                })

        # Entropy vs epistemic state (Spearman + Somers' D)
        if 'epistemic_state' in df.columns:
            conf_sub = df.dropna(subset=['epistemic_state', 'entropy']).copy()
            conf_sub['conf_ord'] = conf_sub['epistemic_state'].map(epistemic_state_mapping)
            conf_sub = conf_sub.dropna(subset=['conf_ord'])
            if len(conf_sub) > 2:
                r, p = stats.spearmanr(conf_sub['entropy'], conf_sub['conf_ord'])
                stats_list.append({
                    'Relationship': 'Entropy vs. Epistemic State',
                    'Method': 'Spearman', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(conf_sub),
                })
                d_result = stats.somersd(conf_sub['entropy'], conf_sub['conf_ord'])
                stats_list.append({
                    'Relationship': 'Entropy vs. Epistemic State',
                    'Method': "Somers' D", 'Correlation': round(d_result.statistic, 4),
                    'P-Value': round(d_result.pvalue, 6),
                    'Significant': d_result.pvalue < 0.05, 'N': len(conf_sub),
                })

        # Entropy vs searches
        if 'num_searches' in df.columns:
            search_sub = df.dropna(subset=['entropy', 'num_searches'])
            if len(search_sub) > 2:
                r, p = stats.pearsonr(search_sub['entropy'], search_sub['num_searches'])
                stats_list.append({
                    'Relationship': 'Entropy vs. Search Count',
                    'Method': 'Pearson', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(search_sub),
                })
                r, p = stats.spearmanr(search_sub['entropy'], search_sub['num_searches'])
                stats_list.append({
                    'Relationship': 'Entropy vs. Search Count',
                    'Method': 'Spearman', 'Correlation': round(r, 4),
                    'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(search_sub),
                })

    # Non-entropy correlations
    # No-search confidence vs searches
    if 'no_search_confidence' in df.columns and 'num_searches' in df.columns:
        subset = df.dropna(subset=['no_search_confidence', 'num_searches'])
        if len(subset) > 1:
            r, p = stats.pearsonr(subset['no_search_confidence'], subset['num_searches'])
            stats_list.append({
                'Relationship': 'No-Search Confidence vs. Searches',
                'Method': 'Pearson', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(subset),
            })
            r, p = stats.spearmanr(subset['no_search_confidence'], subset['num_searches'])
            stats_list.append({
                'Relationship': 'No-Search Confidence vs. Searches',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(subset),
            })

    # Epistemic state vs search count
    if 'epistemic_state' in df.columns and 'num_searches' in df.columns:
        es_sub = df.dropna(subset=['epistemic_state', 'num_searches']).copy()
        es_sub['conf_ord'] = es_sub['epistemic_state'].map(epistemic_state_mapping)
        es_sub = es_sub.dropna(subset=['conf_ord'])
        if len(es_sub) > 2:
            r, p = stats.spearmanr(es_sub['conf_ord'], es_sub['num_searches'])
            stats_list.append({
                'Relationship': 'Epistemic State vs. Search Count',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(es_sub),
            })

    # No-search confidence vs epistemic state
    if 'no_search_confidence' in df.columns and 'epistemic_state' in df.columns:
        subset = df.dropna(subset=['no_search_confidence', 'epistemic_state']).copy()
        subset['state_ordinal'] = subset['epistemic_state'].map(epistemic_state_mapping)
        subset = subset.dropna(subset=['state_ordinal'])
        if len(subset) > 1:
            r, p = stats.spearmanr(subset['no_search_confidence'], subset['state_ordinal'])
            stats_list.append({
                'Relationship': 'No-Search Confidence vs. Epistemic State',
                'Method': 'Spearman', 'Correlation': round(r, 4),
                'P-Value': round(p, 6), 'Significant': p < 0.05, 'N': len(subset),
            })

    if stats_list:
        corr_df = pd.DataFrame(stats_list)
        corr_df.to_csv(os.path.join(output_dir, 'correlations.csv'), index=False)
        content_parts.append(f"### Correlations\n\n{corr_df.to_markdown(index=False)}\n")
        print("Saved correlations.csv")

        # Plot correlation bar charts
        for relationship in corr_df['Relationship'].unique():
            rel_df = corr_df[corr_df['Relationship'] == relationship]
            method = rel_df['Method'].iloc[0]
            rel_df = rel_df[rel_df['Method'] == method]
            if len(rel_df) < 1:
                continue
            fig, ax = plt.subplots(figsize=(8, max(3, len(rel_df) * 0.5)))
            colors = ['#2ecc71' if s else '#e74c3c' for s in rel_df['Significant']]
            bars = ax.barh(range(len(rel_df)), rel_df['Correlation'].values, color=colors)
            ax.set_yticks(range(len(rel_df)))
            ax.set_yticklabels(rel_df['Relationship'].values)
            ax.set_xlabel(f'{method} Correlation')
            ax.set_title(f'{relationship}')
            ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.5)
            for bar_item, pval in zip(bars, rel_df['P-Value']):
                width = bar_item.get_width()
                ax.text(width + 0.01 * np.sign(width) if width != 0 else 0.01,
                        bar_item.get_y() + bar_item.get_height() / 2,
                        f'p={pval:.4f}', va='center', fontsize=8)
            plt.tight_layout()
            safe_name = relationship.lower().replace(' ', '_').replace('.', '')
            fname = f'correlation_{safe_name}.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"![]({fname})\n")

    # ── AUROC (entropy only) ──
    if has_entropy and 'epistemic_state' in df.columns:
        plot_df = df.dropna(subset=['epistemic_state', 'entropy']).copy()
        valid_states = [s for s in EPISTEMIC_STATE_ORDER if s in plot_df['epistemic_state'].values]
        plot_df = plot_df[plot_df['epistemic_state'].isin(valid_states)]

        if not plot_df.empty and len(valid_states) >= 2:
            auroc_records = []
            for state in valid_states:
                binary = (plot_df['epistemic_state'] == state).astype(int)
                if binary.sum() == 0 or binary.sum() == len(binary):
                    continue
                try:
                    auc = roc_auc_score(binary, plot_df['entropy'])
                    auroc_records.append({
                        'Epistemic State': state, 'AUROC': round(auc, 4),
                        'N_positive': int(binary.sum()), 'N_total': len(plot_df),
                    })
                except ValueError:
                    pass

            if auroc_records:
                auroc_df = pd.DataFrame(auroc_records)
                auroc_df.to_csv(os.path.join(output_dir, 'pairwise_auroc_results.csv'), index=False)

                fig, ax = plt.subplots(figsize=(8, 5))
                sns.barplot(data=auroc_df, x='Epistemic State', y='AUROC',
                            palette='viridis', ax=ax, order=valid_states)
                ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
                ax.set_title('One-vs-Rest AUROC: Entropy vs Epistemic State')
                ax.set_ylim(0, 1)
                for c_item in ax.containers:
                    ax.bar_label(c_item, fmt='%.3f', label_type='edge', fontsize=8)
                plt.tight_layout()
                fname = 'pairwise_auroc.png'
                plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
                plt.close()
                content_parts.append(
                    f"### AUROC\n\n{auroc_df.to_markdown(index=False)}\n\n![]({fname})\n"
                )
                print(f"Generated {fname}")

    # ── Logistic Regression (entropy only) ──
    if has_entropy and 'epistemic_state' in df.columns:
        plot_df = df.dropna(subset=['epistemic_state', 'entropy']).copy()
        valid_states = [s for s in EPISTEMIC_STATE_ORDER if s in plot_df['epistemic_state'].values]
        plot_df = plot_df[plot_df['epistemic_state'].isin(valid_states)]

        if not plot_df.empty and len(valid_states) >= 2:
            X = plot_df[['entropy']].values
            y = plot_df['epistemic_state'].values

            try:
                lr = LogisticRegression(solver='lbfgs', max_iter=1000)
                lr.fit(X, y)
                accuracy = lr.score(X, y)

                lr_records = []
                for cls_idx, cls in enumerate(lr.classes_):
                    coef_idx = cls_idx if lr.coef_.shape[0] > 1 else 0
                    intercept_idx = cls_idx if lr.intercept_.shape[0] > 1 else 0
                    lr_records.append({
                        'Class': cls,
                        'Coefficient': round(float(lr.coef_[coef_idx][0]), 4),
                        'Intercept': round(float(lr.intercept_[intercept_idx]), 4),
                        'Accuracy': round(accuracy, 4), 'N': len(plot_df),
                    })
                lr_df = pd.DataFrame(lr_records)
                lr_df.to_csv(os.path.join(output_dir, 'logistic_regression_results.csv'), index=False)

                # Plot predicted probabilities
                entropy_range = np.linspace(
                    plot_df['entropy'].min(), plot_df['entropy'].max(), 200
                ).reshape(-1, 1)
                probs = lr.predict_proba(entropy_range)

                fig, ax = plt.subplots(figsize=(8, 5))
                for cls_idx, cls in enumerate(lr.classes_):
                    ax.plot(entropy_range, probs[:, cls_idx], label=cls, linewidth=2)
                ax.set_title(f"Logistic Regression: P(Epistemic State | Entropy)\n(acc={accuracy:.3f}, n={len(plot_df)})")
                ax.set_xlabel('Semantic Entropy')
                ax.set_ylabel('Predicted Probability')
                ax.legend(fontsize=8, loc='best')
                ax.set_ylim(0, 1)
                plt.tight_layout()
                fname = 'logistic_regression.png'
                plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
                plt.close()
                content_parts.append(
                    f"### Logistic Regression\n\n{lr_df.to_markdown(index=False)}\n\n![]({fname})\n"
                )
                print(f"Generated {fname}")
            except Exception as e:
                print(f"  Logistic regression failed: {e}")

    # ── ECE / Reliability Diagram (entropy only) ──
    if has_entropy and mean_acc_col:
        plot_df = df.dropna(subset=[mean_acc_col, 'entropy']).copy()
        if not plot_df.empty:
            max_entropy = np.log2(5)
            plot_df['confidence'] = (1 - plot_df['entropy'] / max_entropy).clip(0, 1)

            n_bins = 10
            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_accs, bin_confs, bin_counts = [], [], []

            for j in range(n_bins):
                lo, hi = bin_edges[j], bin_edges[j + 1]
                if j < n_bins - 1:
                    in_bin = plot_df[(plot_df['confidence'] >= lo) & (plot_df['confidence'] < hi)]
                else:
                    in_bin = plot_df[(plot_df['confidence'] >= lo) & (plot_df['confidence'] <= hi)]
                if len(in_bin) > 0:
                    bin_accs.append(in_bin[mean_acc_col].mean())
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

            ece_record = {
                'ECE': round(ece, 4) if not np.isnan(ece) else np.nan,
                'N': total,
                'Mean_Confidence': round(plot_df['confidence'].mean(), 4),
                'Mean_Accuracy': round(plot_df[mean_acc_col].mean(), 4),
            }
            ece_df = pd.DataFrame([ece_record])
            ece_df.to_csv(os.path.join(output_dir, 'ece_results.csv'), index=False)

            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
            valid = [i for i in range(n_bins) if bin_counts[i] > 0]
            if valid:
                ax.plot(
                    [bin_confs[i] for i in valid],
                    [bin_accs[i] for i in valid],
                    'o-', label=f"ECE={ece:.3f}", markersize=5,
                )
            ax.set_xlabel('Confidence (1 - entropy/log2(5))')
            ax.set_ylabel('Mean Accuracy')
            ax.set_title('Reliability Diagram (ECE)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8, loc='lower right')
            ax.set_aspect('equal')
            plt.tight_layout()
            fname = 'reliability_diagram.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(
                f"### ECE / Reliability Diagram\n\n"
                f"ECE = {ece:.4f} (N={total})\n\n![]({fname})\n"
            )
            print(f"Generated {fname}")

    # ── Risk-Coverage Curve (entropy only) ──
    if has_entropy and mean_acc_col:
        plot_df = df.dropna(subset=[mean_acc_col, 'entropy']).copy()
        if not plot_df.empty:
            sub = plot_df.sort_values('entropy').reset_index(drop=True)
            n = len(sub)
            coverage = np.arange(1, n + 1) / n
            cumulative_acc = sub[mean_acc_col].expanding().mean().values

            fig, ax = plt.subplots(figsize=(9, 6))
            ax.plot(coverage, cumulative_acc, linewidth=1.5)
            baseline_acc = sub[mean_acc_col].mean()
            ax.axhline(y=baseline_acc, color='gray', linestyle=':', alpha=0.5)
            ax.set_xlabel('Coverage (sorted by entropy, most confident first)')
            ax.set_ylabel('Cumulative Mean Accuracy')
            ax.set_title('Risk-Coverage Curve')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.05)
            plt.tight_layout()
            fname = 'risk_coverage_curve.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(f"### Risk-Coverage Curve\n\n![]({fname})\n")
            print(f"Generated {fname}")

    if content_parts:
        report.add_section("Statistical Tests", "\n".join(content_parts))
    else:
        report.add_section("Statistical Tests", "Skipped: insufficient data.\n")


# ─── Section D: Stability Analysis ───────────────────────────────────────────────


def section_stability(df: pd.DataFrame, output_dir: str, report: ReportBuilder):
    """Analyze stability across multiple runs (5-run baseline + no-search)."""
    content_parts = []
    stability_records = []

    for prefix, label in [('no_search', 'No-Search'), ('baseline', 'Baseline')]:
        agree_col = f'{prefix}_correctness_agreement'
        flip_col = f'{prefix}_answer_flip_rate'
        num_runs_col = f'{prefix}_num_runs'

        if agree_col not in df.columns:
            continue

        sub = df.dropna(subset=[agree_col])
        if sub.empty:
            continue

        n_runs = sub[num_runs_col].median() if num_runs_col in sub.columns else 'N/A'

        # Correctness agreement distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(sub[agree_col], bins=20, edgecolor='black', alpha=0.7)
        ax.set_title(f'{label} Correctness Agreement Distribution (n={len(sub)})')
        ax.set_xlabel('Agreement (fraction of runs agreeing with majority)')
        ax.set_ylabel('Count')
        ax.axvline(x=sub[agree_col].mean(), color='red', linestyle='--',
                    label=f'Mean={sub[agree_col].mean():.3f}')
        ax.legend()
        plt.tight_layout()
        fname = f'{prefix}_correctness_agreement.png'
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
        plt.close()
        content_parts.append(
            f"### {label} Correctness Agreement\n\n"
            f"Median runs per problem: {n_runs}\n\n![]({fname})\n"
        )
        print(f"Generated {fname}")

        # Answer flip rate distribution
        if flip_col in df.columns:
            flip_sub = df.dropna(subset=[flip_col])
            if not flip_sub.empty:
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(flip_sub[flip_col], bins=20, edgecolor='black', alpha=0.7, color='salmon')
                ax.set_title(f'{label} Answer Flip Rate Distribution (n={len(flip_sub)})')
                ax.set_xlabel('Flip Rate (fraction of adjacent pairs with different answers)')
                ax.set_ylabel('Count')
                ax.axvline(x=flip_sub[flip_col].mean(), color='red', linestyle='--',
                            label=f'Mean={flip_sub[flip_col].mean():.3f}')
                ax.legend()
                plt.tight_layout()
                fname = f'{prefix}_answer_flip_rate.png'
                plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
                plt.close()
                content_parts.append(f"![]({fname})\n")
                print(f"Generated {fname}")

        stability_records.append({
            'Type': label,
            'N_problems': len(sub),
            'Mean_agreement': round(sub[agree_col].mean(), 4),
            'Mean_flip_rate': round(sub[flip_col].mean(), 4) if flip_col in sub.columns else 'N/A',
        })

    # Baseline search variance by epistemic state
    if 'baseline_search_var' in df.columns and 'epistemic_state' in df.columns:
        plot_df = df.dropna(subset=['baseline_search_var', 'epistemic_state']).copy()
        valid_states = [s for s in EPISTEMIC_STATE_ORDER if s in plot_df['epistemic_state'].values]
        plot_df = plot_df[plot_df['epistemic_state'].isin(valid_states)]

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=plot_df, x='epistemic_state', y='baseline_search_var',
                        palette='Set3', ax=ax, order=valid_states)
            ax.set_title('Baseline Search Count Variance by Epistemic State')
            ax.set_xlabel('Epistemic State')
            ax.set_ylabel('Search Count Variance (across runs)')
            plt.tight_layout()
            fname = 'baseline_search_variance.png'
            plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
            plt.close()
            content_parts.append(
                f"### Baseline Search Variance by Epistemic State\n\n![]({fname})\n"
            )
            print(f"Generated {fname}")

    # Epistemic state stability across per-run analysis CSVs
    if 'run_id' in df.columns and df['run_id'].nunique() > 1 and 'epistemic_state' in df.columns:
        n_analysis_runs = df['run_id'].nunique()
        es_grouped = df.groupby('problem_id')['epistemic_state']

        es_stability = es_grouped.agg(
            mode=lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan,
            agreement=lambda x: (x == x.mode().iloc[0]).mean() if not x.mode().empty else np.nan,
            n_runs='count',
            n_unique='nunique',
        ).reset_index()

        es_stability.to_csv(os.path.join(output_dir, 'epistemic_state_stability.csv'), index=False)

        mean_agreement = es_stability['agreement'].mean()
        fully_stable = (es_stability['agreement'] == 1.0).mean() * 100

        # Distribution of agreement rates
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(es_stability['agreement'], bins=20, edgecolor='black', alpha=0.7, color='mediumpurple')
        ax.set_title(f'Epistemic State Agreement Across {n_analysis_runs} Runs (n={len(es_stability)} problems)')
        ax.set_xlabel('Agreement Rate (fraction of runs with mode state)')
        ax.set_ylabel('Number of Problems')
        ax.axvline(x=mean_agreement, color='red', linestyle='--',
                    label=f'Mean={mean_agreement:.3f}')
        ax.legend()
        plt.tight_layout()
        fname = 'epistemic_state_agreement.png'
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
        plt.close()
        content_parts.append(
            f"### Epistemic State Stability (Per-Run Analysis)\n\n"
            f"Across {n_analysis_runs} analysis runs:\n"
            f"- Mean epistemic state agreement: {mean_agreement:.3f}\n"
            f"- Fully stable (100% agreement): {fully_stable:.1f}%\n\n"
            f"![]({fname})\n"
        )
        print(f"Generated {fname}")

        # Mode epistemic state distribution
        fig, ax = plt.subplots(figsize=(8, 5))
        mode_counts = es_stability['mode'].value_counts()
        mode_counts = mode_counts.reindex(
            [s for s in EPISTEMIC_STATE_ORDER if s in mode_counts.index]
        )
        mode_counts.plot(kind='bar', ax=ax, color='mediumpurple', edgecolor='black', alpha=0.7)
        ax.set_title('Mode Epistemic State Distribution')
        ax.set_ylabel('Number of Problems')
        ax.set_xlabel('Mode Epistemic State')
        ax.tick_params(axis='x', rotation=30)
        for i, v in enumerate(mode_counts.values):
            ax.text(i, v + 0.5, str(v), ha='center', fontsize=9)
        plt.tight_layout()
        fname = 'epistemic_state_mode_distribution.png'
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
        plt.close()
        content_parts.append(f"![]({fname})\n")
        print(f"Generated {fname}")

        stability_records.append({
            'Type': 'Epistemic State',
            'N_problems': len(es_stability),
            'Mean_agreement': round(mean_agreement, 4),
            'Mean_flip_rate': 'N/A',
        })

    if stability_records:
        stab_df = pd.DataFrame(stability_records)
        stab_df.to_csv(os.path.join(output_dir, 'stability_summary.csv'), index=False)
        content_parts.append(
            f"### Stability Summary\n\n{stab_df.to_markdown(index=False)}\n"
        )
        print("Saved stability_summary.csv")

    if content_parts:
        report.add_section("Stability Analysis (5-Run)", "\n".join(content_parts))
    else:
        report.add_section("Stability Analysis (5-Run)", "Skipped: no multi-run data.\n")


# ─── Section E: Cross-Dataset Comparison ──────────────────────────────────────────


def section_cross_dataset(df: pd.DataFrame, output_dir: str, report: ReportBuilder):
    """Cross-dataset comparison (only with --aggregate and multiple datasets)."""
    datasets = df['dataset'].unique()
    if len(datasets) < 2:
        report.add_section("Cross-Dataset Comparison", "Skipped: only one dataset.\n")
        return

    content_parts = []
    records = []

    mean_acc_col = 'no_search_mean_correct' if 'no_search_mean_correct' in df.columns else None

    for ds in sorted(datasets):
        sub = df[df['dataset'] == ds]
        record = {
            'Dataset': ds,
            'N': len(sub),
        }
        if mean_acc_col and mean_acc_col in sub.columns:
            record['Mean_Accuracy'] = round(sub[mean_acc_col].dropna().mean(), 4)
        if 'entropy' in sub.columns:
            record['Mean_Entropy'] = round(sub['entropy'].dropna().mean(), 4)
        if 'num_searches' in sub.columns:
            record['Mean_Searches'] = round(sub['num_searches'].dropna().mean(), 4)
        records.append(record)

    comp_df = pd.DataFrame(records)
    comp_df.to_csv(os.path.join(output_dir, 'dataset_comparison.csv'), index=False)
    content_parts.append(f"### Dataset Comparison\n\n{comp_df.to_markdown(index=False)}\n")

    # Per-dataset accuracy bar
    if mean_acc_col and mean_acc_col in df.columns:
        per_ds = df.groupby('dataset')[mean_acc_col].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=per_ds, x='dataset', y=mean_acc_col, palette='Set2', ax=ax)
        ax.set_title('Mean Accuracy by Dataset')
        ax.set_ylabel('Mean Accuracy')
        ax.set_xlabel('Dataset')
        ax.set_ylim(0, 1.05)
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', label_type='edge', fontsize=8)
        plt.tight_layout()
        fname = 'per_dataset_accuracy.png'
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
        plt.close()
        content_parts.append(f"![]({fname})\n")
        print(f"Generated {fname}")

    # Per-dataset entropy bar
    if 'entropy' in df.columns and not df['entropy'].isna().all():
        per_ds = df.groupby('dataset')['entropy'].mean().reset_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=per_ds, x='dataset', y='entropy', palette='Set2', ax=ax)
        ax.set_title('Mean Semantic Entropy by Dataset')
        ax.set_ylabel('Mean Entropy')
        ax.set_xlabel('Dataset')
        for c in ax.containers:
            ax.bar_label(c, fmt='%.3f', label_type='edge', fontsize=8)
        plt.tight_layout()
        fname = 'per_dataset_entropy.png'
        plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight', dpi=150)
        plt.close()
        content_parts.append(f"![]({fname})\n")
        print(f"Generated {fname}")

    report.add_section("Cross-Dataset Comparison", "\n".join(content_parts))


# ─── Main ─────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Unified analysis script for parametric search tradeoff experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  uv run python scripts/unified_analysis.py \\
    --model-name "Gemini 3 Pro" \\
    --datasets \\
      "facts_one_hop:logs/foh/analysis.csv:logs/foh/json::logs/entropy/foh.csv" \\
      "popqa:logs/popqa/baseline_run_*_analysis.csv:logs/popqa/json::logs/entropy/popqa.csv" \\
    --output-dir results/gemini_3_pro \\
    --aggregate

Dataset spec format: LABEL:analysis_csv_glob:json_dir[:traces_json][:entropy_csv]
  The analysis_csv_glob field supports glob patterns for multiple per-run CSVs.
  Use empty string to skip optional fields (traces, entropy).
        """,
    )
    parser.add_argument('--model-name', required=True, help='Display name for titles/filenames')
    parser.add_argument(
        '--datasets', required=True, nargs='+',
        help='Colon-delimited dataset specs: LABEL:analysis_csv_glob:json_dir[:traces_json][:entropy_csv]',
    )
    parser.add_argument('--output-dir', required=True, help='Output directory for all results')
    parser.add_argument(
        '--no-search-glob', default='*no_search*run_*.json',
        help='Glob pattern for no-search JSONs in json_dir (default: *no_search*run_*.json)',
    )
    parser.add_argument(
        '--baseline-glob', default='*baseline*run_*.json',
        help='Glob pattern for baseline JSONs in json_dir (default: *baseline*run_*.json)',
    )
    parser.add_argument('--aggregate', action='store_true', help='Pool questions across datasets')
    parser.add_argument(
        '--variant', default='vanilla', choices=['vanilla', 'with_sys_instruct'],
        help='Analysis variant (default: vanilla)',
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Model: {args.model_name}")
    print(f"Variant: {args.variant}")
    print(f"Output: {args.output_dir}")
    print(f"Aggregate: {args.aggregate}")
    print(f"No-search glob: {args.no_search_glob}")
    print(f"Baseline glob: {args.baseline_glob}")

    # Parse and build per-dataset frames
    dataset_frames = []
    dataset_labels = []

    for spec_str in args.datasets:
        spec = parse_dataset_spec(spec_str)
        dataset_labels.append(spec['label'])
        frame = build_dataset_frame(spec, args.no_search_glob, args.baseline_glob, args.variant)
        if frame is not None:
            dataset_frames.append(frame)

    if not dataset_frames:
        print("No data loaded. Exiting.")
        return

    # Concatenate all datasets
    df = pd.concat(dataset_frames, ignore_index=True)
    print(f"\nFinal merged dataset: {len(df)} rows, {df['dataset'].nunique()} dataset(s)")
    print(f"Columns: {df.columns.tolist()}")

    # Build report
    report = ReportBuilder(args.model_name, dataset_labels, len(df))

    # Section 1: Data summary
    summary_parts = []
    for ds_label in sorted(df['dataset'].unique()):
        sub = df[df['dataset'] == ds_label]
        has_entropy = 'entropy' in sub.columns and sub['entropy'].notna().any()
        has_ns = 'no_search_mean_correct' in sub.columns and sub['no_search_mean_correct'].notna().any()
        has_bl = 'baseline_mean_correct' in sub.columns and sub['baseline_mean_correct'].notna().any()
        n_runs = sub['run_id'].nunique() if 'run_id' in sub.columns else 1
        n_problems = sub['problem_id'].nunique() if 'problem_id' in sub.columns else len(sub)
        summary_parts.append(
            f"- **{ds_label}**: {len(sub)} rows ({n_problems} problems x {n_runs} run(s)), "
            f"entropy={'yes' if has_entropy else 'no'}, "
            f"no-search runs={'yes' if has_ns else 'no'}, "
            f"baseline runs={'yes' if has_bl else 'no'}"
        )
    report.add_section("Data Summary", "\n".join(summary_parts))

    # Run analysis sections
    section_epistemic_state(df, args.output_dir, report)
    section_semantic_entropy(df, args.output_dir, report, args.aggregate)
    section_statistical_tests(df, args.output_dir, report)
    section_stability(df, args.output_dir, report)

    if args.aggregate and df['dataset'].nunique() > 1:
        section_cross_dataset(df, args.output_dir, report)

    # Save merged data
    df.to_csv(os.path.join(args.output_dir, 'merged_data.csv'), index=False)
    print(f"\nSaved merged_data.csv ({len(df)} rows)")

    # Save report
    report_path = os.path.join(args.output_dir, 'report.md')
    with open(report_path, 'w') as f:
        f.write(report.build())
    print(f"Saved report to {report_path}")

    print(f"\nAll outputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
