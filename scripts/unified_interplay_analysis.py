"""
unified_interplay_analysis.py

Computes all interplay signatures for MusiQue and MusiQue-Natural datasets,
generates comparison plots (all plots show both datasets side by side), runs
hypothesis tests, and writes a markdown report.

Usage:
    uv run python scripts/unified_interplay_analysis.py \\
        --datasets \\
            musique:results/musique/interplay_stage/interplay_summary.csv \\
            musique-natural:results/musique_natural/interplay_stage/interplay_summary.csv \\
        --example-metrics \\
            musique:results/musique/interplay_stage/example_metrics.csv \\
            musique-natural:results/musique_natural/interplay_stage/example_metrics.csv \\
        --matched-examples \\
            musique:results/musique/interplay_stage/matched_examples_gemini-3-pro-preview.json \\
        --output-dir results/consolidated_analysis \\
        --parallel-hop-filter 2

Metrics computed:
  Original: SCC, SCC_q, POR, SER, CovGap, QBS, SIR_mean, SIR_cv
  Extended: EWOI, SBE, SAE_eeu, SAE_q, Graded-POR (extreme/borderline), P_brittle, IQSRS
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats
def proportions_ztest(counts, nobs):
    """Two-proportion z-test (manual implementation, no statsmodels dependency)."""
    p1, p2 = counts[0] / nobs[0], counts[1] / nobs[1]
    p_pool = sum(counts) / sum(nobs)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
    z = (p1 - p2) / se if se > 0 else 0.0
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return z, p

warnings.filterwarnings("ignore", category=FutureWarning)

# ─── Constants ────────────────────────────────────────────────────────────────

MAX_ENTROPY = math.log2(5)  # log₂(5) ≈ 2.322 bits — maximum entropy for 5-sample clustering
# Entropy is computed in bits (log₂); the 7 possible discrete values are:
# {0, 0.722, 0.971, 1.371, 1.522, 1.922, 2.322}
# corresponding to cluster partitions [5], [4,1], [3,2], [3,1,1], [2,2,1], [2,1,1,1], [1,1,1,1,1]
ENTROPY_LEVELS = [0.0, 0.722, 0.971, 1.371, 1.522, 1.922, 2.322]  # all valid discrete values

VALID_MODELS = {
    "gemini-3-pro-preview",
    "nemotron-3-nano_30b",
    "nemotron-3-nano",
    "qwen3.5_122b",
}

MODEL_DISPLAY = {
    "gemini-3-pro-preview": "Gemini 3 Pro",
    "nemotron-3-nano_30b": "Nemotron Nano",
    "nemotron-3-nano": "Nemotron Nano",
    "qwen3.5_122b": "Qwen 3.5",
}

DATASET_DISPLAY = {
    "musique": "MusiQue",
    "sharechat": "ShareChat",
    "musique-natural": "MusiQue (Natural)",
}

# Canonical (dataset, model) keys in display order
COMBO_ORDER = [
    ("musique", "gemini-3-pro-preview"),
    ("musique", "nemotron-3-nano_30b"),
    ("musique", "qwen3.5_122b"),
    ("sharechat", "gemini-3-pro-preview"),
    ("sharechat", "nemotron-3-nano"),
    ("musique-natural", "gemini-3-pro-preview"),
    ("musique-natural", "nemotron-3-nano_30b"),
    ("musique-natural", "qwen3.5_122b"),
]

COMBO_LABELS = {
    ("musique", "gemini-3-pro-preview"):          "MusiQue\nGemini",
    ("musique", "nemotron-3-nano_30b"):            "MusiQue\nNemotron",
    ("musique", "qwen3.5_122b"):                  "MusiQue\nQwen",
    ("sharechat", "gemini-3-pro-preview"):         "ShareChat\nGemini",
    ("sharechat", "nemotron-3-nano"):              "ShareChat\nNemotron",
    ("musique-natural", "gemini-3-pro-preview"):   "MusiQue-Nat\nGemini",
    ("musique-natural", "nemotron-3-nano_30b"):    "MusiQue-Nat\nNemotron",
    ("musique-natural", "qwen3.5_122b"):           "MusiQue-Nat\nQwen",
}

COMBO_COLORS = {
    ("musique", "gemini-3-pro-preview"):          "#2196F3",
    ("musique", "nemotron-3-nano_30b"):            "#FF9800",
    ("musique", "qwen3.5_122b"):                  "#4CAF50",
    ("sharechat", "gemini-3-pro-preview"):         "#F44336",
    ("sharechat", "nemotron-3-nano"):              "#9C27B0",
    ("musique-natural", "gemini-3-pro-preview"):   "#1565C0",
    ("musique-natural", "nemotron-3-nano_30b"):    "#E65100",
    ("musique-natural", "qwen3.5_122b"):           "#2E7D32",
}

QUADRANT_COLORS = {
    "E":  "#4CAF50",  # Effective — green
    "CP": "#2196F3",  # Correct Parametric — blue
    "PR": "#FF9800",  # Parametric Redundant — orange
    "M":  "#F44336",  # Missed — red
}

sns.set_theme(style="whitegrid", font_scale=1.1)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--datasets",
        nargs="+",
        metavar="NAME:CSV",
        help=(
            "One or more datasets to include, each as name:path/to/csv. "
            "Recognised names: musique, sharechat, musique-natural. "
            "Example: --datasets musique:results/musique_parametric/interplay_analysis/interplay_summary.csv "
            "musique-natural:results/musique-natural/interplay_analysis/interplay_summary.csv"
        ),
    )
    p.add_argument("--output-dir", default="results/unified_interplay")
    p.add_argument(
        "--example-metrics",
        nargs="+",
        metavar="NAME:CSV",
        help=(
            "One or more example-level CSVs as name:path/to/example_metrics.csv. "
            "Required for redundancy_overlap.png and sequential_redundancy_distribution.png."
        ),
    )
    p.add_argument(
        "--matched-examples",
        nargs="+",
        metavar="NAME:JSON",
        help=(
            "One or more matched_examples JSON files as name:path. "
            "Required for multi_hop_query_attribution.png."
        ),
    )
    p.add_argument(
        "--mq-certainty-mode",
        choices=["joint", "entropy_only"],
        default="joint",
        help="joint: entropy=0 AND no-search correct. entropy_only: entropy=0 only.",
    )
    p.add_argument(
        "--parallel-hop-filter", type=int, default=None,
        metavar="N",
        help=(
            "Also generate a parallel set of plots filtered to N-hop questions only, "
            "for all musique-family datasets. Auto-detected from musique-natural if omitted."
        ),
    )
    return p.parse_args()


def resolve_datasets(args: argparse.Namespace) -> dict[str, str]:
    """Return {dataset_name: csv_path} from --datasets or legacy individual flags."""
    resolved: dict[str, str] = {}

    if args.datasets:
        for token in args.datasets:
            if ":" not in token:
                raise ValueError(f"--datasets entries must be name:path, got: {token!r}")
            name, path = token.split(":", 1)
            resolved[name] = path
    else:
        # Fall back to legacy flags with hardcoded defaults
        resolved["musique"] = args.musique_csv or "results/musique_parametric/interplay_analysis/interplay_summary.csv"
        resolved["sharechat"] = args.sharechat_csv or "results/sharechat/atomic_fact_confidence.csv"

    return resolved


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_musique(path: str, certainty_mode: str = "joint") -> pd.DataFrame:
    """Load MusiQue interplay_summary.csv → unified EEU frame."""
    df = pd.read_csv(path)
    df["model"] = df["model"].str.replace(":", "_", regex=False)
    df = df[df["model"].isin(VALID_MODELS)].copy()

    df = df.rename(columns={
        "example_id": "question_id",
        "semantic_entropy": "entropy",
        "searched": "search_attributed",
        "queries_assigned_count": "queries_assigned",
    })

    df["dataset"] = "musique"
    df["search_attributed"] = df["search_attributed"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool)

    if certainty_mode == "joint":
        df["certain"] = (df["entropy"] == 0.0) & (df["parametric_certain"].astype(str) == "True")
    else:
        df["certain"] = df["entropy"] == 0.0

    df["certainty_score"] = (1.0 - df["entropy"] / MAX_ENTROPY).clip(lower=0.0)

    # Question-level has_search: all MusiQue questions have search by design
    df["has_search"] = True

    return df[[
        "dataset", "model", "question_id", "hop_index", "num_hops",
        "entropy", "certainty_score", "certain", "search_attributed",
        "queries_assigned", "has_search", "aggregate_correct",
    ]]


def load_sharechat(path: str) -> pd.DataFrame:
    """Load ShareChat atomic_fact_confidence.csv → unified EEU frame."""
    df = pd.read_csv(path)
    # Drop corrupted rows
    df = df[df["model"].isin(VALID_MODELS)].copy()

    df = df.rename(columns={
        "problem": "question_id",
        "attributed_to_search": "search_attributed",
    })

    df["dataset"] = "sharechat"
    df["search_attributed"] = df["search_attributed"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool)
    df["entropy"] = df["entropy"].astype(float)
    df["search_calls"] = df["search_calls"].astype(int)

    # certainty = entropy == 0 (post-attribution; see caveat in metrics doc)
    df["certain"] = df["entropy"] == 0.0
    df["certainty_score"] = (1.0 - df["entropy"] / MAX_ENTROPY).clip(lower=0.0)

    # has_search at question level
    qs_map = df.groupby("question_id")["search_attributed"].any()
    df["has_search"] = df["question_id"].map(qs_map)

    df["hop_index"] = np.nan
    df["num_hops"] = np.nan
    df["queries_assigned"] = np.nan  # fact-level assignment not available
    df["aggregate_correct"] = np.nan

    return df[[
        "dataset", "model", "question_id", "hop_index", "num_hops",
        "entropy", "certainty_score", "certain", "search_attributed",
        "queries_assigned", "has_search", "aggregate_correct",
        "search_calls",
    ]]


def load_musique_natural(path: str, certainty_mode: str = "joint") -> pd.DataFrame:
    """Load musique-natural interplay_summary.csv → unified EEU frame.

    Produced by analyze_parametric_search_interplay.py run on natural-language traces
    staged via stage_musique_natural_interplay.py.
    """
    df = load_musique(path, certainty_mode=certainty_mode)
    df["dataset"] = "musique-natural"
    # Recompute has_search from data — baseline agent doesn't search on every question
    qs_map = df.groupby("question_id")["search_attributed"].any()
    df["has_search"] = df["question_id"].map(qs_map)
    return df


DATASET_LOADERS = {
    "musique": load_musique,
    "sharechat": load_sharechat,
    "musique-natural": load_musique_natural,
}


def load_example_metrics(path: str) -> pd.DataFrame:
    """Load example_metrics.csv produced by analyze_parametric_search_interplay.py."""
    return pd.read_csv(path)


def load_matched_examples(path: str) -> list:
    """Load matched_examples_<model>.json checkpoint for multi-hop attribution data."""
    import json
    with open(path) as f:
        return json.load(f)


def build_unified_frame(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True, sort=False)


def auto_detect_hop_filter(df: pd.DataFrame) -> int | None:
    """Return N if musique-natural is present and contains only N-hop questions, else None."""
    if "musique-natural" not in df["dataset"].unique():
        return None
    hops = df.loc[df["dataset"] == "musique-natural", "num_hops"].dropna().unique()
    return int(hops[0]) if len(hops) == 1 else None


def build_hop_filtered_frame(df: pd.DataFrame, hop_n: int) -> pd.DataFrame:
    """Return a copy of df where musique-family datasets are filtered to num_hops == hop_n
    and relabelled as '{dataset}-{hop_n}hop'. Non-musique datasets are kept unfiltered."""
    frames = []
    for dataset in df["dataset"].unique():
        subset = df[df["dataset"] == dataset].copy()
        if dataset.startswith("musique"):
            subset = subset[subset["num_hops"] == hop_n]
            if len(subset) == 0:
                continue
            subset["dataset"] = f"{dataset}-{hop_n}hop"
        frames.append(subset)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


# Muted palette for hop-filtered variants (paired with their originals)
_HOP_COLOR_MUTE = {
    "#2196F3": "#90CAF9",  # musique/gemini → light blue
    "#FF9800": "#FFCC80",  # musique/nemotron → light orange
    "#4CAF50": "#A5D6A7",  # musique/qwen → light green
    "#1565C0": "#5E92F3",  # musique-natural/gemini → lighter blue
    "#E65100": "#FF8A50",  # musique-natural/nemotron → lighter orange
    "#2E7D32": "#66BB6A",  # musique-natural/qwen → lighter green
}


def register_hop_filtered_combos(hop_n: int) -> None:
    """Dynamically register display entries for *-{hop_n}hop dataset variants."""
    base_datasets = [k for k in DATASET_DISPLAY if not k.endswith("hop")]
    for base in base_datasets:
        if not base.startswith("musique"):
            continue
        filtered_name = f"{base}-{hop_n}hop"
        DATASET_DISPLAY[filtered_name] = f"{DATASET_DISPLAY[base]} ({hop_n}-hop)"
        for (ds, model), label in list(COMBO_LABELS.items()):
            if ds == base:
                filtered_label = label.replace("\n", f" ({hop_n}h)\n", 1)
                COMBO_LABELS[(filtered_name, model)] = filtered_label
                base_color = COMBO_COLORS.get((ds, model), "#888888")
                COMBO_COLORS[(filtered_name, model)] = _HOP_COLOR_MUTE.get(base_color, base_color)
                if (filtered_name, model) not in COMBO_ORDER:
                    COMBO_ORDER.append((filtered_name, model))


# ─── Quadrant assignment ───────────────────────────────────────────────────────

def assign_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'quadrant' column: E, PR, M, CP.

    Binary classification framing — each EEU is a search decision:
        ground truth positive  = uncertain (entropy > 0)
        ground truth negative  = certain   (entropy = 0)
        predicted positive     = searched
        predicted negative     = not searched

        E  = TP  (uncertain  & searched)      — correct: search was needed
        M  = FN  (uncertain  & not searched)  — error:   gap left open
        PR = FP  (certain    & searched)      — error:   wasted search
        CP = TN  (certain    & not searched)  — correct: correctly skipped

    Standard metrics derived from these quadrants:
        Precision  = TP / (TP + FP)  = E  / (E  + PR)   [formerly SER]
        FDR        = FP / (TP + FP)  = PR / (E  + PR)   [formerly POR; = 1 − Precision]
        Recall     = TP / (TP + FN)  = E  / (E  + M )   [new]
        Specificity= TN / (TN + FP)  = CP / (CP + PR)   [new]
        F1         = 2 × Precision × Recall / (P + R)   [new]
    """
    df = df.copy()
    conditions = [
        (~df["certain"]) & df["search_attributed"],
        (~df["certain"]) & (~df["search_attributed"]),
        df["certain"] & df["search_attributed"],
        df["certain"] & (~df["search_attributed"]),
    ]
    choices = ["E", "M", "PR", "CP"]
    df["quadrant"] = np.select(conditions, choices, default="?")
    return df


# ─── Signatures ───────────────────────────────────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(values: list[float], stat_fn, n_boot: int = 2000, ci: float = 0.95) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(42)
    boots = [stat_fn(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    lo = (1 - ci) / 2
    return (float(np.quantile(boots, lo)), float(np.quantile(boots, 1 - lo)))


def compute_signatures(df: pd.DataFrame) -> dict:
    """Compute all signatures for a single (dataset, model) slice."""
    n = len(df)
    if n == 0:
        return {}

    searched = df["search_attributed"]
    certain = df["certain"]

    n_E  = int((~certain & searched).sum())
    n_PR = int((certain & searched).sum())
    n_M  = int((~certain & ~searched).sum())
    n_CP = int((certain & ~searched).sum())
    assert n_E + n_PR + n_M + n_CP == n

    # ── Original signatures ──────────────────────────────────────────────────
    # SCC (EEU level)
    rho_eeu, p_scc = sp_stats.spearmanr(df["entropy"], df["search_attributed"].astype(int))

    # SCC_q (question level, ShareChat only — MusiQue all has_search=True)
    q_frame = df.groupby("question_id").agg(
        mean_entropy=("entropy", "mean"),
        has_search=("has_search", "first"),
    ).reset_index()
    if q_frame["has_search"].nunique() > 1:
        rho_q, p_scc_q = sp_stats.spearmanr(q_frame["mean_entropy"], q_frame["has_search"].astype(int))
    else:
        rho_q, p_scc_q = np.nan, np.nan

    # ── Binary classifier metrics ────────────────────────────────────────────
    n_searched = n_E + n_PR
    n_uncertain = n_E + n_M

    # Precision = TP/(TP+FP)   (formerly SER)
    precision = n_E / n_searched if n_searched > 0 else np.nan
    # FDR = FP/(TP+FP) = 1 − Precision  (formerly POR)
    fdr = n_PR / n_searched if n_searched > 0 else np.nan
    fdr_ci = wilson_ci(n_PR, n_searched)
    # Recall = TP/(TP+FN)
    recall = n_E / n_uncertain if n_uncertain > 0 else np.nan
    recall_ci = wilson_ci(n_E, n_uncertain)
    # Specificity = TN/(TN+FP)
    specificity = n_CP / (n_CP + n_PR) if (n_CP + n_PR) > 0 else np.nan
    # F1
    if precision is not np.nan and recall is not np.nan and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = np.nan
    # Bootstrap CI for F1
    def _f1_stat(v):
        tp = (np.array(v) == 1).sum()
        fp = (np.array(v) == 2).sum()
        fn = (np.array(v) == 3).sum()
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0
        return 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0
    # Encode: E→1, PR→2, M→3, CP→0
    quad_codes = df["quadrant"].map({"E": 1, "PR": 2, "M": 3, "CP": 0}).fillna(0).tolist()
    f1_ci = bootstrap_ci(quad_codes, _f1_stat)

    # AUROC: discriminability of entropy as a predictor of search_attributed
    pos_ent = df.loc[searched, "entropy"].values
    neg_ent = df.loc[~searched, "entropy"].values
    if len(pos_ent) > 0 and len(neg_ent) > 0:
        u_stat, _ = sp_stats.mannwhitneyu(pos_ent, neg_ent, alternative="greater")
        auroc = u_stat / (len(pos_ent) * len(neg_ent))
    else:
        auroc = np.nan

    # CovGap = FN / N  (miss rate normalised by total, not by positive class)
    cov_gap = n_M / n
    cov_gap_ci = wilson_ci(n_M, n)

    # QBS
    correct = n_E + n_CP
    incorrect = n_PR + n_M
    if incorrect > 0:
        qbs = math.log(correct / incorrect) if correct > 0 else -np.inf
    else:
        qbs = np.inf
    qbs_vals = df["quadrant"].map({"E": 1, "CP": 1, "PR": -1, "M": -1}).fillna(0).tolist()
    qbs_ci = bootstrap_ci(qbs_vals, lambda v: math.log(max(1e-9, (v == 1).mean() / max(1e-9, (v == -1).mean()))))

    # SIR (question level)
    dataset = df["dataset"].iloc[0]
    if "search_calls" in df.columns and not dataset.startswith("musique"):
        # Use search_calls column (one value per question, repeated over facts)
        q_search = df.groupby("question_id")["search_calls"].first()
    else:
        q_search = df.groupby("question_id")["queries_assigned"].sum()
    sir_mean = float(q_search.mean())
    sir_cv = float(q_search.std() / q_search.mean()) if q_search.mean() > 0 else np.nan

    # ── Extended signatures ──────────────────────────────────────────────────
    # EWOI: mean certainty score of searched EEUs
    searched_mask = df["search_attributed"]
    ewoi = float(df.loc[searched_mask, "certainty_score"].mean()) if searched_mask.any() else np.nan

    # SBE: E-EEUs (entropy > 0 AND searched) per total queries
    total_queries = float(q_search.sum())
    n_E_entropy = int((searched_mask & (df["entropy"] > 0)).sum())
    sbe = n_E_entropy / total_queries if total_queries > 0 else np.nan

    # SAE (EEU level)
    if searched_mask.any() and (~searched_mask).any():
        sae_eeu = float(df.loc[searched_mask, "entropy"].mean() - df.loc[~searched_mask, "entropy"].mean())
    else:
        sae_eeu = np.nan

    # SAE (question level)
    if q_frame["has_search"].nunique() > 1:
        sae_q = float(
            q_frame.loc[q_frame["has_search"], "mean_entropy"].mean()
            - q_frame.loc[~q_frame["has_search"], "mean_entropy"].mean()
        )
    else:
        sae_q = np.nan

    # Graded POR — bins aligned to the 7 discrete entropy levels produced by 5-sample clustering.
    # Entropy is in bits: {0, 0.722, 0.971, 1.371, 1.522, 1.922, 2.322}
    # The (0, 0.5] bin used previously was structurally empty (minimum non-zero step = 0.722).
    #   por_certain:    H = 0          → [5] partition, 5/5 samples agree
    #   por_near:       H = 0.722      → [4,1] partition, 4/5 agree, 1 outlier
    #   por_split:      H ∈ {0.971, 1.371, 1.522}  → [3,2], [3,1,1], [2,2,1] — partial agreement
    #   por_uncertain:  H ∈ {1.922, 2.322}          → [2,1,1,1], [1,1,1,1,1] — near-random
    por_extreme    = float((searched_mask & (df["entropy"] == 0.0)).sum() / n_searched) if n_searched > 0 else np.nan
    por_near       = float((searched_mask & (df["entropy"].round(3) == 0.722)).sum() / n_searched) if n_searched > 0 else np.nan
    por_split      = float((searched_mask & (df["entropy"] > 0.722) & (df["entropy"] < 1.9)).sum() / n_searched) if n_searched > 0 else np.nan
    por_uncertain  = float((searched_mask & (df["entropy"] >= 1.9)).sum() / n_searched) if n_searched > 0 else np.nan
    # Legacy alias for backward compat with report / graded_por plot
    por_moderate   = por_near
    por_borderline = float((searched_mask & (df["entropy"] > 0.722)).sum() / n_searched) if n_searched > 0 else np.nan

    # P_brittle: entropy=0 AND searched AND NOT joint-certain (MusiQue only)
    if "parametric_certain" in df.columns or dataset == "musique":
        # Requires re-computing with entropy_only to get the gap
        n_pr_extreme = int((searched_mask & (df["entropy"] == 0.0)).sum())
        p_brittle = (n_pr_extreme - n_PR) / n_searched if n_searched > 0 else np.nan
    else:
        p_brittle = np.nan

    # IQSRS: within-question routing score (MusiQue 3+ hop only)
    iqsrs, p_iqsrs = compute_iqsrs(df) if dataset.startswith("musique") else (np.nan, np.nan)

    return {
        # Quadrant counts and fractions
        "n_E": n_E, "n_PR": n_PR, "n_M": n_M, "n_CP": n_CP, "n_total": n,
        "frac_E": n_E / n, "frac_PR": n_PR / n, "frac_M": n_M / n, "frac_CP": n_CP / n,
        # ── Standard binary classifier metrics ──
        "precision": precision,                                                  # TP/(TP+FP) — of searches, fraction on uncertain EEUs
        "fdr": fdr, "fdr_ci_lo": fdr_ci[0], "fdr_ci_hi": fdr_ci[1],            # FP/(TP+FP) = 1−precision (formerly POR)
        "recall": recall, "recall_ci_lo": recall_ci[0], "recall_ci_hi": recall_ci[1],  # TP/(TP+FN)
        "specificity": specificity,                                              # TN/(TN+FP)
        "f1": f1, "f1_ci_lo": f1_ci[0], "f1_ci_hi": f1_ci[1],
        "auroc": auroc,                                                          # AUC(entropy → search_attributed)
        # Backward-compat aliases (deprecated — prefer precision/fdr above)
        "por": fdr, "por_ci_lo": fdr_ci[0], "por_ci_hi": fdr_ci[1],
        "ser": precision,
        # ── Calibration ──
        "scc": rho_eeu, "p_scc": p_scc,
        "scc_q": rho_q, "p_scc_q": p_scc_q,
        # ── Coverage ──
        "cov_gap": cov_gap, "cov_gap_ci_lo": cov_gap_ci[0], "cov_gap_ci_hi": cov_gap_ci[1],
        # ── Log-odds accuracy (formerly QBS) ──
        "log_odds_acc": qbs, "log_odds_acc_ci_lo": qbs_ci[0], "log_odds_acc_ci_hi": qbs_ci[1],
        "qbs": qbs, "qbs_ci_lo": qbs_ci[0], "qbs_ci_hi": qbs_ci[1],  # alias
        # ── Volume ──
        "sir_mean": sir_mean, "sir_cv": sir_cv,
        # ── Magnitude-aware / extended signatures ──
        "ewoi": ewoi,
        "sbe": sbe,
        "sae_eeu": sae_eeu,
        "sae_q": sae_q,
        "por_extreme": por_extreme,
        "por_moderate": por_moderate,
        "por_borderline": por_borderline,
        "p_brittle": p_brittle,
        "iqsrs": iqsrs, "p_iqsrs": p_iqsrs,
        "n_E_entropy": n_E_entropy, "total_queries": total_queries,
    }


def compute_iqsrs(df: pd.DataFrame) -> tuple[float, float]:
    """Within-question Spearman ρ between hop entropy and queries_assigned.
    Only for 3+ hop MusiQue examples with non-NaN queries_assigned."""
    valid = df[df["num_hops"].fillna(0) >= 3].dropna(subset=["queries_assigned"])
    rhos = []
    for _, grp in valid.groupby("question_id"):
        if len(grp) < 3:
            continue
        rho, _ = sp_stats.spearmanr(grp["entropy"], grp["queries_assigned"])
        if not np.isnan(rho):
            rhos.append(rho)
    if len(rhos) < 10:
        return np.nan, np.nan
    t_stat, p_val = sp_stats.ttest_1samp(rhos, 0)
    return float(np.mean(rhos)), float(p_val)


def run_all_signatures(df: pd.DataFrame) -> pd.DataFrame:
    """Compute signatures for every (dataset, model) combination."""
    records = []
    for (dataset, model), grp in df.groupby(["dataset", "model"]):
        sig = compute_signatures(grp)
        sig["dataset"] = dataset
        sig["model"] = model
        sig["label"] = COMBO_LABELS.get((dataset, model), f"{dataset}/{model}")
        records.append(sig)
    return pd.DataFrame(records)


# ─── Statistical tests ────────────────────────────────────────────────────────

def bh_correct(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg FDR correction. Returns adjusted p-values."""
    n = len(p_values)
    if n == 0:
        return []
    order = np.argsort(p_values)
    adjusted = np.array(p_values, dtype=float)
    for i, idx in enumerate(order):
        adjusted[idx] = min(1.0, p_values[idx] * n / (i + 1))
    # Enforce monotonicity
    for i in range(len(order) - 2, -1, -1):
        adjusted[order[i]] = min(adjusted[order[i]], adjusted[order[i + 1]])
    return adjusted.tolist()


def run_statistical_tests(df: pd.DataFrame, sigs: pd.DataFrame) -> pd.DataFrame:
    """Run all hypothesis tests, grouped into three BH families."""
    tests = []

    def get_slice(dataset, model):
        return df[(df["dataset"] == dataset) & (df["model"] == model)]

    def get_sig(dataset, model, key):
        row = sigs[(sigs["dataset"] == dataset) & (sigs["model"] == model)]
        return float(row[key].iloc[0]) if len(row) > 0 else np.nan

    mq_models = ["gemini-3-pro-preview", "nemotron-3-nano_30b", "qwen3.5_122b"]
    sc_models = ["gemini-3-pro-preview", "nemotron-3-nano"]

    present_datasets = set(df["dataset"].unique())
    present_combos = set(zip(df["dataset"], df["model"]))

    def has_slice(dataset, model):
        return (dataset, model) in present_combos

    # ── Family 1: Cross-dataset (Gemini only) ──────────────────────────────
    def fisher_z_test(r1, n1, r2, n2):
        """Two-sided Fisher z-transform test for difference of Spearman ρ."""
        if n1 <= 3 or n2 <= 3 or np.isnan(r1) or np.isnan(r2):
            return np.nan, np.nan, r1 - r2 if not (np.isnan(r1) or np.isnan(r2)) else np.nan
        z1 = np.arctanh(np.clip(r1, -0.9999, 0.9999))
        z2 = np.arctanh(np.clip(r2, -0.9999, 0.9999))
        se = math.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
        z = (z1 - z2) / se
        p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
        return z, p, r1 - r2

    mq_gem = get_slice("musique", "gemini-3-pro-preview")
    sc_gem = get_slice("sharechat", "gemini-3-pro-preview")

    # Pre-compute musique Gemini EWOI array — used in both Family 1 and HN tests
    ewoi_mq = df.loc[(df["dataset"] == "musique") & (df["model"] == "gemini-3-pro-preview") & df["search_attributed"], "certainty_score"].values

    # ── Family 1 tests only run when both musique and sharechat are present ──
    _run_family1 = has_slice("musique", "gemini-3-pro-preview") and has_slice("sharechat", "gemini-3-pro-preview")

    # H1: SCC difference (EEU level)
    if _run_family1:
        z, p, eff = fisher_z_test(
            get_sig("musique", "gemini-3-pro-preview", "scc"), len(mq_gem),
            get_sig("sharechat", "gemini-3-pro-preview", "scc"), len(sc_gem),
        )
        tests.append({"id": "H1", "family": 1, "hypothesis": "SCC_MusiQue ≠ SCC_ShareChat (EEU)",
                      "statistic": z, "p_raw": p, "effect": eff, "note": "Δρ (MQ−SC)"})

    if _run_family1:
        # H2: POR difference
        n_pr_mq = int(get_sig("musique", "gemini-3-pro-preview", "n_PR"))
        n_s_mq  = int(get_sig("musique", "gemini-3-pro-preview", "n_E") + n_pr_mq)
        n_pr_sc = int(get_sig("sharechat", "gemini-3-pro-preview", "n_PR"))
        n_s_sc  = int(get_sig("sharechat", "gemini-3-pro-preview", "n_E") + n_pr_sc)
        por_mq = get_sig("musique", "gemini-3-pro-preview", "por")
        por_sc = get_sig("sharechat", "gemini-3-pro-preview", "por")
        z2, p2 = proportions_ztest([n_pr_mq, n_pr_sc], [n_s_mq, n_s_sc])
        cohens_h = 2 * (math.asin(math.sqrt(por_mq)) - math.asin(math.sqrt(por_sc)))
        tests.append({"id": "H2", "family": 1, "hypothesis": "POR_MusiQue ≠ POR_ShareChat",
                      "statistic": z2, "p_raw": p2, "effect": cohens_h, "note": "Cohen's h"})

        # H3: CovGap difference
        n_m_mq = int(get_sig("musique", "gemini-3-pro-preview", "n_M"))
        n_m_sc = int(get_sig("sharechat", "gemini-3-pro-preview", "n_M"))
        n_tot_mq = int(get_sig("musique", "gemini-3-pro-preview", "n_total"))
        n_tot_sc = int(get_sig("sharechat", "gemini-3-pro-preview", "n_total"))
        cg_mq = get_sig("musique", "gemini-3-pro-preview", "cov_gap")
        cg_sc = get_sig("sharechat", "gemini-3-pro-preview", "cov_gap")
        z3, p3 = proportions_ztest([n_m_mq, n_m_sc], [n_tot_mq, n_tot_sc])
        h3 = 2 * (math.asin(math.sqrt(cg_mq)) - math.asin(math.sqrt(cg_sc)))
        tests.append({"id": "H3", "family": 1, "hypothesis": "CovGap_MusiQue ≠ CovGap_ShareChat",
                      "statistic": z3, "p_raw": p3, "effect": h3, "note": "Cohen's h"})

        # H4: QBS (via CI non-overlap)
        qbs_mq_lo = get_sig("musique", "gemini-3-pro-preview", "qbs_ci_lo")
        qbs_mq_hi = get_sig("musique", "gemini-3-pro-preview", "qbs_ci_hi")
        qbs_sc_lo = get_sig("sharechat", "gemini-3-pro-preview", "qbs_ci_lo")
        qbs_sc_hi = get_sig("sharechat", "gemini-3-pro-preview", "qbs_ci_hi")
        overlap = max(0, min(qbs_mq_hi, qbs_sc_hi) - max(qbs_mq_lo, qbs_sc_lo))
        delta_qbs = get_sig("musique", "gemini-3-pro-preview", "qbs") - get_sig("sharechat", "gemini-3-pro-preview", "qbs")
        tests.append({"id": "H4", "family": 1, "hypothesis": "QBS_MusiQue ≠ QBS_ShareChat",
                      "statistic": float(overlap == 0), "p_raw": 0.05 if overlap == 0 else 0.5,
                      "effect": delta_qbs, "note": "ΔQBS; sig if CI non-overlap"})

        # H5: EWOI cross-dataset (bootstrap two-sample)
        ewoi_sc = df.loc[(df["dataset"] == "sharechat") & (df["model"] == "gemini-3-pro-preview") & df["search_attributed"], "certainty_score"].values
        if len(ewoi_mq) > 0 and len(ewoi_sc) > 0:
            stat_h5, p_h5 = sp_stats.mannwhitneyu(ewoi_mq, ewoi_sc, alternative="two-sided")
            eff_h5 = ewoi_mq.mean() - ewoi_sc.mean()
            tests.append({"id": "H5", "family": 1, "hypothesis": "EWOI_MusiQue_Gemini ≠ EWOI_ShareChat_Gemini",
                          "statistic": stat_h5, "p_raw": p_h5, "effect": eff_h5, "note": "Δ mean certainty score; MW-U"})

    # H6: SAE EEU level — run for any dataset/model combo present
    def sae_mannwhitney(ds, model):
        sl = get_slice(ds, model)
        s = sl.loc[sl["search_attributed"], "entropy"].values
        ns = sl.loc[~sl["search_attributed"], "entropy"].values
        if len(s) < 2 or len(ns) < 2:
            return np.nan, np.nan
        stat, p = sp_stats.mannwhitneyu(s, ns, alternative="greater")
        return stat, p

    for ds in present_datasets:
        if has_slice(ds, "gemini-3-pro-preview"):
            stat_h6, p_h6 = sae_mannwhitney(ds, "gemini-3-pro-preview")
            ds_label = DATASET_DISPLAY.get(ds, ds)
            tests.append({"id": f"H6_{ds[:6]}", "family": 1,
                          "hypothesis": f"SAE_{ds_label}_Gemini > 0 (searched hops more uncertain)",
                          "statistic": stat_h6, "p_raw": p_h6,
                          "effect": get_sig(ds, "gemini-3-pro-preview", "sae_eeu"), "note": "MW-U one-sided"})

    # HN: musique vs musique-natural phrasing effect (Gemini)
    if has_slice("musique", "gemini-3-pro-preview") and has_slice("musique-natural", "gemini-3-pro-preview"):
        mq_nat_gem = get_slice("musique-natural", "gemini-3-pro-preview")
        z_n, p_n, eff_n = fisher_z_test(
            get_sig("musique", "gemini-3-pro-preview", "scc"), len(mq_gem),
            get_sig("musique-natural", "gemini-3-pro-preview", "scc"), len(mq_nat_gem),
        )
        tests.append({"id": "HN1", "family": 1,
                      "hypothesis": "SCC: MusiQue-benchmark ≠ MusiQue-natural (Gemini)",
                      "statistic": z_n, "p_raw": p_n, "effect": eff_n, "note": "Δρ (MQ−MQ-nat)"})

        ewoi_mq_nat = df.loc[(df["dataset"] == "musique-natural") & (df["model"] == "gemini-3-pro-preview") & df["search_attributed"], "certainty_score"].values
        if len(ewoi_mq) > 0 and len(ewoi_mq_nat) > 0:
            stat_hn2, p_hn2 = sp_stats.mannwhitneyu(ewoi_mq, ewoi_mq_nat, alternative="two-sided")
            tests.append({"id": "HN2", "family": 1,
                          "hypothesis": "EWOI: MusiQue-benchmark ≠ MusiQue-natural (Gemini)",
                          "statistic": stat_hn2, "p_raw": p_hn2,
                          "effect": ewoi_mq.mean() - ewoi_mq_nat.mean(), "note": "Δ mean certainty; MW-U"})

    # ── Family 2: Within-MusiQue ────────────────────────────────────────────
    # W1: Entropy distributions across models
    ent_groups = [
        get_slice("musique", m)["entropy"].values
        for m in mq_models if len(get_slice("musique", m)) > 0
    ]
    if len(ent_groups) >= 2:
        stat_w1, p_w1 = sp_stats.kruskal(*ent_groups)
        tests.append({"id": "W1", "family": 2, "hypothesis": "Entropy distributions differ across MusiQue models",
                      "statistic": stat_w1, "p_raw": p_w1, "effect": np.nan, "note": "Kruskal-Wallis"})

    # W5: EWOI differs across MusiQue models (certainty scores of searched hops)
    ewoi_groups = [
        df.loc[(df["dataset"] == "musique") & (df["model"] == m) & df["search_attributed"], "certainty_score"].values
        for m in mq_models
    ]
    ewoi_groups = [g for g in ewoi_groups if len(g) > 0]
    if len(ewoi_groups) >= 2:
        stat_w5, p_w5 = sp_stats.kruskal(*ewoi_groups)
        tests.append({"id": "W5", "family": 2, "hypothesis": "EWOI differs across MusiQue models",
                      "statistic": stat_w5, "p_raw": p_w5, "effect": np.nan, "note": "KW on certainty_score of searched"})

    # W6: EWOI increases with hop count
    mq_all = df[df["dataset"] == "musique"].copy()
    hop_ewoi = (
        mq_all[mq_all["search_attributed"]]
        .groupby(["model", "num_hops"])["certainty_score"]
        .mean()
        .reset_index()
    )
    if len(hop_ewoi) >= 3:
        rho_w6, p_w6 = sp_stats.spearmanr(hop_ewoi["num_hops"], hop_ewoi["certainty_score"])
        tests.append({"id": "W6", "family": 2, "hypothesis": "EWOI increases with hop count",
                      "statistic": rho_w6, "p_raw": p_w6, "effect": rho_w6, "note": "Spearman ρ"})

    # W7: IQSRS > 0
    for model in mq_models:
        sl = get_slice("musique", model)
        iqsrs_val = get_sig("musique", model, "iqsrs")
        p_iq = get_sig("musique", model, "p_iqsrs")
        if not np.isnan(iqsrs_val):
            tests.append({"id": f"W7_{model[:4]}", "family": 2,
                          "hypothesis": f"IQSRS > 0 for {MODEL_DISPLAY.get(model, model)}",
                          "statistic": iqsrs_val, "p_raw": p_iq / 2,  # one-sided
                          "effect": iqsrs_val, "note": "t-test vs 0, one-sided"})

    # W8: SAE_Nemotron > SAE_Gemini (entropy of searched hops differs)
    s_gem = df.loc[(df["dataset"] == "musique") & (df["model"] == "gemini-3-pro-preview") & df["search_attributed"], "entropy"].values
    s_nem = df.loc[(df["dataset"] == "musique") & (df["model"] == "nemotron-3-nano_30b") & df["search_attributed"], "entropy"].values
    if len(s_gem) > 0 and len(s_nem) > 0:
        stat_w8, p_w8 = sp_stats.mannwhitneyu(s_nem, s_gem, alternative="greater")
        eff_w8 = s_nem.mean() - s_gem.mean()
        tests.append({"id": "W8", "family": 2, "hypothesis": "SAE: Nemotron searches more uncertain hops than Gemini",
                      "statistic": stat_w8, "p_raw": p_w8, "effect": eff_w8, "note": "MW-U one-sided; Δ mean entropy"})

    # ── Family 3: Within-ShareChat ──────────────────────────────────────────
    if "sharechat" not in present_datasets:
        pass  # skip Family 3 entirely
    # S1: SCC_q > 0 (already in sigs)
    for model in sc_models if "sharechat" in present_datasets else []:
        rho_q_val = get_sig("sharechat", model, "scc_q")
        p_q_val = get_sig("sharechat", model, "p_scc_q")
        if not np.isnan(rho_q_val) and not np.isnan(p_q_val):
            tests.append({"id": f"S1_{model[:4]}", "family": 3,
                          "hypothesis": f"SCC_q > 0 for ShareChat {MODEL_DISPLAY.get(model, model)}",
                          "statistic": rho_q_val, "p_raw": p_q_val / 2,
                          "effect": rho_q_val, "note": "Spearman ρ, one-sided"})

    # S6: SAE_q > 0 (question-level)
    for model in sc_models:
        sl = get_slice("sharechat", model)
        q_frame = sl.groupby("question_id").agg(
            mean_entropy=("entropy", "mean"), has_search=("has_search", "first")
        )
        if q_frame["has_search"].nunique() > 1:
            s_vals = q_frame.loc[q_frame["has_search"], "mean_entropy"].values
            ns_vals = q_frame.loc[~q_frame["has_search"], "mean_entropy"].values
            stat_s6, p_s6 = sp_stats.mannwhitneyu(s_vals, ns_vals, alternative="greater")
            eff_s6 = s_vals.mean() - ns_vals.mean()
            tests.append({"id": f"S6_{model[:4]}", "family": 3,
                          "hypothesis": f"SAE_q > 0 for ShareChat {MODEL_DISPLAY.get(model, model)}",
                          "statistic": stat_s6, "p_raw": p_s6, "effect": eff_s6,
                          "note": "MW-U one-sided; Δ mean question-level entropy"})

    # S7: Search call count ~ question entropy
    for model in sc_models if "sharechat" in present_datasets else []:
        sl = get_slice("sharechat", model)
        q_frame = sl.groupby("question_id").agg(
            mean_entropy=("entropy", "mean"), search_calls=("search_calls", "first")
        )
        rho_s7, p_s7 = sp_stats.spearmanr(q_frame["mean_entropy"], q_frame["search_calls"])
        tests.append({"id": f"S7_{model[:4]}", "family": 3,
                      "hypothesis": f"Search count ~ question entropy for {MODEL_DISPLAY.get(model, model)}",
                      "statistic": rho_s7, "p_raw": p_s7, "effect": rho_s7,
                      "note": "Spearman ρ"})

    # ── Family 4: Search-rate uniformity across entropy levels (χ² per combo) ──
    # H₀: P(searched) is constant across the 7 discrete entropy levels.
    # Tests whether the model at all modulates its search behaviour by uncertainty.
    rate_stats = search_rate_by_entropy_level(df)
    for combo, st in rate_stats.items():
        ds_disp  = DATASET_DISPLAY.get(combo[0], combo[0])
        mod_disp = MODEL_DISPLAY.get(combo[1], combo[1])
        tests.append({
            "id": f"U_{combo[0][:2]}_{combo[1][:4]}",
            "family": 4,
            "hypothesis": f"Search rate uniform across entropy levels — {ds_disp} / {mod_disp}",
            "statistic": st["chi2"],
            "p_raw": st["chi2_p"],
            "effect": float(np.nanmax(st["rates"]) - np.nanmin(st["rates"])) if any(~np.isnan(r) for r in st["rates"]) else np.nan,
            "note": f"χ²({st['chi2_dof']}); effect = max−min search rate",
        })

    # ── Family 5: All-or-nothing behaviour (H=0 vs H>0) ──────────────────────
    # Two-part test per combo:
    #   A) Proportions z-test: P(searched | H=0) vs P(searched | H>0)
    #      Significant → jump exists at the certainty boundary
    #   B) χ² uniformity within H>0 only
    #      Non-significant → no further grading beyond the binary certain/uncertain split
    # Both A significant + B non-significant = pure all-or-nothing behaviour.
    for combo, st in rate_stats.items():
        ds_disp  = DATASET_DISPLAY.get(combo[0], combo[0])
        mod_disp = MODEL_DISPLAY.get(combo[1], combo[1])
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]

        certain   = sl[sl["entropy"].round(3) == 0.0]
        uncertain = sl[sl["entropy"].round(3) != 0.0]

        n_s_cert  = int(certain["search_attributed"].sum())
        n_cert    = len(certain)
        n_s_unc   = int(uncertain["search_attributed"].sum())
        n_unc     = len(uncertain)

        # Part A: proportions z-test H=0 vs H>0
        if n_cert > 0 and n_unc > 0:
            z_ao, p_ao = proportions_ztest([n_s_cert, n_s_unc], [n_cert, n_unc])
            rate_cert = n_s_cert / n_cert
            rate_unc  = n_s_unc  / n_unc
            eff_ao    = rate_unc - rate_cert          # positive = model searches more when uncertain
        else:
            z_ao, p_ao, eff_ao = np.nan, np.nan, np.nan

        tests.append({
            "id": f"AO_A_{combo[0][:2]}_{combo[1][:4]}",
            "family": 5,
            "hypothesis": f"P(searched|H=0) ≠ P(searched|H>0) — {ds_disp} / {mod_disp}",
            "statistic": z_ao, "p_raw": p_ao, "effect": eff_ao,
            "note": "z-test; effect = rate(H>0) − rate(H=0)",
        })

        # Part B: χ² uniformity within H>0 only
        contingency_uncertain: list[list[int]] = []
        for lv in ENTROPY_LEVELS[1:]:           # skip H=0
            at_lv = sl[sl["entropy"].round(3) == round(lv, 3)]
            n_tot_lv = len(at_lv)
            n_s_lv   = int(at_lv["search_attributed"].sum())
            if n_tot_lv > 0:
                contingency_uncertain.append([n_s_lv, n_tot_lv - n_s_lv])

        if len(contingency_uncertain) >= 2:
            try:
                chi2_unc, p_unc, dof_unc, _ = sp_stats.chi2_contingency(contingency_uncertain)
            except Exception:
                chi2_unc, p_unc, dof_unc = np.nan, np.nan, np.nan
        else:
            chi2_unc, p_unc, dof_unc = np.nan, np.nan, np.nan

        unc_rates = [r for r, lv in zip(st["rates"], ENTROPY_LEVELS) if lv > 0 and not np.isnan(r)]
        eff_unc = float(max(unc_rates) - min(unc_rates)) if len(unc_rates) >= 2 else np.nan

        tests.append({
            "id": f"AO_B_{combo[0][:2]}_{combo[1][:4]}",
            "family": 5,
            "hypothesis": f"Search rate uniform within H>0 — {ds_disp} / {mod_disp}",
            "statistic": chi2_unc, "p_raw": p_unc, "effect": eff_unc,
            "note": f"χ²({int(dof_unc) if not np.isnan(dof_unc) else '?'}); effect = max−min rate within H>0",
        })

    # Apply BH correction within each family
    tests_df = pd.DataFrame(tests)
    for fam in [1, 2, 3, 4, 5]:
        mask = tests_df["family"] == fam
        if mask.any():
            adj = bh_correct(tests_df.loc[mask, "p_raw"].fillna(1.0).tolist())
            tests_df.loc[mask, "p_adj_bh"] = adj
    tests_df["significant"] = tests_df["p_adj_bh"] < 0.05
    return tests_df


# ─── Plots ────────────────────────────────────────────────────────────────────

def _savefig(fig, output_dir: Path, name: str):
    for ext in ("png", "pdf"):
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", dpi=150)
    plt.close(fig)


def plot_quadrant_taxonomy(df: pd.DataFrame, sigs: pd.DataFrame, output_dir: Path):
    """Figure 1: Stacked bar of E/PR/M/CP per (dataset, model)."""
    combos = [(d, m) for d, m in COMBO_ORDER if len(df[(df["dataset"] == d) & (df["model"] == m)]) > 0]
    labels = [COMBO_LABELS.get(c, f"{c[0]}/{c[1]}") for c in combos]
    colors_q = [QUADRANT_COLORS[q] for q in ["E", "CP", "PR", "M"]]
    keys = ["frac_E", "frac_CP", "frac_PR", "frac_M"]
    q_labels = ["Effective (E)", "Correct Parametric (CP)", "Param. Redundant (PR)", "Missed (M)"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros(len(combos))
    for key, color, qlabel in zip(keys, colors_q, q_labels):
        vals = []
        for combo in combos:
            row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
            vals.append(float(row[key].iloc[0]) if len(row) > 0 else 0.0)
        ax.bar(labels, vals, bottom=bottoms, color=color, label=qlabel, width=0.6)
        # Annotate if visible
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(i, b + v / 2, f"{v:.1%}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        bottoms += np.array(vals)

    ax.set_ylabel("Fraction of EEUs")
    ax.set_title("EEU Quadrant Decomposition — All Models & Datasets")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=4, framealpha=0.9, fontsize=9)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    _savefig(fig, output_dir, "fig1_quadrant_taxonomy")


def plot_overhead_intensity(df: pd.DataFrame, output_dir: Path):
    """Figure 2: Violin of certainty_score among searched EEUs — shows EWOI distribution."""
    searched_df = df[df["search_attributed"]].copy()
    searched_df["combo"] = searched_df.apply(
        lambda r: COMBO_LABELS.get((r["dataset"], r["model"]), f"{r['dataset']}/{r['model']}"), axis=1
    )
    # Filter to valid combos
    valid_labels = [COMBO_LABELS[c] for c in COMBO_ORDER if c in COMBO_LABELS]
    searched_df = searched_df[searched_df["combo"].isin(valid_labels)]

    combo_order = [COMBO_LABELS[c] for c in COMBO_ORDER if COMBO_LABELS.get(c) in searched_df["combo"].unique()]
    palette = {COMBO_LABELS[c]: COMBO_COLORS[c] for c in COMBO_ORDER if c in COMBO_COLORS}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: violin of certainty scores
    ax = axes[0]
    sns.violinplot(
        data=searched_df, x="combo", y="certainty_score",
        order=combo_order, palette=palette, ax=ax, cut=0, inner="quartile",
    )
    ax.axhline(0.5, ls="--", color="gray", alpha=0.6, label="Mid certainty")
    ax.set_xlabel("")
    ax.set_ylabel("Certainty Score  (1 − entropy / ln5)")
    ax.set_title("Overhead Intensity — Distribution of Certainty Scores\nAmong Searched EEUs")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.legend()

    # Right: EWOI vs POR comparison
    ax = axes[1]
    combos = [(d, m) for d, m in COMBO_ORDER if len(df[(df["dataset"] == d) & (df["model"] == m)]) > 0]
    x = np.arange(len(combos))
    ewoi_vals, por_vals = [], []
    for combo in combos:
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        sl_s = sl[sl["search_attributed"]]
        ewoi_vals.append(sl_s["certainty_score"].mean() if len(sl_s) > 0 else np.nan)
        certain_and_searched = sl_s[sl_s["certain"]]
        n_s = len(sl_s)
        por_vals.append(len(certain_and_searched) / n_s if n_s > 0 else np.nan)

    w = 0.35
    bars1 = ax.bar(x - w / 2, ewoi_vals, w, label="EWOI (continuous)", color=[COMBO_COLORS.get(c, "gray") for c in combos], alpha=0.9)
    bars2 = ax.bar(x + w / 2, por_vals, w, label="FDR / 1−Precision (binary)", color=[COMBO_COLORS.get(c, "gray") for c in combos], alpha=0.45, hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([COMBO_LABELS.get(c, str(c)) for c in combos], rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title("EWOI vs FDR — Magnitude vs Binary Overhead")
    ax.set_ylim(0, 1.1)
    ax.legend()
    for bar, v in zip(bars1, ewoi_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    for bar, v in zip(bars2, por_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    fig.tight_layout()
    _savefig(fig, output_dir, "fig2_overhead_intensity")


def _compute_query_split(df_slice: pd.DataFrame) -> tuple[float, float]:
    """Return (effective_qpq, overhead_qpq) — queries per question split by efficiency.

    For MusiQue: queries_assigned exists per EEU; sum directly by quadrant.
    For ShareChat: no per-EEU query count; distribute question-level search_calls
      proportionally by the ratio of E-facts to (E+PR)-facts within each question.
    """
    dataset = df_slice["dataset"].iloc[0]
    n_q = df_slice["question_id"].nunique()
    if n_q == 0:
        return np.nan, np.nan

    if dataset.startswith("musique"):
        eff  = df_slice[df_slice["quadrant"] == "E"]["queries_assigned"].fillna(0).sum() / n_q
        ovhd = df_slice[df_slice["quadrant"] == "PR"]["queries_assigned"].fillna(0).sum() / n_q
    else:
        q_agg = df_slice.groupby("question_id").agg(
            search_calls=("search_calls", "first"),
            n_E=("quadrant",  lambda x: (x == "E").sum()),
            n_PR=("quadrant", lambda x: (x == "PR").sum()),
        )
        q_agg["n_sf"] = q_agg["n_E"] + q_agg["n_PR"]
        q_agg["eff_q"]  = np.where(q_agg["n_sf"] > 0,
                                   q_agg["search_calls"] * q_agg["n_E"]  / q_agg["n_sf"], 0)
        q_agg["ovhd_q"] = np.where(q_agg["n_sf"] > 0,
                                   q_agg["search_calls"] * q_agg["n_PR"] / q_agg["n_sf"], 0)
        eff  = float(q_agg["eff_q"].mean())
        ovhd = float(q_agg["ovhd_q"].mean())
    return float(eff), float(ovhd)


def plot_search_budget_efficiency(df: pd.DataFrame, sigs: pd.DataFrame, output_dir: Path):
    """Figure 3: Stacked bar of queries/question split into effective vs overhead.

    Each bar's total height = SIR_mean (average queries per question).
    The green bottom segment = queries directed at genuinely uncertain EEUs (E-quadrant).
    The orange top segment   = queries directed at certain EEUs (PR-quadrant / overhead).

    Two values are annotated:
      - Green segment centre: query precision = eff_queries / total_queries  [0, 1]
        "What fraction of queries went to uncertain EEUs?"
      - Above the bar: SBE from signatures = n_E_EEUs / total_queries  [small, ~0.05–0.9]
        "How many distinct uncertain EEUs does each query address?"

    SBE ≪ query-precision when many queries are redundantly assigned to the same EEU
    (e.g. Gemini MusiQue: ~74% of queries go to uncertain hops but 16 queries per hop
    on average → SBE = 0.045 despite high query precision).

    For ShareChat, per-question search_calls are distributed proportionally by the
    fraction of attributed facts that are E vs PR (marked with *).
    """
    combos = [(d, m) for d, m in COMBO_ORDER if len(df[(df["dataset"] == d) & (df["model"] == m)]) > 0]
    labels = [COMBO_LABELS.get(c, str(c)) for c in combos]
    colors = [COMBO_COLORS.get(c, "gray") for c in combos]

    eff_vals, ovhd_vals, qprec_vals, sbe_vals = [], [], [], []
    has_sharechat = False
    for combo in combos:
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        eff, ovhd = _compute_query_split(sl)
        eff_vals.append(eff)
        ovhd_vals.append(ovhd)
        total = eff + ovhd
        # Query precision: fraction of queries that go to uncertain EEUs (NOT the same as SBE)
        qprec_vals.append(eff / total if total > 0 else np.nan)
        # SBE from signatures: n_E_EEUs / total_queries  (distinct uncertain EEUs per query issued)
        # This is lower than query-precision because many queries can map to the same EEU
        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        sbe_vals.append(float(row["sbe"].iloc[0]) if len(row) > 0 else np.nan)
        if combo[0] == "sharechat":
            has_sharechat = True

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(combos))
    w = 0.55

    bars_eff  = ax.bar(x, eff_vals,  w, color="#4CAF50", alpha=0.88, label="Effective queries (→ uncertain EEUs)")
    bars_ovhd = ax.bar(x, ovhd_vals, w, bottom=eff_vals, color="#FF9800", alpha=0.88, label="Overhead queries (→ certain EEUs)")

    # Annotate query-precision in the centre of the effective (green) segment
    # and SBE (distinct E-EEUs per query) above each bar
    for i, (eff, ovhd, qprec, sbe) in enumerate(zip(eff_vals, ovhd_vals, qprec_vals, sbe_vals)):
        total = eff + ovhd
        if not np.isnan(qprec) and eff > 0.05:
            mid_y = eff / 2
            ax.text(i, mid_y, f"{qprec:.0%}\nof queries", ha="center", va="center",
                    fontsize=8, color="white", fontweight="bold")
        # Total height + SBE above the bar
        if not np.isnan(total):
            sbe_str = f"SBE={sbe:.3f}" if not np.isnan(sbe) else ""
            ax.text(i, total + max(total * 0.03, 0.15), f"{total:.1f}q\n{sbe_str}",
                    ha="center", fontsize=7.5, color="black", linespacing=1.3)

    ax.set_xticks(x)
    # ShareChat combos get an asterisk on their label
    tick_labels = []
    for c, lbl in zip(combos, labels):
        tick_labels.append(lbl + ("*" if c[0] == "sharechat" else ""))
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Queries per question")
    ax.set_title("Search Budget Decomposition — Effective vs Overhead Queries\n"
                 "(green % = query precision; SBE above bar = distinct E-EEUs per query)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.35)

    # Give headroom above the tallest bar + its 2-line annotation
    max_total = max((e + o) for e, o in zip(eff_vals, ovhd_vals) if not np.isnan(e))
    ax.set_ylim(0, max_total * 1.35)

    if has_sharechat:
        ax.text(0.01, 0.01, "* ShareChat: queries distributed proportionally by E/PR fact counts",
                transform=ax.transAxes, fontsize=7, color="gray", style="italic", va="bottom")

    fig.tight_layout()
    _savefig(fig, output_dir, "fig3_search_budget_efficiency")


def search_rate_by_entropy_level(df: pd.DataFrame) -> dict:
    """Compute P(searched | entropy=X) with 95% Wilson CIs and chi-squared test per combo.

    The chi-squared test answers: is search rate constant across the 7 entropy levels?
    Null hypothesis: search decision is independent of entropy level.
    A significant result means the model does discriminate; non-significant means uniform behaviour.

    Returns dict keyed by (dataset, model) with keys:
      rates, ci_lo, ci_hi  — lists of 7 floats (nan where level has 0 EEUs)
      ns                   — list of 7 ints (total EEUs at each level)
      chi2, chi2_p, chi2_dof
    """
    result = {}
    combos = [(d, m) for d, m in COMBO_ORDER if len(df[(df["dataset"] == d) & (df["model"] == m)]) > 0]

    for combo in combos:
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])].copy()
        rates, ci_lo, ci_hi, ns = [], [], [], []
        contingency_rows: list[list[int]] = []

        for lv in ENTROPY_LEVELS:
            at_level = sl[sl["entropy"].round(3) == round(lv, 3)]
            n_tot = len(at_level)
            n_s = int(at_level["search_attributed"].sum())
            if n_tot == 0:
                rates.append(np.nan)
                ci_lo.append(np.nan)
                ci_hi.append(np.nan)
                ns.append(0)
            else:
                rate = n_s / n_tot
                lo, hi = wilson_ci(n_s, n_tot)
                rates.append(rate)
                ci_lo.append(lo)
                ci_hi.append(hi)
                ns.append(n_tot)
                contingency_rows.append([n_s, n_tot - n_s])

        # Chi-squared test of independence: search rate × entropy level
        # Only use levels that actually appear in the data (n_tot > 0)
        if len(contingency_rows) >= 2:
            try:
                chi2, p, dof, _ = sp_stats.chi2_contingency(contingency_rows)
            except Exception:
                chi2, p, dof = np.nan, np.nan, np.nan
        else:
            chi2, p, dof = np.nan, np.nan, np.nan

        result[combo] = dict(
            rates=rates, ci_lo=ci_lo, ci_hi=ci_hi, ns=ns,
            chi2=chi2, chi2_p=p, chi2_dof=int(dof) if not np.isnan(dof) else np.nan,
        )

    return result


def _pval_label(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    if p < 0.001:
        return "p<0.001"
    if p < 0.01:
        return "p<0.01"
    if p < 0.05:
        return "p<0.05"
    return f"p={p:.2f} (n.s.)"


def plot_graded_por(df: pd.DataFrame, output_dir: Path):
    """Figure 6: Two complementary views of search behaviour across entropy levels.

    Left panel  — P(entropy=X | searched): composition of the search budget.
      "Of the searches the model made, how many were at each certainty level?"
      Answers: how wasteful were the searches?

    Right panel — P(searched | entropy=X): search rate conditional on entropy.
      "At each entropy level, what fraction of EEUs triggered a search?"
      Answers: does the model search MORE when it is uncertain?
      This is the direct calibration question.

    Entropy is discrete with 7 possible values (5-sample clustering, bits):
      0, 0.722, 0.971, 1.371, 1.522, 1.922, 2.322
    """
    combos = [(d, m) for d, m in COMBO_ORDER if len(df[(df["dataset"] == d) & (df["model"] == m)]) > 0]
    labels = [COMBO_LABELS.get(c, str(c)) for c in combos]

    # 4 tiers aligned to discrete entropy structure
    tier_specs = [
        ("H=0  (5/5 agree)",        lambda s: s["entropy"] == 0.0,                           "#E53935"),
        ("H=0.72  (4/5 agree)",     lambda s: s["entropy"].round(2) == 0.72,                 "#FB8C00"),
        ("H∈[0.97,1.52]  (split)",  lambda s: (s["entropy"] > 0.72) & (s["entropy"] < 1.9), "#FDD835"),
        ("H∈[1.92,2.32]  (random)", lambda s: s["entropy"] >= 1.9,                           "#66BB6A"),
    ]

    tier_vals = [[] for _ in tier_specs]
    for combo in combos:
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        s = sl[sl["search_attributed"]]
        n_s = len(s)
        for i, (_, fn, _) in enumerate(tier_specs):
            tier_vals[i].append(len(s[fn(s)]) / n_s if n_s > 0 else 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    # ── Left: composition of search budget (P(entropy=X | searched)) ──────────
    ax = axes[0]
    x = np.arange(len(combos))
    w = 0.6
    bottoms = np.zeros(len(combos))
    for i, (tier_label, _, color) in enumerate(tier_specs):
        vals = np.array(tier_vals[i])
        ax.bar(x, vals, w, bottom=bottoms, label=tier_label, color=color, alpha=0.9)
        for j, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.05:
                ax.text(j, b + v / 2, f"{v:.0%}", ha="center", va="center",
                        fontsize=8, color="white" if i < 2 else "black", fontweight="bold")
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of searched EEUs  [P(H=X | searched)]")
    ax.set_title("Search Budget Composition\n(of searches made, what certainty level?)")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=9)

    # ── Right: search rate per entropy level (P(searched | entropy=X)) ────────
    ax = axes[1]
    level_labels = ["0\n(5/5)", "0.72\n(4/5)", "0.97\n(3/2)", "1.37\n(3/1/1)", "1.52\n(2/2/1)", "1.92\n(2/1/1/1)", "2.32\n(1/1/1/1/1)"]
    x2 = np.arange(len(ENTROPY_LEVELS))
    bar_width = 0.15

    rate_stats = search_rate_by_entropy_level(df)

    for ci_idx, combo in enumerate(combos):
        st = rate_stats[combo]
        rates = np.array(st["rates"], dtype=float)
        lo    = np.array(st["ci_lo"], dtype=float)
        hi    = np.array(st["ci_hi"], dtype=float)

        # Asymmetric error bars: (rate - lo, hi - rate), mask NaNs
        valid = ~np.isnan(rates)
        yerr_lo = np.where(valid, rates - lo, 0.0)
        yerr_hi = np.where(valid, hi - rates, 0.0)
        plot_rates = np.where(valid, rates, 0.0)

        color = COMBO_COLORS.get(combo, "gray")
        lbl = COMBO_LABELS.get(combo, str(combo)).replace("\n", " ")
        xpos = x2 + ci_idx * bar_width

        ax.bar(xpos, plot_rates, bar_width, label=lbl, color=color, alpha=0.85)
        ax.errorbar(xpos, plot_rates, yerr=[yerr_lo, yerr_hi],
                    fmt="none", ecolor="black", elinewidth=0.8, capsize=2.5, alpha=0.7)

        # Overall search rate (dashed reference)
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        overall_rate = sl["search_attributed"].mean()
        ax.axhline(overall_rate, color=color, lw=1.0, ls="--", alpha=0.45)

    ax.set_xticks(x2 + bar_width * (len(combos) - 1) / 2)
    ax.set_xticklabels(level_labels, fontsize=8)
    ax.set_ylabel("Search rate  [P(searched | H=X)]")
    ax.set_xlabel("Entropy level (bits) — cluster partition")
    ax.set_title("Search Rate per Entropy Level  (95% Wilson CI)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc="upper left")
    ax.text(0.99, 0.01, "Dashed = overall search rate per combo",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="gray", style="italic")
    # Shade H=0 bin to highlight the certain/uncertain boundary
    ax.axvspan(-0.5, 0.5, color="red", alpha=0.06, zorder=0)
    ax.axvline(0.5, color="red", lw=1.0, ls=":", alpha=0.5)
    ax.text(0.0, 1.02, "certain\n(H=0)", ha="center", va="bottom", fontsize=7,
            color="red", alpha=0.7, style="italic")

    fig.tight_layout()
    _savefig(fig, output_dir, "fig6_graded_por")


def plot_all_or_nothing(df: pd.DataFrame, output_dir: Path):
    """Figure 7: All-or-nothing vs graded search behaviour.

    Left panel  — Cross-dataset: P(searched|H=0) vs P(searched|H>0) per combo.
      Shows whether there is a jump at the certainty boundary (H=0 vs any disagreement).
      MusiQue models should show a clear gap; ShareChat models should be flat.

    Right panel — Within MusiQue only: P(searched|H=X) for X > 0 per model.
      A flat line = all-or-nothing (binary trigger).
      A rising line = graded (model responds to degree of uncertainty).
    """
    rate_stats = search_rate_by_entropy_level(df)
    combos = [(d, m) for d, m in COMBO_ORDER if (d, m) in rate_stats]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # ── Left: H=0 vs H>0 comparison (all combos) ─────────────────────────────
    ax = axes[0]
    x = np.arange(len(combos))
    bar_w = 0.35
    labels = [COMBO_LABELS.get(c, str(c)) for c in combos]

    cert_rates, cert_lo, cert_hi = [], [], []
    unc_rates,  unc_lo,  unc_hi  = [], [], []

    for combo in combos:
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        certain   = sl[sl["entropy"].round(3) == 0.0]
        uncertain = sl[sl["entropy"].round(3) != 0.0]

        n_s_c, n_c = int(certain["search_attributed"].sum()), len(certain)
        n_s_u, n_u = int(uncertain["search_attributed"].sum()), len(uncertain)

        r_c = n_s_c / n_c if n_c > 0 else np.nan
        r_u = n_s_u / n_u if n_u > 0 else np.nan
        lo_c, hi_c = wilson_ci(n_s_c, n_c) if n_c > 0 else (np.nan, np.nan)
        lo_u, hi_u = wilson_ci(n_s_u, n_u) if n_u > 0 else (np.nan, np.nan)

        cert_rates.append(r_c); cert_lo.append(lo_c); cert_hi.append(hi_c)
        unc_rates.append(r_u);  unc_lo.append(lo_u);  unc_hi.append(hi_u)

    cert_rates = np.array(cert_rates, dtype=float)
    unc_rates  = np.array(unc_rates,  dtype=float)

    def _asym_err(rates, lo, hi):
        lo_arr = np.array(lo, dtype=float)
        hi_arr = np.array(hi, dtype=float)
        return [np.where(~np.isnan(rates), rates - lo_arr, 0.0),
                np.where(~np.isnan(rates), hi_arr - rates, 0.0)]

    bars_c = ax.bar(x - bar_w / 2, np.nan_to_num(cert_rates), bar_w,
                    label="H=0  (5/5 agree — certain)", color="#E53935", alpha=0.85)
    ax.errorbar(x - bar_w / 2, np.nan_to_num(cert_rates),
                yerr=_asym_err(cert_rates, cert_lo, cert_hi),
                fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    bars_u = ax.bar(x + bar_w / 2, np.nan_to_num(unc_rates), bar_w,
                    label="H>0  (any disagreement — uncertain)", color="#42A5F5", alpha=0.85)
    ax.errorbar(x + bar_w / 2, np.nan_to_num(unc_rates),
                yerr=_asym_err(unc_rates, unc_lo, unc_hi),
                fmt="none", ecolor="black", elinewidth=1.0, capsize=3)

    # Draw a connector line between the two bars and annotate significance
    for i, combo in enumerate(combos):
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        certain   = sl[sl["entropy"].round(3) == 0.0]
        uncertain = sl[sl["entropy"].round(3) != 0.0]
        n_s_c = int(certain["search_attributed"].sum());   n_c = len(certain)
        n_s_u = int(uncertain["search_attributed"].sum()); n_u = len(uncertain)
        if n_c > 0 and n_u > 0:
            _, p_ao = proportions_ztest([n_s_c, n_s_u], [n_c, n_u])
            sig_lbl = "***" if p_ao < 0.001 else ("**" if p_ao < 0.01 else ("*" if p_ao < 0.05 else "n.s."))
            y_top = max(cert_rates[i] if not np.isnan(cert_rates[i]) else 0,
                        unc_rates[i]  if not np.isnan(unc_rates[i])  else 0) + 0.06
            ax.plot([i - bar_w / 2, i + bar_w / 2], [y_top, y_top], "k-", lw=0.8)
            ax.text(i, y_top + 0.01, sig_lbl, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Dataset boundary lines between different datasets
    prev_ds = None
    for i, (ds, _) in enumerate(combos):
        if prev_ds is not None and ds != prev_ds:
            ax.axvline(i - 0.5, color="gray", lw=1.2, ls="--", alpha=0.5)
        prev_ds = ds

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Search rate  [P(searched)]")
    ax.set_title("Binary Certainty Boundary: H=0 vs H>0\n(*** p<0.001, n.s. = not significant after BH)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)

    # ── Right: within H>0, per MusiQue model (graded vs flat) ────────────────
    ax = axes[1]
    mq_models = [m for d, m in COMBO_ORDER if d == "musique" and (d, m) in rate_stats]
    uncertain_levels     = ENTROPY_LEVELS[1:]          # skip H=0
    uncertain_level_lbls = ["0.72\n(4/5)", "0.97\n(3/2)", "1.37\n(3/1/1)",
                             "1.52\n(2/2/1)", "1.92\n(2/1/1/1)", "2.32\n(1/1/1/1/1)"]
    x3 = np.arange(len(uncertain_levels))

    for model in mq_models:
        combo = ("musique", model)
        st = rate_stats[combo]
        # rates[1:] = H>0 levels
        rates_unc = np.array(st["rates"][1:], dtype=float)
        lo_unc    = np.array(st["ci_lo"][1:], dtype=float)
        hi_unc    = np.array(st["ci_hi"][1:], dtype=float)
        color = COMBO_COLORS.get(combo, "gray")
        lbl   = MODEL_DISPLAY.get(model, model)

        valid = ~np.isnan(rates_unc)
        ax.plot(x3[valid], rates_unc[valid], "o-", color=color, lw=2, ms=6, label=lbl)
        ax.fill_between(x3[valid],
                        lo_unc[valid], hi_unc[valid],
                        color=color, alpha=0.15)

    ax.set_xticks(x3)
    ax.set_xticklabels(uncertain_level_lbls, fontsize=8)
    ax.set_ylabel("Search rate  [P(searched | H=X)]")
    ax.set_xlabel("Entropy level (bits) — cluster partition  [H>0 only]")
    ax.set_title("Within Uncertain EEUs — Graded vs Flat\n(MusiQue only; shaded = 95% Wilson CI)")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.3)
    ax.text(0.99, 0.04,
            "Flat line → all-or-nothing  |  Rising line → graded",
            transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
            color="gray", style="italic")

    fig.tight_layout()
    _savefig(fig, output_dir, "fig7_all_or_nothing")


def plot_iqsrs(df: pd.DataFrame, output_dir: Path):
    """Figure 8: IQSRS — within-question routing scores for musique-family datasets, 3+ hop."""
    mq_datasets = [d for d in df["dataset"].unique() if d.startswith("musique")]
    mq = df[df["dataset"].isin(mq_datasets) & (df["num_hops"] >= 3)].copy()
    if len(mq) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    # Build (dataset, model) combos in order
    mq_combos = [(d, m) for d, m in COMBO_ORDER
                 if d in mq_datasets and len(mq[(mq["dataset"] == d) & (mq["model"] == m)]) > 0]
    mq_models = sorted(mq["model"].unique())

    # Left: distribution of per-question routing scores
    ax = axes[0]
    all_rhos = {}
    for combo in mq_combos:
        ds, model = combo
        sl = mq[(mq["dataset"] == ds) & (mq["model"] == model)].dropna(subset=["queries_assigned"])
        rhos = []
        for _, grp in sl.groupby("question_id"):
            if len(grp) < 3:
                continue
            rho, _ = sp_stats.spearmanr(grp["entropy"], grp["queries_assigned"])
            if not np.isnan(rho):
                rhos.append(rho)
        all_rhos[combo] = rhos
        color = COMBO_COLORS.get(combo, "gray")
        label = COMBO_LABELS.get(combo, f"{ds}/{model}").replace("\n", " ")
        ax.hist(rhos, bins=30, alpha=0.6, color=color, label=f"{label} (n={len(rhos)})", density=True)

    ax.axvline(0, color="black", ls="--", lw=1.5, label="ρ = 0")
    ax.set_xlabel("Within-question Spearman ρ  (entropy vs queries)")
    ax.set_ylabel("Density")
    ax.set_title("IQSRS Distribution\n(per-question routing calibration)")
    ax.legend(fontsize=9)

    # Right: mean IQSRS per combo with CI
    ax = axes[1]
    means, cis, colors_list, m_labels = [], [], [], []
    for combo in mq_combos:
        rhos = all_rhos.get(combo, [])
        if len(rhos) < 5:
            continue
        m = np.mean(rhos)
        se = np.std(rhos) / np.sqrt(len(rhos))
        means.append(m)
        cis.append(1.96 * se)
        colors_list.append(COMBO_COLORS.get(combo, "gray"))
        m_labels.append(COMBO_LABELS.get(combo, f"{combo[0]}/{combo[1]}").replace("\n", " "))

    ax.barh(m_labels, means, xerr=cis, color=colors_list, alpha=0.85, capsize=5)
    ax.axvline(0, color="black", ls="--", lw=1.5)
    ax.set_xlabel("Mean within-question Spearman ρ")
    ax.set_title("Mean IQSRS ± 95% SE\n(positive = query volume tracks hop uncertainty)")
    for i, (m, ci) in enumerate(zip(means, cis)):
        ax.text(m + ci + 0.01, i, f"{m:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    _savefig(fig, output_dir, "fig8_iqsrs")


def _compute_roc_curve(df_slice: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """ROC curve using entropy as a continuous predictor of search_attributed.

    Entropy is discrete (7 levels), so the curve has at most 8 operating points.
    Higher entropy → more uncertain → should predict positive (searched).
    """
    y_true  = df_slice["search_attributed"].astype(int).values
    y_score = df_slice["entropy"].values
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    # Sweep thresholds from high to low entropy; predict positive if entropy >= threshold
    thresholds = sorted(set(y_score), reverse=True)
    fprs, tprs = [0.0], [0.0]
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        fprs.append(fp / n_neg)
        tprs.append(tp / n_pos)
    fprs.append(1.0)
    tprs.append(1.0)
    return np.array(fprs), np.array(tprs)


def plot_search_decision_roc(df: pd.DataFrame, sigs: pd.DataFrame, output_dir: Path):
    """Figure 10b: ROC curves — can uncertainty (entropy) predict search decisions?

    Left panel  — overlaid ROC curves, one per (dataset, model).
      X: FPR (rate of searching certain EEUs)
      Y: TPR / recall (rate of searching uncertain EEUs)
      Each curve uses 7 discrete entropy thresholds → 8 operating points.
      A model with perfect calibration sits near the top-left (high TPR, low FPR).
      A reflexive model (searches everything) lies on the diagonal.

    Right panel — AUROC bar chart summary.
      AUROC = P(entropy_searched > entropy_not_searched) via Mann-Whitney U.
      AUROC = 0.5 → search decision is independent of uncertainty (diagonal ROC).
      AUROC → 1.0 → uncertainty perfectly predicts search.
    """
    combos = [(d, m) for d, m in COMBO_ORDER if len(df[(df["dataset"] == d) & (df["model"] == m)]) > 0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: overlaid ROC curves ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random (AUC=0.5)")

    for combo in combos:
        sl = df[(df["dataset"] == combo[0]) & (df["model"] == combo[1])]
        fpr, tpr = _compute_roc_curve(sl)
        color = COMBO_COLORS.get(combo, "gray")
        label_base = COMBO_LABELS.get(combo, str(combo)).replace("\n", " ")

        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        auc_val = float(row["auroc"].iloc[0]) if len(row) > 0 else np.nan
        auc_str = f"{auc_val:.3f}" if not np.isnan(auc_val) else "n/a"

        ax.plot(fpr, tpr, "o-", color=color, lw=2, ms=5, alpha=0.85,
                label=f"{label_base}  (AUC={auc_str})")

    ax.set_xlabel("FPR  —  P(searched | certain)")
    ax.set_ylabel("TPR / Recall  —  P(searched | uncertain)")
    ax.set_title("ROC: entropy → search decision\n(operating points = 7 discrete entropy levels)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_aspect("equal")

    # ── Right: AUROC bar chart ────────────────────────────────────────────────
    ax = axes[1]
    auroc_vals = []
    for combo in combos:
        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        auroc_vals.append(float(row["auroc"].iloc[0]) if len(row) > 0 else np.nan)

    x = np.arange(len(combos))
    colors_list = [COMBO_COLORS.get(c, "gray") for c in combos]
    labels = [COMBO_LABELS.get(c, str(c)) for c in combos]
    bars = ax.bar(x, auroc_vals, color=colors_list, alpha=0.85, width=0.55)
    ax.axhline(0.5, color="black", lw=1.2, ls="--", alpha=0.6, label="Random baseline (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_title("AUROC Summary\n(entropy predicting search decision)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    for bar, v in zip(bars, auroc_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=9)

    fig.tight_layout()
    _savefig(fig, output_dir, "fig10b_search_decision_roc")


def plot_signature_heatmap(sigs: pd.DataFrame, output_dir: Path):
    """Figure 9: Heatmap of all signatures per (dataset, model)."""
    sig_cols = ["scc", "auroc", "precision", "recall", "f1", "ewoi", "sbe", "sae_eeu", "cov_gap", "sir_mean"]
    col_labels = ["SCC", "AUROC", "Precision", "Recall", "F1", "EWOI", "SBE", "SAE", "CovGap", "SIR_mean"]

    combos = [(d, m) for d, m in COMBO_ORDER if len(sigs[(sigs["dataset"] == d) & (sigs["model"] == m)]) > 0]
    row_labels = [COMBO_LABELS.get(c, str(c)).replace("\n", " ") for c in combos]

    mat = []
    for combo in combos:
        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        mat.append([float(row[c].iloc[0]) if len(row) > 0 else np.nan for c in sig_cols])

    mat = np.array(mat, dtype=float)
    # Normalise each column to [0,1] for display
    mat_norm = np.zeros_like(mat)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        valid = col[~np.isnan(col)]
        if len(valid) > 1:
            lo, hi = valid.min(), valid.max()
            mat_norm[:, j] = (col - lo) / (hi - lo + 1e-9)
        else:
            mat_norm[:, j] = col

    fig, ax = plt.subplots(figsize=(13, 4))
    im = ax.imshow(mat_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    for i in range(len(combos)):
        for j in range(len(sig_cols)):
            v = mat[i, j]
            txt = f"{v:.2f}" if not np.isnan(v) else "—"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < mat_norm[i, j] < 0.8 else "white")
    ax.set_title("Signature Heatmap — All Metrics (colour = column-normalised)")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Normalised score")
    fig.tight_layout()
    _savefig(fig, output_dir, "fig9_signature_heatmap")


def plot_sli(usefulness_df: pd.DataFrame, output_dir: Path):
    """Figure 10: Search Leverage Index — MusiQue only."""
    if usefulness_df is None:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    models = usefulness_df["model"].unique()
    x = np.arange(len(usefulness_df["certainty_level"].unique()))

    for model in models:
        sl = usefulness_df[usefulness_df["model"] == model].copy()
        sl = sl.sort_values("certainty_level")
        color = COMBO_COLORS.get(("musique", model), "gray")
        label = MODEL_DISPLAY.get(model, model)
        ax.plot(range(len(sl)), sl["sli"], "o-", color=color, lw=2, ms=7, label=label)
        ax.fill_between(range(len(sl)), 0, sl["sli"], alpha=0.15, color=color)

    ax.axhline(0, color="black", ls="--", lw=1)
    ax.set_xticks(range(len(sl)))
    ax.set_xticklabels(sl["certainty_level"].tolist(), rotation=15, ha="right")
    ax.set_ylabel("SLI  (search accuracy − no-search accuracy)")
    ax.set_title("Search Leverage Index by Certainty Level (MusiQue)")
    ax.legend()
    _savefig(fig, output_dir, "fig10_sli")


# ─── Comparison plots (ported from analyze_parametric_search_interplay.py) ────
# Each function takes raw_hop_dfs: dict[dataset_name → interplay_summary DataFrame]
# and shows musique vs musique-natural side by side.

_DATASET_PALETTE = {
    "musique":         "#2196F3",
    "musique-natural": "#1565C0",
    "sharechat":       "#F44336",
}


def _cmp_datasets(raw_hop_dfs: dict) -> list[str]:
    """Return dataset names in display order."""
    return [d for d in ["musique", "musique-natural", "sharechat"] if d in raw_hop_dfs]


def _search_calibration_extended_one(dfs: dict[str, pd.DataFrame], plot_dir: Path,
                                     hop_label: str = ""):
    """Render one set of search_calibration_extended plots for the given (pre-filtered) dfs.

    Produces one figure per model — each figure shows 6 bar groups (taxonomy categories)
    with one bar per dataset inside each group, distinguished by hatch pattern.
    """
    sextets = [
        ("Necessary\nsearch",  False, True,  False, None,  "#2ecc71"),
        ("Correct\nskip",      True,  False, None,  None,  "#3498db"),
        ("Truly\nmissed",      False, False, None,  False, "#e67e22"),
        ("Cross-hop\ncovered", False, False, None,  True,  "#9b59b6"),
        ("Seq.\nredundant",    False, True,  True,  None,  "#f1c40f"),
        ("Wasteful\nsearch",   True,  True,  None,  None,  "#e74c3c"),
    ]
    datasets = [d for d in ("musique", "musique-natural") if d in dfs]
    if not datasets:
        return

    all_models = sorted({m for ds in datasets for m in dfs[ds]["model"].unique()})
    hatches = ["", "///"]
    x = np.arange(len(sextets))
    cat_labels = [s[0] for s in sextets]
    cat_colors = [s[5] for s in sextets]

    hop_suffix = f"_{hop_label}" if hop_label else ""
    title_hop = f" — {hop_label}" if hop_label else ""

    for model in all_models:
        fig, ax = plt.subplots(figsize=(12, 5))
        n_ds = len(datasets)
        w = 0.35
        offsets = np.linspace(-(n_ds - 1) * w / 2, (n_ds - 1) * w / 2, n_ds)

        legend_handles = []
        y_max = 0.0
        for di, ds in enumerate(datasets):
            sub = dfs[ds]
            sub = sub[sub["model"] == model]
            total = len(sub)
            if total == 0:
                continue

            proportions, counts = [], []
            for _, certain_val, searched_val, seq_val, cross_hop_val, _ in sextets:
                mask = (sub["parametric_certain"] == certain_val) & (sub["searched"] == searched_val)
                if seq_val is not None:
                    mask &= (sub["seq_redundant"] == seq_val)
                if cross_hop_val is not None:
                    col_name = "missed_search_answer_found_cross_hop"
                    mask &= (sub[col_name] == cross_hop_val) if col_name in sub.columns else False
                cnt = int(mask.sum())
                proportions.append(100 * cnt / total)
                counts.append(cnt)

            hatch = hatches[di % len(hatches)]
            xpos = x + offsets[di]
            bars = ax.bar(xpos, proportions, w * 0.92, color=cat_colors, alpha=0.85,
                          hatch=hatch, edgecolor="white" if not hatch else "#444")
            for bar, pct, cnt in zip(bars, proportions, counts):
                if pct >= 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                            f"{pct:.1f}%\n(n={cnt})", ha="center", va="bottom", fontsize=7.5)
            y_max = max(y_max, max(proportions))

            ds_label = DATASET_DISPLAY.get(ds, ds)
            legend_handles.append(mpatches.Patch(
                facecolor="lightgray", hatch=hatch,
                edgecolor="#444" if hatch else "lightgray",
                label=f"{ds_label}  (n={total} hops)",
            ))

        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=9.5)
        ax.set_ylim(0, y_max * 1.3 + 5)
        ax.set_ylabel("% of Total Hops")
        ax.axvline(1.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(handles=legend_handles, fontsize=9, loc="upper right")
        fig.suptitle(
            f"Search Calibration (Extended){title_hop} — {MODEL_DISPLAY.get(model, model)}\n"
            "(left of dashed line: correct decisions; right: search errors)",
            fontsize=11,
        )
        plt.tight_layout()
        safe_model = re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_")
        _savefig(fig, plot_dir, f"search_calibration_extended_{safe_model}{hop_suffix}")


def plot_search_calibration_extended_cmp(raw_hop_dfs: dict, plot_dir: Path,
                                         produce_hop_variants: bool = True):
    """6-bar taxonomy per model — side-by-side dataset comparison, one figure per model.

    Produces:
      • Overall plots (all hops combined)
      • Per-hop-count plots (2hop, 3hop, 4hop) in hop_N/ subdirs (when produce_hop_variants=True)
    """
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return

    # Normalise bool columns once
    dfs: dict[str, pd.DataFrame] = {}
    for ds in datasets:
        df = raw_hop_dfs[ds].copy()
        for col in ("parametric_certain", "searched", "seq_redundant",
                    "missed_search_answer_found_cross_hop"):
            if col in df.columns:
                df[col] = df[col].astype(bool)
        dfs[ds] = df

    # Overall (all hops combined)
    _search_calibration_extended_one(dfs, plot_dir)

    # Per-hop-count variants — skipped when the caller already passes filtered data
    if produce_hop_variants:
        hop_counts = sorted({h for df in dfs.values() if "num_hops" in df.columns
                             for h in df["num_hops"].unique()})
        for h in hop_counts:
            hop_dir = plot_dir / f"hop_{h}"
            hop_dir.mkdir(exist_ok=True)
            filtered = {ds: df[df["num_hops"] == h].copy() for ds, df in dfs.items()
                        if "num_hops" in df.columns and (df["num_hops"] == h).any()}
            if filtered:
                _search_calibration_extended_one(filtered, hop_dir, hop_label=f"{h}hop")


def plot_accuracy_outcomes_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Aggregate accuracy per (certainty, searched) group, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    groups = [
        ("Certain\n+Searched",   True,  True),
        ("Certain\n+No search",  True,  False),
        ("Uncertain\n+Searched", False, True),
        ("Uncertain\n+No search",False, False),
    ]
    group_colors = ["#FF9800", "#2196F3", "#4CAF50", "#F44336"]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        models = sorted(df["model"].unique())
        x = np.arange(len(groups))
        w = 0.8 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            accs = []
            ns = []
            for _, cert, searched in groups:
                grp = sub[(sub["parametric_certain"] == cert) & (sub["searched"] == searched)]
                accs.append(grp["aggregate_correct"].mean() if len(grp) > 0 else np.nan)
                ns.append(len(grp))

            color = COMBO_COLORS.get((ds, model), "gray")
            label = MODEL_DISPLAY.get(model, model)
            bars = ax.bar(x + offsets[i], accs, w * 0.9, color=color, alpha=0.8, label=label)
            for bar, acc_v, n_v in zip(bars, accs, ns):
                if not np.isnan(acc_v) and n_v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, acc_v + 0.01,
                            f"n={n_v}", ha="center", fontsize=6, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_ylabel("Aggregate accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Accuracy by Certainty × Search Status", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "accuracy_by_certainty_search")


def plot_accuracy_vs_queries_bars_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Aggregate accuracy binned by total search queries per example, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        # Aggregate to example level
        ex = df.groupby(["model", "example_id"]).agg(
            total_queries=("queries_assigned_count", "sum"),
            aggregate_correct=("aggregate_correct", "first"),
        ).reset_index()

        models = sorted(ex["model"].unique())
        bins = [0, 1, 2, 3, 5, 100]
        bin_labels = ["0", "1", "2", "3–4", "5+"]
        x = np.arange(len(bin_labels))
        w = 0.8 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = ex[ex["model"] == model].copy()
            sub["bin"] = pd.cut(sub["total_queries"], bins=bins, labels=bin_labels, right=False)
            grouped = sub.groupby("bin", observed=True)["aggregate_correct"].mean()
            accs = [grouped.get(b, np.nan) for b in bin_labels]
            color = COMBO_COLORS.get((ds, model), "gray")
            ax.bar(x + offsets[i], accs, w * 0.9, color=color, alpha=0.8,
                   label=MODEL_DISPLAY.get(model, model))

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels)
        ax.set_xlabel("Total search queries per example")
        ax.set_ylabel("Aggregate accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Accuracy vs Number of Search Queries", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "accuracy_vs_queries_bars")


def plot_accuracy_per_hop_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Parametric accuracy per hop type (certain/uncertain) and search status, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    groups = [
        ("Certain\n+No search", True,  False),
        ("Certain\n+Searched",  True,  True),
        ("Uncertain\n+Searched",False, True),
        ("Uncertain\n+No search", False, False),
    ]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        models = sorted(df["model"].unique())
        x = np.arange(len(groups))
        w = 0.8 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            accs = []
            for _, cert, searched in groups:
                grp = sub[(sub["parametric_certain"] == cert) & (sub["searched"] == searched)]
                accs.append(grp["parametric_accuracy"].mean() if len(grp) > 0 else np.nan)
            color = COMBO_COLORS.get((ds, model), "gray")
            ax.bar(x + offsets[i], accs, w * 0.9, color=color, alpha=0.8,
                   label=MODEL_DISPLAY.get(model, model))

        ax.set_xticks(x)
        ax.set_xticklabels([g[0] for g in groups], fontsize=8)
        ax.set_ylabel("Parametric accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Hop Parametric Accuracy by Certainty × Search Status", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "accuracy_per_hop")


def plot_searched_vs_unsearched_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Aggregate accuracy for searched vs unsearched examples at question level, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        # Aggregate to example level: searched if any hop was searched
        ex = df.groupby(["model", "example_id"]).agg(
            any_searched=("searched", "any"),
            aggregate_correct=("aggregate_correct", "first"),
        ).reset_index()

        models = sorted(ex["model"].unique())
        x = np.arange(2)
        w = 0.7 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = ex[ex["model"] == model]
            acc_searched = sub[sub["any_searched"]]["aggregate_correct"].mean()
            acc_unsearched = sub[~sub["any_searched"]]["aggregate_correct"].mean()
            color = COMBO_COLORS.get((ds, model), "gray")
            ax.bar(x + offsets[i], [acc_searched, acc_unsearched], w * 0.9,
                   color=color, alpha=0.8, label=MODEL_DISPLAY.get(model, model))

        ax.set_xticks(x)
        ax.set_xticklabels(["Searched", "Not searched"])
        ax.set_ylabel("Aggregate accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Accuracy: Searched vs Unsearched Examples", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "searched_vs_unsearched")


def plot_entropy_bucket_queries_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Mean query count binned by semantic entropy level, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(9 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    level_labels = ["H=0", "0.72", "0.97", "1.37", "1.52", "1.92", "2.32"]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        models = sorted(df["model"].unique())
        x = np.arange(len(ENTROPY_LEVELS))
        w = 0.8 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            means = []
            for lv in ENTROPY_LEVELS:
                at_lv = sub[sub["semantic_entropy"].round(3) == round(lv, 3)]
                means.append(at_lv["queries_assigned_count"].mean() if len(at_lv) > 0 else np.nan)
            color = COMBO_COLORS.get((ds, model), "gray")
            ax.bar(x + offsets[i], means, w * 0.9, color=color, alpha=0.8,
                   label=MODEL_DISPLAY.get(model, model))

        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, rotation=30, ha="right")
        ax.set_xlabel("Semantic entropy level")
        ax.set_ylabel("Mean queries assigned")
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Queries Per Hop by Entropy Bucket", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "entropy_bucket_queries")


def plot_multi_hop_query_attribution_cmp(matched_examples_by_dataset: dict, plot_dir: Path):
    """Fraction of queries attributed as multi-hop vs single-hop, per dataset."""
    datasets = [d for d in ["musique", "musique-natural"] if d in matched_examples_by_dataset]
    if not datasets:
        print("  Skipping multi_hop_query_attribution: no matched_examples provided")
        return

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        examples = matched_examples_by_dataset[ds]
        # Group by model
        model_counts: dict = {}
        for ex in examples:
            model = ex.get("model", "unknown")
            queries = ex.get("queries", [])
            if model not in model_counts:
                model_counts[model] = {"single": 0, "multi": 0}
            for q in queries:
                if q.get("multi_hop_relevant", False):
                    model_counts[model]["multi"] += 1
                else:
                    model_counts[model]["single"] += 1

        models = sorted(model_counts.keys())
        x = np.arange(len(models))
        single_fracs = [model_counts[m]["single"] / max(1, model_counts[m]["single"] + model_counts[m]["multi"])
                        for m in models]
        multi_fracs  = [model_counts[m]["multi"]  / max(1, model_counts[m]["single"] + model_counts[m]["multi"])
                        for m in models]

        model_labels = [MODEL_DISPLAY.get(m, m) for m in models]
        colors = [COMBO_COLORS.get((ds, m), "gray") for m in models]
        ax.bar(x, single_fracs, 0.6, label="Single-hop", color=colors, alpha=0.85)
        ax.bar(x, multi_fracs,  0.6, bottom=single_fracs, label="Multi-hop", color=colors, alpha=0.45, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(model_labels, rotation=15, ha="right")
        ax.set_ylabel("Fraction of queries")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)

    fig.suptitle("Multi-Hop vs Single-Hop Query Attribution", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "multi_hop_query_attribution")


def plot_redundancy_overlap_cmp(example_dfs: dict, plot_dir: Path):
    """Parametric vs sequential redundancy overlap (stacked bar), per dataset."""
    datasets = [d for d in ["musique", "musique-natural"] if d in example_dfs]
    if not datasets:
        print("  Skipping redundancy_overlap: no example_metrics provided")
        return

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        df = example_dfs[ds]
        models = sorted(df["model"].unique())
        x = np.arange(len(models))
        w = 0.6

        necessary  = [df[df["model"] == m]["necessary_count"].mean() for m in models]
        par_only   = [df[df["model"] == m]["parametric_only_count"].mean() for m in models]
        seq_only   = [df[df["model"] == m]["sequential_only_count"].mean() for m in models]
        both       = [df[df["model"] == m]["both_redundant_count"].mean() for m in models]

        ax.bar(x, necessary, w, label="Necessary",           color="#4CAF50", alpha=0.85)
        ax.bar(x, par_only,  w, bottom=necessary,            label="Param. redundant", color="#FF9800", alpha=0.85)
        bot2 = [a + b for a, b in zip(necessary, par_only)]
        ax.bar(x, seq_only,  w, bottom=bot2,                 label="Seq. redundant",   color="#2196F3", alpha=0.85)
        bot3 = [a + b for a, b in zip(bot2, seq_only)]
        ax.bar(x, both,      w, bottom=bot3,                 label="Both redundant",   color="#F44336", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models], rotation=15, ha="right")
        ax.set_ylabel("Mean queries per example")
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Search Budget Composition: Necessary vs Redundant Queries", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "redundancy_overlap")


def plot_searches_per_entropy_level_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Search rate P(searched) per entropy level with 95% Wilson CI bands, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(10 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    level_labels = ["H=0", "0.72", "0.97", "1.37", "1.52", "1.92", "2.32"]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        models = sorted(df["model"].unique())
        x = np.arange(len(ENTROPY_LEVELS))
        w = 0.8 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            rates, ci_lo_vals, ci_hi_vals = [], [], []
            for lv in ENTROPY_LEVELS:
                at_lv = sub[sub["semantic_entropy"].round(3) == round(lv, 3)]
                n_tot = len(at_lv)
                n_s = int(at_lv["searched"].sum()) if n_tot > 0 else 0
                if n_tot == 0:
                    rates.append(np.nan)
                    ci_lo_vals.append(np.nan)
                    ci_hi_vals.append(np.nan)
                else:
                    lo, hi = wilson_ci(n_s, n_tot)
                    rates.append(n_s / n_tot)
                    ci_lo_vals.append(lo)
                    ci_hi_vals.append(hi)

            rates_arr = np.array(rates, dtype=float)
            lo_arr = np.array(ci_lo_vals, dtype=float)
            hi_arr = np.array(ci_hi_vals, dtype=float)
            valid = ~np.isnan(rates_arr)
            x_pos = x + offsets[i]
            color = COMBO_COLORS.get((ds, model), "gray")

            ax.plot(x_pos[valid], rates_arr[valid], "o-", color=color, lw=2, ms=6,
                    label=MODEL_DISPLAY.get(model, model))
            ax.fill_between(x_pos[valid], lo_arr[valid], hi_arr[valid],
                            color=color, alpha=0.15)

        ax.set_xticks(x)
        ax.set_xticklabels(level_labels, rotation=30, ha="right")
        ax.set_xlabel("Semantic entropy level")
        ax.set_ylabel("Search rate P(searched)")
        ax.set_ylim(-0.02, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax.axvspan(-0.5, 0.5, color="red", alpha=0.05)
        ax.text(0.99, 0.01, "Shaded = 95% Wilson CI", transform=ax.transAxes,
                fontsize=7, ha="right", va="bottom", color="gray", style="italic")

    fig.suptitle("Search Rate per Entropy Level (95% Wilson CI)", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "searches_per_entropy_level")


def plot_searches_per_certainty_level_cmp(raw_hop_dfs: dict, plot_dir: Path):
    """Search rate for certain vs uncertain hops, per dataset."""
    datasets = _cmp_datasets(raw_hop_dfs)
    if not datasets:
        return
    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        df = raw_hop_dfs[ds].copy()
        models = sorted(df["model"].unique())
        x = np.arange(2)
        w = 0.7 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]
            certain_rate   = sub[sub["parametric_certain"] == True]["searched"].mean()
            uncertain_rate = sub[sub["parametric_certain"] == False]["searched"].mean()
            color = COMBO_COLORS.get((ds, model), "gray")
            ax.bar(x + offsets[i], [certain_rate, uncertain_rate], w * 0.9,
                   color=color, alpha=0.8, label=MODEL_DISPLAY.get(model, model))

        ax.set_xticks(x)
        ax.set_xticklabels(["Certain\n(H=0, all correct)", "Uncertain"])
        ax.set_ylabel("Search rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Search Rate per Certainty Level", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "searches_per_certainty_level")


def plot_sequential_redundancy_distribution_cmp(example_dfs: dict, plot_dir: Path):
    """Distribution of sequential redundant search counts, excluding the zero case.

    Only includes examples that had at least one redundant search, showing the
    1-2 / 3 / 4 / … / 10+ breakdown to reveal how deep the tail extends.
    """
    datasets = [d for d in ["musique", "musique-natural"] if d in example_dfs]
    if not datasets:
        print("  Skipping sequential_redundancy_distribution: no example_metrics provided")
        return

    # Bins: "1–2" consolidated, then 3–9 explicit, then "10+"
    explicit_vals = list(range(3, 10))  # 3,4,5,6,7,8,9
    bin_labels = ["1–2"] + [str(v) for v in explicit_vals] + ["10+"]
    n_bins = len(bin_labels)

    n = len(datasets)
    fig, axes = plt.subplots(1, n, figsize=(max(10, 1.1 * n_bins) * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        df = example_dfs[ds].copy()
        # Only examples that actually had redundant searches
        df = df[(df["total_search_calls"] > 0) & (df["sequential_redundant_count"] >= 1)].copy()
        models = sorted(df["model"].unique())
        x = np.arange(n_bins)
        w = 0.8 / len(models)
        offsets = np.linspace(-(len(models) - 1) * w / 2, (len(models) - 1) * w / 2, len(models))

        for i, model in enumerate(models):
            sub = df[df["model"] == model]["sequential_redundant_count"].dropna()
            total = len(sub)
            if total == 0:
                continue
            fracs = (
                [(sub <= 2).sum() / total]
                + [(sub == v).sum() / total for v in explicit_vals]
                + [(sub >= 10).sum() / total]
            )
            color = COMBO_COLORS.get((ds, model), "gray")
            mean_val = sub.mean()
            pct_gt2 = 100 * (sub > 2).mean()
            label = f"{MODEL_DISPLAY.get(model, model)} (n={total}, μ={mean_val:.1f}, {pct_gt2:.0f}%>2)"
            bars = ax.bar(x + offsets[i], fracs, w * 0.9, color=color, alpha=0.8, label=label)
            for bar, frac in zip(bars, fracs):
                if frac >= 0.04:
                    ax.text(bar.get_x() + bar.get_width() / 2, frac + 0.01,
                            f"{frac:.0%}", ha="center", fontsize=7, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_xlabel("Sequential redundant searches per example")
        ax.set_ylabel("Fraction of examples with ≥1 redundant search")
        ax.set_title(DATASET_DISPLAY.get(ds, ds))
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.text(0.99, 0.98, "n = examples with ≥1 redundant search\nμ = mean, %>2 = fraction in tail",
                transform=ax.transAxes, fontsize=7, ha="right", va="top", color="gray", style="italic")

    fig.suptitle("Sequential Redundancy Distribution (examples with ≥1 redundant search)", fontsize=12)
    fig.tight_layout()
    _savefig(fig, plot_dir, "sequential_redundancy_distribution")


# ─── Report ───────────────────────────────────────────────────────────────────

def f(v, fmt=".3f"):
    return f"{v:{fmt}}" if not (isinstance(v, float) and np.isnan(v)) else "—"


def generate_report(sigs: pd.DataFrame, tests: pd.DataFrame, output_dir: Path) -> str:
    lines = ["# Unified Interplay Analysis — Report", ""]
    lines.append(f"Generated from `unified_interplay_analysis.py`. Datasets: MusiQue × ShareChat.\n")

    # ── Signatures table ──────────────────────────────────────────────────────
    lines.append("## 1. Interplay Signatures\n")
    header = "| Dataset | Model | E% | PR% | M% | CP% | SCC | AUROC | Prec | Recall | F1 | FDR | EWOI | SBE | SAE | CovGap |"
    sep    = "|---------|-------|:--:|:---:|:--:|:---:|:---:|:-----:|:----:|:------:|:--:|:---:|:----:|:---:|:---:|:------:|"
    lines += [header, sep]
    for combo in COMBO_ORDER:
        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lines.append(
            f"| {DATASET_DISPLAY[combo[0]]} | {MODEL_DISPLAY.get(combo[1], combo[1])} "
            f"| {r['frac_E']:.1%} | {r['frac_PR']:.1%} | {r['frac_M']:.1%} | {r['frac_CP']:.1%} "
            f"| {f(r['scc'])} | {f(r['auroc'])} | {f(r['precision'])} | {f(r['recall'])} "
            f"| {f(r['f1'])} | {f(r['fdr'])} | {f(r['ewoi'])} | {f(r['sbe'])} "
            f"| {f(r['sae_eeu'])} | {f(r['cov_gap'])} |"
        )
    lines.append("")

    # ── Extended metrics detail ───────────────────────────────────────────────
    lines.append("## 2. Extended Metrics Detail\n")
    lines.append("### EWOI vs FDR (overhead magnitude)\n")
    lines.append("| Dataset | Model | FDR (1−Prec) | EWOI | Δ (EWOI−FDR) | PR_extreme% | PR_borderline% | P_brittle% |")
    lines.append("|---------|-------|:------------:|:----:|:------------:|:-----------:|:--------------:|:----------:|")
    for combo in COMBO_ORDER:
        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        delta = (r["ewoi"] - r["fdr"]) if not np.isnan(r["ewoi"]) and not np.isnan(r["fdr"]) else np.nan
        lines.append(
            f"| {DATASET_DISPLAY[combo[0]]} | {MODEL_DISPLAY.get(combo[1], combo[1])} "
            f"| {f(r['fdr'])} | {f(r['ewoi'])} | {f(delta, '+.3f')} "
            f"| {f(r['por_extreme'], '.1%')} | {f(r['por_borderline'], '.1%')} "
            f"| {f(r['p_brittle'], '.1%') if not np.isnan(r.get('p_brittle', np.nan)) else '—'} |"
        )
    lines.append("")

    lines.append("### Search Budget Efficiency (SBE)\n")
    lines.append("| Dataset | Model | SBE | SIR_mean | N_E (entropy>0) | Total queries |")
    lines.append("|---------|-------|:---:|:--------:|:---------------:|:-------------:|")
    for combo in COMBO_ORDER:
        row = sigs[(sigs["dataset"] == combo[0]) & (sigs["model"] == combo[1])]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lines.append(
            f"| {DATASET_DISPLAY[combo[0]]} | {MODEL_DISPLAY.get(combo[1], combo[1])} "
            f"| {f(r['sbe'])} | {f(r['sir_mean'], '.2f')} "
            f"| {int(r['n_E_entropy'])} | {int(r['total_queries'])} |"
        )
    lines.append("")

    lines.append("### IQSRS — Within-Question Routing Score (MusiQue 3+ hop)\n")
    lines.append("| Model | IQSRS | p (one-sided) |")
    lines.append("|-------|:-----:|:-------------:|")
    for model in ["gemini-3-pro-preview", "nemotron-3-nano_30b", "qwen3.5_122b"]:
        row = sigs[(sigs["dataset"] == "musique") & (sigs["model"] == model)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        p_val = r.get("p_iqsrs", np.nan)
        p_one = p_val / 2 if not np.isnan(p_val) else np.nan
        lines.append(f"| {MODEL_DISPLAY.get(model, model)} | {f(r['iqsrs'])} | {f(p_one)} |")
    lines.append("")

    # ── Statistical tests ─────────────────────────────────────────────────────
    lines.append("## 3. Statistical Tests\n")
    for fam, fam_name in [(1, "Cross-Dataset"), (2, "Within-MusiQue"), (3, "Within-ShareChat")]:
        lines.append(f"### Family {fam}: {fam_name}\n")
        fam_tests = tests[tests["family"] == fam]
        lines.append("| ID | Hypothesis | Statistic | p_raw | p_adj_BH | Significant | Effect |")
        lines.append("|----|-----------|:---------:|:-----:|:--------:|:-----------:|:------:|")
        for _, t in fam_tests.iterrows():
            sig_mark = "✓" if t.get("significant", False) else "✗"
            lines.append(
                f"| {t['id']} | {t['hypothesis']} "
                f"| {f(t['statistic'])} | {f(t['p_raw'])} "
                f"| {f(t.get('p_adj_bh', np.nan))} | {sig_mark} | {f(t['effect'])} |"
            )
        lines.append("")

    # ── Key findings ─────────────────────────────────────────────────────────
    lines.append("## 4. Key Findings\n")

    gem_mq = sigs[(sigs["dataset"] == "musique") & (sigs["model"] == "gemini-3-pro-preview")]
    nem_mq = sigs[(sigs["dataset"] == "musique") & (sigs["model"] == "nemotron-3-nano_30b")]
    gem_sc = sigs[(sigs["dataset"] == "sharechat") & (sigs["model"] == "gemini-3-pro-preview")]
    nem_sc = sigs[(sigs["dataset"] == "sharechat") & (sigs["model"] == "nemotron-3-nano")]

    def gv(df_row, col):
        return float(df_row[col].iloc[0]) if len(df_row) > 0 else np.nan

    lines.append(
        f"1. **Overhead magnitude (EWOI)**: When Gemini searches on MusiQue, those hops are on average "
        f"**{gv(gem_mq, 'ewoi'):.1%} as certain as maximally-certain hops** (EWOI = {gv(gem_mq, 'ewoi'):.3f}). "
        f"Nemotron's EWOI is {gv(nem_mq, 'ewoi'):.3f} — a {abs(gv(gem_mq,'ewoi')-gv(nem_mq,'ewoi')):.3f} "
        f"point difference that binary FDR ({gv(gem_mq,'fdr'):.3f} vs {gv(nem_mq,'fdr'):.3f}) understates.\n"
    )
    lines.append(
        f"2. **Epistemic bimodality**: The entropy distribution among searched EEUs is near-binary — "
        f"either entropy = 0 (PR_extreme: {gv(gem_mq,'por_extreme'):.1%} for Gemini, "
        f"{gv(nem_mq,'por_extreme'):.1%} for Nemotron) or entropy > 0.5. "
        f"There are almost no searched EEUs with moderate uncertainty, revealing that models have "
        f"all-or-nothing confidence rather than graded uncertainty.\n"
    )
    lines.append(
        f"3. **Budget efficiency gap (SBE)**: Gemini issues {gv(gem_mq,'sir_mean'):.1f} queries/question "
        f"but only {gv(gem_mq,'sbe'):.3f} E-EEUs per query. Nemotron achieves {gv(nem_mq,'sbe'):.3f} — "
        f"a **{gv(nem_mq,'sbe')/gv(gem_mq,'sbe'):.1f}× efficiency advantage** despite similar E-EEU fractions "
        f"({gv(gem_mq,'frac_E'):.1%} vs {gv(nem_mq,'frac_E'):.1%}).\n"
    )
    if not np.isnan(gv(gem_sc, "sae_eeu")):
        lines.append(
            f"4. **Coarse-grained activation (SAE)**: On MusiQue, Gemini's search activation entropy "
            f"(SAE = {gv(gem_mq,'sae_eeu'):+.3f}) is far below Nemotron's ({gv(nem_mq,'sae_eeu'):+.3f}). "
            f"On ShareChat, Gemini's EEU-level SAE is near-zero ({gv(gem_sc,'sae_eeu'):+.3f}), "
            f"confirming that search decisions are made at question level (coarse) not fact level (fine-grained)."
        )
        if not np.isnan(gv(nem_sc, "sae_eeu")):
            lines.append(
                f" Nemotron on ShareChat also shows near-zero EEU-level SAE ({gv(nem_sc,'sae_eeu'):+.3f}),"
                f" with {gv(nem_sc,'frac_PR'):.1%} PR and {gv(nem_sc,'sir_mean'):.1f} queries/question.\n"
            )
        else:
            lines.append("\n")

    return "\n".join(lines)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(exist_ok=True)
    plot_dir = output_dir / "plots"

    dataset_paths = resolve_datasets(args)
    frames = []
    for name, path in dataset_paths.items():
        loader = DATASET_LOADERS.get(name)
        if loader is None:
            raise ValueError(f"Unknown dataset {name!r}. Known: {list(DATASET_LOADERS)}")
        kwargs = {"certainty_mode": args.mq_certainty_mode} if name in ("musique", "musique-natural") else {}
        print(f"Loading {name} from {path}")
        frame = loader(path, **kwargs)
        print(f"  {len(frame)} EEUs across {frame['model'].nunique()} models")
        frames.append(frame)

    df = build_unified_frame(*frames)
    df = assign_quadrants(df)
    print(f"Unified frame: {len(df)} EEUs total")

    print("Computing signatures...")
    sigs = run_all_signatures(df)

    print("Running statistical tests...")
    tests = run_statistical_tests(df, sigs)

    # Load raw hop DataFrames for comparison plots (need columns not in unified frame)
    raw_hop_dfs: dict = {}
    for name, path in dataset_paths.items():
        if name in ("musique", "musique-natural"):
            print(f"Loading raw hop data for {name} from {path}")
            raw_df = pd.read_csv(path)
            raw_df["model"] = raw_df["model"].str.replace(":", "_", regex=False)
            raw_hop_dfs[name] = raw_df

    # Load example-level metrics for redundancy plots
    example_dfs: dict = {}
    for spec in (args.example_metrics or []):
        name, path = spec.split(":", 1)
        print(f"Loading example metrics for {name} from {path}")
        df_e = load_example_metrics(path)
        df_e["model"] = df_e["model"].str.replace(":", "_", regex=False)
        example_dfs[name] = df_e

    # Load matched_examples for multi-hop attribution plot
    matched_examples_by_dataset: dict = {}
    for spec in (args.matched_examples or []):
        name, path = spec.split(":", 1)
        print(f"Loading matched examples for {name} from {path}")
        matched_examples_by_dataset[name] = load_matched_examples(path)

    print("Generating plots...")
    # ── Selected unified plots (signatures, EWOI vs FDR, quadrant, ROC) ────────
    plot_quadrant_taxonomy(df, sigs, plot_dir)
    plot_overhead_intensity(df, plot_dir)
    plot_search_budget_efficiency(df, sigs, plot_dir)
    plot_all_or_nothing(df, plot_dir)
    plot_iqsrs(df, plot_dir)
    plot_search_decision_roc(df, sigs, plot_dir)

    # ── Comparison plots (musique vs musique-natural side by side) ─────────────
    plot_search_calibration_extended_cmp(raw_hop_dfs, plot_dir)
    plot_accuracy_outcomes_cmp(raw_hop_dfs, plot_dir)
    plot_accuracy_vs_queries_bars_cmp(raw_hop_dfs, plot_dir)
    plot_accuracy_per_hop_cmp(raw_hop_dfs, plot_dir)
    plot_searched_vs_unsearched_cmp(raw_hop_dfs, plot_dir)
    plot_entropy_bucket_queries_cmp(raw_hop_dfs, plot_dir)
    plot_searches_per_entropy_level_cmp(raw_hop_dfs, plot_dir)
    plot_searches_per_certainty_level_cmp(raw_hop_dfs, plot_dir)
    plot_redundancy_overlap_cmp(example_dfs, plot_dir)
    plot_sequential_redundancy_distribution_cmp(example_dfs, plot_dir)
    plot_multi_hop_query_attribution_cmp(matched_examples_by_dataset, plot_dir)

    # ── Parallel hop-filtered plots ────────────────────────────────────────────
    hop_filter = args.parallel_hop_filter or auto_detect_hop_filter(df)
    if hop_filter is not None:
        print(f"\nBuilding hop-filtered frame (num_hops == {hop_filter})...")
        register_hop_filtered_combos(hop_filter)
        df_hop = build_hop_filtered_frame(df, hop_filter)
        if len(df_hop) > 0:
            # Align musique to the musique-natural example subset for a fair apples-to-apples
            # comparison. Without this, musique includes all N-hop examples (e.g. 200 2-hop
            # questions) while musique-natural only has its specific paraphrased subset (e.g. 103).
            # The two populations have different certainty/search distributions, causing spurious
            # shifts in the quadrant taxonomy that are selection artefacts, not paraphrasing effects.
            mq_hop_label = f"musique-{hop_filter}hop"
            mq_nat_hop_label = f"musique-natural-{hop_filter}hop"
            if mq_hop_label in df_hop["dataset"].unique() and mq_nat_hop_label in df_hop["dataset"].unique():
                nat_qids = set(df_hop[df_hop["dataset"] == mq_nat_hop_label]["question_id"].unique())
                df_hop = df_hop[
                    (df_hop["dataset"] != mq_hop_label) | df_hop["question_id"].isin(nat_qids)
                ].copy()
                print(f"  Aligned musique-{hop_filter}hop to {len(nat_qids)} matched question IDs (musique-natural subset)")

            df_hop = assign_quadrants(df_hop)
            n_eeus = len(df_hop)
            n_models = df_hop["model"].nunique()
            print(f"  {n_eeus} EEUs across {n_models} models after filtering")
            sigs_hop = run_all_signatures(df_hop)
            hop_plot_dir = output_dir / f"plots/hop_{hop_filter}"
            hop_plot_dir.mkdir(parents=True, exist_ok=True)
            # Also build filtered raw_hop_dfs for comparison plots
            raw_hop_dfs_hop = {
                name: df_r[df_r["num_hops"] == hop_filter].copy()
                for name, df_r in raw_hop_dfs.items()
            }
            # Align musique raw hops to musique-natural example IDs
            if "musique" in raw_hop_dfs_hop and "musique-natural" in raw_hop_dfs_hop:
                nat_eids = set(raw_hop_dfs_hop["musique-natural"]["example_id"].unique())
                raw_hop_dfs_hop["musique"] = raw_hop_dfs_hop["musique"][
                    raw_hop_dfs_hop["musique"]["example_id"].isin(nat_eids)
                ].copy()
                print(f"  Aligned musique raw hops to {len(nat_eids)} matched example IDs")
            example_dfs_hop = {
                name: df_e[df_e["num_hops"] == hop_filter].copy()
                for name, df_e in example_dfs.items()
            }
            # Align musique example metrics to musique-natural example IDs
            if "musique" in example_dfs_hop and "musique-natural" in example_dfs_hop:
                nat_eids_ex = set(example_dfs_hop["musique-natural"]["example_id"].unique())
                example_dfs_hop["musique"] = example_dfs_hop["musique"][
                    example_dfs_hop["musique"]["example_id"].isin(nat_eids_ex)
                ].copy()
                print(f"  Aligned musique example metrics to {len(nat_eids_ex)} matched example IDs")
            print(f"Generating {hop_filter}-hop parallel plots → {hop_plot_dir}/")
            plot_quadrant_taxonomy(df_hop, sigs_hop, hop_plot_dir)
            plot_overhead_intensity(df_hop, hop_plot_dir)
            plot_search_budget_efficiency(df_hop, sigs_hop, hop_plot_dir)
            plot_all_or_nothing(df_hop, hop_plot_dir)
            plot_search_decision_roc(df_hop, sigs_hop, hop_plot_dir)
            plot_search_calibration_extended_cmp(raw_hop_dfs_hop, hop_plot_dir, produce_hop_variants=False)
            plot_accuracy_outcomes_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_accuracy_vs_queries_bars_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_accuracy_per_hop_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_searched_vs_unsearched_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_entropy_bucket_queries_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_searches_per_entropy_level_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_searches_per_certainty_level_cmp(raw_hop_dfs_hop, hop_plot_dir)
            plot_redundancy_overlap_cmp(example_dfs_hop, hop_plot_dir)
            plot_sequential_redundancy_distribution_cmp(example_dfs_hop, hop_plot_dir)
            sigs_hop.to_csv(output_dir / f"interplay_signatures_{hop_filter}hop.csv", index=False)
            df_hop.to_csv(output_dir / f"unified_eeu_frame_{hop_filter}hop.csv", index=False)
        else:
            print(f"  WARNING: No EEUs remain after filtering to {hop_filter}-hop — skipping parallel plots.")

    print("Writing outputs...")
    df.to_csv(output_dir / "unified_eeu_frame.csv", index=False)
    sigs.to_csv(output_dir / "interplay_signatures.csv", index=False)
    tests.to_csv(output_dir / "statistical_tests.csv", index=False)

    report = generate_report(sigs, tests, plot_dir)
    (output_dir / "report.md").write_text(report)

    print(f"\nDone. Outputs in {output_dir}/")
    print(f"  {len(sigs)} signature rows, {len(tests)} statistical tests")
    print(f"  Plots: {list(plot_dir.glob('*.png'))}")

    # Print signature summary to stdout
    print("\n── Signature Summary ──")
    display_cols = ["dataset", "model", "frac_E", "frac_PR", "frac_M", "frac_CP",
                    "scc", "auroc", "precision", "recall", "f1", "fdr", "ewoi", "sbe", "sae_eeu", "cov_gap"]
    available = [c for c in display_cols if c in sigs.columns]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.max_columns", 20, "display.width", 160):
        print(sigs[available].to_string(index=False))


if __name__ == "__main__":
    main()
