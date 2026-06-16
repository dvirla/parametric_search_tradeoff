"""
Shared visualization + analysis helpers for the paper figures.

Single source of truth for:
  - the 4-colour ColorBrewer RdBu diverging palette used across every figure,
  - the matplotlib/seaborn theme,
  - model slug -> display-name maps,
  - the statistics helpers (Wilson CI, two-proportion test, Mantel-Haenszel, ...),
  - the E/CP/PR/M per-hop cell assignment and a light cell-fraction roll-up,
  - loaders for interplay_summary.csv and matched_examples_*.json.

Everything that `scripts/make_paper_figures.py` needs lives here so the figure
script stays declarative and no palette/stat/cell logic is duplicated.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# ─── Palette (ColorBrewer RdYlBu, 6-class diverging) ────────────────────────────
# Diverging good -> bad: blues = correct decisions, yellow/orange/red = errors.
# Six distinct colours let every category get its own fill, so figures distinguish
# sub-categories by hue rather than by hatching/dashed fills.
BLUE       = "#4575b4"   # strong blue   (best)
LIGHT_BLUE = "#91bfdb"   # light blue
PALE_BLUE  = "#e0f3f8"   # pale blue
YELLOW     = "#fee090"   # yellow
ORANGE     = "#fc8d59"   # orange
RED        = "#d73027"   # strong red    (worst)
SALMON     = ORANGE      # backward-compatible alias

# Semantic aliases (so figure code reads intent, not hex). The six-cell taxonomy
# maps cleanly onto the six-class scale, good -> bad:
EFFECTIVE      = BLUE        # E  — searched an uncertain hop (correct)
CP             = LIGHT_BLUE  # CP — correctly skipped a certain hop (correct)
CROSS_HOP      = PALE_BLUE   # XH — uncertain hop resolved via a sibling hop's search
PR             = YELLOW      # PR — wasted search on a certain hop (mild error)
CASCADE        = ORANGE      # Mc — missed, but an upstream hop was already wrong
MISSED         = RED         # M / Mg — genuinely skipped an uncertain hop (error)

BENCHMARK      = BLUE        # MuSiQue / formal phrasing
NATURAL        = RED         # user-like phrasing

GENUINE_MISSED = RED         # missed with all prior hops correct
# (CASCADE, above, doubles as the cascade-possible colour)

HIGH_COST      = RED         # missed hop that was answered wrong in-trace
LOW_COST       = LIGHT_BLUE  # missed hop that was answered right anyway

QUADRANT_COLORS = {"E": EFFECTIVE, "CP": CP, "PR": PR, "M": MISSED}

# Greys reused for error bars / annotations
ERR_DARK = "#444444"
ERR_MUTE = "#777777"

# ─── Model identity ─────────────────────────────────────────────────────────────
SLUG_TO_LABEL = {
    "gemini-3-pro-preview": "Gemini 3.1 Pro",
    "nemotron-3-nano_30b":  "Nemotron Nano 30B",
    "nemotron-3-nano":      "Nemotron Nano 30B",
    "nemotron-3-nano-musique-v3-aug_latest": "Nemotron Nano (SFT)",
    "qwen3.5_122b":         "Qwen 3.5 122B",
}
# The three production models, in the order figures should present them.
MODEL_ORDER = ["Gemini 3.1 Pro", "Nemotron Nano 30B", "Qwen 3.5 122B"]
BASE_MODEL_SLUGS = ["gemini-3-pro-preview", "nemotron-3-nano_30b", "qwen3.5_122b"]


def display_name(slug: str) -> str:
    return SLUG_TO_LABEL.get(slug, slug)


# ─── Theme ──────────────────────────────────────────────────────────────────────
def apply_theme() -> None:
    """Headless backend + consistent seaborn whitegrid theme."""
    import matplotlib
    matplotlib.use("Agg")
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=1.1)


def savefig(fig, output_dir, name: str, exts=("png", "pdf")) -> None:
    """Save a figure under output_dir as <name>.<ext> for each ext, then close it."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(output_dir / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved {output_dir}/{name}.{'/'.join(exts)}")


# ─── Statistics ─────────────────────────────────────────────────────────────────
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a binomial proportion. Returns (lo, hi)."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def sig_stars(p: float) -> str:
    """APA-style significance stars; 'ns' when not significant, '' when undefined."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


# alias kept for call sites that historically said pval_stars
pval_stars = sig_stars


def two_prop_p(k1, n1, k2, n2) -> float:
    """Two-sided two-proportion z-test p-value."""
    if min(n1, n2) == 0:
        return float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (p1 - p2) / se
    return float(2 * (1 - sp_stats.norm.cdf(abs(z))))


def mantel_haenszel(strata: list[tuple[int, int, int, int]]) -> tuple[float, float]:
    """Cochran-Mantel-Haenszel common odds ratio + p-value across strata.

    Each stratum is (a, b, c, d) = (correct|exposed, wrong|exposed,
    correct|unexposed, wrong|unexposed). OR<1 means exposure lowers P(correct).
    """
    num_or = den_or = 0.0
    sum_a = sum_ea = sum_var = 0.0
    for a, b, c, d in strata:
        n = a + b + c + d
        if n == 0:
            continue
        num_or += a * d / n
        den_or += b * c / n
        sum_a += a
        sum_ea += (a + b) * (a + c) / n
        sum_var += ((a + b) * (c + d) * (a + c) * (b + d)) / (n**2 * (n - 1)) if n > 1 else 0
    or_mh = num_or / den_or if den_or > 0 else float("nan")
    if sum_var > 0:
        chi2 = (abs(sum_a - sum_ea) - 0.5) ** 2 / sum_var
        p = float(1 - sp_stats.chi2.cdf(chi2, df=1))
    else:
        p = float("nan")
    return or_mh, p


def mean_ci95(arr: np.ndarray) -> tuple[float, float]:
    """Return (mean, 95% CI half-width) using the t-distribution."""
    arr = np.asarray(arr)
    n = len(arr)
    if n < 2:
        return (float(arr.mean()) if n else float("nan"), float("nan"))
    sem = arr.std(ddof=1) / np.sqrt(n)
    t_crit = sp_stats.t.ppf(0.975, df=n - 1)
    return float(arr.mean()), float(t_crit * sem)


# ─── Per-hop cell assignment ────────────────────────────────────────────────────
def assign_quadrants(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'quadrant' column (E, M, PR, CP) from `certain` x `search_attributed`.

        E  = uncertain & searched      (correct: search was needed)
        M  = uncertain & not searched  (error:   gap left open)
        PR = certain   & searched      (error:   wasted search)
        CP = certain   & not searched  (correct: correctly skipped)
    """
    df = df.copy()
    conditions = [
        (~df["certain"]) & df["search_attributed"],
        (~df["certain"]) & (~df["search_attributed"]),
        df["certain"] & df["search_attributed"],
        df["certain"] & (~df["search_attributed"]),
    ]
    df["quadrant"] = np.select(conditions, ["E", "M", "PR", "CP"], default="?")
    return df


def cell_fractions(df: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, model) fraction of hops in each cell.

    Returns columns: dataset, model, n, frac_E, frac_CP, frac_PR, frac_M.
    Expects `quadrant` already assigned.
    """
    rows = []
    for (dataset, model), sub in df.groupby(["dataset", "model"]):
        n = len(sub)
        rows.append({
            "dataset": dataset, "model": model, "n": n,
            **{f"frac_{q}": (sub["quadrant"] == q).sum() / n if n else 0.0
               for q in ["E", "CP", "PR", "M"]},
        })
    return pd.DataFrame(rows)


def add_cost_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Add uncertain / cross_hop_covered / truly_missed / effective flags.

    `truly_missed` = uncertain & not searched & not covered by another hop's search.
    `effective`    = uncertain & searched.
    """
    df = df.copy()
    df["uncertain"] = ~df["parametric_certain"].astype(bool)
    df["search_attributed"] = df["search_attributed"].astype(bool)
    df["cross_hop_covered"] = df["missed_search_answer_found_cross_hop"].astype(bool)
    df["truly_missed"] = df["uncertain"] & (~df["search_attributed"]) & (~df["cross_hop_covered"])
    df["effective"] = df["uncertain"] & df["search_attributed"]
    return df


# ─── Loaders ──────────────────────────────────────────────────────────────────--
VAL_SET_SIZE = 600  # first N records in each uncertainty JSON are the validation set

# Authoritative per-hop entropy — overrides stale interplay_summary.csv values so the
# benchmark and natural variants use identical entropy for the same (model, ex, hop).
CANONICAL_ENTROPY_JSONS = {
    "gemini-3-pro-preview": "results/musique_parametric/musique_parametric_uncertainty_gemini-3-pro-preview.json",
    "nemotron-3-nano_30b":  "results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano_30b.json",
    "nemotron-3-nano-musique-v3-aug_latest": "results/musique_parametric/musique_parametric_uncertainty_nemotron-3-nano-musique-v3-aug_latest.json",
    "qwen3.5_122b":         "results/musique_parametric/musique_parametric_uncertainty_qwen3.5_122b.json",
}

_CANONICAL_ENTROPY: dict | None = None


def load_canonical_entropy(jsons: dict[str, str] | None = None) -> dict[tuple[str, str, int], float]:
    """Build {(model, example_id, hop_index): semantic_entropy} from uncertainty JSONs.

    When the SQLite results store exists and no explicit ``jsons`` override is
    given, this delegates to ``results_db.load_canonical_entropy`` (the same
    values, sourced from the ingested parametric runs). It falls back to reading
    the JSON files directly when the DB is absent, so behaviour is unchanged on
    machines that have not run ``scripts/ingest_results.py`` yet.
    """
    global _CANONICAL_ENTROPY
    if jsons is None and _CANONICAL_ENTROPY is not None:
        return _CANONICAL_ENTROPY
    if jsons is None and os.path.exists("results/results.db"):
        try:
            from src import results_db as _rdb  # lazy: avoids circular import
            conn = _rdb.connect()
            canon = _rdb.load_canonical_entropy(conn)
            conn.close()
            if canon:
                _CANONICAL_ENTROPY = canon
                return canon
        except Exception:
            pass  # fall back to reading the JSON files directly
    jsons = jsons or CANONICAL_ENTROPY_JSONS
    canon: dict[tuple[str, str, int], float] = {}
    for slug, path in jsons.items():
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for rec in data[:VAL_SET_SIZE]:
            eid = rec.get("example_id", "")
            for sq in rec.get("sub_questions_results", []):
                hop = sq.get("hop_index")
                ent = sq.get("semantic_entropy")
                if hop is not None and ent is not None:
                    canon[(slug, eid, int(hop))] = float(ent)
    if jsons is CANONICAL_ENTROPY_JSONS or jsons == CANONICAL_ENTROPY_JSONS:
        _CANONICAL_ENTROPY = canon
    return canon


def load_interplay_summary(path: str, dataset_label: str,
                           override_entropy: bool = True,
                           certain_rule: str = "joint") -> pd.DataFrame:
    """Load an interplay_summary.csv and attach entropy/search/certain/quadrant.

    Keeps all original columns (so downstream code can use parametric_certain,
    missed_search_answer_found_cross_hop, aggregate_correct, num_hops, ...), and
    adds: dataset, entropy, search_attributed (bool), certain, quadrant.

    `certain_rule` selects how parametric knowledge ("certain") is defined:
      - "joint" (default): entropy == 0 AND the parametric probe marked the hop
        certain (which itself bakes in all-runs-correct). Reproduces prior figures.
      - "entropy": semantic entropy alone (entropy == 0). The `parametric_certain`
        column is overridden to match so *every* figure (calibration, cost-cells,
        quadrant) adopts the entropy-only rule consistently.
    """
    df = pd.read_csv(path)
    df["model"] = df["model"].str.replace(":", "_", regex=False)
    df["dataset"] = dataset_label
    df["entropy"] = df["semantic_entropy"]
    df["search_attributed"] = df["searched"].map(
        {True: True, False: False, "True": True, "False": False, 1: True, 0: False}
    ).fillna(False).astype(bool)

    if override_entropy:
        canon = load_canonical_entropy()
        if canon:
            def _ovr(row):
                hop = int(row["hop_index"]) if pd.notna(row["hop_index"]) else -1
                return canon.get((row["model"], row["example_id"], hop), row["entropy"])
            df["entropy"] = df.apply(_ovr, axis=1)

    if certain_rule == "entropy":
        # Parametric knowledge from semantic entropy alone. Override the
        # parametric_certain column too so calibration / cost-cell figures
        # (which read it directly) adopt the same rule.
        df["parametric_certain"] = (df["entropy"] == 0.0)
        df["certain"] = (df["entropy"] == 0.0)
    else:  # "joint" — zero entropy AND the parametric probe marked the hop certain.
        df["certain"] = (df["entropy"] == 0.0) & (df["parametric_certain"].astype(str) == "True")
    return assign_quadrants(df)


def load_search_counts_by_example(path: str) -> tuple[str, dict[str, int]]:
    """From a matched_examples_<model>.json -> (model_slug, {example_id: n_queries})."""
    with open(path) as f:
        data = json.load(f)
    model = (data[0]["model"] if data else "unknown").replace(":", "_")
    return model, {ex["example_id"]: len(ex.get("query_records", [])) for ex in data}


def paired_search_counts(a_map: dict[str, int], b_map: dict[str, int]):
    """Align two {example_id: count} maps -> (a_arr, b_arr, common_ids)."""
    common = sorted(set(a_map) & set(b_map))
    return (np.array([a_map[e] for e in common]),
            np.array([b_map[e] for e in common]),
            common)
