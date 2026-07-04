#!/usr/bin/env python3
"""Collapsed cue-FAMILY star plots for FRAMES prompt-cue sensitivity.

Collapses the 24 paired contrasts of Table 2 into 7 mechanism families and draws
one star (radar) plot per model whose vertices are the families.

Combining significance *properly* (not by averaging p-values):

  For each contrast i in a family we take the paired per-example delta vector
  d_i = search_calls(condition_a) − search_calls(condition_b) over the example_ids
  common to that pair, its two-sided Wilcoxon p_i, and a *signed* z-score
        z_i = sign(mean d_i) · Φ⁻¹(1 − p_i/2).
  Contrasts that agree in direction reinforce; contrasts that conflict cancel — so
  the axis measures *consistent directional* evidence for the cue.

  The contrasts in a family are NOT independent (they share the same examples and the
  PLAIN baseline), so a naive Stouffer/Fisher combination is anti-conservative. We
  apply a dependency correction using the empirical correlation ρ_ij between the
  per-example delta vectors (intersected ids):
        Z_fam = Σ z_i / sqrt( k + 2 Σ_{i<j} ρ_ij )
        p_fam = 2·(1 − Φ(|Z_fam|))   (two-sided)

  Radius on the star = −log10(p_fam); a dashed ring marks p=0.05. Vertex colour =
  sign(Z_fam): red = the cue reduces search, blue = increases search.

The 6 pairwise *dissociation* contrasts (POLITE−NATURAL, DIRECT−NATURAL,
DIRECT−POLITE) are differences of main effects, not a single cue vs a baseline, so
they are intentionally excluded from the collapse and reported separately.

Usage:
    uv run python scripts/plot_frames_cue_family_stars.py \
        --results-root results/frames_cues \
        --output-dir results/frames_cues/star_plots
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_frames_cues_grid import load_condition, _wilcoxon_p  # noqa: E402

P_THRESHOLD = 0.05
THRESH_R = -math.log10(P_THRESHOLD)  # 1.301
DOWN_COLOR = "#ca0020"  # reduces search
UP_COLOR = "#0571b0"    # increases search

# Each family: list of (condition_a, condition_b) contrasts; Δ = a − b.
# Order chosen so the star reads roughly weakest→strongest cue clockwise.
FAMILIES: dict[str, list[tuple[str, str]]] = {
    "Epistemic\n(boost/hedge − neutral)": [
        ("epi_strong_boost", "terse_plain"),
        ("epi_strong_hedge", "terse_plain"),
    ],
    "Phrasing\n(terse − verbose)": [
        ("terse_plain", "verbose_plain"),
        ("terse_natural", "verbose_natural"),
        ("terse_elaborate", "verbose_elaborate"),
        ("terse_query", "verbose_query"),
    ],
    "Structured\n(QUERY − PLAIN)": [
        ("verbose_query", "verbose_plain"),
        ("terse_query", "terse_plain"),
    ],
    "Length directive\n(NAT/ELAB − PLAIN)": [
        ("verbose_natural", "verbose_plain"),
        ("terse_natural", "terse_plain"),
        ("verbose_elaborate", "verbose_plain"),
        ("terse_elaborate", "terse_plain"),
    ],
    "Output length\n(ELAB − NAT)": [
        ("verbose_elaborate", "verbose_natural"),
        ("terse_elaborate", "terse_natural"),
    ],
    "Politeness\n(POLITE − PLAIN)": [
        ("verbose_polite", "verbose_plain"),
        ("terse_polite", "terse_plain"),
    ],
    "Answer-directive\n(DIRECT − PLAIN)": [
        ("verbose_direct", "verbose_plain"),
        ("terse_direct", "terse_plain"),
    ],
}

# Reported alongside but NOT collapsed onto the star (differences of main effects).
DISSOCIATION = {
    "POLITE − NATURAL": [("verbose_polite", "verbose_natural"), ("terse_polite", "terse_natural")],
    "DIRECT − NATURAL": [("verbose_direct", "verbose_natural"), ("terse_direct", "terse_natural")],
    "DIRECT − POLITE": [("verbose_direct", "verbose_polite"), ("terse_direct", "terse_polite")],
}

def _slugify(dirname: str) -> str:
    """Model dir name → slug used in titles/filenames (matches SUMMARY_<slug>.md)."""
    return dirname.replace("-", "_").replace(".", "_")


def discover_model_dirs(root: Path) -> dict[str, str]:
    """{slug: dirname} for every model subdir under ``root`` that holds result JSONs.

    Auto-discovery (instead of a hardcoded map) means new models — e.g.
    nemotron-3-super — are picked up automatically as soon as their results land.
    """
    out: dict[str, str] = {}
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        if sub.name == "star_plots":
            continue
        if not any(sub.glob("frames-cues_baseline_*.json")):
            continue
        out[_slugify(sub.name)] = sub.name
    return out


def _diff_map(a: dict, b: dict) -> dict[str, float]:
    """{example_id: search_calls(a) − search_calls(b)} over common ids."""
    ids = set(a) & set(b)
    return {i: a[i]["search_calls"] - b[i]["search_calls"] for i in ids}


def _signed_z(a: dict, b: dict) -> tuple[float, float, float, dict[str, float]]:
    """Return (signed z, two-sided p, mean delta, diff map) for one contrast."""
    dmap = _diff_map(a, b)
    ids = sorted(dmap)
    av = np.array([a[i]["search_calls"] for i in ids], float)
    bv = np.array([b[i]["search_calls"] for i in ids], float)
    p = _wilcoxon_p(av, bv)
    mean_d = float((av - bv).mean()) if len(ids) else 0.0
    if p >= 1.0 or mean_d == 0.0:
        z = 0.0
    else:
        # isf is accurate for tiny p (ppf(1 - p/2) saturates to inf below ~1e-16,
        # which large-n Wilcoxon p-values routinely reach)
        z = math.copysign(norm.isf(max(p, 1e-300) / 2), mean_d)
    return z, p, mean_d, dmap


def combine_family(conds: dict, members: list[tuple[str, str]]) -> dict | None:
    """Dependency-corrected signed-Stouffer combination across a family's contrasts."""
    zs, ps, means, dmaps = [], [], [], []
    for ca, cb in members:
        if conds.get(ca) is None or conds.get(cb) is None:
            continue
        z, p, m, dmap = _signed_z(conds[ca], conds[cb])
        zs.append(z)
        ps.append(p)
        means.append(m)
        dmaps.append(dmap)
    k = len(zs)
    if k == 0:
        return None

    # pairwise empirical correlation between per-example delta vectors (intersected ids)
    rho_sum = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            common = sorted(set(dmaps[i]) & set(dmaps[j]))
            if len(common) >= 3:
                vi = np.array([dmaps[i][c] for c in common], float)
                vj = np.array([dmaps[j][c] for c in common], float)
                if vi.std() > 0 and vj.std() > 0:
                    rho_sum += float(np.corrcoef(vi, vj)[0, 1])
    denom_var = max(k + 2 * rho_sum, 1e-6)
    z_fam = sum(zs) / math.sqrt(denom_var)
    p_fam = 2 * (1 - norm.cdf(abs(z_fam)))
    p_fam = min(max(p_fam, 1e-12), 1.0)
    return {
        "k": k,
        "z_fam": z_fam,
        "p_fam": p_fam,
        "mean_delta": float(np.mean(means)),
        "rho_sum": rho_sum,
        "members_p": ps,
        "members_mean": means,
    }


def load_conditions(results_dir: Path, cues_dir: str) -> dict:
    needed = set()
    for fam in list(FAMILIES.values()) + list(DISSOCIATION.values()):
        for ca, cb in fam:
            needed.add(ca)
            needed.add(cb)
    return {c: load_condition(str(results_dir), c, cues_dir) for c in needed}


def plot_star(fam_results: dict, model: str, ax: plt.Axes) -> None:
    labels = [k for k in FAMILIES if fam_results.get(k)]
    n = len(labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    scores = np.array([-math.log10(fam_results[k]["p_fam"]) for k in labels])
    closed_s = np.concatenate([scores, scores[:1]])
    closed_a = np.concatenate([angles, angles[:1]])

    rmax = max(scores.max() * 1.15, THRESH_R + 0.6)

    ax.plot(closed_a, closed_s, color="#444444", lw=1.5, zorder=3)
    ax.fill(closed_a, closed_s, color="#999999", alpha=0.15, zorder=1)

    ring_ang = np.linspace(0, 2 * np.pi, 200)
    ax.plot(ring_ang, np.full(200, THRESH_R), color="green", ls="--", lw=1.0, alpha=0.7, zorder=2)

    for ang, sc, k in zip(angles, scores, labels):
        r = fam_results[k]
        color = DOWN_COLOR if r["z_fam"] < 0 else UP_COLOR
        sig = r["p_fam"] < P_THRESHOLD
        ax.scatter(ang, sc, s=120 if sig else 55, color=color,
                   edgecolors="black" if sig else "none", linewidths=1.0, zorder=5)

    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, rmax)
    ax.set_rlabel_position(180 / n)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_title(model, fontsize=13, pad=26, fontweight="bold")
    ax.text(np.pi / 2, THRESH_R, "p=0.05", color="green", fontsize=7, ha="center", va="bottom")


def fmt_p(p: float) -> str:
    return f"{p:.2g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/frames_cues")
    ap.add_argument("--output-dir", default="results/frames_cues/star_plots")
    ap.add_argument("--cues-dir", default="data/frames_cues")
    args = ap.parse_args()

    root = Path(args.results_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = discover_model_dirs(root)
    if not model_dirs:
        raise SystemExit(f"No model result subdirs found under {root}")

    per_model: dict[str, dict] = {}
    per_model_diss: dict[str, dict] = {}
    for model, dirname in model_dirs.items():
        rdir = root / dirname
        conds = load_conditions(rdir, args.cues_dir)
        per_model[model] = {fam: combine_family(conds, mem) for fam, mem in FAMILIES.items()}
        per_model_diss[model] = {name: combine_family(conds, mem) for name, mem in DISSOCIATION.items()}

    legend_handles = [
        plt.Line2D([], [], marker="o", ls="", color=DOWN_COLOR, markeredgecolor="black",
                   label="↓ reduces search (Z<0)"),
        plt.Line2D([], [], marker="o", ls="", color=UP_COLOR, markeredgecolor="black",
                   label="↑ increases search (Z>0)"),
        plt.Line2D([], [], color="green", ls="--", label="p=0.05 threshold"),
        plt.Line2D([], [], marker="o", ls="", color="gray", markeredgecolor="black",
                   label="ringed = significant (p_fam<0.05)"),
    ]

    # individual
    for model, fam_results in per_model.items():
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        plot_star(fam_results, model, ax)
        fig.legend(handles=legend_handles, loc="lower center", ncol=2, fontsize=8,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.suptitle("FRAMES cue-family significance star — radius = −log₁₀(p_fam), "
                     "dependency-corrected signed Stouffer", fontsize=10, y=0.98)
        fig.tight_layout(rect=(0, 0.04, 1, 0.96))
        out = out_dir / f"family_star_{model}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out}")

    # grid
    n_models = len(per_model)
    ncols = 2
    nrows = math.ceil(n_models / ncols)
    fig = plt.figure(figsize=(8 * ncols, 8 * nrows))
    for i, (model, fam_results) in enumerate(per_model.items()):
        ax = fig.add_subplot(nrows, ncols, i + 1, polar=True)
        plot_star(fam_results, model, ax)
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("FRAMES cue-family significance stars — 7 collapsed cue families · "
                 "radius = −log₁₀(p_fam) (dependency-corrected signed Stouffer) · "
                 "colour = effect direction on search_calls", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    out = out_dir / "family_star_grid_all_models.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")

    # numbers table
    md = ["# FRAMES cue-family collapse — combined significance",
          "",
          "Per family: dependency-corrected **signed Stouffer** combination of its member "
          "contrasts (Δ on `search_calls`). `Z_fam>0` ⇒ cue **increases** search; `<0` ⇒ "
          "**reduces**. `p_fam` is two-sided. Correction inflates the Stouffer denominator by the "
          "empirical between-contrast correlation (Σρ) — contrasts share the same examples + PLAIN "
          "baseline, so this guards against anti-conservative combination.",
          ""]
    for model, fam_results in per_model.items():
        md.append(f"## {model}")
        md.append("")
        md.append("| Family | k | Z_fam | p_fam | dir | mean meanΔ | Σρ |")
        md.append("|---|---|---|---|---|---|---|")
        for fam in FAMILIES:
            r = fam_results.get(fam)
            if not r:
                continue
            label = fam.replace("\n", " ")
            direction = "↑ more" if r["z_fam"] > 0 else "↓ less"
            sig = " **sig**" if r["p_fam"] < P_THRESHOLD else ""
            md.append(f"| {label} | {r['k']} | {r['z_fam']:+.2f} | {fmt_p(r['p_fam'])}{sig} | "
                      f"{direction} | {r['mean_delta']:+.2f} | {r['rho_sum']:+.2f} |")
        md.append("")
        md.append("Dissociation contrasts (not on star):")
        md.append("")
        md.append("| Contrast | k | Z | p | dir |")
        md.append("|---|---|---|---|---|")
        for name, r in per_model_diss[model].items():
            if not r:
                continue
            direction = "↑" if r["z_fam"] > 0 else "↓"
            sig = " **sig**" if r["p_fam"] < P_THRESHOLD else ""
            md.append(f"| {name} | {r['k']} | {r['z_fam']:+.2f} | {fmt_p(r['p_fam'])}{sig} | {direction} |")
        md.append("")
    out = out_dir / "FAMILY_COMBINED.md"
    out.write_text("\n".join(md))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
