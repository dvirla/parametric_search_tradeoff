"""Predict per-example search-call counts from question text alone (FRAMES cue grid).

Assembles (model, condition, example_id, problem, search_calls) rows from a frames_cues_full
results tree, extracts hand-crafted text features, and evaluates ridge / GBM regressors with
GroupKFold CV grouped on base example_id (cue paraphrases of one question never straddle
train/test). Optionally adds precomputed text embeddings (see embed_questions.py) as features.

Feature-set ablation per model: cue-condition one-hots only, hand features only, hand+condition,
and (with --embeddings) embeddings only and embeddings+hand.

Usage:
    uv run python scripts/predict_search_calls.py
    uv run python scripts/predict_search_calls.py \
        --embeddings results/search_call_prediction/embeddings_bge-base.npz \
        --output-dir results/search_call_prediction

Budget-exhausted rows (stop_reason set, uniformly search_calls=100) are dropped, matching
summarize_frames_cues_grid.py's "completed" convention.
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

DEFAULT_RESULTS = "results/frames_cues_full"
DEFAULT_CUES_DIR = "data/frames_cues"


# ---------------------------------------------------------------- data assembly

def source_jsonl(condition: str) -> str:
    # Full-run anchors (501 examples). summarize_frames_cues_grid.py uses the 50-example
    # variants for the smoke run; frames_cues_full rows map through the *_full files.
    if condition.startswith("verbose_"):
        return "orig_phrasing_full.jsonl"
    if condition.startswith("terse_"):
        return "neutral_matched_full.jsonl"
    return f"{condition}.jsonl"


def problem_to_id(cues_dir: str, condition: str) -> dict[str, str]:
    path = os.path.join(cues_dir, source_jsonl(condition))
    mapping = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    mapping[str(r["text"]).strip()] = str(r["example_id"])
    return mapping


def load_all(results_dir: str, cues_dir: str) -> pd.DataFrame:
    rows, unmapped = [], {}
    for model_dir in sorted(glob.glob(os.path.join(results_dir, "*", ""))):
        model = os.path.basename(model_dir.rstrip("/"))
        for path in sorted(glob.glob(os.path.join(model_dir, "frames-cues_baseline_*.json"))):
            m = re.match(r"frames-cues_baseline_.+?_((?:terse|verbose)_\w+|epi_strong_\w+)\.json",
                         os.path.basename(path))
            if not m:
                continue
            condition = m.group(1)
            with open(path) as f:
                results = json.load(f)
            p2i = None
            for r in results:
                ex = r.get("example_id")
                if ex is None:
                    if p2i is None:
                        p2i = problem_to_id(cues_dir, condition)
                    ex = p2i.get(str(r.get("problem", "")).strip())
                    if ex is None:
                        unmapped[(model, condition)] = unmapped.get((model, condition), 0) + 1
                        continue
                rows.append({
                    "model": model,
                    "condition": condition,
                    "example_id": str(ex),
                    "problem": str(r.get("problem", "")),
                    "search_calls": int(r.get("sampler_search_calls", 0) or 0),
                    "stopped": bool(r.get("stop_reason")),
                })
    if unmapped:
        print("UNMAPPED rows (dropped):")
        for k, v in sorted(unmapped.items()):
            print(f"  {k}: {v}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- hand features

SUPERLATIVE_RE = re.compile(r"\b(first|last|largest|smallest|oldest|youngest|highest|lowest"
                            r"|longest|shortest|most|least|best|worst|earliest|latest)\b", re.I)
ORDINAL_RE = re.compile(r"\b\d+(st|nd|rd|th)\b")
YEAR_RE = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
HEDGE_RE = re.compile(r"\b(surely|perhaps|maybe|i think|i believe|i'm not sure|not sure"
                      r"|you must know|you probably|of course|obviously|i guess|wondering)\b", re.I)
POLITE_RE = re.compile(r"\b(please|could you|would you|kindly|thanks|thank you|appreciate)\b", re.I)
RELCLAUSE_RE = re.compile(r"\b(whose|which|that|who|where|when)\b", re.I)
NAMED_ENTITY_RE = re.compile(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+")


def featurize(text: str) -> dict:
    words = text.split()
    n_words = max(len(words), 1)
    feats = {
        "n_chars": len(text),
        "n_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / n_words,
        "n_commas": text.count(","),
        "n_of_the": len(re.findall(r"\bof the\b", text, re.I)),
        "n_relclause": len(RELCLAUSE_RE.findall(text)),
        "n_capitalized_mid": len(NAMED_ENTITY_RE.findall(text)),
        "n_digits": sum(c.isdigit() for c in text),
        "n_years": len(YEAR_RE.findall(text)),
        "n_ordinals": len(ORDINAL_RE.findall(text)),
        "n_superlatives": len(SUPERLATIVE_RE.findall(text)),
        "n_quotes": text.count('"') + text.count("“"),
        "n_hedges": len(HEDGE_RE.findall(text)),
        "n_polite": len(POLITE_RE.findall(text)),
        "ends_qmark": int(text.rstrip().endswith("?")),
        "n_sentences": max(len(re.findall(r"[.!?]+", text)), 1),
        "n_and": len(re.findall(r"\band\b", text, re.I)),
    }
    tl = text.lower()
    for wh in ["what", "who", "which", "when", "where", "how many"]:
        feats[f"wh_{wh.replace(' ', '_')}"] = int(wh in tl)
    return feats


# ---------------------------------------------------------------- evaluation

def cv_predict(X: np.ndarray, y: np.ndarray, groups: np.ndarray, kind: str,
               n_splits: int = 5) -> np.ndarray:
    pred = np.zeros_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if kind == "ridge":
            # Standardize only low-dim hand/condition features (heterogeneous scales).
            # Per-dim standardization of L2-normalized embeddings amplifies noise
            # dimensions and destroys the signal — leave embedding blocks unscaled.
            if X.shape[1] <= 64:
                sc = StandardScaler().fit(X[tr])
                Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
            else:
                Xtr, Xte = X[tr], X[te]
            m = RidgeCV(alphas=np.logspace(-2, 6, 17)).fit(Xtr, np.log1p(y[tr]))
            pred[te] = np.expm1(m.predict(Xte))
        elif kind == "gbm":
            m = HistGradientBoostingRegressor(loss="poisson", max_depth=4, max_iter=300,
                                              learning_rate=0.05, random_state=0)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        else:  # mean baseline
            pred[te] = y[tr].mean()
    return pred


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    rho, p = stats.spearmanr(y, pred)
    return {"spearman": rho, "spearman_p": p,
            "mae": float(np.mean(np.abs(y - pred))),
            "r2": float(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))}


def identity_oracle(sub: pd.DataFrame) -> float:
    """Spearman of leave-one-out per-question mean — ceiling for question-level predictors."""
    g = sub.groupby("example_id").search_calls
    s, n = g.transform("sum"), g.transform("count")
    loo = (s - sub.search_calls) / (n - 1).clip(lower=1)
    return float(stats.spearmanr(sub.search_calls, loo)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--cues-dir", default=DEFAULT_CUES_DIR)
    ap.add_argument("--embeddings", default=None,
                    help="npz with arrays 'texts' and 'embeddings' from embed_questions.py")
    ap.add_argument("--regressor", choices=["ridge", "gbm"], default="ridge")
    ap.add_argument("--output-dir", default=None, help="write assembled table + metrics CSVs here")
    args = ap.parse_args()

    df = load_all(args.results_dir, args.cues_dir)
    n0 = len(df)
    df = df[~df.stopped].reset_index(drop=True)
    print(f"assembled {n0} rows; dropped {n0 - len(df)} budget-exhausted; {len(df)} remain "
          f"({df.example_id.nunique()} questions x {df.condition.nunique()} conditions "
          f"x {df.model.nunique()} models)")

    emb_lookup = None
    if args.embeddings:
        z = np.load(args.embeddings, allow_pickle=True)
        emb_lookup = {t: e for t, e in zip(z["texts"], z["embeddings"])}
        missing = sum(1 for t in df.problem.unique() if t not in emb_lookup)
        print(f"embeddings: {len(emb_lookup)} texts, dim={z['embeddings'].shape[1]}, "
              f"{missing} unique problems missing")

    all_rows = []
    for model in sorted(df.model.unique()):
        sub = df[df.model == model].reset_index(drop=True)
        y = sub.search_calls.values.astype(float)
        groups = sub.example_id.values
        Xh = pd.DataFrame([featurize(t) for t in sub.problem]).values.astype(float)
        # Pre-standardize hand features (text statistics only — no target involved) so the
        # embed+hand concatenation doesn't mix raw counts with unit-norm embedding dims.
        Xh = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-9)
        Xc = pd.get_dummies(sub.condition).values.astype(float)
        feature_sets = {
            "condition_only": Xc,
            "hand_only": Xh,
            "hand+condition": np.hstack([Xh, Xc]),
        }
        if emb_lookup is not None:
            ok = sub.problem.isin(emb_lookup).values
            if not ok.all():  # evaluate embeddings on the covered subset only
                print(f"  [{model}] {len(ok) - ok.sum()} rows lack embeddings — excluded "
                      f"from embedding feature sets")
            Xe = np.stack([emb_lookup[t] for t in sub.problem[ok]])
            feature_sets["embed_only"] = (Xe, y[ok], groups[ok])
            feature_sets["embed+hand"] = (np.hstack([Xe, Xh[ok]]), y[ok], groups[ok])

        print(f"\n### {model}  (n={len(y)}, mean={y.mean():.2f}, sd={y.std():.2f}, "
              f"identity-oracle rho={identity_oracle(sub):.3f})")
        base = metrics(y, cv_predict(Xh, y, groups, "baseline"))
        print(f"{'features':<18}{'spearman':>10}{'MAE':>8}{'R2':>8}")
        print(f"{'mean_baseline':<18}{'--':>10}{base['mae']:>8.2f}{base['r2']:>8.3f}")
        all_rows.append({"model": model, "features": "mean_baseline", **base})
        for name, spec in feature_sets.items():
            X, yy, gg = spec if isinstance(spec, tuple) else (spec, y, groups)
            m = metrics(yy, cv_predict(X, yy, gg, args.regressor))
            print(f"{name:<18}{m['spearman']:>10.3f}{m['mae']:>8.2f}{m['r2']:>8.3f}")
            all_rows.append({"model": model, "features": name, **m})

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        df.to_csv(os.path.join(args.output_dir, "frames_search_calls_table.csv"), index=False)
        pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, "prediction_metrics.csv"),
                                      index=False)
        print(f"\nwrote table + metrics to {args.output_dir}/")


if __name__ == "__main__":
    main()
