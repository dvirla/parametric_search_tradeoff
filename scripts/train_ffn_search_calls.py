"""Per-model feed-forward regressor: predict search-call count from a prompt embedding.

Trains one small torch MLP per model on the FRAMES cue grid, embeddings-only, and reports
out-of-fold Spearman / MAE / R2 under GroupKFold grouped on base example_id (cue paraphrases of
one question never straddle train/test — the anti-leakage guarantee; ungrouped CV inflates
embedding scores ~4x via question-identity leakage).

Headline comparison: for each model it runs the FFN on
  - `templated`   embeddings of the reconstructed prompt   (looked up by the `prompt` column)
  - `untemplated` embeddings of the bare question `problem` (looked up by the `problem` column)
side by side, plus the identity-oracle ceiling and (if available) the prior hand-feature Spearman
from prediction_metrics.csv — i.e. does giving the embedder the runtime output-directive cue add
predictive signal.

Target is a heavy-tailed count: trained on log1p, reported on the expm1 scale; Spearman + MAE are
the headline metrics (R2 ~ 0 expected). L2-normalized embedding dims are fed through a learnable
input LayerNorm — NOT per-dim StandardScaler, which prior work showed amplifies noise dims and
destroys the signal.

Usage:
    uv run python scripts/train_ffn_search_calls.py \
        --table results/search_call_prediction/frames_reconstructed_table.csv \
        --templated-emb   results/search_call_prediction/embeddings_qwen3-8b_templated.npz \
        --untemplated-emb results/search_call_prediction/embeddings_qwen3-8b_untemplated.npz \
        --out results/search_call_prediction/ffn_metrics_qwen3_8b.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.predict_search_calls import metrics, identity_oracle  # noqa: E402


# ---------------------------------------------------------------- model

class FFN(nn.Module):
    def __init__(self, in_dim: int, hidden=(512, 128), dropout=0.2):
        super().__init__()
        layers = [nn.LayerNorm(in_dim)]
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_one(Xtr, ytr, Xva, yva, in_dim, device, epochs, patience, lr, wd, batch):
    """Train an FFN on log1p targets with early stopping on val MSE; return best model."""
    model = FFN(in_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.as_tensor(np.log1p(ytr), dtype=torch.float32, device=device)
    Xva_t = torch.as_tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(np.log1p(yva), dtype=torch.float32, device=device)

    n = len(Xtr_t)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(Xva_t), yva_t).item()
        if v < best_val - 1e-5:
            best_val, best_state, bad = v, {k: t.detach().clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        p = model(torch.as_tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
    return np.expm1(p)  # back to count scale


# ---------------------------------------------------------------- CV driver

def cv_ffn(X, y, groups, device, args):
    """Out-of-fold predictions from per-fold FFNs (GroupKFold on example_id)."""
    oof = np.zeros_like(y, dtype=float)
    gkf = GroupKFold(n_splits=args.folds)
    for tr, te in gkf.split(X, y, groups):
        # inner group-aware val split for early stopping
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=0)
        itr, iva = next(gss.split(X[tr], y[tr], groups[tr]))
        model = train_one(X[tr][itr], y[tr][itr], X[tr][iva], y[tr][iva],
                          in_dim=X.shape[1], device=device, epochs=args.epochs,
                          patience=args.patience, lr=args.lr, wd=args.weight_decay,
                          batch=args.batch_size)
        oof[te] = predict(model, X[te], device)
    return oof


def load_emb(path):
    z = np.load(path, allow_pickle=True)
    return {str(t): np.asarray(e, dtype=np.float32) for t, e in zip(z["texts"], z["embeddings"])}


def build_X(sub, lookup, text_col):
    """Return (X, mask) where mask marks rows whose text has an embedding."""
    texts = sub[text_col].astype(str).values
    mask = np.array([t in lookup for t in texts])
    X = np.stack([lookup[t] for t in texts[mask]]) if mask.any() else np.empty((0, 0))
    return X, mask


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default="results/search_call_prediction/frames_reconstructed_table.csv")
    ap.add_argument("--templated-emb", required=True,
                    help="npz of embeddings of the reconstructed `prompt` column")
    ap.add_argument("--untemplated-emb", default=None,
                    help="npz of embeddings of the bare `problem` column (baseline comparison)")
    ap.add_argument("--prior-metrics", default="results/search_call_prediction/prediction_metrics.csv",
                    help="prior hand-feature metrics for context (optional)")
    ap.add_argument("--out", default="results/search_call_prediction/ffn_metrics_qwen3_8b.csv")
    ap.add_argument("--ckpt-dir", default="results/search_call_prediction/ffn_ckpts")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda | cpu (auto-detect if unset)")
    ap.add_argument("--models", default=None,
                    help="comma-separated subset of model names to run (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    df = pd.read_csv(args.table)
    sources = {"templated": (args.templated_emb, "prompt")}
    if args.untemplated_emb:
        sources["untemplated"] = (args.untemplated_emb, "problem")
    lookups = {name: load_emb(path) for name, (path, _) in sources.items()}

    prior = {}
    if os.path.exists(args.prior_metrics):
        pm = pd.read_csv(args.prior_metrics)
        for _, r in pm[pm.features == "hand+condition"].iterrows():
            prior[str(r["model"])] = float(r["spearman"])

    os.makedirs(args.ckpt_dir, exist_ok=True)
    model_names = sorted(df.model.unique())
    if args.models:
        wanted = {m.strip() for m in args.models.split(",")}
        model_names = [m for m in model_names if m in wanted]
    rows = []
    for model_name in model_names:
        sub = df[df.model == model_name].reset_index(drop=True)
        y = sub.search_calls.values.astype(float)
        groups = sub.example_id.astype(str).values
        oracle = identity_oracle(sub)
        print(f"\n### {model_name}  (n={len(sub)}, mean={y.mean():.2f}, "
              f"oracle_rho={oracle:.3f}, prior_hand+cond={prior.get(model_name, float('nan')):.3f})")
        print(f"{'source':<14}{'spearman':>10}{'MAE':>8}{'R2':>8}{'n':>7}")
        for name, (_, text_col) in sources.items():
            X, mask = build_X(sub, lookups[name], text_col)
            if mask.sum() < len(sub):
                print(f"  [{name}] {len(sub) - int(mask.sum())} rows lack embeddings — excluded")
            yy, gg = y[mask], groups[mask]
            oof = cv_ffn(X, yy, gg, device, args)
            m = metrics(yy, oof)
            print(f"{name:<14}{m['spearman']:>10.3f}{m['mae']:>8.2f}{m['r2']:>8.3f}{len(yy):>7}")
            rows.append({"model": model_name, "source": name, "n": len(yy),
                         "identity_oracle": oracle, "prior_hand_cond": prior.get(model_name),
                         **m})
            # deployable checkpoint: final FFN trained on all rows of this source
            gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=0)
            itr, iva = next(gss.split(X, yy, gg))
            final = train_one(X[itr], yy[itr], X[iva], yy[iva], in_dim=X.shape[1], device=device,
                              epochs=args.epochs, patience=args.patience, lr=args.lr,
                              wd=args.weight_decay, batch=args.batch_size)
            slug = model_name.replace("/", "_").replace(":", "_")
            torch.save({"state_dict": final.state_dict(), "in_dim": X.shape[1],
                        "source": name, "model": model_name},
                       os.path.join(args.ckpt_dir, f"ffn_{slug}_{name}.pt"))

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote metrics to {args.out}; checkpoints in {args.ckpt_dir}/")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
