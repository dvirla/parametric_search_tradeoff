"""Per-model feed-forward regressor: predict search-call count from a prompt embedding
(optionally fused with hand-crafted + template features).

Trains one small torch MLP per model on the FRAMES cue grid and reports out-of-fold Spearman /
MAE / R2 under GroupKFold grouped on base example_id (cue paraphrases of one question never
straddle train/test — the anti-leakage guarantee; ungrouped CV inflates scores ~4x via
question-identity leakage).

Two feature regimes:
  (default, embeddings-only) compares
    - `templated`   embeddings of the reconstructed prompt   (looked up by the `prompt` column)
    - `untemplated` embeddings of the bare question `problem` (looked up by the `problem` column)
  (--with-hand) adds a hand/template feature block and a TWO-TOWER fusion model, comparing
    - `templated`       (embedding only)
    - `templated+hand`  (embedding fused with hand+template features)
    - `hand_only`       (hand+template features only, no embedding)

Fusion weighting: a raw concat of ~42 hand dims with a 4096-dim embedding lets the embedding
dominate. Instead each modality gets its own learnable projection — embedding -> LayerNorm ->
Linear(emb_proj=256), hand -> Linear(hand_proj=64) — then the projected vectors are concatenated
into the trunk. emb_proj/hand_proj set the balance; the network learns the relative weight.

Hand block = continuous text features from featurize() computed on the *templated* prompt
(captures politeness / hedge / length / directive words) + the cue-condition one-hot (the explicit
template feature). The continuous block is StandardScaler-fit per fold (heterogeneous scales); the
one-hot is left raw; the embedding is never per-dim standardized (LayerNorm handles its scale).

Target is a heavy-tailed count: trained on log1p, reported on the expm1 scale; Spearman + MAE are
the headline metrics (R2 ~ 0 expected).

Usage:
    # embeddings-only (templated vs untemplated)
    uv run python scripts/train_ffn_search_calls.py \
        --table results/search_call_prediction/frames_reconstructed_table.csv \
        --templated-emb   results/search_call_prediction/embeddings_qwen3-8b_templated.npz \
        --untemplated-emb results/search_call_prediction/embeddings_qwen3-8b_untemplated.npz \
        --out results/search_call_prediction/ffn_metrics_qwen3_8b.csv

    # embed + hand/template fusion (reuses the templated npz; GPU for the FFN)
    uv run python scripts/train_ffn_search_calls.py \
        --templated-emb results/search_call_prediction/embeddings_qwen3-8b_templated.npz \
        --with-hand --out results/search_call_prediction/ffn_metrics_qwen3_8b_hand.csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.predict_search_calls import metrics, identity_oracle, featurize  # noqa: E402


# ---------------------------------------------------------------- model

class FusionFFN(nn.Module):
    """Two-tower late-fusion MLP. Either tower may be absent (embed-only / hand-only)."""

    def __init__(self, emb_dim=None, hand_dim=None, emb_proj=256, hand_proj=64,
                 hidden=128, dropout=0.2):
        super().__init__()
        fused = 0
        if emb_dim:
            self.emb_tower = nn.Sequential(
                nn.LayerNorm(emb_dim), nn.Linear(emb_dim, emb_proj), nn.GELU(), nn.Dropout(dropout))
            fused += emb_proj
        else:
            self.emb_tower = None
        if hand_dim:
            self.hand_tower = nn.Sequential(
                nn.Linear(hand_dim, hand_proj), nn.GELU(), nn.Dropout(dropout))
            fused += hand_proj
        else:
            self.hand_tower = None
        self.trunk = nn.Sequential(
            nn.Linear(fused, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, emb, hand):
        parts = []
        if self.emb_tower is not None:
            parts.append(self.emb_tower(emb))
        if self.hand_tower is not None:
            parts.append(self.hand_tower(hand))
        return self.trunk(torch.cat(parts, dim=-1)).squeeze(-1)


def _t(a, device):
    return None if a is None else torch.as_tensor(a, dtype=torch.float32, device=device)


def train_one(emb_tr, hand_tr, ytr, emb_va, hand_va, yva, emb_dim, hand_dim, device, args):
    """Train a FusionFFN on log1p targets with early stopping on val MSE; return best model."""
    model = FusionFFN(emb_dim=emb_dim, hand_dim=hand_dim,
                      emb_proj=args.emb_proj, hand_proj=args.hand_proj).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = nn.MSELoss()
    emb_tr, hand_tr = _t(emb_tr, device), _t(hand_tr, device)
    emb_va, hand_va = _t(emb_va, device), _t(hand_va, device)
    ytr_t = torch.as_tensor(np.log1p(ytr), dtype=torch.float32, device=device)
    yva_t = torch.as_tensor(np.log1p(yva), dtype=torch.float32, device=device)

    n = len(ytr)
    best_val, best_state, bad = float("inf"), None, 0
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(model(emb_tr[idx] if emb_tr is not None else None,
                               hand_tr[idx] if hand_tr is not None else None), ytr_t[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = lossf(model(emb_va, hand_va), yva_t).item()
        if v < best_val - 1e-5:
            best_val = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict(model, emb, hand, device):
    model.eval()
    with torch.no_grad():
        p = model(_t(emb, device), _t(hand, device)).cpu().numpy()
    return np.expm1(p)


def scale_cont(cont, tr_idx):
    """Fit StandardScaler on the training rows only; return a transform closure."""
    if cont is None:
        return None
    sc = StandardScaler().fit(cont[tr_idx])
    return sc


def make_hand(cont_scaled, onehot):
    """Assemble the hand modality from the (already-scaled) continuous block + raw one-hot."""
    if cont_scaled is None and onehot is None:
        return None
    blocks = [b for b in (cont_scaled, onehot) if b is not None]
    return np.hstack(blocks).astype(np.float32)


# ---------------------------------------------------------------- CV driver

def cv_ffn(emb, cont, onehot, y, groups, device, args):
    """Out-of-fold predictions from per-fold FusionFFNs (GroupKFold on example_id).

    emb: (n,D) or None. cont: (n,c) continuous hand features or None. onehot: (n,k) or None.
    """
    oof = np.zeros_like(y, dtype=float)
    emb_dim = emb.shape[1] if emb is not None else None
    hand_dim = (0 if cont is None else cont.shape[1]) + (0 if onehot is None else onehot.shape[1])
    hand_dim = hand_dim or None
    for tr, te in GroupKFold(n_splits=args.folds).split(y, y, groups):
        gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=0)
        itr, iva = next(gss.split(tr, tr, groups[tr]))
        tr_fit, tr_val = tr[itr], tr[iva]
        sc = scale_cont(cont, tr_fit)
        def hand_of(idx):
            cs = None if cont is None else sc.transform(cont[idx])
            oh = None if onehot is None else onehot[idx]
            return make_hand(cs, oh)
        model = train_one(
            None if emb is None else emb[tr_fit], hand_of(tr_fit), y[tr_fit],
            None if emb is None else emb[tr_val], hand_of(tr_val), y[tr_val],
            emb_dim, hand_dim, device, args)
        oof[te] = predict(model, None if emb is None else emb[te], hand_of(te), device)
    return oof


def final_model(emb, cont, onehot, y, groups, device, args):
    """Train one deployable FusionFFN on all rows (group-aware val split for early stop)."""
    emb_dim = emb.shape[1] if emb is not None else None
    hand_dim = (0 if cont is None else cont.shape[1]) + (0 if onehot is None else onehot.shape[1])
    hand_dim = hand_dim or None
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_frac, random_state=0)
    itr, iva = next(gss.split(y, y, groups))
    sc = scale_cont(cont, itr)
    def hand_of(idx):
        cs = None if cont is None else sc.transform(cont[idx])
        oh = None if onehot is None else onehot[idx]
        return make_hand(cs, oh)
    m = train_one(None if emb is None else emb[itr], hand_of(itr), y[itr],
                  None if emb is None else emb[iva], hand_of(iva), y[iva],
                  emb_dim, hand_dim, device, args)
    return m, emb_dim, hand_dim


# ---------------------------------------------------------------- assembly

def load_emb(path):
    z = np.load(path, allow_pickle=True)
    return {str(t): np.asarray(e, dtype=np.float32) for t, e in zip(z["texts"], z["embeddings"])}


def emb_matrix(sub, lookup, text_col):
    texts = sub[text_col].astype(str).values
    mask = np.array([t in lookup for t in texts])
    X = np.stack([lookup[t] for t in texts[mask]]) if mask.any() else None
    return X, mask


def hand_blocks(sub):
    """Continuous text features on the templated prompt + cue-condition one-hot (raw)."""
    cont = pd.DataFrame([featurize(t) for t in sub.prompt.astype(str)]).values.astype(np.float32)
    onehot = pd.get_dummies(sub.condition).values.astype(np.float32)
    return cont, onehot


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", default="results/search_call_prediction/frames_reconstructed_table.csv")
    ap.add_argument("--templated-emb", required=True)
    ap.add_argument("--untemplated-emb", default=None)
    ap.add_argument("--with-hand", action="store_true",
                    help="add templated+hand fusion and hand_only sources (two-tower model)")
    ap.add_argument("--prior-metrics", default="results/search_call_prediction/prediction_metrics.csv")
    ap.add_argument("--out", default="results/search_call_prediction/ffn_metrics_qwen3_8b.csv")
    ap.add_argument("--ckpt-dir", default="results/search_call_prediction/ffn_ckpts")
    ap.add_argument("--emb-proj", type=int, default=256, help="embedding-tower output width")
    ap.add_argument("--hand-proj", type=int, default=64, help="hand-tower output width (the weight knob)")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default=None, help="cuda | cpu (auto-detect if unset)")
    ap.add_argument("--models", default=None, help="comma-separated subset of model names")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  with_hand={args.with_hand}  emb_proj={args.emb_proj}  hand_proj={args.hand_proj}")

    df = pd.read_csv(args.table)
    templated = load_emb(args.templated_emb)
    untemplated = load_emb(args.untemplated_emb) if args.untemplated_emb else None

    # source spec: (name, emb_lookup_or_None, text_col, use_hand)
    if args.with_hand:
        sources = [("templated", templated, "prompt", False),
                   ("templated+hand", templated, "prompt", True),
                   ("hand_only", None, "prompt", True)]
    else:
        sources = [("templated", templated, "prompt", False)]
        if untemplated is not None:
            sources.append(("untemplated", untemplated, "problem", False))

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
        cont_all, onehot_all = hand_blocks(sub)
        print(f"\n### {model_name}  (n={len(sub)}, mean={y.mean():.2f}, "
              f"oracle_rho={oracle:.3f}, prior_hand+cond={prior.get(model_name, float('nan')):.3f})")
        print(f"{'source':<16}{'spearman':>10}{'MAE':>8}{'R2':>8}{'n':>7}")
        for name, lookup, text_col, use_hand in sources:
            if lookup is not None:
                emb, mask = emb_matrix(sub, lookup, text_col)
                if mask.sum() < len(sub):
                    print(f"  [{name}] {len(sub) - int(mask.sum())} rows lack embeddings — excluded")
            else:
                emb, mask = None, np.ones(len(sub), dtype=bool)
            cont = cont_all[mask] if use_hand else None
            onehot = onehot_all[mask] if use_hand else None
            yy, gg = y[mask], groups[mask]
            oof = cv_ffn(emb, cont, onehot, yy, gg, device, args)
            m = metrics(yy, oof)
            print(f"{name:<16}{m['spearman']:>10.3f}{m['mae']:>8.2f}{m['r2']:>8.3f}{len(yy):>7}")
            rows.append({"model": model_name, "source": name, "n": len(yy),
                         "identity_oracle": oracle, "prior_hand_cond": prior.get(model_name), **m})
            fm, ed, hd = final_model(emb, cont, onehot, yy, gg, device, args)
            slug = model_name.replace("/", "_").replace(":", "_")
            torch.save({"state_dict": fm.state_dict(), "emb_dim": ed, "hand_dim": hd,
                        "emb_proj": args.emb_proj, "hand_proj": args.hand_proj,
                        "source": name, "model": model_name},
                       os.path.join(args.ckpt_dir, f"ffn_{slug}_{name.replace('+', '_')}.pt"))

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nwrote metrics to {args.out}; checkpoints in {args.ckpt_dir}/")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
