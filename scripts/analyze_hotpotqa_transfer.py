#!/usr/bin/env python3
"""HotpotQA cue-robustness TRANSFER: gemma4:31b baseline vs the 7-condition FRAMES-SFT.

The SFT was trained ONLY on FRAMES cues and never on HotpotQA, so this measures two generalization
axes at once. Of HotpotQA's 8 cues, 5 were in the SFT's training conditions (natural, elaborate,
polite, direct, query) and 3 were NOT (confident_parametric, multiturn, searchmulti) -- so the
seen/unseen split crossed with in-domain FRAMES vs out-of-domain HotpotQA gives a 2x2.

Reads the offline-graded per-row table, NOT the raw JSONs, because grade_hotpotqa_regex.py also
corrects `search_calls` for the mocked-history tool call that inflates `searchmulti` (commit
f0e71ec). Run that first:

    uv run python scripts/grade_hotpotqa_regex.py --results-root results/hotpotqa_cue_grid
    uv run python scripts/analyze_hotpotqa_transfer.py

`plain_rep2` is the run-to-run FLOOR: a configuration-identical second plain pass whose paired
difference is the empirical null every cue effect must beat. Only the baseline has one.

Write-up: docs/frames_cue_robustness_sft.md ("TRANSFER -- HotpotQA").
"""
import pandas as pd, numpy as np
from scipy.stats import wilcoxon, binomtest

df = pd.read_csv("results/hotpotqa_cue_grid_regex/per_row.csv")
BASE, SFT = "gemma4_31b", "gemma4-frames-robust-q4km_latest"
# 7-cond SFT trained on: polite, terse, natural, elaborate, query, direct  (+plain)
SEEN = ["natural", "elaborate", "polite", "direct", "query"]
UNSEEN = ["confident_parametric", "multiturn", "searchmulti"]
CUES = SEEN + UNSEEN

def stars(p): return "***" if p<1e-3 else "**" if p<1e-2 else "*" if p<5e-2 else "  "
def mcnemar(a,b):
    x=sum(1 for u,v in zip(a,b) if u and not v); y=sum(1 for u,v in zip(a,b) if v and not u)
    return 1.0 if x+y==0 else binomtest(min(x,y),x+y,0.5).pvalue

ACC = "strict" if "strict" in df.columns else [c for c in df.columns if "strict" in c][0]
SC  = "search_calls"
def cell(model, cond):
    d = df[(df.model==model) & (df.run_name==cond)].set_index("example_id")
    return d

for model, label in ((BASE,"BASELINE gemma4:31b"), (SFT,"SFT frames-robust (7-cond)")):
    pl = cell(model,"plain")
    print(f"\n{'='*92}\n### {label}   plain={pl[SC].mean():.2f} calls  zero-search={100*(pl[SC]==0).mean():.1f}%  acc={pl[ACC].mean():.3f}")
    print(f"{'cue':<22}{'Dcalls':>9}{'%plain':>9}{'p':>11}   {'Dacc pp':>9}{'p':>9}   seen?")
    res={}
    for cond in CUES + (["plain_rep2"] if len(cell(model,"plain_rep2")) else []):
        cu = cell(model,cond)
        ids = sorted(set(pl.index)&set(cu.index))
        if not ids: continue
        a=pl.loc[ids,SC].values.astype(float); b=cu.loc[ids,SC].values.astype(float); d=b-a
        p = wilcoxon(d).pvalue if np.any(d) else 1.0
        oa=pl.loc[ids,ACC].astype(bool).tolist(); ob=cu.loc[ids,ACC].astype(bool).tolist()
        tag = "FLOOR" if cond=="plain_rep2" else ("seen" if cond in SEEN else "UNSEEN")
        print(f"{cond:<22}{d.mean():>+9.2f}{100*d.mean()/a.mean():>+8.1f}%{p:>11.2e}{stars(p)}"
              f"{100*(np.mean(ob)-np.mean(oa)):>+9.1f}{mcnemar(oa,ob):>9.3f}{stars(mcnemar(oa,ob))}   {tag}")
        res[cond]=(abs(d.mean()), abs(100*d.mean()/a.mean()), p)
    for grp,name in ((SEEN,"SEEN cues"),(UNSEEN,"UNSEEN cues"),(CUES,"ALL 8")):
        v=[res[c] for c in grp if c in res]
        print(f"  {name:<12} mean|D|={np.mean([x[0] for x in v]):.2f} calls  "
              f"mean|D%|={np.mean([x[1] for x in v]):.1f}%  #sig={sum(1 for x in v if x[2]<.05)}/{len(v)}")
